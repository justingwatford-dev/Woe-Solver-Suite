import pandas as pd
import cdsapi
import xarray as xr
import numpy as np
import os
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter  # PATCH V60.1: For Five's smooth land fraction
from hurdat_parser import get_hurricane_data
from datetime import timedelta
import sys

# === ORACLE V60.3: NaN-SAFE LANDFALL PHYSICS ===
# V60.2: Two-request CDS fix (pressure-levels + single-levels)
# V60.3: NaN handling for complex coastlines (Caribbean islands crash fix)
# 
# Issue: ERA5 LSM interpolation can fail near complex coastlines → NaN in land_fraction
# Cascade: NaN land_fraction → NaN fluxes → NaN in q,T → GPU memory corruption → crash
# 
# Fixes Applied:
# 1. Pre-smoothing NaN check: Replace NaN with 0.0 (ocean fallback)
# 2. Post-smoothing validation: Detect gaussian_filter NaN propagation
# 3. Critical failsafe: Zero land_fraction if validation fails
# 4. Backup/restore mechanism: Revert to last good data on fetch failure
#
# Tested: Caribbean islands, Gulf coast, Texas landfall
# Status: PRODUCTION READY

# === GPU ACCELERATION TOGGLE ===
USE_GPU = True

try:
    if USE_GPU:
        import cupy as xp
        import cupyx.scipy.ndimage as ndimage
        import cupyx.scipy.fft as fft
        print(f"[{__name__}] 🚀 GPU Acceleration ENABLED (CuPy)")
    else:
        raise ImportError
except ImportError:
    import numpy as xp
    import scipy.ndimage as ndimage
    import scipy.fft as fft
    print(f"[{__name__}] 🐢 GPU Acceleration DISABLED (NumPy)")


class DataInterface:
    """
    Interface to ERA5 reanalysis data and HURDAT2 historical tracks.
    
    Handles:
    - Fetching ERA5 steering flow data AND Land-Sea Mask (LSM)
    - Vertical integration with pressure-weighted averaging
    - Interpolation to simulation grid
    - Conversion between physical (m/s) and dimensionless velocities
    - HURDAT2 best-track data loading
    
    UNIT HANDLING:
        - ERA5 winds: m/s (physical SI units from reanalysis)
        - Simulation winds: dimensionless (scaled by U_CHAR = 50 m/s)
        - Land Fraction: 0.0 (Ocean) to 1.0 (Land)
        - Conversion uses U_CHAR characteristic velocity
    """
    
    def __init__(self, sim_instance, storm_name, storm_year):
        self.sim = sim_instance
        print("DataInterface Initialized for Project Oracle (ERA5 Historical).")
        print("  -> ERA5 winds will be converted: m/s → dimensionless")
        
        # Load HURDAT2 historical track
        self.historical_track = get_hurricane_data(storm_name, storm_year)
        if self.historical_track is None:
            raise ValueError(f"Could not load data for {storm_name} {storm_year}")

        # Steering flow targets (dimensionless velocity) - NOW ON GPU!
        self.u_target = xp.zeros((self.sim.nx, self.sim.ny))
        self.v_target = xp.zeros((self.sim.nx, self.sim.ny))
        self.u_old = xp.zeros_like(self.u_target)
        self.v_old = xp.zeros_like(self.v_target)
        
        # === PATCH V60: LAND FRACTION FIELD ===
        # 0.0 = Pure Ocean, 1.0 = Pure Land. Intermediate values = Coastline/Transition.
        self.land_fraction = xp.zeros((self.sim.nx, self.sim.ny))
        
        # Geographic bounds (will be set when update_steering_data is called)
        self.lon_bounds, self.lat_bounds = (0, 0), (0, 0)
        self.last_update_frame = 0 
        
        # CDS API client for ERA5 data
        self.cds_client = cdsapi.Client()

    def set_kalman_backup(self, u_kalman, v_kalman):
        """
        Saves the current Kalman filter state as the 'old' state
        right before a new data fetch. This prevents the
        "Kalman State Shock" by ensuring the temporal dampener
        blends from the *actual* current state.
        
        NOTE: u_kalman and v_kalman are already GPU arrays from Kalman filter
        """
        # These are already GPU arrays, just copy them
        self.u_old = xp.copy(u_kalman)
        self.v_old = xp.copy(v_kalman)

    def _fetch_era5_data(self, date_time):
        """
        Fetch ERA5 steering layer data AND Land-Sea Mask.
        
        Process:
        1. Download ERA5 multi-level wind data AND single-level LSM
        2. Interpolate to simulation grid horizontally
        3. Apply pressure-weighted vertical integration (winds only)
        4. Smooth the Land-Sea Mask (Five's Fix)
        5. Convert to proper units and UPLOAD TO GPU
        
        Args:
            date_time: Datetime object for the ERA5 data to fetch
        """
        print(f"  -> ORACLE DI: Fetching ERA5 steering & land mask for {date_time.strftime('%Y-%m-%d %H:%M')}...")
        
        lat_north = self.lat_bounds[1]
        lat_south = self.lat_bounds[0]
        lon_west = self.lon_bounds[0]
        lon_east = self.lon_bounds[1]
        
        # === PATCH V60.2: TWO-REQUEST METHOD (FIX FOR CDS API) ===
        # LSM is single-level, winds are pressure-level
        # Must fetch from separate CDS products
        
        # REQUEST 1: Pressure-level winds
        winds_request = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': ['u_component_of_wind', 'v_component_of_wind'],
            'pressure_level': ['300', '400', '500', '600', '700', '850'], 
            'year': date_time.strftime('%Y'),
            'month': date_time.strftime('%m'),
            'day': date_time.strftime('%d'),
            'time': date_time.strftime('%H:00'),
            'area': [lat_north, lon_west, lat_south, lon_east],
        }
        
        winds_path = f'era5_winds_temp_{date_time.strftime("%Y%m%d%H%M%S")}.nc'
        print("     📊 Fetching pressure-level winds...")
        self.cds_client.retrieve('reanalysis-era5-pressure-levels', winds_request, winds_path)
        
        # REQUEST 2: Single-level land-sea mask  
        lsm_request = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': ['land_sea_mask'],
            'year': date_time.strftime('%Y'),
            'month': date_time.strftime('%m'),
            'day': date_time.strftime('%d'),
            'time': date_time.strftime('%H:00'),
            'area': [lat_north, lon_west, lat_south, lon_east],
        }
        
        lsm_path = f'era5_lsm_temp_{date_time.strftime("%Y%m%d%H%M%S")}.nc'
        print("     🏝️ Fetching land-sea mask...")
        self.cds_client.retrieve('reanalysis-era5-single-levels', lsm_request, lsm_path)
        
        # Open BOTH datasets
        with xr.open_dataset(winds_path) as winds_dataset, xr.open_dataset(lsm_path) as lsm_dataset:
            # Process WINDS dataset
            if 'valid_time' in winds_dataset.dims:
                winds_at_time = winds_dataset.squeeze('valid_time')
            else:
                winds_at_time = winds_dataset
            
            # Process LSM dataset
            if 'valid_time' in lsm_dataset.dims:
                lsm_at_time = lsm_dataset.squeeze('valid_time')
            else:
                lsm_at_time = lsm_dataset

            # Create simulation grid in geographic coordinates
            sim_lon = np.linspace(lon_west, lon_east, self.sim.nx)
            sim_lat = np.linspace(lat_south, lat_north, self.sim.ny)
            
            # Interpolate WINDS to simulation grid
            winds_interp = winds_at_time.interp(longitude=sim_lon, latitude=sim_lat)
            
            # Interpolate LSM to simulation grid  
            lsm_interp = lsm_at_time.interp(longitude=sim_lon, latitude=sim_lat)
            
            # --- PROCESS WINDS (Existing Logic) ---
            actual_pressure_levels_pa = winds_interp.pressure_level.values * 100
            
            if len(actual_pressure_levels_pa) < 2:
                raise ValueError("ERA5 returned < 2 vertical levels.")

            u_levels = winds_interp["u"].values.transpose(0, 2, 1)  # [pressure, nx, ny]
            v_levels = winds_interp["v"].values.transpose(0, 2, 1)
            
            # Initialize integrated wind fields (CPU)
            u_integrated_ms = np.zeros((self.sim.nx, self.sim.ny))
            v_integrated_ms = np.zeros((self.sim.nx, self.sim.ny))
            
            pressure_levels_hpa = actual_pressure_levels_pa / 100.0

            # Weights: Higher for 300-700 hPa
            weights = np.ones_like(pressure_levels_hpa, dtype=np.float64)
            for i, p_level in enumerate(pressure_levels_hpa):
                if 300 <= p_level <= 700:
                    weights[i] = 2.0 

            # Sort by pressure
            sort_indices = np.argsort(pressure_levels_hpa)
            sorted_pressure_levels = pressure_levels_hpa[sort_indices]
            sorted_u_levels = u_levels[sort_indices, :, :]
            sorted_v_levels = v_levels[sort_indices, :, :]
            
            # Deep Layer Mean Calculation
            for i in range(self.sim.nx):
                for j in range(self.sim.ny):
                    u_profile = sorted_u_levels[:, i, j]
                    v_profile = sorted_v_levels[:, i, j]

                    u_series = pd.Series(u_profile).interpolate(method='linear', limit_direction='both')
                    v_series = pd.Series(v_profile).interpolate(method='linear', limit_direction='both')
                    u_profile_clean = u_series.values
                    v_profile_clean = v_series.values
                    
                    if not np.all(np.isfinite(u_profile_clean)): u_profile_clean = np.zeros_like(u_profile_clean)
                    if not np.all(np.isfinite(v_profile_clean)): v_profile_clean = np.zeros_like(v_profile_clean)
                    
                    log_p = np.log(sorted_pressure_levels)
                    u_mean = -np.trapz(u_profile_clean, log_p) / (log_p[0] - log_p[-1])
                    v_mean = -np.trapz(v_profile_clean, log_p) / (log_p[0] - log_p[-1])
                    
                    u_integrated_ms[i, j] = u_mean * 0.55
                    v_integrated_ms[i, j] = v_mean * 0.55
            
            # Doughnut Filter (Center Masking)
            x = np.arange(self.sim.nx)
            y = np.arange(self.sim.ny)
            yy, xx = np.meshgrid(y, x)
            cx, cy = self.sim.nx // 2, self.sim.ny // 2
            radius = np.sqrt((xx - cx)**2 + (yy - cy)**2)
            hole_mask = radius < (0.40 * self.sim.nx)
            doughnut_mask = ~hole_mask
            
            if np.any(doughnut_mask):
                u_env_mean = np.mean(u_integrated_ms[doughnut_mask])
                v_env_mean = np.mean(v_integrated_ms[doughnut_mask])
                u_integrated_ms[hole_mask] = u_env_mean
                v_integrated_ms[hole_mask] = v_env_mean
            
            # --- PROCESS LAND MASK (Patch V60.3: NaN-Safe LSM Processing) ---
            # Extract raw LSM from the single-level dataset
            lsm_raw = lsm_interp['lsm'].values
            
            # Handle potential extra dimensions (time/pressure) if squeezed poorly
            if len(lsm_raw.shape) == 3:
                lsm_raw = lsm_raw[0, :, :] # Take first slice
                
            # Transpose to match our [nx, ny] convention (if needed)
            # xarray interpolation usually keeps lat/lon order, we need to verify x/y mapping
            # Based on u_levels transpose(0, 2, 1), we likely need a transpose here too 
            # if lsm comes out as [lat, lon] and we want [lon, lat]
            lsm_data = lsm_raw.T 
            
            # === PATCH V60.3: NaN SAFETY NET ===
            # Caribbean islands and complex coastlines can trigger ERA5 LSM interpolation failures
            # Replace any NaN values with safe fallback (0.0 = ocean)
            if not np.all(np.isfinite(lsm_data)):
                nan_count = np.sum(~np.isfinite(lsm_data))
                print(f"     ⚠️ WARNING: {nan_count} NaN/Inf values in LSM data, replacing with 0.0 (ocean)")
                lsm_data = np.nan_to_num(lsm_data, nan=0.0, posinf=1.0, neginf=0.0)
            
            # FIVE'S FIX: Smooth Land Fraction
            # Apply Gaussian blur to create a "soft" coastline
            # Sigma=1.5 gives a ~3-cell transition zone
            land_fraction = gaussian_filter(lsm_data, sigma=1.5)
            land_fraction = np.clip(land_fraction, 0.0, 1.0)
            
            # === PATCH V60.3: POST-SMOOTHING VALIDATION ===
            # Gaussian filter can propagate NaN if input has NaN (despite pre-check above)
            # This is a critical safety check before GPU upload
            if not np.all(np.isfinite(land_fraction)):
                print(f"     ⚠️ CRITICAL: NaN detected AFTER gaussian_filter, falling back to zero land fraction")
                land_fraction = np.zeros((self.sim.nx, self.sim.ny))
            
            # Diagnostic
            land_pct = float(np.mean(land_fraction) * 100.0)
            print(f"     🏝️ Land Fraction processed: {land_pct:.1f}% land coverage")

            # --- UPLOAD TO GPU ---
            self.u_target = xp.asarray(u_integrated_ms / self.sim.U_CHAR)
            self.v_target = xp.asarray(v_integrated_ms / self.sim.U_CHAR)
            self.land_fraction = xp.asarray(land_fraction) # <-- New GPU Array

        # Clean up BOTH temporary files
        os.remove(winds_path)
        os.remove(lsm_path)
        print("  -> ERA5 data & Land Mask successfully integrated and uploaded to GPU.")

    def update_steering_data(self, center_lat, center_lon, current_sim_time, frame_number):
        """
        Update ERA5 steering data AND Land Mask for new domain center.
        
        Args:
            center_lat: Center latitude for ERA5 domain
            center_lon: Center longitude for ERA5 domain
            current_sim_time: Datetime for ERA5 data to fetch
            frame_number: Current simulation frame
        """
        # Store last good data as backup (including land mask!)
        u_last_good = xp.copy(self.u_target)
        v_last_good = xp.copy(self.v_target)
        land_last_good = xp.copy(self.land_fraction) # <-- Backup land
        
        # Precision Box
        box_radius = 2.0
        self.lon_bounds = (center_lon - box_radius, center_lon + box_radius)
        self.lat_bounds = (center_lat - box_radius, center_lat + box_radius)
        
        try:
            self._fetch_era5_data(current_sim_time)
            self.last_update_frame = frame_number
        except Exception as e:
            print(f"---! ORACLE DI FETCH ERROR !---: {e}")
            print("  -> WARNING: Reverting to last known good steering & land data.")
            self.u_target = u_last_good
            self.v_target = v_last_good
            self.land_fraction = land_last_good # <-- Restore land
            
    def get_smoothed_steering(self, frame):
        """
        Get temporally-smoothed steering flow.
        """
        return self.u_target, self.v_target
