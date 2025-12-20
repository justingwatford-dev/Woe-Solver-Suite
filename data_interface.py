import pandas as pd
import cdsapi
import xarray as xr
import numpy as np
import os
from scipy.interpolate import CubicSpline
from hurdat_parser import get_hurricane_data
from datetime import timedelta
import sys

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
    - Fetching ERA5 steering flow data from CDS API
    - Vertical integration with pressure-weighted averaging
    - Interpolation to simulation grid
    - Conversion between physical (m/s) and dimensionless velocities
    - HURDAT2 best-track data loading
    
    UNIT HANDLING:
        - ERA5 winds: m/s (physical SI units from reanalysis)
        - Simulation winds: dimensionless (scaled by U_CHAR = 50 m/s)
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
        
        # Geographic bounds (will be set when update_steering_data is called)
        self.lon_bounds, self.lat_bounds = (0, 0), (0, 0)
        self.last_update_frame = 0 # <-- ADD THIS
        
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
        Fetch ERA5 steering layer data and convert to dimensionless velocities.
        
        Process:
        1. Download ERA5 multi-level wind data (500-850 hPa)
        2. Interpolate to simulation grid horizontally
        3. Apply pressure-weighted vertical integration
        4. Convert from m/s to dimensionless simulation units
        5. UPLOAD TO GPU for simulation use
        
        Args:
            date_time: Datetime object for the ERA5 data to fetch
        """
        print(f"  -> ORACLE DI: Fetching ERA5 steering layer data for {date_time.strftime('%Y-%m-%d %H:%M')}...")
        
        lat_north = self.lat_bounds[1]
        lat_south = self.lat_bounds[0]
        lon_west = self.lon_bounds[0]
        lon_east = self.lon_bounds[1]
        
        era5_request = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': ['u_component_of_wind', 'v_component_of_wind'],
            'pressure_level': ['300', '400', '500', '600', '700', '850'],  # PATCH V28: Added 300, 400 hPa
            'year': date_time.strftime('%Y'),
            'month': date_time.strftime('%m'),
            'day': date_time.strftime('%d'),
            'time': date_time.strftime('%H:00'),
            'area': [lat_north, lon_west, lat_south, lon_east],
        }
        
        # === FIX: Use a unique filename to avoid permission errors ===
        #download_path = 'era5_data.nc'
        download_path = f'era5_data_temp_{date_time.strftime("%Y%m%d%H%M%S")}.nc'
        # -- END FIX ---
        self.cds_client.retrieve('reanalysis-era5-pressure-levels', era5_request, download_path)
        
        with xr.open_dataset(download_path) as era5_dataset:
            # Handle different time dimension names
            if 'valid_time' in era5_dataset.dims:
                era5_at_time = era5_dataset.squeeze('valid_time')
            else:
                era5_at_time = era5_dataset

            # Create simulation grid in geographic coordinates
            sim_lon = np.linspace(lon_west, lon_east, self.sim.nx)
            sim_lat = np.linspace(lat_south, lat_north, self.sim.ny)
            
            # Interpolate ERA5 data to simulation grid
            interpolated_data = era5_at_time.interp(longitude=sim_lon, latitude=sim_lat)
            
            # Get pressure levels (convert from hPa to Pa)
            actual_pressure_levels_pa = interpolated_data.pressure_level.values * 100
            
            if len(actual_pressure_levels_pa) < 2:
                raise ValueError("ERA5 returned < 2 vertical levels.")

            # Get wind components (u, v) at all pressure levels
            # ERA5 winds are in m/s (SI units)
            u_levels = interpolated_data['u'].values.transpose(0, 2, 1)  # [pressure, nx, ny]
            v_levels = interpolated_data['v'].values.transpose(0, 2, 1)
            
            # Initialize integrated wind fields (will be in m/s after integration)
            # Keep these on CPU during processing since we're using scipy/pandas
            u_integrated_ms = np.zeros((self.sim.nx, self.sim.ny))
            v_integrated_ms = np.zeros((self.sim.nx, self.sim.ny))
            
            pressure_levels_hpa = actual_pressure_levels_pa / 100.0

            # === GHOST GUIDANCE TUNING ===
            # Define pressure-level weights
            # Higher weights for 300-700 hPa (mid-troposphere steering layer)
            weights = np.ones_like(pressure_levels_hpa, dtype=np.float64)
            for i, p_level in enumerate(pressure_levels_hpa):
                if 300 <= p_level <= 700:
                    weights[i] = 2.0  # Double weight for key steering layers

            # Sort by pressure (for proper vertical integration)
            sort_indices = np.argsort(pressure_levels_hpa)
            sorted_pressure_levels = pressure_levels_hpa[sort_indices]
            sorted_u_levels = u_levels[sort_indices, :, :]
            sorted_v_levels = v_levels[sort_indices, :, :]
            sorted_weights = weights[sort_indices]
            
            # === PATCH V28: ROBUST DEEP LAYER MEAN ===
            # Replaced complex spline interpolation with standard meteorological
            # mass-weighted mean (trapezoidal integration on log-pressure).
            # This is the operational method used by NHC and research centers.
            
            # Process each grid point
            for i in range(self.sim.nx):
                for j in range(self.sim.ny):
                    u_profile = sorted_u_levels[:, i, j]
                    v_profile = sorted_v_levels[:, i, j]

                    # Handle NaN values
                    u_series = pd.Series(u_profile).interpolate(method='linear', limit_direction='both')
                    v_series = pd.Series(v_profile).interpolate(method='linear', limit_direction='both')
                    u_profile_clean = u_series.values
                    v_profile_clean = v_series.values
                    
                    # Fallback for all-NaN profiles
                    if not np.all(np.isfinite(u_profile_clean)):
                        u_profile_clean = np.zeros_like(u_profile_clean)
                    if not np.all(np.isfinite(v_profile_clean)):
                        v_profile_clean = np.zeros_like(v_profile_clean)
                    
                    # Mass-weighted integration using log-pressure
                    # Log-pressure spacing approximates mass between levels
                    log_p = np.log(sorted_pressure_levels)
                    
                    # Trapezoidal integration (negative because p decreases with height)
                    u_mean = -np.trapz(u_profile_clean, log_p) / (log_p[0] - log_p[-1])
                    v_mean = -np.trapz(v_profile_clean, log_p) / (log_p[0] - log_p[-1])
                    
                    # === MAGNITUDE SCALER (0.55) ===
                    # ERA5 synoptic winds are often 1.5-2× too strong for hurricane steering
                    # This empirical scaler aligns with observed translation speeds
                    u_integrated_ms[i, j] = u_mean * 0.55
                    v_integrated_ms[i, j] = v_mean * 0.55
            
            # === PATCH V26: THE DOUGHNUT FILTER ===
            # Mask out the storm center (inner 30%) and replace with environmental mean.
            # This prevents the "Self-Steering Ghost" feedback loop where the storm's
            # own circulation contaminates the environmental steering calculation.
            
            # Create coordinate grid
            x = np.arange(self.sim.nx)
            y = np.arange(self.sim.ny)
            yy, xx = np.meshgrid(y, x)  # Note meshgrid ordering
            
            # Calculate distance from center
            cx, cy = self.sim.nx // 2, self.sim.ny // 2
            radius = np.sqrt((xx - cx)**2 + (yy - cy)**2)
            
            # Define Doughnut: Inner 40% is the hole (storm vortex)
            # PATCH V28: Increased from 30% to 40% because box is smaller (2° not 3°)
            hole_mask = radius < (0.40 * self.sim.nx)
            doughnut_mask = ~hole_mask
            
            # Calculate mean flow of the environment (Doughnut only)
            if np.any(doughnut_mask):
                u_env_mean = np.mean(u_integrated_ms[doughnut_mask])
                v_env_mean = np.mean(v_integrated_ms[doughnut_mask])
                
                # Overwrite the hole with the environmental mean
                # This gives the storm a uniform "background flow" to follow
                u_integrated_ms[hole_mask] = u_env_mean
                v_integrated_ms[hole_mask] = v_env_mean
                print(f"     🍩 Doughnut Filter: Core masked, Env=({u_env_mean:.2f}, {v_env_mean:.2f}) m/s")
            # === END PATCH V26 ===
            
            # === CRITICAL: CONVERT FROM m/s TO DIMENSIONLESS AND UPLOAD TO GPU ===
            # Convert using the characteristic velocity U_CHAR, then upload to GPU
            self.u_target = xp.asarray(u_integrated_ms / self.sim.U_CHAR)
            self.v_target = xp.asarray(v_integrated_ms / self.sim.U_CHAR)

        # Clean up temporary file
        os.remove(download_path)
        print("  -> ERA5 steering layer successfully integrated, converted, and uploaded to GPU.")
        
        # Diagnostic: print typical steering magnitude
        steering_magnitude_ms = np.mean(np.sqrt(u_integrated_ms**2 + v_integrated_ms**2))
        # Download a small amount of data for diagnostics
        steering_magnitude_dim = float(xp.mean(xp.sqrt(self.u_target**2 + self.v_target**2)))
        print(f"     Average steering: {steering_magnitude_ms:.2f} m/s → {steering_magnitude_dim:.6f} dimensionless")

    def update_steering_data(self, center_lat, center_lon, current_sim_time, frame_number):
        """
        Update ERA5 steering data for new domain center.
        
        PATCH V25: Fetches new ERA5 data centered on (center_lat, center_lon) with 6° box.
        Reduced from 15° to focus on local environmental steering and avoid:
        - Distant synoptic features (subtropical ridge, westerlies)
        - Contamination from the real hurricane present in ERA5 reanalysis
        
        If fetch fails, reverts to last known good data.
        
        Args:
            center_lat: Center latitude for ERA5 domain
            center_lon: Center longitude for ERA5 domain
            current_sim_time: Datetime for ERA5 data to fetch
            frame_number: Current simulation frame
        """
        # Store last good data as backup in case of fetch failure (keep on GPU)
        u_last_good = xp.copy(self.u_target)
        v_last_good = xp.copy(self.v_target)
        # self.u_old and self.v_old are now set EXTERNALLY
        # by set_kalman_backup()

        # === PATCH V28: PRECISION BOX ===
        # Reduced from 3.0° to 2.0° (4° total box)
        # This captures LOCAL environment without synoptic noise
        # WAS V25: ±3.0° (6° box) - Still too large
        # NOW V28: ±2.0° (4° box) - Optimal for hurricane steering
        box_radius = 2.0
        self.lon_bounds = (center_lon - box_radius, center_lon + box_radius)
        self.lat_bounds = (center_lat - box_radius, center_lat + box_radius)
        
        try:
            self._fetch_era5_data(current_sim_time)
            self.last_update_frame = frame_number # <-- ADD THIS
        except Exception as e:
            print(f"---! ORACLE DI FETCH ERROR !---: {e}")
            print("  -> WARNING: Reverting to last known good steering data.")
            self.u_target = u_last_good
            self.v_target = v_last_good
            
    def get_smoothed_steering(self, frame):
        """
        Get temporally-smoothed steering flow.
        
        Smoothly transitions between old and new ERA5 data over 100 frames
        after each update to avoid sudden jumps.
        
        Args:
            frame: Current simulation frame
            
        Returns:
            (u_smooth, v_smooth): Smoothed steering flow in dimensionless units (GPU arrays)
        """
        # === ENSEMBLE: "Guidance Dampener" ===
        # This smoothing logic is now handled in _apply_guidance_force
        # inside World_woe_main_adaptive.py.
        # We can now return the raw target data directly (as GPU arrays).
        return self.u_target, self.v_target