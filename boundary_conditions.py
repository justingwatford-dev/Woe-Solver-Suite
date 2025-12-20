import numpy as np
import sys

# === GPU ACCELERATION TOGGLE ===
USE_GPU = True

try:
    if USE_GPU:
        import cupy as xp
        import cupyx.scipy.ndimage as ndimage
        import cupyx.scipy.fft as fft
        print(f"[{__name__}] GPU Acceleration ENABLED (CuPy)")
    else:
        raise ImportError
except ImportError:
    import numpy as xp
    import scipy.ndimage as ndimage
    import scipy.fft as fft
    print(f"[{__name__}] GPU Acceleration DISABLED (NumPy)")

# Import unit conversion utilities
from utils import ms_to_kts

class BoundaryConditions:
    """
    Ocean-atmosphere boundary layer physics.
    
    Handles:
    - Surface flux calculations (momentum, moisture, heat)
    - Saturation vapor pressure computations
    - Wind-dependent drag coefficients
    - Thermodynamic regulators to prevent unrealistic intensification
    
    UNIT HANDLING:
        - Input velocities: dimensionless (from simulation)
        - Wind speed for physics: m/s (converted explicitly)
        - Temperature: Celsius internally, Kelvin for calculations
        - Pressure: Pascals
        - Fluxes: SI units (kg/(m²·s) for moisture, W/m² for heat)
    """
    
    def __init__(self, sim_instance):
        self.sim = sim_instance
        
        # === PHYSICAL CONSTANTS ===
        # === ENSEMBLE: Progressive Equilibrium - Sensible Flux ===
        # Heat exchange coefficient (dimensionless)
        # C_h = 0.0011 provides ~100 W/m² sensible heat (observed reality)
        # We set a *base* value which will be throttled by the Governor Protocol
        self.C_h_base = 0.0018  # ENSEMBLE FIX (per Grok): "Major Cruise" sensible heat
        
        # Air density at sea level (kg/m³)
        self.air_density = 1.225
        
        print("BoundaryConditions initialized with SI-compliant flux calculations.")

    def calculate_saturation_humidity(self, T_celsius, P_pascals=101325):
        """
        Calculate saturation specific humidity using Clausius-Clapeyron relation.
        
        Uses Bolton's (1980) approximation for saturation vapor pressure.
        
        Args:
            T_celsius: Temperature in degrees Celsius
            P_pascals: Atmospheric pressure in Pascals (default: sea level)
            
        Returns:
            Saturation specific humidity (kg water / kg air, dimensionless)
            
        References:
            Bolton, D. (1980). The computation of equivalent potential temperature.
            Monthly Weather Review, 108(7), 1046-1053.
        """
        T_kelvin = T_celsius + 273.15
        
        # Prevent overflow in exponential (clip to reasonable temperature range)
        # ENSEMBLE FIX (per Grok): Use tighter clip for hurricane range
        temp_term = xp.clip(T_celsius, -40, 40)
        
        # Bolton's formula for saturation vapor pressure (Pa)
        # ENSEMBLE: Corrected formula using T_celsius + 243.5
        e_sat = 611.2 * xp.exp(17.67 * temp_term / (temp_term + 243.5 + 1e-9))
        
        # Convert to specific humidity using mixing ratio approximation
        # q_sat = 0.622 * e_sat / (P - 0.378 * e_sat)
        q_sat = (0.622 * e_sat) / (P_pascals - 0.378 * e_sat + 1e-9)
        
        return q_sat

    def apply_surface_fluxes(self, q, T, q_flux_boost_factor):
        """
        Calculate and apply ocean-atmosphere surface fluxes.
        
        Computes:
        - Moisture flux (evaporation) driven by humidity deficit
        - Sensible heat flux driven by temperature difference
        - Wind-dependent bulk transfer coefficients
        - Thermodynamic regulator to prevent unrealistic intensification
        
        This is the CRITICAL ocean-atmosphere coupling that fuels hurricanes.
        
        Args:
            q: 3D specific humidity field (kg/kg)
            T: 3D temperature field (°C)
            q_flux_boost_factor: Multiplier for moisture flux (default: 1.0)
            
        Returns:
            Tuple: (q_updated, T_updated, mean_q_flux, mean_h_flux)
                - q_updated: Updated humidity field
                - T_updated: Updated temperature field
                - mean_q_flux: Mean moisture flux (kg/(m²·s))
                - mean_h_flux: Mean heat flux (W/m²)
        """
        # === INPUT VALIDATION ===
        if not xp.all(xp.isfinite(self.sim.u)) or not xp.all(xp.isfinite(q)) or not xp.all(xp.isfinite(T)):
            print("    --- Boundary Condition Warning: Invalid input data (NaN/Inf) ---")
            return q, T, 0, 0, 1.0  # ENSEMBLE: Return dampening_factor=1.0

        # === EXTRACT SURFACE FIELDS ===
        q_air_surface = q[:, :, 0]
        T_air_surface = T[:, :, 0]

        # === "Global Kill-Switch" Brake ===
        # The brake must see the absolute fastest wind in the
        # entire 3D grid, matching the logic in storm_tracker.py.
        
        # 1. Get the 3D horizontal wind speed (dimensionless)
        wind_speed_dim_3d = xp.sqrt(self.sim.u**2 + self.sim.v**2)
        
        # 2. Convert to physical (m/s)
        wind_speed_ms_3d = wind_speed_dim_3d * self.sim.U_CHAR
        
        # 3. Find the single GLOBAL maximum wind speed (a scalar)
        #    This is the "authoritative" wind speed for the brakes.
        global_max_wind_ms = float(xp.max(wind_speed_ms_3d))
        
        # 4. We still need a 2D map of wind speeds for the *flux* calculation
        #    (This is the wind at the surface, k=0)
        wind_speed_ms_surface = wind_speed_ms_3d[:, :, 0]
        # === END "Global Kill-Switch" Brake ===
        
        # === ENSEMBLE: The "Governor Protocol" ===
        # This throttle *should* use the 2D surface wind map, as it
        # represents the local flux.
        #
        # --- *** ENSEMBLE V5: AGGRESSIVE GOVERNOR *** ---
        # The previous [1.0, 0.7, 0.6] ramp was too generous and
        # allowed the storm to reach hypercane speeds before the
        # WISDOM brake could engage. This new ramp cuts fuel
        # much earlier to force stabilization at H5 speeds.
        governor_throttle = xp.interp(
            xp.ascontiguousarray(wind_speed_ms_surface),
            xp.array([0.0, 35.0, 70.0, 90.0]),  # Wind speed (m/s)
            xp.array([1.0, 1.0,  0.5,  0.2])    # Throttle factor
        )
        # --- *** END ENSEMBLE V5 FIX *** ---
        
        # 2. Apply the throttle to the "Grok" parameters
        q_flux_boost_dynamic = q_flux_boost_factor * governor_throttle
        C_h_dynamic = self.C_h_base * governor_throttle
        # === END GOVERNOR PROTOCOL ===
        
        # === OCEAN SURFACE CONDITIONS ===
        T_ocean_surface = self.sim.SST  # Sea Surface Temperature (°C)
        q_sat_ocean = self.calculate_saturation_humidity(T_ocean_surface)
        
        # === BULK TRANSFER COEFFICIENTS ===
        # Drag coefficient increases with wind speed
        C_d_dynamic = xp.where(
            wind_speed_ms_surface < 20,  # m/s threshold
            1.6e-3,                      # Lower winds
            2.0e-3                       # Higher winds (increased surface roughness)
        )
        
        # === ENSEMBLE: Progressive Equilibrium - Moisture Flux (The Fuel) ===
        q_flux = (
            C_d_dynamic * self.air_density * wind_speed_ms_surface * (q_sat_ocean - q_air_surface)
        ) * q_flux_boost_dynamic
        
        # === ENSEMBLE: Progressive Equilibrium - Sensible Heat Flux ===
        h_flux = (
            C_h_dynamic * self.air_density * self.sim.c_p * wind_speed_ms_surface * (T_ocean_surface - T_air_surface)
        )
        
        # === ENSEMBLE: Progressive Equilibrium - WISDOM Regulator (The Safety Net) ===
        # === V50.4 PATCH 2: AGGRESSIVE WISDOM RAMP (Opus's Fix) ===
        # Problem: Original 90-110 m/s range allowed 202 kt hypercanes
        # Solution: Start damping earlier (80 m/s = 155 kts) and ramp more aggressively
        
        # === "Global Kill-Switch" FIX ===
        # The dampening_factor is now a single SCALAR value based
        # on the GLOBAL max wind speed.
        wisdom_threshold_ms = 80.0  # Start braking at ~155 kts (mid-H5) - LOWERED from 90
        wisdom_cap_ms = 95.0        # Force 0% flux at ~185 kts (realistic max) - LOWERED from 110

        # Linearly ramp from 1.0 (full flux) down to 0.0 (no flux)
        dampening_factor_scalar = xp.interp(
            xp.ascontiguousarray(xp.array([global_max_wind_ms])),
            xp.array([wisdom_threshold_ms, wisdom_cap_ms]),
            xp.array([1.0, 0.0])
        )[0]
        # === END "Global Kill-Switch" FIX ===
        
        # Clip values outside this range
        dampening_factor = float(xp.clip(dampening_factor_scalar, 0.0, 1.0))
        
        # === V51 PATCH 1: RATE-LIMITED WISDOM DIAGNOSTIC ===
        # Problem in V50.4: Logged EVERY frame when active (console spam)
        # Solution: Log only every 100 frames to keep console readable
        # Convert global_max_wind_ms to knots for readability
        max_wind_kts = ms_to_kts(global_max_wind_ms)
        if dampening_factor < 0.99 and max_wind_kts > 150:
            # Initialize counter if it doesn't exist
            if not hasattr(self, '_wisdom_log_counter'):
                self._wisdom_log_counter = 0
            self._wisdom_log_counter += 1
            
            # Only print every 100th activation
            if self._wisdom_log_counter % 100 == 0:
                print(f"    🛑 V51 WISDOM ACTIVE: Wind={max_wind_kts:.0f}kts, Dampening={dampening_factor:.2f} ({(1-dampening_factor)*100:.0f}% flux cut) [x{self._wisdom_log_counter}]")
        # === END V51 PATCH 1 ===

        # Apply the single scalar dampening factor to the entire flux field
        q_flux *= dampening_factor
        h_flux *= dampening_factor
        
        # === FLUX VALIDATION ===
        if not xp.all(xp.isfinite(q_flux)) or not xp.all(xp.isfinite(h_flux)):
            print("    --- Boundary Condition Warning: NaN flux detected ---")
            return q, T, 0, 0, 1.0  # ENSEMBLE: Return dampening_factor=1.0

        # === APPLY FLUXES TO SURFACE LAYER ===
        dt_physical_s = self.sim.dt_solver * self.sim.T_CHAR
        dz_physical_m = self.sim.domain_scaler.dimensionless_to_physical_z(self.sim.dz)

        q[:, :, 0] += q_flux * dt_physical_s / (dz_physical_m * self.air_density)
        T[:, :, 0] += h_flux * dt_physical_s / (dz_physical_m * self.air_density * self.sim.c_p)
        
        # Return the scalar dampening_factor for diagnostics
        return q, T, float(xp.mean(q_flux)), float(xp.mean(h_flux)), dampening_factor

    def calculate_surface_drag(self, u_surface, v_surface):
        """
        Calculate surface drag stress components.
        
        Computes the momentum flux (drag stress) at the ocean surface due to
        wind friction.
        
        Args:
            u_surface: Dimensionless u-velocity at surface (lowest level)
            v_surface: Dimensionless v-velocity at surface (lowest level)
            
        Returns:
            Tuple: (drag_stress_x, drag_stress_y)
                - drag_stress_x: X-component of drag stress (Pa = N/m²)
                - drag_stress_y: Y-component of drag stress (Pa = N/m²)
        """
        # === CONVERT WIND SPEED TO PHYSICAL UNITS (m/s) ===
        # Calculate dimensionless wind speed magnitude
        wind_speed_dim = xp.sqrt(u_surface**2 + v_surface**2)
        
        # Convert to physical velocity (m/s) using U_CHAR
        wind_speed_ms = wind_speed_dim * self.sim.U_CHAR
        
        # === BULK TRANSFER COEFFICIENTS ===
        # Drag coefficient increases with wind speed
        C_d_dynamic = xp.where(
            wind_speed_ms < 20,  # m/s threshold
            1.6e-3,              # Lower winds
            2.0e-3               # Higher winds (increased surface roughness)
        )
        
        # === CALCULATE DRAG STRESS ===
        # τ = C_d * ρ * |V| * V  [Pa = N/m²]
        drag_stress_x = C_d_dynamic * self.air_density * wind_speed_ms * u_surface * self.sim.U_CHAR
        drag_stress_y = C_d_dynamic * self.air_density * wind_speed_ms * v_surface * self.sim.U_CHAR
        
        return drag_stress_x, drag_stress_y