import numpy as np
import sys

# === ORACLE V61: ZOMBIE STORM FIX (LANDFALL PHYSICS) ===
# Ensemble contributions:
# - Gemini: "Flywheel Effect" Diagnosis, Momentum Anchor Decoupling, Terrain Roughness
# - Five: Nonlinear scaling refinement (land_fraction**2), Coastal coherence protection
# - Claude: Integration, Consensus Building, Core Sample Diagnostic
#
# Changes in this file:
# - Patch V61: Momentum Anchor weakened by land_fraction**2 (line ~1260)
# - Patch V61.1: Terrain Roughness adds local viscosity over land (line ~1350)
# - Diagnostic: "Core Sample" print to verify mask orientation (line ~1160)
#
# Status: READY FOR "ZOMBIE HARVEY" TEST

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

import random
import os
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass # CPU only function

from core_solver import CoreSolver
from boundary_conditions import BoundaryConditions
from data_interface import DataInterface
from storm_tracker import StormTracker
from visualizer import Visualizer
from amr_handler import AMRHandler
from kalman_filter import KalmanFilter

# === ORACLE V4 IMPORTS ===
from oracle_memory import OracleMemory
from oracle_adaptive import AdaptiveOracle

# === UNIT CONVERSION IMPORTS ===
from utils import (
    kts_to_ms, ms_to_kts,
    km_to_m, m_to_km,
    haversine_distance_km,
    DomainScaler
)

# === LOGGING SETUP ===
try:
    from oracle_logger import setup_global_logger, log_info
    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False
    def log_info(msg):
        print(f"[INFO] {msg}")  # Safe fallback - no recursion!
    print("WARNING: oracle_logger.py not found - logging to console only")

class Simulation3D:
    def __init__(self, nx=128, ny=128, nz=64, num_frames=10000, mu=0.15, initial_wind_kts=50.00,
                 storm_name='HUGO', storm_year=1989,
                 oracle_mode='adaptive', oracle_params_file=None, collect_memory=True, random_seed=1989,
                 simulation_mode='reconstruction', ablation_mode=False):
        """
        Hurricane simulation with Oracle V4 adaptive learning.
        
        *** UNIT SYSTEM SPECIFICATION ***
        ALL INTERNAL PHYSICS OPERATES IN SI UNITS:
            - Velocity: m/s
            - Distance: m
            - Pressure: Pa
            - Temperature: K
            - Time: s
        
        EXTERNAL INTERFACES (conversions at boundaries):
            - Input wind speed: knots → converted to m/s
            - Output wind speed: m/s → converted to knots for logging
            - Track errors: computed in km or degrees
        
        DOMAIN GEOMETRY:
            - Dimensionless domain: [0, lx] × [0, ly] × [0, lz]
            - Physical domain: 2000 km × 2000 km × 20 km
            - Velocities in dimensionless units are "grid units per timestep"
        
        Args:
            storm_name: Name of the hurricane (e.g., 'HUGO', 'ANDREW', 'KATRINA')
            storm_year: Year of the hurricane (e.g., 1989, 1992, 2005)
            oracle_mode: 'fixed' (V3), 'adaptive' (V4), or 'learning' (data collection)
            oracle_params_file: Path to learned parameters (for adaptive mode)
            collect_memory: Whether to record Oracle decisions for learning
            random_seed: Random seed for reproducibility (default: 1989)
        """
        
        # === RANDOM SEED INITIALIZATION ===
        if random_seed is not None:
            # Use legacy RandomState for exact reproducibility with training data
            legacy_rng = np.random.RandomState(random_seed)
            np.random.set_state(legacy_rng.get_state())
            random.seed(random_seed)
            
            # === GPU FIX: Seed CuPy as well ===
            if USE_GPU:
                xp.random.seed(random_seed)
        
        # === STORM PARAMETERS ===

        # === V55: SIMULATION MODE & ABLATION ===
        self.simulation_mode = simulation_mode
        self.ablation_mode = ablation_mode
        
        # Validate simulation mode
        valid_modes = ['free_run', 'assisted', 'reconstruction']
        if self.simulation_mode not in valid_modes:
            raise ValueError(f"Invalid simulation_mode: {simulation_mode}. Must be one of {valid_modes}")
        
        # Log configuration
        if self.ablation_mode:
            log_info("⚠️ ABLATION MODE ACTIVE: Heuristic nudges (Stall Breaker, Ghost Nudge) DISABLED")
        log_info(f"📋 SIMULATION MODE: {self.simulation_mode.upper()}")
        
        if self.simulation_mode == 'reconstruction':
            log_info("   → Mode C: Historical anchoring + full guidance")
        elif self.simulation_mode == 'assisted':
            log_info("   → Mode B: Oracle guidance without anchoring")
        elif self.simulation_mode == 'free_run':
            log_info("   → Mode A: Pure forecast, no assistance")
        
        # === V55: HELPER STATISTICS TRACKING ===
        self.stall_breaker_count = 0
        self.ghost_nudge_count = 0
        self.ghost_nudge_history = []
        self.guidance_force_history = []

        self.storm_name = storm_name
        self.storm_year = storm_year
        
        # === SETUP LOGGING ===
        if LOGGING_AVAILABLE:
            self.logger = setup_global_logger(
                storm_name=storm_name,
                storm_year=storm_year,
                run_id=random_seed
            )
        else:
            self.logger = None
        
        # === PHYSICAL CONSTANTS (SI UNITS) ===
        self.g = 9.81                           # Gravitational acceleration (m/s²)
        self.c_p = 1004.0                       # Specific heat of air at constant pressure (J/(kg·K))
        self.L_v = 2.4e6                        # Latent heat of vaporization (J/kg)
        self.D_q = 1e-5                         # Moisture diffusivity (m²/s)
        self.kappa_T = 1e-5                     # Thermal diffusivity (m²/s)
        
        log_info(f"Initializing WoeSolver Suite: Oracle V4 Protocol (Storm: {storm_name} {storm_year}, Seed: {initial_wind_kts} kts)...")
        log_info(f"  -> Random seed fixed to {random_seed} for reproducibility")
        log_info("  -> All internal physics in SI units (m/s, Pa, K, m, s)")
        
        # === GRID PARAMETERS (Dimensionless) ===
        self.nx, self.ny, self.nz = nx, ny, nz
        self.lx, self.ly, self.lz = 2.0, 2.0, 1.0  # Dimensionless domain extents
        self.dx, self.dy, self.dz = self.lx / self.nx, self.ly / self.ny, self.lz / self.nz
        
        # === PHYSICAL DOMAIN PARAMETERS ===
        self.physical_domain_x_km = 2000.0      # Physical x-extent (km)
        self.physical_domain_y_km = 2000.0      # Physical y-extent (km)
        self.physical_domain_z_km = 20.0        # Physical z-extent (km)
        
        # === PHYSICAL SCALING CONSTANTS ===
        # The simulation uses characteristic scaling for non-dimensionalization
        self.U_CHAR = 50.0  # Characteristic velocity (m/s) - THE FUNDAMENTAL SCALING CONSTANT
        self.physical_domain_x_m = km_to_m(self.physical_domain_x_km)
        self.L_CHAR = self.physical_domain_x_m  # Characteristic length (m) # <--- UNIFIED FIX
        self.T_CHAR = self.L_CHAR / self.U_CHAR  # Characteristic time (s) # <--- UNIFIED FIX
        
        # === UNIT SCALERS ===
        self.dt_solver = 1e-4  # Dimensionless solver timestep (dt')
        self.domain_scaler = DomainScaler(
            self.lx, self.ly, self.lz,
            self.physical_domain_x_km, self.physical_domain_y_km, self.physical_domain_z_km
        )
        
        log_info(f"  -> Characteristic velocity U_CHAR: {self.U_CHAR} m/s")
        log_info(f"  -> Characteristic time T_CHAR: {self.T_CHAR:.1f} s")
        
        # <--- UNIFIED FIX: Removed the D_q_dim and kappa_T_dim definitions here ---
        
        # === SIMULATION PARAMETERS ===
        self.num_frames = num_frames
        self.rho = 1.0                          # Density (kept at 1.0 for dimensionless equations)
        
        # === ENSEMBLE: Progressive Equilibrium - Vortex Sharpness ===
        self.mu = mu                            # ENSEMBLE: mu = 0.15 for sharper gradients
        self.mu_current = mu                    # V12: Initialize smoothing state for soft-brake
        log_info(f"  -> VISCOSITY (mu) SET TO: {self.mu}")    
        
        # === ENSEMBLE: Progressive Equilibrium - Engine Timing ===
        self.tau_condensation_s = 5400.0        # ENSEMBLE: 90-minute precipitation lag    
        
        # === INITIAL CONDITIONS ===
        # Store initial wind speed in BOTH units
        self.initial_wind_kts = initial_wind_kts
        self.initial_wind_ms = kts_to_ms(initial_wind_kts)
        log_info(f"  -> Initial wind speed: {self.initial_wind_kts} kts = {self.initial_wind_ms:.2f} m/s")
        
        # === ORACLE V4 INITIALIZATION ===
        self.oracle_mode = oracle_mode
        self.frames_of_confusion = 0
        self.oracle_trigger_count = 0
        
        log_info(f"\n=== Oracle Mode: {oracle_mode.upper()} ===")
        
        if oracle_mode == 'off':
            # P0.2 FIX: Truly disable Oracle
            self.oracle = None
            self.drift_threshold_km = float('inf')  # Never trigger
            self.confusion_threshold_frames = float('inf')  # Never trigger
            log_info("  → Oracle DISABLED (off mode)")
            
        elif oracle_mode == 'adaptive':
            # Load learned parameters
            if oracle_params_file and os.path.exists(oracle_params_file):
                self.oracle = AdaptiveOracle.from_file(oracle_params_file)
                log_info(f"  ✓ Loaded learned parameters from {oracle_params_file}")
            else:
                # Use adaptive with fallback defaults
                self.oracle = AdaptiveOracle(fallback_drift_km=75.0, fallback_patience=200)
                log_info("  ⚠ No learned parameters found, using adaptive mode with defaults")
        elif oracle_mode == 'learning':
            # Fixed thresholds for data collection
            self.drift_threshold_km = 75.0
            self.confusion_threshold_frames = 200
            self.oracle = None
            log_info(f"  → Learning mode: Fixed thresholds (drift={self.drift_threshold_km}km, patience=200)")
        else:  # 'fixed' mode (V3) - default fallback
            self.drift_threshold_km = 75.0
            self.confusion_threshold_frames = 200
            self.oracle = None
            log_info(f"  → Fixed mode (V3): drift={self.drift_threshold_km}km, patience=200")
            
        # Memory collection - will be initialized after data_interface is created
        self.collect_memory = collect_memory
        self.oracle_memory = None
        
        # === V42 PARK BUSTER: Recovery Boost Tracking ===
        self.just_intervened = False
        self.recovery_frames_left = 0
        self.recovery_fatigue_min = 1.0  # No override by default
        self.recovery_guidance_min = 0.0  # No override by default
            
        # === SIMULATION CONTROL PARAMETERS ===
        self.kalman_blend_interval = 60
        self.nest_shift_interval = 250
        self.nest_lockdown_delay_frames = 2000
        self.initial_era5_blend_frames = 200

        self.k_nudge_smoothed = 0.0

        # === DATA INTERFACE ===
        self.data_interface = DataInterface(self, self.storm_name, self.storm_year)
        
        # Get start date from the data interface (first record of the storm)
        try:
            start_record = self.data_interface.historical_track.iloc[0]
            initial_lat, initial_lon = start_record['latitude'], start_record['longitude']
            self.sim_start_time = start_record['datetime']
            log_info(f"  -> Loaded storm data: {self.storm_name} ({self.storm_year})")
            log_info(f"  -> Start time: {self.sim_start_time}")
            log_info(f"  -> Initial position: ({initial_lat:.2f}°N, {initial_lon:.2f}°W)")
        except Exception as e:
            log_info(f"CRITICAL ERROR: Could not load start data for {self.storm_name} ({self.storm_year})")
            log_info(f"Check 'hurdat2.txt' for storm availability. Error: {e}")
            raise
        
        # === ORACLE MEMORY INITIALIZATION ===
        if self.collect_memory:
            self.oracle_memory = OracleMemory(storm_name=self.storm_name, year=self.storm_year)
            self.oracle_memory.set_simulation_params(
                start_time=self.sim_start_time,
                dt_solver=self.dt_solver,
                historical_track=self.data_interface.historical_track,
                t_char=self.T_CHAR  # V17: Fix time-travel bug
            )
            log_info("  ✓ Oracle Memory recording enabled (Prime Directive activated)")
        else:
            self.oracle_memory = None

        # === BETA-PLANE PARAMETERS ===
        center_lat_rad = np.deg2rad(initial_lat)
        omega = 7.2921e-5
        R_earth = 6.371e6  # Earth radius in meters
        self.f0 = 2 * omega * np.sin(center_lat_rad)
        self.beta = 2 * omega * np.cos(center_lat_rad) / R_earth
        log_info(f"  -> Beta-Plane Initialized for lat {initial_lat:.2f}: f0={self.f0:.2e}, beta={self.beta:.2e}")

        # === FIELD INITIALIZATION ===
        # All velocity fields in dimensionless units (grid units per timestep)
        self.u, self.v, self.w = [xp.zeros((nx, ny, nz), dtype=xp.float64) for _ in range(3)]
        
        # Specific humidity (dimensionless, kg water / kg air)
        self.q = xp.full((nx, ny, nz), 0.01, dtype=xp.float64)
        
        # Temperature in Celsius (will be converted to Kelvin when needed)
        self.T = xp.full((nx, ny, nz), 25.0, dtype=xp.float64)
        
        # === OCEAN PARAMETERS ===
        # Sea Surface Temperature (°C)
        self.SST = xp.zeros((nx, ny), dtype=xp.float64)
        
        # Ocean Heat Content (kJ/cm²)
        self.OHC = xp.full((nx, ny), 120.0, dtype=xp.float64)
        
        # OHC mixing coefficient (dimensionless, tuned for this system)
        self.ohc_mixing_coeff = 1.5e-5
        
        self._update_sst_from_ohc()

        # === DIAGNOSTIC HISTORY ===
        self.frame_history, self.max_wind_history, self.latent_heat_history = [], [], []
        
        # === PATCH V60: LANDFALL PHYSICS TOGGLE ===
        # Five's suggestion: Enable null testing (compare to baseline)
        # Set to False to run pure-ocean simulation (V59 behavior)
        # Set to True to enable land/ocean physics (V60 landfall)
        self.use_landfall_physics = True  # <-- Toggle for testing
        
        if self.use_landfall_physics:
            log_info("🏝️ LANDFALL PHYSICS ENABLED (V60): Ocean/Land blended surface fluxes")
        else:
            log_info("🌊 OCEAN-ONLY MODE (V59): Pure ocean simulation for baseline comparison")
        
        # === OUTPUT CONFIGURATION ===
        self.plot_dir = "world_woe_plots"
        if not os.path.exists(self.plot_dir): 
            os.makedirs(self.plot_dir)
        self.log_file = f"woesolver_oracle_v4_{oracle_mode}_run_{self.initial_wind_kts}kts_log.txt"
        with open(self.log_file, 'w', encoding='utf-8') as f: 
            f.write(f"--- Oracle V4 ({oracle_mode}) Protocol Log ---\n")
            f.write(f"--- ALL INTERNAL PHYSICS IN SI UNITS ---\n")

        # === MODULE INITIALIZATION ===
        self.solver = CoreSolver(self)
        self.boundaries = BoundaryConditions(self)
        self.storm_tracker = StormTracker(self)
        self.visualizer = Visualizer(self)
        self.amr = AMRHandler(self)
        
        # Kalman filter for ERA5 steering flow
        # Process noise and measurement noise are in dimensionless velocity units
        # Measurement noise: approximately (5 m/s)² converted to dimensionless
        measurement_noise_ms = 5.0  # m/s
        measurement_noise_dim = (measurement_noise_ms / self.U_CHAR)**2
        self.kalman = KalmanFilter(nx, ny, process_noise=1e-5, measurement_noise=measurement_noise_dim)
        self.frames_since_kalman_update = 0

        # === INITIALIZATION ROUTINES ===
        self._initialize_environment()
        self._initialize_storm_system()
        
        log_info(f"\n--- Priming simulation for {self.sim_start_time} ---")
        self.current_center_lat, self.current_center_lon = initial_lat, initial_lon
        # === PATCH 1: Fix __init__ call to update_steering_data ===
        self.data_interface.update_steering_data(self.current_center_lat, self.current_center_lon, self.sim_start_time, 0)
        log_info("--- Initialization Complete ---")
        
    def _initialize_environment(self):
        """
        Initialize environmental vorticity field.
        
        Creates a weak vortex patch to provide initial asymmetry.
        All velocities are in dimensionless units.
        """
        log_info("  -> Fine-tuning the Cradle...")
        cx, cy = self.lx / 2, self.ly / 2
        patch_cx, patch_cy = cx + 0.03 * self.lx, cy
        
        # Vorticity amplitude in dimensionless units
        # Physical interpretation: weak environmental shear
        vort_amp_physical_ms = 0.075  # m/s vorticity scale
        vort_amp = vort_amp_physical_ms / self.U_CHAR  # Convert to dimensionless
        
        vort_sigma = 0.15 * self.lx
        
        x, y = xp.arange(self.nx)*self.dx, xp.arange(self.ny)*self.dy
        xx, yy = xp.meshgrid(x, y, indexing='ij')
        patch_vort = vort_amp * xp.exp(-((xx - patch_cx)**2 + (yy - patch_cy)**2) / (2 * vort_sigma**2))
        r_patch = xp.sqrt((xx - patch_cx)**2 + (yy - patch_cy)**2)
        v_theta = (patch_vort * r_patch / 2.0)
        u_p = -v_theta * xp.sin(xp.arctan2(yy - patch_cy, xx - patch_cx))
        v_p = v_theta * xp.cos(xp.arctan2(yy - patch_cy, xx - patch_cx))
        
        # Add small random noise
        # === FIX: Use pure GPU noise if available ===
        if USE_GPU:
             noise = xp.random.normal(0, 0.03, (self.nx, self.ny, 2))
        else:
             noise = xp.asarray(np.random.normal(0, 0.03, (self.nx, self.ny, 2)))

        self.u[..., self.nz//2] += u_p + noise[...,0]
        self.v[..., self.nz//2] += v_p + noise[...,1]

    def _update_ohc(self):
        """
        Update Ocean Heat Content based on wind-driven mixing.
        
        OHC depletion is proportional to wind speed cubed (wind stress).
        Units: OHC in kJ/cm², wind speed converted to m/s for calculation.
        """
        # Get surface wind speed in dimensionless units
        wind_speed_dim = xp.sqrt(self.u[:, :, 0]**2 + self.v[:, :, 0]**2)
        
        # Convert to physical wind speed (m/s)
        wind_speed_ms = wind_speed_dim * self.U_CHAR
        
        # OHC depletion: proportional to wind stress (wind³)
        # Coefficient is tuned empirically for OHC in kJ/cm²
        ohc_depletion = self.dt_solver * self.ohc_mixing_coeff * wind_speed_ms**3
        
        # === V52 PATCH 1: THERMODYNAMIC SAFETY FLOOR (Gemini's Lazarus Protocol) ===
        # Problem: Storm parking at Cayman depleted OHC to zero, causing collapse
        #   Real Ivan: Survived 36-hour stall due to deep Caribbean warm pool
        #   Sim Ivan: Died in 28 hours from "thermodynamic cannibalism"
        #
        # Solution: Enforce minimum OHC representing deep warm pool
        #   Caribbean warm pool: 26°C isotherm at 150-200m depth
        #   Even extreme surface depletion can't access this reserve instantly
        #   This simulates mixing from below + horizontal advection
        
        MIN_OHC_DEEP_POOL = 40.0  # kJ/cm² (deep warm pool reserve)
        # At OHC=40: SST = 26.0 + 40/50 = 26.8°C (sufficient for H3-H4 survival)
        
        self.OHC = xp.maximum(MIN_OHC_DEEP_POOL, self.OHC - ohc_depletion)
        # === END V52 PATCH 1 ===

    def _update_sst_from_ohc(self):
        """
        Update Sea Surface Temperature from Ocean Heat Content.
        
        Empirical relationship: SST (°C) = 26 + OHC/50
        This is a simplified parameterization where:
        - OHC in kJ/cm²
        - SST in °C
        """
        self.SST = 26.0 + (self.OHC / 50.0)
    
    def _initialize_storm_system(self):
        """
        Initialize hurricane vortex with realistic structure.
        
        Creates a Rankine vortex with:
        - Maximum winds at initial_wind_ms (m/s)
        - Radius of maximum wind ~60 km
        - Enhanced moisture and temperature in the core
        """
        log_info("Executing Genesis Protocol...")
        cx, cy = self.lx / 2, self.ly / 2
        
        # Convert initial wind speed to dimensionless velocity
        v_max_dim = self.initial_wind_ms / self.U_CHAR  # Convert to dimensionless
        
        # Radius of maximum wind: 60 km
        R_max_km = 60.0
        R_max_dim = (R_max_km / self.physical_domain_x_km) * self.lx
        
        # Create Rankine vortex
        x, y = xp.arange(self.nx)*self.dx, xp.arange(self.ny)*self.dy
        xx, yy = xp.meshgrid(x, y, indexing='ij')
        r = xp.sqrt((xx - cx)**2 + (yy - cy)**2)
        
        # Tangential velocity profile
        v_theta = xp.where(r <= R_max_dim, 
                          v_max_dim * (r/R_max_dim),              # Solid body rotation inside
                          v_max_dim * (R_max_dim/(r + 1e-9)))     # Decaying outside
        
        # Convert to u,v components
        u_vortex = -v_theta * xp.sin(xp.arctan2(yy-cy, xx-cx))
        v_vortex =  v_theta * xp.cos(xp.arctan2(yy-cy, xx-cx))
        
        # Add to velocity field (broadcast to all levels)
        self.u += u_vortex[..., xp.newaxis]
        self.v += v_vortex[..., xp.newaxis]
        
        # Add moisture and temperature anomalies
        z = xp.arange(self.nz) * self.dz
        xx_3d, yy_3d, zz_3d = xp.meshgrid(x, y, z, indexing='ij')
        
        # Gaussian distribution of moisture and heat
        rad_h, rad_v = self.lx/15.0, self.lz/4.0
        gauss = xp.exp(-((xx_3d-cx)**2/(2*rad_h**2) + 
                        (yy_3d-cy)**2/(2*rad_h**2) + 
                        (zz_3d-(self.lz/2.5))**2/(2*rad_v**2)))
        
        self.q += 0.004 * gauss          # Add ~0.4% moisture
        self.T += 3.0 * gauss            # Add up to 3°C warming
        
        # Temperature lapse rate (decrease with height)
        self.T -= (z / self.lz * 40.0)[xp.newaxis, xp.newaxis, :]
        
    def calculate_era5_quality_metrics(self):
        """
        PATCH V42: Environmental Quality Assessment (KWAI's Diagnostic)
        
        Detects weak/incoherent ERA5 steering that causes parking at 19-20°N, 83°W.
        This is the ROOT CAUSE identified by the ensemble analysis.
        
        Returns:
            steering_magnitude_ms: Average environmental flow speed (m/s)
            steering_coherence: Flow uniformity (0=chaotic, 1=uniform)
        """
        # Get current ERA5 steering field (already on GPU)
        u_steer = self.data_interface.u_target
        v_steer = self.data_interface.v_target
        
        # Metric 1: Steering Magnitude
        # How strong is the environmental flow?
        # Weak flow (<3 m/s = 6 kts) causes parking
        steering_speed_dim = xp.sqrt(u_steer**2 + v_steer**2)
        steering_magnitude_ms = float(xp.mean(steering_speed_dim)) * self.U_CHAR
        
        # Metric 2: Steering Coherence  
        # How uniform is the flow direction?
        # Incoherent fields have high spatial gradient variance
        du_dx = xp.diff(u_steer, axis=0).flatten()
        dv_dy = xp.diff(v_steer, axis=1).flatten()
        gradient_variance = float(xp.std(xp.concatenate([du_dx, dv_dy])))
        
        # Normalize: 0 = chaotic, 1 = uniform
        # Typical: 0.01 = coherent, 0.10 = turbulent
        steering_coherence = 1.0 / (1.0 + 10.0 * gradient_variance)
        
        return steering_magnitude_ms, steering_coherence
        
    def _apply_guidance_force(self, frame):
        """
        Oracle V4.4: "Deep Layer Open Loop" Protocol.
        Includes Patch V30 (Dynamic Scale) and Patch V33c (Graduated Throttle).
        
        The "Ensemble Consensus" (Gemini + Grok):
        1. CORRECTION PHASE (Oracle): Uses "Open Loop" Vector Injection.
           - Ignores local storm velocity to prevent "Eyewall Sampling" noise.
           - Applies pure directional momentum to snap out of dead zones.
        2. GUIDANCE PHASE (Physics): Uses "Deep Layer" Coupling.
           - Steers based on Kalman (Deep) vs Surface mismatch.
           - Ensures the storm 'feels' the upper-level ridge breakdown.
        """
        if frame < self.nest_lockdown_delay_frames:
            return

        # === PATCH V33c: GRADUATED GUIDANCE THROTTLE ===
        # Diagnosis: Binary lockout creates "Cliff Effect" and parking lots.
        # Fix: Scale guidance force proportionally to tracking confidence (Lock Score).
        # Logic: 
        #   Lock < 0.05: DNR (0% Force) - Storm is lost.
        #   Lock 0.05-0.20: Linear Ramp (25% - 100% Force) - Gentle nudging.
        #   Lock > 0.20: Full Active Guidance (100% Force).
        
        current_lock = self.storm_tracker.get_lock_score()
        guidance_strength = 0.0
        
        if current_lock < 0.05:
            # DNR Protocol: Storm is truly lost (<5% lock).
            if frame % 1000 == 0:
                 log_info(f"    [DNR PROTOCOL] Lock={current_lock:.2f} < 0.05. Guidance DISABLED.")
            return # Exit function, apply no force
            
        elif current_lock < 0.20:
            # Degraded Guidance: Scale strength linearly
            # 0.05 -> 0.25 strength
            # 0.20 -> 1.00 strength
            guidance_strength = current_lock / 0.20
            if frame % 1000 == 0:
                 log_info(f"    [GRADUATED GUIDANCE] Lock={current_lock:.2f}. Throttle: {guidance_strength*100:.0f}%")
                 
        else:
            # Full Guidance
            guidance_strength = 1.0
            
        # === V42 PARK BUSTER: Recovery Override ===
        # GROK's Fix: Force minimum guidance during recovery
        if hasattr(self, 'recovery_guidance_min') and self.recovery_guidance_min > 0.0:
            guidance_strength = max(guidance_strength, self.recovery_guidance_min)
        # === END V42 RECOVERY OVERRIDE ===
            
        # === END PATCH V33c ===

        if not hasattr(self, 'was_correcting'):
            self.was_correcting = False

        # === 1. MEASURE DRIFT ===
        cx_storm, cy_storm = self.storm_tracker.get_current_center_grid()
        if cx_storm is None: return

        time_elapsed = timedelta(seconds=(frame * self.dt_solver * self.T_CHAR))
        current_sim_time = self.sim_start_time + time_elapsed
        
        # Historical Target
        hist_track = self.data_interface.historical_track
        true_loc = hist_track.iloc[(hist_track['datetime'] - current_sim_time).abs().argsort()[:1]]
        target_lat = true_loc['latitude'].iloc[0]
        target_lon = true_loc['longitude'].iloc[0]
        
        # Convert to grid
        lat_min, lat_max = self.data_interface.lat_bounds
        lon_min, lon_max = self.data_interface.lon_bounds
        target_y_grid = ((target_lat - lat_min) / (lat_max - lat_min)) * self.ny
        target_x_grid = ((target_lon - lon_min) / (lon_max - lon_min)) * self.nx
        
        # === PATCH V30: DYNAMIC SCALE FIX ===
        # Calculate vector in grid units
        vec_x = target_x_grid - cx_storm
        vec_y = target_y_grid - cy_storm
        drift_grid = xp.sqrt(vec_x**2 + vec_y**2)
        
        # Calculate ACTUAL km per pixel based on current bounds
        # (This handles the V28 Precision Box of 4 degrees vs 2000km default)
        lon_min, lon_max = self.data_interface.lon_bounds
        lat_min, lat_max = self.data_interface.lat_bounds
        
        # Calculate domain width in degrees
        deg_width = lon_max - lon_min
        
        # Latitude correction (Cos(lat) scaling)
        # 1 deg lat = 111 km. 1 deg lon = 111 * cos(lat)
        # We use the mid-latitude of the domain for the conversion factor
        mid_lat_rad = np.deg2rad(0.5 * (lat_min + lat_max))
        km_per_deg_lon = 111.0 * np.cos(mid_lat_rad)
        
        real_domain_width_km = deg_width * km_per_deg_lon
        
        km_per_pixel = real_domain_width_km / self.nx
        drift_km = float(drift_grid.item()) * km_per_pixel
        
        # === DIAGNOSTIC LOGGING (VALIDATION) ===
        if frame % 100 == 0:
             # Log directly to console to verify the fix immediately
             print(f"    [GUIDANCE DEBUG] km/px={km_per_pixel:.3f} (Expect ~3.5), drift_km={drift_km:.1f}")
        # === END PATCH V30 ===
        
        # === PATCH V42: ERA5 ENVIRONMENTAL DIAGNOSTICS ===
        # KWAI's Root Cause: Weak/chaotic ERA5 steering causes parking at 19-20°N, 83°W
        steering_magnitude_ms, steering_coherence = self.calculate_era5_quality_metrics()
        
        # Detect weak steering zones (the "parking lot" region)
        is_weak_zone = steering_magnitude_ms < 3.0  # Less than 6 knots
        is_chaotic_zone = steering_coherence < 0.3  # Incoherent flow field
        
        if is_weak_zone or is_chaotic_zone:
            if frame % 200 == 0:
                log_info(f"    ⚠️  [V42 WEAK ERA5] Mag={steering_magnitude_ms:.2f} m/s, Coherence={steering_coherence:.2f}")
        # === END PATCH V42 DIAGNOSTICS ===

        # === 2. DETERMINE TOLERANCE ===
        current_wind_kts = self.storm_tracker.get_max_wind()
        stage = self.storm_tracker.current_stage
        is_in_erc = self.storm_tracker.is_in_erc()
        
        # === V42 PARK BUSTER: Softer Drift Thresholds ===
        # GROK's Fix: Prevent over-intervention (96.5 km caused 701 interventions with 75% parking)
        if stage == "Potential" or current_wind_kts < 64: 
            tolerance_km = 90.0   # Was 60.0 (+50%)
        elif is_in_erc: 
            tolerance_km = 150.0  # Was 120.0 (+25%)
        else: 
            tolerance_km = 60.0   # Was 35.0 (+70%)
        # === END V42 PARK BUSTER THRESHOLDS ===

        # GROK'S TURN DETECTOR: Calculate deep-layer mismatch magnitude
        try:
            idx_x = max(0, min(int(cx_storm), self.nx-1))
            idx_y = max(0, min(int(cy_storm), self.ny-1))
            
            # Deep Layer (Kalman) vs Surface (Sim)
            u_deep = self.kalman.state_estimate[idx_x, idx_y, 0]
            v_deep = self.kalman.state_estimate[idx_x, idx_y, 1]
            u_surf = self.u[idx_x, idx_y, 1]
            v_surf = self.v[idx_x, idx_y, 1]
            
            mismatch = float(xp.sqrt((u_deep - u_surf)**2 + (v_deep - v_surf)**2)) * self.U_CHAR
            
            # If atmosphere and storm disagree by > 8 kts (4 m/s), tighten leash
            if mismatch > 4.0:
                tolerance_km *= 0.5
        except: pass

        # Hysteresis
        active_threshold = tolerance_km * 0.8 if self.was_correcting else tolerance_km
        needs_correction = drift_km > active_threshold
        self.was_correcting = needs_correction
        
        total_u, total_v = 0.0, 0.0

        # === 3. APPLY FORCES ===
        
        if needs_correction:
            # --- BRANCH A: V42 HIGH TORQUE TRACTOR BEAM ---
            # GEMINI's Fix: 2x force authority, 50% stronger base, hurricane-speed limit
            # KWAI's Fix: Adaptive boost (up to 1.5x more) in weak ERA5 zones
            
            norm = drift_grid + 1e-9
            dir_x = vec_x / norm
            dir_y = vec_y / norm
            
            # === V42 HIGH TORQUE: Dynamic Speed Ramp ===
            # Base speed: 8.0 → 12.0 m/s (50% stronger)
            # Ramp factor: /8.0 → /5.0 (60% faster response)
            # Speed cap: 30.0 → 50.0 m/s (hurricane-force steering)
            target_speed_ms = 12.0 + (drift_km / 5.0)
            target_speed_ms = min(target_speed_ms, 50.0)
            target_speed_dim = target_speed_ms / self.U_CHAR
            
            # === V42 HIGH TORQUE: Force Gain ===
            # Base gain DOUBLED: 4.0 → 8.0 (match vortex inertia from V39b)
            base_gain = 8.0
            
            # === V42 ENVIRONMENTAL BOOST: Adaptive authority ===
            # Compensate for weak/chaotic ERA5 steering
            environmental_boost = 1.0
            if is_weak_zone:
                environmental_boost = 1.5  # 50% more force in dead zones
            if is_chaotic_zone:
                environmental_boost = max(environmental_boost, 1.3)  # 30% for chaos
            
            gain = base_gain * environmental_boost
            
            # Pure Force Injection (Open Loop - preserves Gemini's original fix)
            total_u = gain * dir_x * target_speed_dim
            total_v = gain * dir_y * target_speed_dim
            
            if frame % 100 == 0:
                boost_str = f"(BOOSTED {environmental_boost:.1f}x)" if environmental_boost > 1.0 else ""
                log_info(f"    >>> V42 HIGH TORQUE: Drift {drift_km:.1f}km, Gain={gain:.1f} {boost_str}, Speed={target_speed_ms:.1f} m/s <<<")
                
        else:
            # --- BRANCH B: GHOST NUDGE V4 (Deep Layer Lock) ---
            # Grok's Fix: Couple to the Deep Layer (Kalman) to anticipate turns.
            if stage != "Potential" and current_wind_kts > 45 and not self.ablation_mode and self.simulation_mode != 'free_run':
                try:
                    # Calculate error between Deep Layer and Surface
                    # (Already calculated in Turn Detector above)
                    err_u = u_deep - u_surf
                    err_v = v_deep - v_surf
                    
                    # Ultra-aggressive coupling for mature storms
                    # === PATCH V19: Supercharged Steering ===
                    # WAS: 0.045. Increased to 0.09 to force the turn.
                    base_k = 0.09 # Grok's "Lethal" Base
                    maturity = min(1.0, (current_wind_kts - 40.0) / 70.0)
                    lock_factor = max(self.storm_tracker.get_lock_score(), 0.35)
                    
                    k_final = base_k * maturity * (0.7 + lock_factor)
                    
                    total_u = k_final * err_u
                    total_v = k_final * err_v
                    
                    if frame % 500 == 0:
                         # V55: Track statistics
                         self.ghost_nudge_count += 1
                         self.ghost_nudge_history.append(k_final)
                         log_info(f"    >>> GHOST NUDGE V4: Deep Layer Lock (k={k_final:.4f}). <<<")
                except: pass

            elif stage != "Potential" and current_wind_kts > 45:
                if self.ablation_mode:
                    if frame % 500 == 0:
                        log_info(f"    ⚠️ ABLATION: Ghost Nudge conditions met but DISABLED for testing")
                elif self.simulation_mode == 'free_run':
                    if frame % 500 == 0:
                        log_info(f"    📋 FREE RUN: Ghost Nudge disabled in Mode A")

        # === 4. APPLY WITH SOFT MASK ===
        if total_u != 0 or total_v != 0:
            # === V42 HIGH TORQUE: Increased Force Cap ===
            # Old: 3.5 (for 4.0 gain)
            # New: 10.5 (for 8.0 base gain, or 12.0 with environmental boost)
            force_mag = xp.sqrt(total_u**2 + total_v**2)
            max_force = 10.5
            if force_mag > max_force:
                scale = max_force / force_mag
                total_u *= scale
                total_v *= scale
                
            # === V52 PATCH 2: STALL BREAKER (Gemini's Lazarus Protocol) ===
            # Problem: Storm can park in weak steering zones (like Cayman)
            #   Real Ivan: Ridge eventually rebuilt, picked up storm after 36 hrs
            #   ERA5 data: May be too smooth, missing subtle synoptic pickup
            #
            # Solution: Detect extended stall, add gentle "synoptic shove"
            #   Only activates after 500 frames of <5 kt movement
            #   Adds gentle NW vector to help storm resume motion
            
            # Initialize stall counter if needed
            if not hasattr(self, '_stall_counter'):
                self._stall_counter = 0
                self._last_cx = cx_storm
                self._last_cy = cy_storm
            
            # Calculate recent movement (every 100 frames)
            if frame % 100 == 0:
                displacement = xp.sqrt((cx_storm - self._last_cx)**2 + 
                                      (cy_storm - self._last_cy)**2)
                # Convert to physical distance (km)
                # Domain is 2000 km across nx cells
                km_per_cell = self.physical_domain_x_km / self.nx
                displacement_km = float(displacement) * km_per_cell
                # Calculate speed (km/hr then knots)
                time_hours = (100 * self.dt_solver * self.T_CHAR) / 3600
                speed_kts = (displacement_km / time_hours) / 1.852
                
                if speed_kts < 5.0:  # Less than 5 knots = stalled
                    self._stall_counter += 100
                else:
                    self._stall_counter = 0  # Reset if moving
                
                self._last_cx = cx_storm
                self._last_cy = cy_storm
            
            # If stalled for 500+ frames, add synoptic shove
            if self._stall_counter >= 500 and not self.ablation_mode:
                # Gentle northwest vector (typical post-stall recovery direction)
                synoptic_u = 2.0  # dimensionless (~10-15 m/s physical)
                synoptic_v = 2.0  # dimensionless (~10-15 m/s physical)
                
                total_u += synoptic_u
                total_v += synoptic_v
                
                if frame % 200 == 0:
                    # V55: Track statistics
                    self.stall_breaker_count += 1
                    log_info(f"    🚑 V52 STALL BREAKER: Stalled {self._stall_counter} frames, adding synoptic shove")
            elif self._stall_counter >= 500:
                log_info(f"    ⚠️ ABLATION: Stall detected ({self._stall_counter} frames) but breaker DISABLED")

            # === END V52 PATCH 2 ===
            
            x, y = xp.arange(self.nx), xp.arange(self.ny)
            xx, yy = xp.meshgrid(x, y, indexing='ij')
            dist = xp.sqrt((xx - cx_storm)**2 + (yy - cy_storm)**2)
            
            # Gaussian Mask
            sigma = self.nx * 0.12
            mask = xp.exp(-0.5 * (dist / sigma)**2)
            
            # === PATCH V33c: Apply Graduated Throttle ===
            # Scale the force by guidance_strength calculated at function start
            self.u += total_u * guidance_strength * mask[..., xp.newaxis] * self.dt_solver
            self.v += total_v * guidance_strength * mask[..., xp.newaxis] * self.dt_solver

    def _update_moving_nest(self, frame):
        """
        V4.5 "Golden Handcuffs" Protocol.
        
        Updates the domain center to lock onto the HISTORICAL track.
        Prevents the "Blindfold Bug" where following a drifting storm caused
        the simulation to download incorrect environmental data.
        """
        if self.storm_tracker.current_stage == "Potential":
            return
            
        if frame % self.nest_shift_interval != 0: 
            return
        
        # === 1. GET TARGET TIME ===
        time_elapsed = timedelta(seconds=(frame * self.dt_solver * self.T_CHAR))
        current_sim_time = self.sim_start_time + time_elapsed
        
        # === 2. FIND HISTORICAL CENTER (The Anchor) ===
        hist_track = self.data_interface.historical_track
        
        # Find the record closest to current time
        closest_rec = hist_track.iloc[(hist_track['datetime'] - current_sim_time).abs().argsort()[:1]]
        target_lat = closest_rec['latitude'].iloc[0]
        target_lon = closest_rec['longitude'].iloc[0]
        
        # === 3. CALCULATE SHIFT ===
        # We simply force the domain center to match the historical center
        old_lat = self.current_center_lat
        old_lon = self.current_center_lon
        
        # Update domain center
        self.current_center_lat = target_lat
        self.current_center_lon = target_lon
        
        # === 4. KALMAN HANDOFF ===
        # Since the domain is jumping to a specific point, we need to manage the 
        # Kalman filter memory so it doesn't "shock" the system with the sudden shift.
        
        # Save current Kalman state before we fetch new data
        self.data_interface.set_kalman_backup(
            self.kalman.state_estimate[..., 0],
            self.kalman.state_estimate[..., 1]
        )
        
        # === 5. EXECUTE DATA FETCH ===
        log_info(f"  >>> NEST UPDATE: Anchoring to History ({target_lat:.2f}, {target_lon:.2f}) <<<")
        self.data_interface.update_steering_data(
            self.current_center_lat, 
            self.current_center_lon, 
            current_sim_time, 
            frame
        )

    def update(self, frame):
        """
        Main physics update loop.
        """
        
        # === V42 PARK BUSTER: Recovery Boost Management ===
        # GROK's Fix: Force fuel and guidance after Oracle intervention
        if self.just_intervened and self.recovery_frames_left > 0:
            self.recovery_frames_left -= 1
            
            # Override V39b Fatigue to force fuel flow
            # Override V33c Guidance to maintain steering
            self.recovery_fatigue_min = 0.50  # Minimum 50% fuel
            self.recovery_guidance_min = 0.30  # Minimum 30% guidance
            
            if frame % 100 == 0:
                frames_done = 500 - self.recovery_frames_left
                log_info(f"    🔧 [V42 RECOVERY] Frame {frames_done}/500 | Min Fuel={self.recovery_fatigue_min:.0%}, Min Guidance={self.recovery_guidance_min:.0%}")
                
        elif self.recovery_frames_left <= 0 and self.just_intervened:
            # Recovery complete
            self.just_intervened = False
            self.recovery_fatigue_min = 1.0
            self.recovery_guidance_min = 0.0
            log_info("    ✅ [V42 RECOVERY COMPLETE] Returning to normal operations")
        # === END V42 RECOVERY BOOST ===
        
        
                # === ORACLE'S JUDGMENT PROTOCOL V4 - V50.1 FIX ===
        # V50.1: Wire confusion to STRUCTURAL HEALTH, not tracking accuracy
        # Rationale: Only panic if storm is physically broken (low coherence),
        #            not if it's just off-center (low track). Oracle's drift_km
        #            check already handles navigation problems separately.
        if self.storm_tracker.lock_struct < 0.05:  # Physical health threshold
            self.frames_of_confusion += 1
        else:
            self.frames_of_confusion = 0

        # Determine patience threshold based on mode
        if self.oracle_mode == 'adaptive':
            patience_threshold = 200
        else:
            patience_threshold = getattr(self, 'confusion_threshold_frames', 200)

        if self.frames_of_confusion >= patience_threshold:
            log_info("\n>>> AI CONFUSION DETECTED. ENACTING ORACLE'S JUDGMENT. <<<\n")
            
            # === FIX: Use self.T_CHAR for correct time conversion ===
            time_elapsed = timedelta(seconds=(frame * self.dt_solver * self.T_CHAR)) # <--- UNIFIED FIX
            # === END FIX ===
            
            current_sim_time = self.sim_start_time + time_elapsed
            
            hist_track = self.data_interface.historical_track
            true_location = hist_track.iloc[
                (hist_track['datetime'] - current_sim_time).abs().argsort()[:1]
            ]
            true_lat = true_location['latitude'].iloc[0]
            true_lon = true_location['longitude'].iloc[0]

            # === START CRITICAL FIX 1: Use actual storm position for drift ===
            
            # Get the storm's current grid coordinates from the tracker
            cx, cy = self.storm_tracker.get_current_center_grid()
            if cx is None:  # Fallback if tracker hasn't found it yet
                cx, cy = self.nx // 2, self.ny // 2

            # Get the geographic boundaries of the *current* domain
            lon_min, lon_max = self.data_interface.lon_bounds
            lat_min, lat_max = self.data_interface.lat_bounds
            
            # Convert the storm's grid (cx, cy) to a lat/lon
            sim_lon = lon_min + (cx / self.nx) * (lon_max - lon_min)
            sim_lat = lat_min + (cy / self.ny) * (lat_max - lat_min)

            # Calculate drift based on the *storm's* actual position
            drift_km = haversine_distance_km(sim_lon, sim_lat, true_lon, true_lat)
            
            # Use the old domain center *only* for reference
            # This is what the Oracle *thinks* its center is.
            old_lat, old_lon = self.current_center_lat, self.current_center_lon
            # === END CRITICAL FIX 1 ===
            
            # Prepare state for Oracle
            current_intensity = self.storm_tracker.get_max_wind()
            lock_score = self.storm_tracker.get_lock_score()
            
            state = {
                'frame': frame,
                'intensity': current_intensity,
                'dW_dt': self.storm_tracker.get_intensity_trend(),
                'lock': lock_score,
                'in_erc': self.storm_tracker.is_in_erc(),
                'erc_detected': self.storm_tracker.is_in_erc(),
                'stage': self.storm_tracker.current_stage,
                'drift': drift_km,      # This now holds the CORRECT drift
                'sim_lat': sim_lat,     # <--- Use the new sim_lat
                'sim_lon': sim_lon,     # <--- Use the new sim_lon
                'hist_lat': true_lat,
                'hist_lon': true_lon,
                'confusion_frames': self.frames_of_confusion,
                'ohc': float(self.OHC[int(cx), int(cy)])  # <--- Use storm center for OHC
            }
            
            # Make decision based on mode
            if self.oracle_mode == 'adaptive':
                should_intervene, reason = self.oracle.should_intervene(
                    state, drift_km, self.frames_of_confusion
                )
                drift_threshold = reason['drift_threshold_km']
                phase = reason['phase']
                confidence = reason['confidence']
            else:
                # Fixed mode (V3)
                drift_threshold = self.drift_threshold_km
                should_intervene = drift_km > drift_threshold
                phase = 'unknown'
                confidence = 1.0
                reason = {'phase': phase, 'using_learned_params': False}

            # Record decision for learning
            if self.collect_memory:
                self.oracle_memory.record_check(frame, state, should_intervene)

            # Take action
            if should_intervene:
                self.oracle_trigger_count += 1
                
                log_info(f"\n{'='*80}")
                log_info(f">>> ORACLE'S JUDGMENT: CATASTROPHIC DRIFT DETECTED <<<")
                log_info(f"    Event #{self.oracle_trigger_count} | Frame: {frame}")
                if self.oracle_mode == 'adaptive':
                    log_info(f"    Phase: {phase.upper()} (confidence: {confidence:.2f})")
                    if reason['using_learned_params']:
                        log_info(f"    Using LEARNED parameters ✓")
                    else:
                        log_info(f"    Using DEFAULT parameters (phase not trained)")
                log_info(f"    Confusion Duration: {self.frames_of_confusion} frames")
                log_info(f"    Current LOCK: {lock_score:.2f} | Intensity: {current_intensity:.1f} kts")
                log_info(f"    Simulated location: ({sim_lat:.2f}°, {sim_lon:.2f}°) [Domain Center: ({old_lat:.2f}°, {old_lon:.2f}°)]")
                log_info(f"    Historical location: ({true_lat:.2f}°, {true_lon:.2f}°)")
                log_info(f"    Drift distance: {drift_km:.1f} km (~{drift_km/111.0:.2f}°)")
                log_info(f"    Threshold: {drift_threshold:.1f} km")
                log_info(f"    INTERVENTION REQUIRED. Re-centering to historical position.")
                log_info(f"{'='*80}\n")
                
                self.current_center_lat = true_lat
                self.current_center_lon = true_lon
                
                # === ENSEMBLE: "Kalman-Aware" Dampener (Oracle) ===
                # We must also save the Kalman state when the Oracle
                # forces a re-center.
                self.data_interface.set_kalman_backup(
                    self.kalman.state_estimate[..., 0],
                    self.kalman.state_estimate[..., 1]
                )
                # === END "Kalman-Aware" Dampener ===
                
                # === PATCH 3: Fix Oracle crash ===
                self.data_interface.update_steering_data(self.current_center_lat, self.current_center_lon, current_sim_time, frame)
                
                # === V42 PARK BUSTER: Trigger Recovery Mode ===
                # GROK's Fix: 500-frame recovery boost after intervention
                self.just_intervened = True
                self.recovery_frames_left = 500
                log_info("    🔧 [V42 RECOVERY MODE] 500-frame boost initiated after intervention")
                # === END V42 RECOVERY TRIGGER ===
            else:
                log_info(f"    >>> ORACLE'S JUDGMENT: Drift ({drift_km:.1f} km) below threshold ({drift_threshold:.1f} km).")
                if self.oracle_mode == 'adaptive':
                    log_info(f"        Phase: {phase.upper()} | LOCK: {lock_score:.2f} | Allowing natural evolution. <<<\n")
                else:
                    log_info(f"        LOCK: {lock_score:.2f} | Allowing natural evolution. <<<\n")

            self.frames_of_confusion = 0
        # === END ORACLE'S JUDGMENT PROTOCOL V4 ===
        
        # Record state periodically for learning
        if self.collect_memory and frame % 100 == 0:
            self.oracle_memory.record_state(
                frame,
                self.current_center_lat,
                self.current_center_lon,
                self.storm_tracker.get_max_wind(),
                self.storm_tracker.get_lock_score()
            )
        
        # === PHYSICS UPDATE ===
        # Adaptive timestep based on CFL condition
        if frame > 0:
            max_vel = self.solver.get_max_velocity(self.u, self.v, self.w)
            if max_vel > 0 and xp.isfinite(max_vel):
                self.dt_solver = min(1e-4, 0.5 * self.dx / max_vel)

        # <--- UNIFIED FIX: Get PHYSICAL timestep in seconds ---
        dt_physical_s = self.dt_solver * self.T_CHAR

        self._update_ohc()
        self._update_sst_from_ohc()
        self.kalman.predict()
        
        # === ENSEMBLE: Progressive Equilibrium - Apply surface fluxes with new fuel boost ===
        
        # === PATCH V54: LAUNCH CONTROL (Genesis Governor V2) ===
        # Diagnosis: Katrina V1 exploded to 180 kts in 14 hours ("Popcorn Storm").
        # Cause: Turbocharger (Fuel 1.28) engaged too early (at 64 kts).
        # Fix: Enforce strict timeline to force realistic intensification rate.
        #      Phase 1 (0-20h):  Heavy Break-in (Force TS structure)
        #      Phase 2 (20-40h): Soft Launch (Allow Cat 1/2 growth)
        #      Phase 3 (40h+):   Maturation (Unlock Cat 3-4)
        #      Phase 4 (Cat 3+): Turbocharger (Allow RI to Cat 5)
        
        current_wind_kts = self.storm_tracker.get_max_wind()
        
        if frame < 15000:
            # PHASE 1: THE HEAVY BREAK-IN (First ~20 hours)
            # Prevent "Popcorn" spin-up. Force structure building.
            fuel_load = 1.0  # Baseline fuel only
            self.mu_current = 0.25 # High viscosity brake
            
        elif frame < 30000:
            # PHASE 2: SOFT LAUNCH (Hours 20-40)
            # Allow development to Hurricane, but throttle RI.
            # Max fuel 1.10 is enough for Cat 1/2, but not Cat 5.
            fuel_load = 1.10
            # Let dynamic viscosity take over (likely ~0.12)
            
        elif current_wind_kts < 96.0:
            # PHASE 3: MATURATION (After 40h, but still < Cat 3)
            # Keep steady pressure until Major Hurricane status.
            fuel_load = 1.15
            
        else:
            # PHASE 4: TURBOCHARGER (Mature Major Hurricane)
            # Storm has earned the right to rapid intensification.
            fuel_load = 1.28 
        
        # === PATCH V60.4: GET LAND FRACTION FROM DATA INTERFACE ===
        # Land fraction updates automatically with moving nest via ERA5 LSM
        # Conditional on landfall physics toggle for null testing
        if self.use_landfall_physics:
            land_fraction = self.data_interface.land_fraction
        else:
            land_fraction = None  # Triggers all-ocean behavior in boundaries
            
        # === DIAGNOSTIC: LAND FRACTION CORE SAMPLE ===
        if self.use_landfall_physics and frame % 100 == 0:
            cx, cy = self.storm_tracker.get_current_center_grid()
            if cx is not None:
                # Sample the land fraction exactly at the storm center
                idx_x, idx_y = int(cx), int(cy)
                core_land = float(land_fraction[idx_x, idx_y])
                if core_land > 0.1:
                    log_info(f"    ⛰️ LANDFALL CONTACT: Core is over {core_land*100:.1f}% Land")

        self.q, self.T, q_f, h_f, damp_factor = self.boundaries.apply_surface_fluxes(
            self.q, self.T, fuel_load, land_fraction  # <-- ADDED land_fraction
        )
        # === END PATCH V54 ===
        
        # === APPLY SURFACE DRAG ===
        drag_x_pa, drag_y_pa = self.boundaries.calculate_surface_drag(
            self.u[:,:,0], self.v[:,:,0], land_fraction  # <-- ADDED land_fraction
        )
        
        dz_physical_m = self.domain_scaler.dimensionless_to_physical_z(self.dz)
        dt_physical_s = self.dt_solver * self.T_CHAR
        
        # Force = Stress / (rho * dz) -> units of m/s^2 (acceleration)
        # Multiply by dt -> change in velocity (m/s)
        # Divide by U_CHAR -> dimensionless change in velocity
        du_drag = (drag_x_pa / (self.rho * dz_physical_m)) * dt_physical_s / self.U_CHAR
        dv_drag = (drag_y_pa / (self.rho * dz_physical_m)) * dt_physical_s / self.U_CHAR
        
        # Apply drag ONLY to the lowest level (k=0)
        self.u[:,:,0] -= du_drag
        self.v[:,:,0] -= dv_drag
        
        # <--- UNIFIED FIX: Use physical dt for vertical mixing ---
        # === ENSEMBLE FIX (per Grok): "MAJOR CRUISE" FIRM RAMP MIXING ===
        # This is a stronger "brake" ramp designed to handle
        # the new, more powerful fuel load.

        wind_speed_dim_mix = xp.sqrt(self.u**2 + self.v**2)
        wind_speed_ms_mix = wind_speed_dim_mix * self.U_CHAR
        wind_norm = wind_speed_ms_mix / 50.0  # Normalize by U_CHAR

        # 1. Dynamic K_z (Physical m²/s)
        # Ramps from 22.0 up to a max of 260.0
        K_z = 22.0 + 48.0 * xp.power(wind_norm, 1.2)
        K_z = xp.clip(K_z, 0, 260.0) # Hard cap

        # 2. Dynamic C_mix (Numerical Coefficient)
        # Ramps from 0.26 up to a max of 0.72
        C_mix = 0.26 + 0.44 * xp.power(wind_norm, 1.2)
        C_mix = xp.clip(C_mix, 0, 0.49) # Hard cap at stability limit

        # === END GROK FIX ===
        
        # ENSEMBLE FIX: Add these two lines back in
        mixing_height_m = 2000.0
        mixing_height_dim = self.domain_scaler.physical_to_dimensionless_z(mixing_height_m)
        
        # ENSEMBLE FIX: We must mix momentum (u,v) and heat (T) as well!
        dq_mix = xp.zeros_like(self.q)
        dT_mix = xp.zeros_like(self.T)
        
        # Helper dictionary to apply mixing to scalars (q, T)
        fields_to_mix = {
            'q': (self.q, dq_mix),
            'T': (self.T, dT_mix),
        }

        # ENSEMBLE FIX: Implement a "sticky" boundary (k=1) AND
        # a surface "fuel pump" (k=0) to fix starvation.
        
        for field_name, (field_data, mix_array) in fields_to_mix.items():
            
            # --- START CORRECTED LOGIC ---
            
            # 1. Apply standard mixing for all "upper" interior layers (k=2 to nz-2)
            for k in range(2, self.nz - 1):
                if (k * self.dz) < mixing_height_dim:
                    mix_array[:, :, k] = C_mix[:, :, k] * (
                        field_data[:, :, k + 1] - 2 * field_data[:, :, k] + field_data[:, :, k - 1]
                    )

            # 2. Weld layer k=1 to the surface layer k=0 (This stops the explosion)
            if (1 * self.dz) < mixing_height_dim:
                 mix_array[:, :, 1] = C_mix[:, :, 1] * (
                     field_data[:, :, 2] - 2 * field_data[:, :, 1] + field_data[:, :, 0]
                 )

            # 3. Unclog the drain: Mix from k=0 up to k=1 (This ignites the engine)
            # ENSEMBLE FIX: The '* 2.0' was numerically unstable and causing
            # the explosion. Use a stable 1.0 coefficient.
            if (0 * self.dz) < mixing_height_dim:
                 mix_array[:, :, 0] = C_mix[:, :, 0] * (field_data[:, :, 1] - field_data[:, :, 0]) # <-- REMOVED * 2.0

            # --- END CORRECTED LOGIC ---

        # Apply the changes to scalars
        self.q += dq_mix
        self.T += dT_mix

        # === ENSEMBLE FIX: The "Momentum Anchor" ===
        # The previous C_mix logic allowed the wind at k=1 to decouple
        # from the surface drag applied at k=0, creating an unrealistic
        # low-level jet (400+ kts).
        # This fix explicitly anchors the momentum of the lowest layers
        # together, forcing the surface drag to be felt.

        # 1. Get the drag-applied surface velocity (k=0)
        u_surface = self.u[:, :, 0]
        v_surface = self.v[:, :, 0]

        # 2. Get the runaway jet velocity (k=1)
        u_jet = self.u[:, :, 1]
        v_jet = self.v[:, :, 1]
        
        # 3. Define a DYNAMIC coupling factor (The "Dynamic Anchor")
        # The strength of the anchor is now INVERSELY proportional to
        # the jet's wind speed. This allows a high-shear eyewall
        # to form at high intensity without "snapping" the core.
        
        # === PATCH V24: THE MAXIMUM GRIP ===
        # CRITICAL FIX: We must use the MAXIMUM wind speed across the lower levels (k=0 to k=3)
        # to determine anchor strength. Using just k=1 was missing the core intensity.
        # The Storm Tracker uses the 3D maximum to classify intensity - the anchor must too!
        
        # 1. Get wind speed magnitude for the bottom 4 layers
        wind_mag_lower = xp.sqrt(self.u[:, :, :4]**2 + self.v[:, :, :4]**2)
        
        # 2. Find the max wind speed at each (x,y) point across these vertical levels
        #    This gives us a 2D map of "Max Low-Level Wind"
        max_wind_2d = xp.max(wind_mag_lower, axis=2)
        
        # 3. Convert to m/s for the interpolation
        max_wind_ms = max_wind_2d * self.U_CHAR
        
        # === PATCH V24b: CONTIGUITY FIX ===
        # CuPy's interp requires C-contiguous arrays
        # The max operation can create non-contiguous views, so force a copy
        max_wind_ms = xp.ascontiguousarray(max_wind_ms)
        
        # === DEBUG: Print max wind values every 300 frames ===
        if frame % 300 == 0:
            max_wind_value = float(xp.max(max_wind_ms))
            mean_wind_value = float(xp.mean(max_wind_ms))
            log_info(f"    DEBUG ANCHOR: Frame {frame}, Max Wind = {max_wind_value:.1f} m/s, Mean Wind = {mean_wind_value:.1f} m/s")

        # === PATCH V28: THE IRON FLOOR ===
        # Raised minimum from 0.10 to 0.25 to prevent "ice skating" at low intensity
        # With corrected steering (now 7 m/s not 13 m/s), we need stronger grip
        anchor_strength = xp.interp(
            max_wind_ms,  # <--- USING MAX WIND, NOT JET WIND
            xp.asarray([15.0, 33.0, 50.0, 70.0]),  # <--- LOWERED THRESHOLD TO 15 M/S
            xp.asarray([0.25, 0.40, 0.55, 0.70])   # <--- IRON FLOOR: 0.25 minimum!
        )
        
        # === PATCH V61: LAND-BASED DECOUPLING (Zombie Fix) ===
        # Diagnosis: Momentum Anchor was transferring "Flywheel" energy from k=1
        # to k=0, overriding land friction and creating "Zombie Harvey" (145kts overland).
        # Fix: Weaken the anchor over land to allow surface friction to kill the winds.
        if self.use_landfall_physics:
            # Nonlinear scaling: coastlines preserve coherence,
            # deep inland allows surface shear-off.
            # 100% Land = Cut anchor strength by 75%.
            anchor_factor = 1.0 - (0.75 * land_fraction**2)
            anchor_strength *= anchor_factor
            
            # === PATCH V61.1: TERRAIN ROUGHNESS (VISCOSITY BOOST) ===
            # Land creates turbulence that shreds structure.
            # Boost local viscosity (mu) where land is present.
            # Base mu is ~0.12. Land adds up to +0.10.
            terrain_roughness = 0.10 * land_fraction
        else:
            terrain_roughness = 0.0
            
        # === *** END PATCH (V13 "Dynamic Anchor") *** ===

        # 4. Calculate the change:
        #    - The surface (k=0) gets "pulled up" by the jet.
        #    - The jet (k=1) gets "dragged down" by the surface.
        u_change = anchor_strength * (u_jet - u_surface)
        v_change = anchor_strength * (v_jet - v_surface)

        # 5. Apply the anchor
        self.u[:, :, 0] += u_change
        self.v[:, :, 0] += v_change
        self.u[:, :, 1] -= u_change
        self.v[:, :, 1] -= v_change
        
        # 6. Apply standard mixing to all *other* layers (k=2+)
        # (We still need to mix momentum in the rest of the atmosphere)
        for k in range(2, self.nz - 1):
            self.u[:, :, k] += C_mix[:, :, k] * (
                self.u[:, :, k + 1] - 2 * self.u[:, :, k] + self.u[:, :, k - 1]
            )
            self.v[:, :, k] += C_mix[:, :, k] * (
                self.v[:, :, k + 1] - 2 * self.v[:, :, k] + self.v[:, :, k - 1]
            )
        # === END MOMENTUM ANCHOR ===
        
        # === ENSEMBLE: Progressive Equilibrium - Diagnostic Print Block ===
        if frame > 0 and frame % 300 == 0:
            latent_wm2 = q_f * self.L_v  # q_f is mean_q_flux from apply_surface_fluxes
            k_z_mean = xp.mean(K_z)     # <-- ENSEMBLE FIX: Add K_z back
            c_mix_mean = xp.mean(C_mix) # <-- ENSEMBLE FIX: Keep C_mix
            # FIX: damp_factor is a scalar, so we just check it directly
            wisdom_pct = 100.0 if damp_factor < 1.0 else 0.0
            
            # --- DIAGNOSTIC FOR DYNAMIC ANCHOR ---
            # PATCH V24c: Report MAX anchor (core value) instead of domain mean
            anchor_strength_max = float(xp.max(anchor_strength))
            anchor_strength_mean = float(xp.mean(anchor_strength))
            max_wind_value = float(xp.max(max_wind_ms))
            log_info(f"DIAG FRAME {frame}: Latent={latent_wm2:.0f} W/m², K_z_mean={k_z_mean:.1f}, C_mix_mean={c_mix_mean:.2f}, Anchor_MAX={anchor_strength_max:.3f}, Anchor_mean={anchor_strength_mean:.3f}, MaxWind={max_wind_value:.1f}m/s, Tau={self.tau_condensation_s}s, WISDOM%={wisdom_pct:.1f}%")
        
        # Get the raw steering data (smoothing is now handled by the Nudge)
        u_steer_raw = self.data_interface.u_target
        v_steer_raw = self.data_interface.v_target
        
        # Kalman filter update for steering flow
        if frame % self.kalman_blend_interval == 0 and frame > 0:
            # Feed the raw data to the Kalman filter.
            # The *output* of the filter will be smoothed by the Nudge.
            self.kalman.update(xp.stack((u_steer_raw, v_steer_raw), axis=-1))
            self.frames_since_kalman_update = 0

        # === PATCH V34 FIXED: THE SPEED LIMITER (NO MORE 1e-9!) ===
        # Diagnosis: Stalled storms can hit numerical singularities (>3000 kts).
        # Fix: Hard cap velocity at 225 kts (115 m/s).
        # V34 Bug Fix: Removed 1e-9 hack that was AMPLIFYING noise in calm areas!
        # The 1e-9 hack created scale_factors of BILLIONS in near-zero velocity zones,
        # amplifying noise and breaking the limiter. This fix uses xp.where to avoid
        # division by zero naturally - only scaling where actually needed.
        
        # 1. Calculate magnitude (using GPU array xp)
        u_mag = xp.sqrt(self.u**2 + self.v**2)
        
        # 2. Define Max Speed (225 kts converted to dimensionless)
        # 225 kts = 115.75 m/s
        max_speed_dim = 115.75 / self.U_CHAR
        
        # 3. Apply Hard Brake if limit exceeded
        max_current = float(xp.max(u_mag))
        if max_current > max_speed_dim:
            # Identify cells that need braking (velocity > threshold)
            brake_mask = u_mag > max_speed_dim
            
            # FIXED: Use xp.where to compute scale_factor safely
            # - Where brake_mask=True: scale = max_speed_dim / u_mag (proper scaling)
            # - Where brake_mask=False: scale = 1.0 (no change)
            # This avoids division by zero AND the 1e-9 amplification bug!
            scale_factor = xp.where(brake_mask, 
                                    max_speed_dim / u_mag,  # Real scaling for fast cells
                                    1.0)                     # No change for slow cells
            
            # Apply scaling to ALL cells (but scale_factor=1.0 where not needed)
            self.u *= scale_factor
            self.v *= scale_factor
            
            # Log intervention (rate limited)
            if frame % 100 == 0:
                print(f"    V34 LIMITER (FIXED): Capped {max_current * self.U_CHAR * 1.94:.0f} kts -> 225 kts")
        # === END PATCH V34 FIXED ===

        # Advection
        self.u, self.v, self.w, self.q, self.T = [
            self.solver.advect(f, self.u, self.v, self.w) 
            for f in [self.u, self.v, self.w, self.q, self.T]
        ]
        
        # <--- UNIFIED FIX: Use physical dt for horizontal diffusion ---
        # Get physical dx^2 (assuming dx=dy)
        dx_phys_m_sq = self.domain_scaler.dimensionless_to_physical_x(self.dx)**2
        
        # === ENSEMBLE: Dynamic Viscosity Protocol ===
        # The fixed mu=0.12 was too high for genesis (killing the storm)
        # and too low for hypercanes (allowing a "zombie" storm).
        # We now scale mu with intensity.
        
        # Get current intensity in knots
        current_wind_kts = self.storm_tracker.get_max_wind()
        
        # 1. Low mu (0.10) for genesis (0-70 kts) to prevent fizzling.
        # 2. Baseline mu (0.12) for mature storms (70-170 kts).
        # 3. High mu (0.22) for "zombie" hypercanes (170+ kts)
        #    to dissipate runaway numerical energy.
        
        # --- *** START PATCH (V12.1 "The Hard-Deck Safety Net") *** ---
        # Diagnosis: V12 was too soft; storm hit 248 kts (Hypercane) because drag was too low.
        # Fix: Steeper ramp at H5 + Safety Override for Hypercanes.

        # 1. Get current metrics
        current_wind_kts = self.storm_tracker.get_max_wind()
        current_coherence = self.storm_tracker.get_chimera_coherence()
        
        # 2. Steeper Curve (The "Safety Net")
        # 0-130 kts: Gentle (0.08-0.12) - Let it grow naturally.
        # 130-155 kts: Firm (0.12-0.25) - Cap H4.
        # 155-175 kts: Heavy (0.25-0.45) - The "Soft Wall" for H5.
        # 175+ kts: Emergency (0.45-0.75) - Prevent Numerical Explosion.
        mu_target = xp.interp(
            xp.asarray(current_wind_kts),
            xp.asarray([0.0,  130.0, 155.0, 175.0, 200.0]),
            xp.asarray([0.08, 0.12,  0.25,  0.45,  0.75])
        )
        
        # 3. The "Hypercane Override" (Logic Fix)
        # If wind > 155 kts (Strong H5), we MUST brake, regardless of coherence.
        # Only apply the coherence gate if we are below the danger zone.
        if current_wind_kts < 155.0 and current_coherence < 0.85:
            # If storm is weak/messy, keep viscosity low to allow organization
            mu_target = xp.minimum(mu_target, 0.15)

        # 4. Temporal Smoothing (Faster Reaction)
        # Increased from 0.05 to 0.10 to catch rapid explosions faster
        self.mu_current = self.mu_current + 0.10 * (mu_target - self.mu_current)
        
        # 5. Apply to Diffusion (With V61 Terrain Roughness)
        # Combine global dynamic viscosity with local terrain roughness
        # broadcasting terrain_roughness (2D) to 3D shape
        mu_total = self.mu_current + terrain_roughness[..., xp.newaxis]

        for f in [self.u, self.v, self.w]: 
            # Note: technically laplacian(f) is correct, but spatially varying mu 
            # should really be div(mu * grad(f)). For now, this approximation 
            # (boosted dissipation over land) is sufficient for the "shredding" effect.
            f += self.dt_solver * mu_total * self.solver.laplacian(f)
            
        # Debug Print (Every 500 frames)
        if frame % 500 == 0:
            mu_val = float(self.mu_current) if hasattr(self.mu_current, 'item') else float(self.mu_current)
            target_val = float(mu_target) if hasattr(mu_target, 'item') else float(mu_target)
            log_info(f"    PHYSICS DIAG: Wind={current_wind_kts:.1f}kts, COH={current_coherence:.3f} -> Mu_Target={target_val:.3f}, Mu_Actual={mu_val:.3f}")
            
        # --- *** END PATCH (V12.1 "The Hard-Deck Safety Net") *** ---
        
        # Use PHYSICAL timestep, PHYSICAL diffusivities, and PHYSICAL grid spacing
        self.q += (dt_physical_s * self.D_q / dx_phys_m_sq) * self.solver.laplacian(self.q)
        self.T += (dt_physical_s * self.kappa_T / dx_phys_m_sq) * self.solver.laplacian(self.T)
        
        # Coriolis force (beta-plane approximation)
        # Scale y-coordinate to physical meters for beta calculation
        y_physical_m = xp.arange(self.ny) * self.domain_scaler.dimensionless_to_physical_y(self.dy)
        y_center_m = self.ny/2 * self.domain_scaler.dimensionless_to_physical_y(self.dy)
        
        # This is f (units 1/s)
        f_y_physical = self.f0 + self.beta * (y_physical_m - y_center_m)
        
        # === FIX: Non-dimensionalize by multiplying by characteristic time ===
        f_y_dimensionless = f_y_physical * self.T_CHAR

        # ENSEMBLE FIX: Use a stable Euler step for Coriolis force.
        # Using the "new" u to calculate v (a semi-implicit step) was
        # creating a numerical instability (positive feedback loop).
        # We must use the values from the *start* of the timestep (u_old, v_old)
        # for both calculations.
        u_old = self.u.copy()
        v_old = self.v.copy()
        
        self.u += self.dt_solver * (f_y_dimensionless[:,xp.newaxis] * v_old)
        self.v -= self.dt_solver * (f_y_dimensionless[:,xp.newaxis] * u_old)
        
        # === THE PHOENIX PROTOCOL ===
        # Buoyancy with upper-tropospheric temperature control
        T_upper_kelvin = xp.mean(self.T[:,:,self.nz*2//3:]) + 273.15
        eta = xp.clip(0.525 * (220.0 / T_upper_kelvin), 0.05, 0.40)
        
        # === FIX: Calculate non-dimensional buoyancy scaling factor ===
        # This is (g * L_z) / U_char^2, or 1/Fr^2 (Froude number squared)
        
        g_physical = self.g  # 9.81 m/s^2
        L_z_physical = self.domain_scaler.physical_lz_m # 20,000 m
        U_char_sq = self.U_CHAR**2 # 50.0^2 = 2500
        
        buoyancy_scaling_factor = (g_physical * L_z_physical) / (U_char_sq + 1e-9)
        
        # Temperature anomaly term (dimensionless)
        T_anomaly = (self.T - xp.mean(self.T)) / (xp.mean(self.T) + 273.15 + 1e-9)
        
        # ENSEMBLE FIX: Implement a "Stratospheric Lid"
        # The T-w-T bomb is exploding in the upper atmosphere.
        # We must create a vertical profile that weakens buoyancy
        # with altitude, mimicking the real stratosphere.

        # 1. Get the base throttle (our previous fix)
        base_throttle = 0.001

        # 2. Create the vertical "lid" profile
        # z_profile ranges from 0.0 (bottom) to 1.0 (top)
        z_profile = xp.linspace(0, 1, self.nz)[xp.newaxis, xp.newaxis, :]
        
        # 3. Create the throttle profile
        # (1.0 - z_profile**2): 100% power at k=0, ramping
        # quadratically down to 0% power at the top (k=nz-1).
        z_throttle_profile = (1.0 - z_profile**2)

        # 4. Apply the throttles
        # The final throttle is now (base_throttle * z_throttle_profile)
        #
        # --- *** ENSEMBLE V6: THE "GLOBAL THERMOSTAT" FIX *** ---
        # The storm is running away on a pure buoyancy-advection
        # feedback loop, even when latent heat is zero.
        # We MUST connect our "Global Kill-Switch" (damp_factor)
        # to this buoyancy engine as well.
        self.w += self.dt_solver * (
            base_throttle * z_throttle_profile * buoyancy_scaling_factor * T_anomaly * eta
        ) * damp_factor
        # --- *** END ENSEMBLE V6 FIX *** ---
        
        # <--- UNIFIED FIX: Use physical dt for condensation ---
        # === ENSEMBLE: Progressive Equilibrium - The Engine ===
        # Calculate saturation humidity
        q_sat = self.boundaries.calculate_saturation_humidity(self.T)
        
        # ENSEMBLE FIX (per Grok): Add "safety valve" to prevent
        # the "Delayed Supersat Bomb" from "marinating".
        q_supersat = xp.maximum(0, self.q - q_sat)
        q_supersat = xp.minimum(q_supersat, 0.008) # Cap at 0.008 kg/kg
        
        # ENSEMBLE: tau = 5400s = 90 minutes (realistic precipitation lag)

        # ENSEMBLE FIX: BREAK THE T-w-T FEEDBACK BOMB.
        # The 'up_factor' logic was linking the condensation rate
        # *directly* to the updraft speed, which was *caused* by the heat
        # from condensation. This is a runaway positive feedback loop.
        # We are DECOUPLING it. The condensation rate will now *only*
        # Physical condensation rate (units of 1/s)
        cond_rate_phys = (q_supersat / self.tau_condensation_s)
        
        # Change in q and T over the physical timestep
        dq_condense = dt_physical_s * cond_rate_phys
        
        # === *** THE "ENGINE KILL-SWITCH" FIX (UPDATED V13) *** ===
        # The "Global Kill-Switch" (damp_factor) throttles latent heat based on max wind.
        # V13 UPDATE: "Fatigue Protocol" - Throttles heat based on Lock Score.
        
        # 1. Get Structural Health
        current_lock = self.storm_tracker.get_lock_score()
        
        # 2. Calculate Fatigue Factor
        # If Lock Score is healthy (> 0.50), we run at 100% efficiency.
        # If Lock Score drops (crumbling eyewall), we linearly choke the engine.
        # At Lock 0.35 (disorganized), we cut fuel by 50%.
        # At Lock 0.25 (collapse), we cut fuel by 100%.
        if current_lock > 0.50:
            fatigue_factor = 1.0
        else:
            # Ramp from 1.0 down to 0.0 as lock drops from 0.50 to 0.25
            # Use max/min instead of xp.clip since current_lock is a scalar
            fatigue_factor = max(0.0, min(1.0, (current_lock - 0.25) / 0.25))
        
        # === V42 PARK BUSTER: Recovery Override ===
        # GROK's Fix: Force minimum fuel during recovery to rebuild structure
        if hasattr(self, 'recovery_fatigue_min') and self.recovery_fatigue_min < 1.0:
            fatigue_factor = max(fatigue_factor, self.recovery_fatigue_min)
        # === END V42 RECOVERY OVERRIDE ===
            
        # 3. Apply the Fatigue
        # If the storm is disorganized, this forces it to weaken (ERC),
        # allowing the wind speed to drop and the viscosity to reset.
        effective_fuel_throttle = damp_factor * fatigue_factor
        
        # Debug print for Fatigue (only if active)
        if fatigue_factor < 1.0 and frame % 100 == 0:
             log_info(f"    FATIGUE ACTIVE: Lock={current_lock:.2f} -> Fuel Cut={(1.0-fatigue_factor)*100:.0f}%")

        # 4. Execute Heat Release
        dT_heat = dt_physical_s * (self.L_v / self.c_p) * cond_rate_phys * effective_fuel_throttle
        # === *** END "ENGINE KILL-SWITCH" FIX *** ===
        
        # === CONDENSATION AND LATENT HEATING ===
        # Apply changes
        self.T += dT_heat
        self.q -= dq_condense

        # Temperature cap (prevent unrealistic warming)
        self.T = xp.minimum(self.T, 40.0)
        
        # Apply guidance force and projection
        self._apply_guidance_force(frame)

        # --- *** START PATCH (V16 "Genesis-Aware Inverted damper" Damper) *** ---
        # The V12 damper fixed genesis but strangled the ERC recovery.
        # The V15 "Genesis-Aware damper" was in the incorrectly applied fixed with V16
        # This new logic is "Genesis-Aware" (time-based) AND
        # "Structurally-Aware" (lock-based).
        
        # 1. Get current intensity AND lock score
        current_wind_kts = self.storm_tracker.get_max_wind()
        lock_score = self.storm_tracker.get_lock_score()
        
        # 2. Set damping
        if frame < 2500:
            # --- GENESIS PHASE ---
            # We are in the initial spin-up.
            # Force the damper OFF to allow the core to form.
            damping_h_dynamic = 1.0 # Damping is OFF
        else:
            # --- MATURE / ERC PHASE ---
            # Switch to the V10 "Structurally-Aware" logic.
            # Damping is now based on LOCK SCORE to help
            # a messy ERC re-organize.

            # --- *** START PATCH (V16 "Inverted Damper") *** ---
            # The V15 logic was inverted. We need the damper
            # ON (0.835) at LOW lock, and OFF (1.0) at HIGH lock.
            damping_h_dynamic = xp.interp(
                xp.asarray(lock_score),
                xp.asarray([0.60, 0.80]),  # Lock score thresholds
                xp.asarray([0.835, 1.0])   # [ON, OFF] <-- INVERTED!
            )
            # --- *** END PATCH (V16 "Inverted Damper") ---
        
        # ENSEMBLE FIX: Apply damping to H and W.
        # ... (rest of the project() call) ...
        self.u, self.v, self.w, p = self.solver.project(
            self.u, self.v, self.w, 
            damping_factor_h=damping_h_dynamic,  # <-- USE NEW DYNAMIC VALUE
            damping_factor_w=0.3
        )

        # NOTE: The Kalman filter logic was MOVED UP.
        # We only call the nest update here.
        self._update_moving_nest(frame)
        self.frames_since_kalman_update += 1
        
        # === DIAGNOSTICS AND VISUALIZATION ===
        if frame % 100 == 0:
            vort_mag = xp.sqrt(sum(c**2 for c in self.solver.curl(self.u, self.v, self.w)))
            
            # GPU Safety: Move fields to CPU for the tracker
            p_cpu = p.get() if hasattr(p, 'get') else p
            vort_cpu = vort_mag.get() if hasattr(vort_mag, 'get') else vort_mag
            
            def_core = self.storm_tracker.update_metrics(frame, p_cpu, vort_cpu)
            
            self.frame_history.append(frame)
            
            # FIX: Convert GPU scalars to Python floats before appending to lists
            # FIX: Force conversion to CPU using .item() for absolute safety
            max_wind = self.storm_tracker.get_max_wind()
            if hasattr(max_wind, 'item'):
                self.max_wind_history.append(max_wind.item())
            else:
                self.max_wind_history.append(float(max_wind))
            
            # Calculate latent heat mean on GPU, then move to CPU
            mean_latent = xp.mean(dT_heat / dt_physical_s) * 86400
            if hasattr(mean_latent, 'item'):
                self.latent_heat_history.append(mean_latent.item())
            else:
                self.latent_heat_history.append(float(mean_latent))

            # FIX: Pass GPU arrays (p, vort_mag) to AMR, not CPU copies!
            # The AMR handler is now GPU-accelerated and expects CuPy arrays.
            self.amr.find_refinement_regions(p, vort_mag, def_core)
            
            if frame > 0 and frame % 500 == 0:
                log_info("\n---VISUALIZATION CYCLE INITIATED ---")
                # GPU/CPU Transfer: Download all primary fields for visualization
                p_vis = p.get() if USE_GPU else p
                q_vis = self.q.get() if USE_GPU else self.q
                T_vis = self.T.get() if USE_GPU else self.T
                u_vis = self.u.get() if USE_GPU else self.u
                v_vis = self.v.get() if USE_GPU else self.v
                w_vis = self.w.get() if USE_GPU else self.w
                
                # === FIX: Download the DataInterface steering fields for the overlay ===
                # The data_interface fields (u_target, v_target) are CuPy and need .get()
                di_cpu = self.data_interface
                if USE_GPU:
                    # Create a CPU-safe temporary object for the visualizer to read
                    class CpuDataInterface:
                        # Only download the necessary fields for the quiver plot
                        u_target = di_cpu.u_target.get()
                        v_target = di_cpu.v_target.get()
                        u_old = di_cpu.u_old.get()
                        v_old = di_cpu.v_old.get()
                        
                        # Pass through metadata needed for the plot axes/bounds
                        lon_bounds = di_cpu.lon_bounds
                        lat_bounds = di_cpu.lat_bounds
                        
                        # FIX: Pass through the historical track (Already CPU/Pandas)
                        historical_track = di_cpu.historical_track
                    
                    di_overlay = CpuDataInterface()
                else:
                    di_overlay = di_cpu
                # === END FIX ===
                
                self.visualizer.generate_3d_scene(frame, p_vis, q_vis, T_vis, u_vis, v_vis, w_vis)
                self.visualizer.generate_2d_slice(frame, p_vis, u_vis, v_vis)
                self.visualizer.generate_diagnostic_overlay(frame, self.storm_tracker, di_overlay)
                log_info("--- VISUALIZATION CYCLE COMPLETE ---\n")

    def plot_results(self, acc_scores):
        """Generate final diagnostic plots."""
        log_info("Generating and saving final plots...")
        fig, axs = plt.subplots(1, 2, figsize=(20,8))
        rmse_km, _, _, _, _, _, nav_conf = acc_scores # <--- UNIFIED FIX: RMSE is in km
        
        title_mode = f"Oracle V4 ({self.oracle_mode.upper()})"
        # <--- UNIFIED FIX: Report RMSE in km ---
        title = (f'{title_mode} ({self.initial_wind_kts} kts) | Oracle Triggers: {self.oracle_trigger_count}\n'
                f'Nav Confidence: {nav_conf:.1f}% | Track RMSE: {rmse_km:.2f} km')
        fig.suptitle(title, fontsize=16)
        
        # === FIX: Ensure all history lists are plain Python/NumPy arrays ===
        # Convert list of floats to numpy array for safe plotting with Matplotlib
        frame_history_cpu = np.asarray(self.frame_history)
        max_wind_history_cpu = np.asarray(self.max_wind_history)
        latent_heat_history_cpu = np.asarray(self.latent_heat_history)
        
        axs[0].plot(frame_history_cpu, max_wind_history_cpu, color='cyan', label='Max Wind (kts)')
        axs[0].set_title("Maximum Wind Speed")
        axs[0].set_ylabel("Wind Speed (knots)")
        axs[0].grid(True, linestyle=':')
        
        # Saffir-Simpson category lines
        for kts in [64, 83, 96, 113, 137]: # Corrected SS scale: 74, 96, 111, 130, 157
            axs[0].axhline(y=kts, color='gray', linestyle='--', alpha=0.7)
        
        axs[1].plot(frame_history_cpu, latent_heat_history_cpu, color='red', label='Latent Heat (K/day)')
        axs[1].set_title("Thermodynamic Engine")
        axs[1].set_ylabel("Latent Heat Release (K/day)", color='red')
        axs[1].tick_params(axis='y', labelcolor='red')
        axs[1].grid(True, linestyle=':')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename = f"oracle_v4_{self.oracle_mode}_run_{self.initial_wind_kts}kts_metrics.png"
        plt.savefig(os.path.join(self.plot_dir, filename))
        plt.close()
        log_info(f"Plots saved to '{self.plot_dir}'")

    def run(self):
        """Execute the full simulation."""
        log_info(f"\n--- Running Oracle V4 ({self.oracle_mode.upper()}): {self.initial_wind_kts} kts ---")
        try:
            for i in range(self.num_frames): 
                self.update(i)
        finally:
            log_info(f"\n--- Run Complete: {self.initial_wind_kts} kts ---")
            if self.oracle_trigger_count > 0:
                log_info(f"    Oracle's Judgment triggered {self.oracle_trigger_count} times during simulation")
            
            # Print Oracle statistics if in adaptive mode
            if self.oracle_mode == 'adaptive' and self.oracle is not None:
                stats = self.oracle.get_statistics()
                log_info(f"\n=== Oracle V4 Statistics ===")
                log_info(f"  Total checks: {stats['total_checks']}")
                log_info(f"  Total triggers: {stats['total_triggers']}")
                log_info(f"  Trigger rate: {stats['trigger_rate']:.1%}")
                log_info(f"  Phase usage: {stats['phase_usage']}")
                
            # Evaluate and save memory
            if self.collect_memory:
                log_info(f"\n=== Processing Oracle Memory ===")
                if self.oracle_memory.historical_track is None:
                    log_info("  ⚠ Historical track not set! Setting now...")
                    self.oracle_memory.set_simulation_params(
                        start_time=self.sim_start_time,
                        dt_solver=self.dt_solver,
                        historical_track=self.data_interface.historical_track,
                        t_char=self.T_CHAR  # <--- ADD THIS
                    )
                self.oracle_memory.evaluate_interventions()
                memory_file = self.oracle_memory.save()

                # V55: Enrich JSON with simulation mode and helper statistics
                try:
                    import json
                    with open(memory_file, 'r') as f:
                        data = json.load(f)
                    
                    # Add V55 metadata
                    if 'metadata' in data:
                        data['metadata']['simulation_mode'] = self.simulation_mode
                        data['metadata']['ablation_mode'] = self.ablation_mode
                        data['metadata']['helper_statistics'] = {
                            'stall_breaker_activations': self.stall_breaker_count,
                            'ghost_nudge_activations': self.ghost_nudge_count,
                            'ghost_nudge_mean_strength': float(sum(self.ghost_nudge_history) / len(self.ghost_nudge_history)) if self.ghost_nudge_history else 0.0,
                            'ghost_nudge_max_strength': float(max(self.ghost_nudge_history)) if self.ghost_nudge_history else 0.0,
                        }
                    
                    # Write back
                    with open(memory_file, 'w') as f:
                        json.dump(data, f, indent=2)
                except Exception as e:
                    log_info(f"  ⚠️ Could not enrich JSON with V55 metadata: {e}")
                log_info(f"  ✓ Memory saved to: {memory_file}")
            
            acc = self.storm_tracker.calculate_historical_accuracy(0)
            self.storm_tracker.save_path_to_geojson()
            self.plot_results(acc)
            
            # === CLOSE LOGGER ===
            if hasattr(self, 'logger') and self.logger:
                self.logger.close()

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Oracle V4 Hurricane Simulation')
    parser.add_argument('--storm', type=str, default='HUGO', 
                        help='Storm name (e.g., HUGO, KATRINA, ANDREW)')
    parser.add_argument('--year', type=int, default=1989,
                        help='Storm year (e.g., 1989, 2005, 1992)')
    parser.add_argument('--initial-wind', type=float, default=50.0,
                        help='Initial wind speed in knots (default: 50.0)')
    parser.add_argument('--oracle-mode', type=str, default='adaptive',
                        choices=['learning', 'adaptive', 'off'],
                        help='Oracle mode: learning (collect data), adaptive (trained), off (disabled)')
    parser.add_argument('--oracle-params', type=str, default='oracle_learned_params_v4.json',
                        help='Path to learned Oracle parameters file')
    
    args = parser.parse_args()
    
    # Create simulation with parsed arguments
    sim = Simulation3D(
        storm_name=args.storm.upper(),
        storm_year=args.year,
        initial_wind_kts=args.initial_wind,
        mu=0.12,  # ENSEMBLE FIX (per Grok): "Stable RI" viscosity
        oracle_mode=args.oracle_mode,
        oracle_params_file=args.oracle_params if args.oracle_mode == 'adaptive' else None,
        collect_memory=True  # Continue learning
    )
    
    sim.run()
