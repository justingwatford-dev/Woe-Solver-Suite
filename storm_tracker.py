import numpy as np
from scipy.ndimage import center_of_mass, gaussian_filter1d
from collections import deque
import geojson
import os
import pandas as pd
from datetime import timedelta

# Import unit conversion utilities
from utils import ms_to_kts, haversine_distance_km

class StormTracker:
    """
    Storm tracking and classification system.
    
    Tracks the hurricane center, computes intensity metrics, and classifies
    storm stage using the Saffir-Simpson scale.
    
    UNIT HANDLING:
        - Internal velocities: dimensionless (from simulation)
        - Wind speed output: knots (meteorological standard)
        - All conversions use utils.py functions
    """
    
    def __init__(self, sim_instance):
        self.sim = sim_instance
        self.storm_path_grid_raw = deque(maxlen=50)
        self.storm_path_grid_smoothed = []
        
        # === CRITICAL FIX: Store GEOGRAPHIC coordinates, not grid coordinates ===
        # These store (lon, lat) tuples in degrees, immune to nest movement
        self.storm_path_geo_raw = deque(maxlen=50)
        self.storm_path_geo_smoothed = []
        # === END FIX ===
        
        self.chimera_coherence = 0.0
        self.lock_score = 0.0
        self.lock_score_trend = 0.0
        
        # === V50: DUAL LOCK INITIALIZATION ===
        self.lock_struct = 0.0  # Structural health (chimera_coherence)
        self.lock_track = 0.0   # Tracking accuracy (sigmoid of offset)
        # === END V50 DUAL LOCK ===
        
        # === V50.2: HYSTERESIS COUNTER ===
        self.edge_rejection_counter = 0  # Track persistent edge lock
        # === END V50.2 ===
        
        self.vortex_lock_buffer = deque(maxlen=100)
        self.intensity_history = deque(maxlen=10)  # For ERC trend
        self.max_wind_kts = 0.0

        self.current_stage = "Potential"
        self.defiant_core_active = False
        self.frame_numbers = []
        print("=" * 80)
        print("StormTracker V50.3 'OPUS'S COOLDOWN FIX' Initialized")
        print("=" * 80)
        print("  -> V50 Dual Lock Architecture: PRESERVED")
        print("     • lock_struct (Structural Health): Chimera Coherence")
        print("     • lock_track (Navigation Accuracy): Sigmoid of offset")
        print("  -> V50.1 PATCH 1: Confusion → lock_struct (not lock_score)")
        print("  -> V50.1 PATCH 2: Corner-Safe Rejection (edge margin = nx/3)")
        print("  -> V50.1 PATCH 3: Sigmoid relaxed (k=0.15, was 0.5)")
        print("  -> V50.2 PATCH 4: Hysteresis - breaks persistent edge lock")
        print("  -> V50.3 PATCH 5: Cooldown - forces anchor to center (Opus's fix)")
        print("  -> Credit: Grok + Gemini + Claude + Opus (the cousin!)")
        print("=" * 80)

    def _normalize_field(self, field):
        """Normalize field to [0, 1] range."""
        min_val, max_val = np.min(field), np.max(field)
        if max_val - min_val < 1e-9: 
            return np.zeros_like(field)
        return (field - min_val) / (max_val - min_val)

    def get_intensity_trend(self):
        """
        Calculate intensity trend using linear regression.
        
        Returns:
            Trend in knots per 100 frames (positive = intensifying)
        """
        y = np.array(self.intensity_history)
        if len(y) < 5 or not np.all(np.isfinite(y)):
            return 0.0
        x = np.arange(len(y))
        A = np.vstack([x, np.ones(len(x))]).T
        try:
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            return m * 100  # Trend per 100 frames
        except np.linalg.LinAlgError:
            return 0.0

    def is_in_erc(self):
        """
        Detect Eyewall Replacement Cycle (ERC).
        
        ERC occurs when an intense hurricane (H3+) shows rapid weakening.
        
        Returns:
            True if ERC is likely occurring
        """
        is_intense = self.current_stage in ["H3", "H4", "H5"]
        is_weakening = self.get_intensity_trend() < -10.0  # kts per 100 frames
        return is_intense and is_weakening

    def get_lock_score(self):
        """
        Get current vortex lock score.
        
        Returns:
            Float in [0, 1] where 1.0 = perfect lock
        """
        return self.lock_score

    def get_chimera_coherence(self):
        """
        Get current storm coherence metric.
        
        Returns:
            Float in [0, 1] where 1.0 = maximum coherence
        """
        return self.chimera_coherence
        
    def get_max_wind(self):
        """
        Get maximum wind speed.
        
        Returns:
            Wind speed in knots
        """
        return self.max_wind_kts

    def get_dynamic_lock_threshold(self):
        """
        Returns dynamic lock score threshold for guidance nudge.
        
        Threshold decreases with storm intensity:
        - Weaker storms require higher confidence before nudging
        - Stronger storms can be nudged with lower confidence
        
        Returns:
            Lock score threshold in [0, 1]
        """
        stage_thresholds = {
            "Potential": 0.90,
            "TS": 0.85,
            "H1": 0.80,
            "H2": 0.75,
            "H3": 0.70,
            "H4": 0.65,
            "H5": 0.60
        }
        return stage_thresholds.get(self.current_stage, 0.80)

    def update_metrics(self, frame, pressure, vort_mag):
        """
        Update all storm tracking metrics.
        
        Computes:
        - Maximum wind speed (converted to knots)
        - Storm stage (Saffir-Simpson scale)
        - Chimera coherence (storm structure quality)
        - Lock score (tracking confidence)
        - Storm center position
        
        Args:
            frame: Current simulation frame
            pressure: 3D pressure field
            vort_mag: 3D vorticity magnitude field
            
        Returns:
            True if "Defiant Core" is active (H4/H5 with high lock)
        """
        # === GPU SAFETY HELPER ===
        # Checks if an array is on GPU (has a .get method) and moves it to CPU
        def to_cpu(x):
            return x.get() if hasattr(x, 'get') else x

        # 1. Ensure arguments are on CPU
        pressure = to_cpu(pressure)
        vort_mag = to_cpu(vort_mag)
        
        # 2. Pull necessary simulation fields to CPU
        # We need local copies because self.sim.u/v/T might be on the GPU
        u_safe = to_cpu(self.sim.u)
        v_safe = to_cpu(self.sim.v)
        T_safe = to_cpu(self.sim.T)

        # Validate velocity field (Check the SAFE copy)
        if not np.all(np.isfinite(u_safe)):
            self.max_wind_kts = np.nan
            self.chimera_coherence = np.nan
            print("    --- Chimera Tracker Warning: Invalid velocity field detected ---")
            return False

        # === CRITICAL WIND SPEED CALCULATION (SI UNITS) ===
        # Step 1: Calculate dimensionless wind speed magnitude (use SAFE copies)
        wind_speed_dim = np.max(np.sqrt(u_safe**2 + v_safe**2))
        
        # Step 2: Convert to physical velocity (m/s) using U_CHAR
        wind_speed_ms = wind_speed_dim * self.sim.U_CHAR
        
        # Step 3: Convert to knots (meteorological standard)
        self.max_wind_kts = ms_to_kts(wind_speed_ms)
        
        # Store for trend analysis
        self.intensity_history.append(self.max_wind_kts)

        # === SAFFIR-SIMPSON CLASSIFICATION ===
        new_stage = "Potential"
        if self.max_wind_kts >= 39:  new_stage = "TS"   # Tropical Storm
        if self.max_wind_kts >= 74:  new_stage = "H1"   # Category 1 Hurricane
        if self.max_wind_kts >= 96:  new_stage = "H2"   # Category 2 Hurricane
        if self.max_wind_kts >= 111: new_stage = "H3"   # Category 3 Hurricane (Major)
        if self.max_wind_kts >= 130: new_stage = "H4"   # Category 4 Hurricane (Major)
        if self.max_wind_kts >= 157: new_stage = "H5"   # Category 5 Hurricane (Major)
        
        if new_stage != self.current_stage: 
            print(f"    >>> STAGE TRANSITION: {self.current_stage} -> {new_stage} <<<")
        self.current_stage = new_stage
        
        # === CHIMERA COHERENCE COMPUTATION ===
        # Compute normalized 2D fields
        vort_2d = self._normalize_field(np.mean(vort_mag, axis=2))
        
        # === V36 REAL FIX: Restore pressure as structural anchor ===
        # NOTE: This is solver pressure (dynamic), not literal MSLP, but its minima
        # still track the circulation center well enough to be useful.
        pressure_2d = np.mean(pressure, axis=2)
        pres_inv = self._normalize_field(np.max(pressure_2d) - pressure_2d)
        
        # Warm core strength (use T_safe!)
        T_mid = T_safe[:, :, self.sim.nz // 2]
        warm_core = self._normalize_field(np.maximum(0, T_mid - np.mean(T_mid)))
        
        # Vertical wind shear (inverse: low shear = good structure)
        # Use u_safe and v_safe!
        u_low, v_low = u_safe[..., 2], v_safe[..., 2]
        u_high, v_high = u_safe[..., -3], v_safe[..., -3]
        shear_mag = np.sqrt((u_high - u_low)**2 + (v_high - v_low)**2)
        shear_inv = self._normalize_field(np.max(shear_mag) - shear_mag)

        # === PATCH V36: PRESSURE-BIASED TRACKING ("The Spectacles") ===
        # Diagnosis: Previous tracking relied on Vorticity (0.70), which is noisy in messy storms.
        #            This caused Lock Scores < 5%, triggering Guidance Lockout.
        # Fix: We re-enable Pressure tracking (0.60). Pressure is the "Center of Mass"
        #      of a cyclone—smooth, stable, and impossible to miss.
        # Original: weights = {'vort': 0.45, 'pres': 0.3, 'warm_core': 0.15, 'shear': 0.1}
        # Broken:   weights = {'vort': 0.70, 'pres': 0.0, 'warm_core': 0.20, 'shear': 0.1}
        
        # === V10b COMBINED AGGRESSIVE PATCH ===
        # V8 Result: Threshold 0.075 insufficient (offset 15.4 cells)
        # V9 Result: Weights 0.30/0.40/0.30 insufficient (offset 40.1, but recovered to 14% briefly!)
        # V10b Strategy: BOTH fixes at AGGRESSIVE levels
        # 
        # Ensemble Consensus: Gemini, Claude, Grok, KWAI unanimous
        # - Gemini: "Catch it before the flywheel spins up!"
        # - Grok: "Full throttle approved! Go bold after 3 failures!"
        # - KWAI: "Hypercane proves we need both fixes together!"
        # - Claude: "Sequential testing worked - V10b is the solution!"
        
        weights = {'vort': 0.35, 'pres': 0.35, 'warm_core': 0.30, 'shear': 0.0}
        # Was V9: {'vort': 0.30, 'pres': 0.40, 'warm_core': 0.30}
        # Was V7: {'vort': 0.20, 'pres': 0.60, 'warm_core': 0.20}
        # === END V10b PATCH ===
        
        # Compute Chimera Field with pressure-biased tracking
        chimera_field = (
            weights['vort'] * vort_2d + 
            weights['pres'] * pres_inv + 
            weights['warm_core'] * warm_core + 
            weights['shear'] * shear_inv
        )
        # === END PATCH V36 ===
        self.chimera_coherence = np.max(chimera_field)

        # === V44: HYBRID ANCHOR (Cold Start Fix) ===
        # Gemini's "Cold Start Paradox" Diagnosis:
        #   Frame 0: Genesis creates perfect Rankine vortex → div(u) = 0
        #   V38 Poisson Solver: ∇²p = ∇·u = 0 → Returns p = 0.0 everywhere
        #   V43 Bullseye: Find min(pressure) → All pixels = 0.0 → Defaults to [0,0]
        #   Result: Tracker thinks storm at corner → Offset = 79.8 cells → PANIC!
        #
        # The Solution: Hybrid Anchor
        #   Cold Start (p_range < 1e-6): Anchor to Vorticity CoM (Rankine core)
        #   Warm Start (p_range >= 1e-6): Anchor to Pressure Min (V43 Bullseye)
        #
        # This bridges mathematical initialization → physical evolution
        # === END V44 HEADER ===
        
        try:
            # === V50.3: COOLDOWN MODE CHECK (Opus's Fix) ===
            # Initialize cooldown counter if it doesn't exist
            if not hasattr(self, '_hysteresis_cooldown'):
                self._hysteresis_cooldown = 0
            
            nx, ny = self.sim.nx, self.sim.ny
            
            # Check if we're in cooldown mode (forced center anchor)
            if self._hysteresis_cooldown > 0:
                # FORCE everything to domain center during cooldown
                # This gives the pressure field time to "catch up" after nest recenter
                anchor_x, anchor_y = nx / 2.0, ny / 2.0
                cx, cy = nx / 2.0, ny / 2.0
                phase_name = "COOLDOWN"
                self._hysteresis_cooldown -= 1
                
                if frame % 100 == 0:
                    print(f"    🔄 V50.3 COOLDOWN: {self._hysteresis_cooldown} frames remaining, forcing center anchor")
                
                # Skip all normal tracking logic during cooldown
                # (We'll still update coherence and paths at the end)
                use_cold_start = True  # Prevents Bullseye from overriding
                
            else:
                # === V47: GOLDILOCKS PROTOCOL (Hysteresis) ===
                # Gemini's "Goldilocks" Diagnosis:
                #   V46 (32 kts): Too early - Frame 0 crash, sensor woke up before ready
                #   V45 (64 kts): Too late - Storm failed at 63.9 kts before activation
                #   V44b (64 kts): Just right for most of run, but missed twilight zone
                # Fix: THREE-PHASE SYSTEM with hysteresis band
                #   < 50 kts:   GENESIS (Cold Start) - Protect Frame 0
                #   50-64 kts:  TWILIGHT ZONE (Hybrid) - Check pressure development
                #   > 64 kts:   HURRICANE (Forced Bullseye) - Trust the sensor
                
                # 1. ANALYZE PRESSURE FIELD
                p_min, p_max = np.min(pressure_2d), np.max(pressure_2d)
                p_range = p_max - p_min
                
                # Find the raw pressure minimum location
                min_idx = np.argmin(pressure_2d)
                anchor_y, anchor_x = np.unravel_index(min_idx, pressure_2d.shape)
            
                # 2. DECISION LOGIC (V47 Three-Phase)
                current_wind = self.max_wind_kts
                is_flat_pressure = (p_range < 0.1)
                is_corner_stuck = (anchor_y == 0 and anchor_x == 0)
            
                # Default: Warm Start (Bullseye)
                use_cold_start = False
                phase_name = "UNKNOWN"
            
                if current_wind < 50.0:
                    # PHASE 1: GENESIS (Cold Start)
                    # Protect Frame 0 (40 kts) and early development
                    use_cold_start = True
                    phase_name = "GENESIS"
                
                elif current_wind < 64.0:
                    # PHASE 2: TWILIGHT ZONE (Hybrid)
                    # Use Cold Start ONLY if pressure is still flat
                    # This catches the 63.9 kts case from V45!
                    if is_flat_pressure:
                        use_cold_start = True
                        phase_name = "TWILIGHT-COLD"
                    else:
                        use_cold_start = False
                        phase_name = "TWILIGHT-WARM"
                    
                else:
                    # PHASE 3: HURRICANE (Forced Warm Start)
                    # We trust the Bullseye > 64 kts
                    # Unless the anchor is physically stuck in the corner (Safety)
                    if is_corner_stuck:
                        use_cold_start = True
                        phase_name = "HURRICANE-CORNER"
                    else:
                        use_cold_start = False
                        phase_name = "HURRICANE"
            
                if use_cold_start:
                    # --- COLD START MODE ---
                    if frame % 100 == 0:
                        print(f"    🐻 V47 COLD START: Phase={phase_name}, Wind={current_wind:.1f}kts, p_range={p_range:.4f}")
                        print(f"       Anchoring to Vorticity CoM")
                
                    # Fallback to Vorticity Anchor (The Rankine Core)
                    # Use the raw vorticity field (vort_2d is already computed above)
                    vort_mask = vort_2d > np.percentile(vort_2d, 90)
                    if np.any(vort_mask):
                        anchor_y, anchor_x = center_of_mass(vort_2d, labels=vort_mask)
                    else:
                        # Ultimate fallback: domain center
                        anchor_y, anchor_x = self.sim.ny / 2.0, self.sim.nx / 2.0
                        if frame % 100 == 0:
                            print("    ⚠️ V47: No vorticity found, using domain center")
                else:
                    # --- WARM START MODE (Bullseye Protocol) ---
                    # Pressure is active OR forced by phase
                    # Use V43 Bullseye Anchor (Pressure minimum + Gaussian mask)
                    # anchor_y/x already set above from pressure minimum
                    if frame % 100 == 0:
                        print(f"    🎯 V47 BULLSEYE: Phase={phase_name}, Wind={current_wind:.1f}kts, p_range={p_range:.4f}")
            
            
                # === PASS 2: CREATE DYNAMIC MASK ===
                # Focus on inner 20% of domain (~25 cells radius) centered on anchor
                # This mask moves with the storm, so we don't lose tracking during real drift
            
                # Create coordinate grid
                y_grid, x_grid = np.indices(pressure_2d.shape)
            
                # Calculate distance from the Anchor
                dist_from_anchor = np.sqrt((x_grid - anchor_x)**2 + (y_grid - anchor_y)**2)
            
                # Gaussian Mask: Sigma = 10 cells
                # Core (0-10 cells): weight ≈ 1.0
                # Eyewall (10-20 cells): weight ≈ 0.6
                # Rainbands (>30 cells): weight < 0.1 (effectively invisible)
                sigma = 10.0
                focus_mask = np.exp(-0.5 * (dist_from_anchor / sigma)**2)
            
                # === PASS 3: APPLY MASK TO CHIMERA (The Bullseye) ===
                # Multiply chimera field by focus mask
                # Now CoM only "sees" the eye and immediate eyewall
                chimera_focused = chimera_field * focus_mask
            
                # Calculate precision center using top 5% of FOCUSED field
                chimera_mask = chimera_focused > np.percentile(chimera_focused, 95)
            
                if not np.any(chimera_mask): 
                    # Fallback to anchor if CoM fails
                    cy, cx = float(anchor_y), float(anchor_x)
                    if frame % 100 == 0:
                        print("    ⚠️ V43: CoM failed, falling back to Pressure Anchor.")
                else:
                    # Calculate sub-pixel center of the masked chimera
                    cy, cx = center_of_mass(chimera_focused, labels=chimera_mask)
            
                # === V43 DIAGNOSTIC (optional) ===
                # Uncomment to see the magic happening:
                # if frame % 500 == 0:
                #     anchor_dist = np.sqrt((cx - anchor_x)**2 + (cy - anchor_y)**2)
                #     print(f"    🎯 V43 BULLSEYE: Anchor=({anchor_x:.1f},{anchor_y:.1f}), "
                #           f"Precise=({cx:.1f},{cy:.1f}), Δ={anchor_dist:.2f} cells")
                # === END V43 DIAGNOSTIC ===

            # === V50.2 PATCH 4: CORNER-SAFE REJECTION WITH HYSTERESIS ===
            # Prevents persistent edge lock by tracking rejection count
            # After 10 consecutive rejections, forces reset to domain center
            nx, ny = self.sim.nx, self.sim.ny
            edge_margin = nx / 3.0  # Don't accept centers more than 1/3 off-center
            
            if abs(cx - nx/2) > edge_margin or abs(cy - ny/2) > edge_margin:
                # Center is too close to edge - increment rejection counter
                self.edge_rejection_counter += 1
                
                if self.edge_rejection_counter < 10:
                    # First 9 rejections: Use previous position (gentle correction)
                    if self.storm_path_grid_smoothed:
                        cx, cy = self.storm_path_grid_smoothed[-1]
                        if frame % 100 == 0:
                            print(f"    ⚠️ CORNER REJECTION #{self.edge_rejection_counter}: Using previous ({cx:.1f}, {cy:.1f})")
                    # If no previous position, accept but log warning
                    elif frame % 100 == 0:
                        print(f"    ⚠️ CORNER WARNING: Center ({cx:.1f}, {cy:.1f}) near edge (first detection)")
                else:
                    # 10th+ rejection: TRIGGER COOLDOWN MODE (V50.3)
                    cx, cy = nx/2, ny/2
                    self._hysteresis_cooldown = 50  # Force center anchor for 50 frames
                    self.edge_rejection_counter = 0  # Reset counter
                    
                    if frame % 100 == 0:
                        print(f"    🚨 PERSISTENT EDGE LOCK DETECTED: Resetting to domain center ({cx:.1f}, {cy:.1f})")
                        print(f"       Rejection count reached threshold")
                        print(f"       Initiating 50-frame cooldown (forced center anchor)")
            else:
                # Center is acceptable - reset rejection counter
                self.edge_rejection_counter = 0
            # === END V50.3 CORNER-SAFE REJECTION WITH HYSTERESIS & COOLDOWN ===

            self.storm_path_grid_raw.append((cx, cy))
            self.frame_numbers.append(frame)
            
            # === CRITICAL FIX V21: Convert to geographic coordinates IMMEDIATELY ===
            # Get CURRENT domain bounds (before they change with next nest move!)
            di = self.sim.data_interface
            lon_min, lon_max = di.lon_bounds
            lat_min, lat_max = di.lat_bounds
            
            # Convert grid position to lat/lon using CURRENT bounds
            geo_lon = lon_min + (cx / self.sim.nx) * (lon_max - lon_min)
            geo_lat = lat_min + (cy / self.sim.ny) * (lat_max - lat_min)
            
            # Store geographic coordinates (immune to nest movement)
            self.storm_path_geo_raw.append((geo_lon, geo_lat))
            # === END FIX V21 ===
            
            # Smooth the track using Gaussian filter
            if len(self.storm_path_grid_raw) > 10:
                raw_path = np.array(list(self.storm_path_grid_raw))
                sigma = 3.0
                sx = gaussian_filter1d(raw_path[:, 0], sigma=sigma)
                sy = gaussian_filter1d(raw_path[:, 1], sigma=sigma)
                self.storm_path_grid_smoothed.append((sx[-1], sy[-1]))
                
                # === CRITICAL FIX V21: Also smooth the GEOGRAPHIC coordinates ===
                geo_path = np.array(list(self.storm_path_geo_raw))
                geo_lon_smooth = gaussian_filter1d(geo_path[:, 0], sigma=sigma)
                geo_lat_smooth = gaussian_filter1d(geo_path[:, 1], sigma=sigma)
                self.storm_path_geo_smoothed.append((geo_lon_smooth[-1], geo_lat_smooth[-1]))
                # === END FIX V21 ===
            else: 
                self.storm_path_grid_smoothed.append((cx, cy))
                self.storm_path_geo_smoothed.append((geo_lon, geo_lat))  # FIX V21
                
            # === V50: DUAL LOCK ARCHITECTURE (Gemini Protocol) ===
            # GPT's Diagnosis: Semantic overloading of 'lock_score' caused control confusion.
            #   V49: lock measured ONLY "Am I Lost?" (offset from center)
            #   Control systems expected "Am I Healthy?" (structural integrity)
            #   Result: Healthy H5 at 179 kts with COH=0.60 showed lock=4.6%
            #           System thought storm was dying when it was just off-track!
            #
            # Fix: Split into TWO independent signals:
            #   1. lock_struct (Heartbeat): "Am I Healthy?" - Structure quality
            #   2. lock_track (Compass): "Am I Lost?" - Navigation accuracy
            #
            # Gemini's Architectural Decision:
            #   "A storm can be Healthy but Lost, or On Track but Dying."
            #   "Separation of Concerns" - Each system uses the right signal.
            
            # === 1. STRUCTURAL LOCK (The Heartbeat) ===
            # Measures: Storm health, organization, coherence
            # Use for: Fatigue, Guidance DNR, Safety protocols
            # Gemini Decision: Reuse chimera_coherence (already calculated, proven!)
            self.lock_struct = float(self.chimera_coherence)
            
            # === 2. TRACKING LOCK (The Compass) ===
            # Measures: Distance from target track / domain center
            # Use for: Oracle judgment, Phoenix amnesty, Track RMSE
            
            # Calculate offset from domain center
            offset = np.sqrt((cx - self.sim.nx/2)**2 + (cy - self.sim.ny/2)**2)
            
            # Define adaptive thresholds (H5 tightening rule)
            if self.max_wind_kts > 150:
                target_dist = min(self.sim.nx, self.sim.ny) * 0.06  # 7.7 cells (tight for H5)
            else:
                target_dist = min(self.sim.nx, self.sim.ny) * 0.08  # 10.2 cells (relaxed for weaker)
            
            # Calculate Sigmoid Score (V49 formula, V50.1 relaxed)
            # Formula: 1 / (1 + exp(k * (offset - target)))
            # V50.1: k=0.15 provides gentler slope, maintaining analog gradient signal
            # Prevents catastrophic zero-out at moderate offsets (borrowed from V44b's forgiving nature)
            k_steepness = 0.15  # Relaxed from 0.5 - allows guidance to "feel" pull strength
            sigmoid_score = 1.0 / (1.0 + np.exp(k_steepness * (offset - target_dist)))
            self.lock_track = float(sigmoid_score)
            
            # === 3. BUFFER MANAGEMENT ===
            # Buffer stores tracking history for Oracle trend analysis
            self.vortex_lock_buffer.append(self.lock_track)
            
            # === 4. LEGACY SUPPORT ===
            # Keep lock_score for graphing/logging (maps to tracking accuracy)
            # But control systems MUST use lock_struct or lock_track explicitly!
            self.lock_score = float(np.mean(self.vortex_lock_buffer))
            
            # V50 diagnostic logging (every 100 frames)
            if frame % 100 == 0:
                print(f"    🔥 V50 GEMINI: Struct={self.lock_struct:.2f}, Track={self.lock_track:.2f}, "
                      f"Offset={offset:.1f}, COH={self.chimera_coherence:.3f}")
            # === END V50 DUAL LOCK ===
            
        except (ValueError, IndexError) as e: 
            print(f"    --- Chimera Tracker Warning: {e} ---")

        # Compute lock score
        self.lock_score = np.mean(self.vortex_lock_buffer) if self.vortex_lock_buffer else 0
        
        # Log metrics
        log_entry = (
            f"    METRICS: Pmin:{np.min(pressure):.2f}, "
            f"Wmax:{self.max_wind_kts:.1f} kts ({self.current_stage}), "
            f"LOCK:{self.lock_score:.2%}, COH:{self.chimera_coherence:.3f}"
        )
        print(log_entry)
        
        # Defiant Core: Major hurricane with excellent lock
        self.defiant_core_active = (
            self.current_stage in ["H4", "H5"] and 
            self.lock_score > 0.8
        )
        return self.defiant_core_active

    def get_current_center_grid(self):
        """
        Get current storm center in grid coordinates.
        
        Returns:
            (cx, cy): Storm center grid coordinates, or (None, None) if not tracked
        """
        if not self.storm_path_grid_smoothed: 
            return None, None
        return self.storm_path_grid_smoothed[-1]

    def calculate_historical_accuracy(self, mean_flux):
        """
        Calculate track accuracy against historical data using HAVERSINE distance.
        
        Computes RMSE between simulated and observed track using
        time-interpolated historical positions and proper geodesic distance.
        
        Args:
            mean_flux: (unused, legacy parameter)
            
        Returns:
            Tuple: (rmse_km, 0, 0, 0, 0, 0, nav_conf)
                - rmse_km: Root mean square error in KILOMETERS (using haversine)
                - nav_conf: Navigation confidence score (0-100)
        """
        print("\n--- Calculating Final Historical Accuracy Score (Haversine) ---")
        if len(self.storm_path_grid_smoothed) < 2: 
            return (0, 0, 0, 0, 0, 0, 0)
        
        hist_track = self.sim.data_interface.historical_track
        hist_track['datetime'] = pd.to_datetime(hist_track['datetime'])
        sim_start_time = self.sim.sim_start_time
        
        # Store errors in km (squared for RMSE) and biases in degrees
        errors_km_sq = []
        lats_deg_bias = []
        lons_deg_bias = []
        
        di = self.sim.data_interface
        lon_min, lon_max = di.lon_bounds
        lat_min, lat_max = di.lat_bounds
        
        if lon_min == 0: 
            return (0, 0, 0, 0, 0, 0, 0)

        # Compare each simulated position with interpolated historical position
        for i in range(len(self.storm_path_geo_smoothed)):
            # === CRITICAL FIX V21: Use GEOGRAPHIC coordinates directly ===
            sim_lon, sim_lat = self.storm_path_geo_smoothed[i]
            # === END FIX V21 ===
            
            # Calculate simulation time for this frame
            sim_time = sim_start_time + timedelta(
                seconds=(self.frame_numbers[i] * self.sim.dt_solver * self.sim.T_CHAR)
            )
            
            # Find bracketing historical records
            hb = hist_track[hist_track['datetime'] <= sim_time].iloc[-1:]
            ha = hist_track[hist_track['datetime'] >= sim_time].iloc[:1]
            
            if hb.empty or ha.empty or hb.index[0] == ha.index[0]: 
                continue
            
            # Time interpolation
            t1, t2 = hb['datetime'].iloc[0], ha['datetime'].iloc[0]
            frac = (sim_time - t1) / (t2 - t1)
            
            interp_lat = hb['latitude'].iloc[0] + frac * (
                ha['latitude'].iloc[0] - hb['latitude'].iloc[0]
            )
            interp_lon = hb['longitude'].iloc[0] + frac * (
                ha['longitude'].iloc[0] - hb['longitude'].iloc[0]
            )
            
            # === CRITICAL FIX: Use Haversine distance (km) ===
            dist_km = haversine_distance_km(sim_lon, sim_lat, interp_lon, interp_lat)
            errors_km_sq.append(dist_km**2)
            
            # Store bias in degrees (for reference)
            lats_deg_bias.append(sim_lat - interp_lat)
            lons_deg_bias.append(sim_lon - interp_lon)

        if not errors_km_sq: 
            return (0, 0, 0, 0, 0, 0, 0)
        
        # Calculate RMSE in kilometers
        rmse_km = np.sqrt(np.mean(errors_km_sq))
        bias_lat, bias_lon = np.mean(lats_deg_bias), np.mean(lons_deg_bias)
        
        # Navigation confidence score (rescaled for kilometers)
        # Threshold: 500 km (roughly 5° at mid-latitudes)
        nav_conf = max(0, 1 - (rmse_km / 500.0)) * 100
        if rmse_km < 100.0:  # High accuracy: avg error < 100 km
            print("  -> CONFIDENCE BOOST: High track accuracy achieved.")
            nav_conf += 50.0
        nav_conf = min(100, nav_conf)
        
        print(f"  -> Final Track RMSE: {rmse_km:.2f} km (Haversine)")
        print(f"  -> Track Bias (Lat, Lon): ({bias_lat:+.3f}°, {bias_lon:+.3f}°)")
        print(f"  -> Navigation Confidence: {nav_conf:.1f}%")
        
        # Return RMSE in kilometers
        return (rmse_km, 0, 0, 0, 0, 0, nav_conf)

    def save_path_to_geojson(self):
        """
        Export simulated track to GeoJSON format.
        
        Creates a LineString feature containing the smoothed storm track
        in geographic coordinates (lat/lon).
        """
        print("Saving SMOOTHED storm path to GeoJSON...")
        if not self.storm_path_grid_smoothed: 
            print("  -> No storm path was tracked.")
            return
        
        di = self.sim.data_interface
        lon_min, lon_max = di.lon_bounds
        lat_min, lat_max = di.lat_bounds
        
        if lon_min == 0 and lon_max == 0: 
            print("  -> WARNING: Geo-bounds not set.")
            return
        
        # === CRITICAL FIX V21: Use GEOGRAPHIC coordinates directly ===
        path_geo = list(self.storm_path_geo_smoothed)
        # === END FIX V21 ===
        
        # Create GeoJSON feature
        line = geojson.LineString(path_geo)
        feature = geojson.Feature(
            geometry=line,
            properties={"name": "Simulated Track"}
        )
        collection = geojson.FeatureCollection([feature])
        
        # Save to file
        filepath = os.path.join(
            self.sim.plot_dir,
            f"simulated_track_{self.sim.initial_wind_kts}kts.geojson"
        )
        with open(filepath, 'w', encoding='utf-8') as f: 
            geojson.dump(collection, f)
        
        print(f"  -> Path saved to {filepath}")