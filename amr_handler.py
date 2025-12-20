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


class AMRHandler:
    def __init__(self, sim_instance):
        """
        Initializes the multi-level Adaptive Mesh Refinement handler.
        """
        self.sim = sim_instance
        self.pressure_threshold_percentile = 10
        self.vorticity_level1_threshold = 2.5
        self.vorticity_level2_threshold = 3.0
        self.level3_active = False
        print("Multi-Level AMR Handler initialized.")

    def find_refinement_regions(self, pressure, vort_mag, defiant_core_active):
        """
        Identifies regions for mesh refinement using a multi-level trigger system.
        Returns an integer mask: 0=no refinement, 1=level 1, 2=level 2, 3=level 3.
        """
        refinement_mask = xp.zeros_like(pressure, dtype=xp.int8)

        # --- Level 1 Trigger: High vorticity ---
        level1_mask = vort_mag > self.vorticity_level1_threshold
        refinement_mask[level1_mask] = 1

        # --- Level 2 Trigger: High vorticity AND low pressure (eyewall zone) ---
        pressure_threshold = xp.percentile(pressure, self.pressure_threshold_percentile)
        level2_vort_mask = vort_mag > self.vorticity_level2_threshold
        level2_pressure_mask = pressure < pressure_threshold
        level2_mask = xp.logical_and(level2_vort_mask, level2_pressure_mask)
        refinement_mask[level2_mask] = 2

        # --- // ORACLE CODEX //: Level 3 Nest Trigger ---
        # Triggered by Defiant Core. This marks the region for an ultra-fine 1.5km grid.
        if defiant_core_active:
            self.level3_active = True
            
            # === GPT-5.1 PATCH: The "Turbocharger Cap" ===
            # Diagnosis: Uncapped L3 resolution (e.g., 554 tiles) creates grid-scale 
            # feedback loops that act like a turbocharger, overpowering the viscosity physics.
            # Fix: Cap L3 at 250 tiles, prioritizing the highest vorticity cells.
            
            # 1. Count potential candidates
            l3_candidates_mask = level2_mask
            num_candidates = int(xp.sum(l3_candidates_mask))
            max_l3_tiles = 250
            
            if num_candidates > max_l3_tiles:
                # 2. Too many tiles! Engage the Governor.
                # We need to find the vorticity threshold that keeps only the top 250.
                
                # Get all vorticity values in the candidate region
                active_vort = vort_mag[l3_candidates_mask]
                
                # Find the cutoff value (k-th largest element)
                # We want the top N, so we pivot at (size - N)
                pivot_idx = num_candidates - max_l3_tiles
                cutoff_vort = xp.partition(active_vort, pivot_idx)[pivot_idx]
                
                # Create strict L3 mask: Original Candidate AND Vorticity >= cutoff
                l3_strict_mask = xp.logical_and(l3_candidates_mask, vort_mag >= cutoff_vort)
                
                # Upgrade only the elite core to Level 3
                refinement_mask[l3_strict_mask] = 3
                
                # (The rest of level2_mask remains at Level 2, effectively throttled)
            else:
                # 3. Within limits, upgrade the full core
                refinement_mask[l3_candidates_mask] = 3
        else:
            self.level3_active = False

        level1_cells = int(xp.sum(refinement_mask == 1))
        level2_cells = int(xp.sum(refinement_mask == 2))
        level3_cells = int(xp.sum(refinement_mask == 3))
        
        if level1_cells > 0 or level2_cells > 0 or level3_cells > 0:
            log_entry = (
                f"    AMR TRIGGER: L1:{level1_cells}, L2:{level2_cells}, "
                f"L3(Defiant):{level3_cells}"
            )
            print(log_entry)
            with open(self.sim.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry + "\n")
        
        return refinement_mask