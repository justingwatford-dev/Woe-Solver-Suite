# ORACLE V4: EVALUATION & VALIDATION PROTOCOL

**Version:** 1.0 
**Status:** ACTIVE
**Primary Objective:** To validate whether a local High-Resolution CFD core (1.5km) can be successfully driven by Low-Resolution (30km) global reanalysis data using an adaptive AI guidance layer.

---

## 1. SIMULATION MODES (The "Truth" Taxonomy)

To ensure scientific rigor and prevent "truth leakage" in forecasting claims, all simulations must be categorized into one of three strict modes.

###  MODE A: FREE RUN (Pure Forecasting)
* **Goal:** Test pure predictive skill of the physics + steering engine.
* **Input:** Initial State (t=0) + ERA5 Steering Fields only.
* **Constraints:**
    *  No Historical Anchoring (The nest does not shift based on history).
    *  NO Oracle Interventions (AI cannot nudge the track).
    *  NO Heuristic Stabilizers (Stall Breaker/Ghost Nudge disabled).
* **Primary Metric:** Track Error (RMSE) at 24h, 48h, 72h.
* **Note:** High drift expected due to ERA5 smoothing limits.

### 🟡 MODE B: ASSISTED RUN (The "Standard" Configuration)
* **Goal:** Test the "Oracle" AI guidance capabilities and system stability.
* **Input:** Initial State + ERA5 Steering + Adaptive AI Guidance.
* **Constraints:**
    *  Historical Anchoring: ENABLED (Nest follows storm to keep it in domain).
    *  Oracle Guidance: ENABLED (AI corrects drift > 75km).
    *  Numerical Stabilizers: ENABLED.
* **Primary Metric:** Lock Score Stability, Intervention Count, Intensity Fidelity.
* **Use Case:** Long-haul lifecycle validation (e.g., Ivan, Katrina).

###  MODE C: PHYSICS SANDBOX (Data Assimilation)
* **Goal:** Validate internal physics (Intensity, Thermodynamics, Eye Formation).
* **Input:** Continuous Historical Position Data.
* **Constraints:**
    *  Hard Lock to History (Storm is re-centered every 6 hours).
    *  Focus is purely on Intensity/Structure, not Track.
* **Primary Metric:** Pressure Deficit, Maximum Wind Speed (kts), Radius of Maximum Winds (RMW).
* **Use Case:** Tuning the V54 Governor or testing Thermodynamic Floors.

---

## 2. NUMERICAL STABILIZATION (Addressing the "Life Support" Critique)

Ablation studies confirm that standard resolution inputs (ERA5 ~30km) cannot drive High-Res Physics (~1.5km) without numerical stabilization. The following components are **structural requirements**, not "cheats."

###  Synoptic Flow Injector (formerly "Stall Breaker")
* **Problem:** Large simulated vortices generate immense surface drag. Smoothed ERA5 steering currents often lack the local torque to overcome this friction, causing artificial stalls.
* **Mechanism:** Injects synthetic momentum vectors when forward velocity drops near zero in regions of known steering flow.
* **Validation:** In the Wilma Ablation Study, this system activated **850 times** (every ~16 mins), preventing total physics collapse.

###  Deep Layer Coupler (formerly "Ghost Nudge")
* **Problem:** Grid resolution limits can cause the surface low and upper-level high to decouple during shear events ("Decapitation").
* **Mechanism:** Applies a vertical alignment force to maintain vortex tilt coherence.
* **Validation:** Activated **343 times** in the Wilma Study to maintain structural integrity.

---

## 3. VALIDATION LOG (The "Gold Standard" Runs)

| Storm | Year | Mode | Frames | Result | Key Finding |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **IVAN** | 2004 | B | 300,000 |  Success | Validated **Lazarus Protocol** (Stall Survival). Reached Alabama. |
| **KATRINA** | 2005 | B | 134,600 |  Success | Validated **V54 Launch Control**. Prevented hypercane. Perfect track. |
| **RITA** | 2005 | B | 155,000 |  Success | Validated **Landfall Decay**. |
| **WILMA** | 2005 | A/B | 209,400 |  Proven | **Ablation Study** confirmed Stabilizers are necessary for physics viability. |

---

## 4. PUBLICATION DISCLAIMER

*Oracle V4 is a research prototype. It is a single-GPU hurricane lifecycle simulator exploring the limits of local CFD guided by adaptive AI. It is NOT an operational forecast model and relies on ERA5 reanalysis for steering currents. All "Track Accuracy" claims refer to Mode B (Assisted) stability unless otherwise specified.*