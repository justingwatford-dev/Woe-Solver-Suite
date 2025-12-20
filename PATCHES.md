# Oracle V4 Patch History

Complete patch documentation for the Oracle V4 hurricane simulation system. Each patch addresses a specific issue discovered during the Long Haul Campaign.

> **Note:** For full implementation details and code snippets, see [ReadmeOracleFullLife.md](../ReadmeOracleFullLife.md).

---

## Patch Index

| Patch | Name | File(s) | Purpose |
|-------|------|---------|---------|
| V28 | Precision Box | `data_interface.py` | Reduced ERA5 domain from 15° to 4° |
| V30 | Dynamic Drift Scaling | `data_interface.py` | Fixed 4.5x drift calculation error |
| V31 | Genesis Governor | `boundary_conditions.py` | Throttles fuel during genesis phase |
| V32 | Fatigue Re-Coupling | `World_woe_main_adaptive.py` | Cuts fuel when lock is low |
| V33 | Guidance Lockout | `World_woe_main_adaptive.py` | Disables steering when blind |
| V33c | Graduated Throttle | `World_woe_main_adaptive.py` | Replaced V33 with smooth scaling |
| V34 | Speed Limiter | `World_woe_main_adaptive.py` | Hard cap at 225 kts |
| V36 | Pressure-Biased Tracking | `storm_tracker.py` | Enabled pressure-based center detection |
| V38 | Iron Foundation | `core_solver.py` | Fixed k=0 mode amplification bug |
| V39b | Ivan Lifeline | `World_woe_main_adaptive.py` | Softer fatigue ramp (0.15-0.40) |
| V42 | Environmental Rescue | Multiple | ERA5 diagnostics + high torque guidance |
| V43 | Bullseye Protocol | `storm_tracker.py` | Gaussian focus mask for pressure tracking |
| V47 | Goldilocks Protocol | `storm_tracker.py` | Three-phase tracking (Genesis/Twilight/Hurricane) |
| V49 | Phoenix Protocol | Multiple | Sigmoid lock + amnesty + fuel floor |
| V50 | Gemini Protocol | Multiple | Dual-lock architecture (struct + track) |
| V50.1 | Old School Gemini | Multiple | Confusion detection rewired |
| V50.2 | Hysteresis Corner | `storm_tracker.py` | Force reset after 10 edge rejections |
| V50.3 | Cooldown Fix | `storm_tracker.py` | 50-frame cooldown after reset |
| V50.4 | Intensity Regulation | `core_solver.py`, `boundary_conditions.py` | Emergency governor + aggressive WISDOM |
| V50.5 | Surgical Governor | `core_solver.py` | Clamp rotation only, preserve steering |
| V52 | Lazarus Protocol | `World_woe_main_adaptive.py` | Thermodynamic floor + stall breaker |

---

## Foundation Patches (V28-V34)

### V28: Precision Box
**Problem:** ERA5 domain was 15° wide, causing contamination from distant synoptic features.  
**Fix:** Reduced to 4° (±2°) centered on storm.  
**Impact:** Focused on local environmental steering.

### V30: Dynamic Drift Scaling ⚠️ CRITICAL
**Problem:** Guidance assumed fixed 2,000 km domain, but Precision Box is 444 km → 4.5x drift error.  
**Fix:** Calculate `km_per_pixel` dynamically from actual domain bounds.  
**Impact:** Reduced interventions from 301 to ~50-100.

### V31: Genesis Governor
**Problem:** Storms exploded unrealistically during genesis phase.  
**Fix:** Fuel throttle 1.15 (pre-hurricane) → 1.28 (post-hurricane), unlocks at 64 kts.  
**Impact:** Gradual TD → TS → Hurricane evolution.

### V32: Fatigue Re-Coupling
**Problem:** Storms maintained intensity despite disorganization.  
**Fix:** Reduce fuel when lock score is low.  
**Impact:** Created dependency between structure and intensity.

### V33 → V33c: Guidance Control
**V33 Problem:** Binary lockout at lock < 0.15 caused cliff effect.  
**V33c Fix:** Graduated throttle - smooth 25-100% scaling from lock 0.05-0.20.  
**Impact:** Eliminated strobing, provides low-pass filter for noisy tracking.

### V34: Speed Limiter
**Problem:** Numerical singularities reaching 3,000+ kts.  
**Fix:** Hard cap at 225 kts (115 m/s) with proper masked scaling.  
**Impact:** Prevents hypercane formation above physical maximum.

---

## Tracking Breakthrough (V36-V38)

### V36: Pressure-Biased Tracking ("The Spectacles") 👓
**Discovery by:** GPT (fresh code audit)  
**Problem:** Pressure weight was 0.0, tracker relied entirely on noisy vorticity.  
**Fix:** Weights changed to `vort: 0.20, pres: 0.60, warm_core: 0.20, shear: 0.0`.  
**Impact:** Lock scores improved from 5% to 40-60%.

### V38: Iron Foundation
**Discovery by:** KWAI (peer review), validated by Gemini  
**Problem:** k=0 mode hack (`k_squared[0,0,0] = 1e-9`) amplified noise by 1 billion.  
**Fix:** Proper k=0 handling with separated mean flow in Poisson solver.  
**Impact:** Clean pressure field enables V36 tracking to work.

---

## Fatigue & Fuel Control (V39b-V42)

### V39b: Ivan Lifeline
**Discovery by:** Grok  
**Problem:** V32 ramp starved storms at lock < 0.25 (death spiral).  
**Fix:** New ramp: Full fuel > 0.40, zero fuel < 0.15, linear between.  
**Impact:** Allows moderate disorganization (0.20-0.35) to keep 20-80% fuel.

### V42: Environmental Rescue
**Contributors:** KWAI + Gemini + Grok  
**Components:**
1. ERA5 quality diagnostics (weak zone detection)
2. High torque tractor beam (gain 4→8, base 8→12 m/s)
3. Park buster thresholds (60→90, 35→60, 120→150 km)
4. Recovery boost (500 frames of protected fuel after intervention)

---

## Advanced Tracking (V43-V47)

### V43: Bullseye Protocol
**Problem:** Rainbands captured center detection.  
**Fix:** Gaussian focus mask centered on pressure minimum.  
**Impact:** Forces tracking to eyewall, ignores rainband vorticity.

### V47: Goldilocks Protocol
**Problem:** Single threshold caused premature or late Bullseye activation.  
**Fix:** Three-phase tracking with hysteresis:
- **Genesis** (< 50 kts): Cold Start (vorticity CoM)
- **Twilight** (50-64 kts): Hybrid decision based on pressure development
- **Hurricane** (> 64 kts): Forced Bullseye

---

## Resilience Stack (V49-V50.x)

### V49: Phoenix Protocol
**Discovery by:** Gemini  
**Problem:** V48 caused GPU crash from death spiral after Oracle intervention.  
**Fixes:**
1. **Soft Sigmoid Lock:** Continuous 0-1 scoring (not binary)
2. **Post-Intervention Amnesty:** 500 frames of forced full fuel
3. **Fuel Floor (Pilot Light):** Minimum 20% fuel always

### V50: Gemini Protocol (Dual Lock Architecture)
**Discovery by:** GPT, designed by Gemini  
**Problem:** Single `lock_score` represented both health AND navigation → confusion.  
**Fix:** Two separate signals:
- `lock_struct`: Chimera coherence (health)
- `lock_track`: Sigmoid of offset (navigation)

**System Mapping:**
| System | Uses |
|--------|------|
| Fatigue | `lock_struct` |
| Guidance DNR | `lock_struct` |
| Phoenix Amnesty | `lock_track` |

### V50.1: Confusion Detection Rewired
**Problem:** Alarm wired to `lock_score` instead of `lock_struct`.  
**Fix:** Changed confusion threshold to check structural health only.

### V50.2: Hysteresis Corner Rejection
**Problem:** Edge center detection got stuck in feedback loop.  
**Fix:** Force reset to domain center after 10 consecutive rejections.

### V50.3: Cooldown Mode
**Discovery by:** Claude Opus  
**Problem:** V50.2 reset was ineffective (anchor recalculated from stale pressure).  
**Fix:** 50-frame cooldown forcing anchor to center, allowing pressure field to equilibrate.

### V50.4-V50.5: Intensity Regulation
**Problem:** Hypercanes forming (202+ kts) despite WISDOM.  
**V50.4 Fixes:**
- Emergency Governor: Progressive damping above 165 kts
- Aggressive WISDOM: Start at 155 kts, shutoff at 185 kts

**V50.5 Fix:** Separate mean flow before clamping → preserves storm steering.

---

## Thermodynamic Survival (V52)

### V52: Lazarus Protocol
**Discovery by:** Gemini + Justin (historical research)  
**Problem:** Storm "ate itself" during 28-hour stall at Grand Cayman.

**Root Cause:** OHC depleted to zero with no deep pool reserve.

**Fixes:**
1. **Thermodynamic Safety Floor:** MIN_OHC = 40 kJ/cm² (simulates deep pool)
2. **Stall Breaker:** Synoptic shove after 500 frames of < 5 kt movement

**Impact:** Storm survives historical stall events and resumes motion.

---

## Complete Safety Stack

The current production Oracle V4 includes this layered protection:

```
Genesis Governor (V31)     → Prevents early explosion
Fatigue Protocol (V39b)    → Cuts fuel when disorganized
Graduated Guidance (V33c)  → Smooth steering with confidence
Speed Limiter (V34/V50.5)  → Hard cap at 225 kts
Dual Lock (V50)            → Separates health from navigation
Cooldown Mode (V50.3)      → Recovers from edge locks
WISDOM Regulation (V50.4)  → Starts damping at 155 kts
Lazarus Protocol (V52)     → Survives thermodynamic stalls
```


---

*Last Updated: December 2025*
