# Technical Architecture

Deep-dive into Oracle V4's computational fluid dynamics system.

---

## Core Physics Engine

### Navier-Stokes Solver (`core_solver.py`)
- **Method:** Pseudo-spectral (FFT-based derivatives)
- **Pressure projection:** Ensures divergence-free flow
- **Advection:** Semi-Lagrangian (order-1 interpolation)
- **Damping:** Separate horizontal/vertical factors

### Boundary Conditions (`boundary_conditions.py`)
- **Ocean-atmosphere fluxes:** Moisture, heat, momentum
- **Saturation:** Clausius-Clapeyron humidity
- **Transfer coefficients:** Wind-dependent bulk formulas
- **Safety systems:** Phoenix Protocol (Governor + WISDOM)

### Adaptive Mesh Refinement (`amr_handler.py`)
| Level | Trigger | Resolution |
|-------|---------|------------|
| 1 | High vorticity (>2.5) | 15 km |
| 2 | Eyewall zone (vorticity + low pressure) | 5 km |
| 3 | Defiant Core (H4/H5, lock >0.8) | 1.5 km |

---

## Oracle Navigation System

### Storm Tracker (`storm_tracker.py`)

**V50 Dual Lock Architecture:**
```
lock_struct = chimera_coherence     → Health signal
lock_track  = sigmoid(offset)       → Navigation signal
```

**Chimera Field (weighted center detection):**
```python
weights = {'vort': 0.20, 'pres': 0.60, 'warm_core': 0.20, 'shear': 0.0}
chimera = sum(weight * field for field in [vorticity, pressure, warm_core, shear])
```

**V47 Goldilocks Phases:**
- Genesis (<50 kts): Vorticity center of mass
- Twilight (50-64 kts): Hybrid based on pressure development
- Hurricane (>64 kts): Forced Bullseye with Gaussian mask

### Data Interface (`data_interface.py`)
- **Source:** ERA5 reanalysis via CDS API
- **Steering:** Deep layer mean (300-850 hPa, mass-weighted)
- **Doughnut Filter:** 40% core masking (prevents self-steering)
- **Smoothing:** Kalman temporal filtering
- **Domain:** Dynamic re-centering (Lagrangian grid)

---

## Safety Systems: Phoenix Protocol

### Intensity Regulation Stack

```
┌─────────────────────────────────────────────┐
│  Genesis Governor (V31)                      │
│  Fuel throttle 1.15→1.28, unlocks at 64 kts │
├─────────────────────────────────────────────┤
│  V39b Fatigue Ramp                           │
│  Lock 0.40+: 100% │ Lock <0.15: 0%          │
├─────────────────────────────────────────────┤
│  V50.4 WISDOM Regulation                     │
│  Starts 155 kts → Shutoff 185 kts           │
├─────────────────────────────────────────────┤
│  V50.5 Surgical Governor                     │
│  Clamps rotation, preserves steering         │
├─────────────────────────────────────────────┤
│  V52 Lazarus Protocol                        │
│  MIN_OHC=40 + Stall Breaker                  │
└─────────────────────────────────────────────┘
```

### Tracking Safety Stack

```
┌─────────────────────────────────────────────┐
│  V33c Graduated Guidance                     │
│  Lock 0.05-0.20: 25-100% steering strength   │
├─────────────────────────────────────────────┤
│  V49 Fuel Floor (Pilot Light)                │
│  Minimum 20% fuel always                     │
├─────────────────────────────────────────────┤
│  V50.3 Cooldown Mode                         │
│  50-frame forced center after edge lock      │
└─────────────────────────────────────────────┘
```

---

## Hardware Configuration

### Primary Workstation
| Component | Specification |
|-----------|--------------|
| **GPU** | NVIDIA RTX 5070 Ti (16GB VRAM) |
| **Utilization** | 82% sustained during physics |
| **Performance** | ~11,667 frames/hour (3.2 fps) |
| **Acceleration** | CuPy ~20-25x faster than NumPy |

### Performance Metrics
| Run | Frames | Wall Time | Throughput |
|-----|--------|-----------|------------|
| Charley | 120,000 | 7.12 hours | 16,854 f/h |
| Sandy | 175,000 | ~12-15 hours | 11,667-14,583 f/h |
| Ivan | 220,000 | ~9+ hours | TBD |

### Memory Usage
- 128³ grid: ~84 MB per field (~500 MB total GPU)
- 256³ grid: ~670 MB per field (feasible)
- 512³ grid: ~5.4 GB per field (H100/A100 territory)

---

## File Structure

```
oracle_v4/
├── World_woe_main_adaptive.py    # Main orchestrator
├── core_solver.py                # Navier-Stokes solver
├── boundary_conditions.py        # Ocean-atmosphere fluxes
├── storm_tracker.py              # Vortex tracking
├── data_interface.py             # ERA5 integration
├── kalman_filter.py              # Temporal smoothing
├── amr_handler.py                # Adaptive mesh
├── visualizer.py                 # VTK output
│
├── oracle_adaptive.py            # Phase-aware interventions
├── oracle_learner.py             # Statistical training
├── oracle_memory.py              # Intervention logging
│
├── run_*_long_haul.py            # Mission scripts
├── oracle_learned_params_v4.json # Trained thresholds
├── oracle_memory_db/             # History database
└── logs/                         # Simulation logs
```

---

## Validation Philosophy

> **Not intended to compete with or replace forecast data from NOAA/NHC."**

**Goals:**
- Explore alternative computational approaches
- Test machine learning integration
- Validate physics at mesoscale (1.5-15 km)
- Demonstrate consumer-grade GPU capability
- Advance open-source atmospheric modeling

**Success Criteria:**
- Mean track RMSE < 50 km at 72h
- Intensity error < ±15 kts MAE
- Structural feature capture > 80%
- Zero hypercane formations

---

*For implementation details, see [PATCHES.md](./PATCHES.md)*
