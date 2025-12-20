# Oracle V4

GPU-accelerated hurricane lifecycle simulation system.

[![Status](https://img.shields.io/badge/Campaign-Active-brightgreen)]()
[![Storm](https://img.shields.io/badge/Current-Ivan%202004-blue)]()
[![Frames](https://img.shields.io/badge/Target-300K%20Frames-orange)]()

---

### Physics-First Core

**Navier-Stokes Solver** (`core_solver.py`)
- Pseudo-spectral method: `∂u/∂x ≈ iFFT(ik_x · FFT(u))`
- Pressure projection for divergence-free flow: `∇²p = ∇·u*`
- Semi-Lagrangian advection: `uⁿ⁺¹(x) = uⁿ(x − uⁿ Δt)`
- Separate horizontal/vertical damping factors
- Spectral accuracy in space, explicit time stepping

**Boundary Conditions** (`boundary_conditions.py`)
- Ocean-atmosphere flux calculations (moisture, heat, momentum)
- Clausius-Clapeyron saturation humidity
- Wind-dependent bulk transfer coefficients
- **Phoenix Protocol:** Multi-layer intensity regulation system

**Adaptive Mesh Refinement** (`amr_handler.py`)
- **Level 1:** High vorticity regions (>2.5 threshold)
- **Level 2:** Eyewall zone (vorticity + low pressure)
- **Level 3:** Defiant Core (H4/H5, lock >0.8, 250-tile cap)
- Dynamic resolution: 1.5-15 km
- Memory-efficient nested grid design

---

### Oracle Navigation System

**Adaptive Oracle** (`oracle_adaptive.py`)
- Phase-aware intervention thresholds
- Storm phase classification: genesis, intensification, ERC, mature, weakening
- Learned parameters from training campaigns (JSON-serialized memory)
- Context-dependent drift tolerance
- Confidence-aware decision making

**Storm Tracker** (`storm_tracker.py`)
- Vortex center detection (pressure minimum)
- Saffir-Simpson classification
- Geographic coordinate tracking (immune to nest movement)
- Haversine distance calculations for RMSE
- **Lock score:** Vortex centering confidence (0-1 scale)
- **Chimera coherence:** Structural quality metric

**Data Interface** (`data_interface.py`)
- ERA5 reanalysis data via CDS API
- Deep layer mean steering (300-850 hPa, mass-weighted)
- **Doughnut Filter:** 40% core masking to prevent self-steering
- Kalman filtering for temporal smoothing
- Dynamic domain re-centering (Lagrangian grid)
- **Precision Box:** 4° domain (±2°) for local environmental focus

---

## 💻 Hardware & Performance

### Primary Workstation
**GPU:** NVIDIA RTX 5070 Ti (16GB VRAM)
- Utilization: 82% sustained during physics compute
- Thermal: Stable (periodic cooldown during ERA5 fetches)
- Performance: ~11,667 frames/hour (3.2 fps)
- CuPy acceleration: ~23x faster than NumPy CPU

**CPU:** Intel i9 (specifications TBD)
- Used for: ERA5 data fetch, interpolation, I/O operations
- Physics offloaded entirely to GPU

**Storage:** NVMe SSD (recommended for log/output files)

### Performance Benchmarks

| Simulation Length | Frames | Wall Time | Hardware |
|------------------|---------|-----------|----------|
| 3 days (typical) | ~65,000 | ~2 hours | RTX 4090 |
| 5.5 days (Charley) | 120,000 | 7.12 hours | RTX 5070 Ti |
| 8 days (Sandy) | 175,000 | 14.8 hours | RTX 5070 Ti |
| 10 days (Ivan) | 220,000 | ~16 hours | RTX 5070 Ti |

**Scaling Insights:**
- Linear scaling with frame count
- GPU memory stable (<10 GB typical)
- Thermal management crucial for marathon runs
- ERA5 fetch operations provide natural GPU cooldown periods

---

## Installation & Quick Start

### Prerequisites

```bash
# Core requirements
Python 3.9+
CUDA Toolkit 11.x+ (or later)
NVIDIA GPU (recommended: RTX 3080+)

# Python packages
numpy
cupy-cuda11x  # or cupy-cuda12x depending on CUDA version
scipy
matplotlib
```

**ERA5 Access:**
1. Create CDS account: https://cds.climate.copernicus.eu/
2. Place `.cdsapirc` in your home directory

### Setup

```bash
# Clone repository
git clone https://github.com/justingwatford-dev/Woe-Solver-Suite.git
cd oracle-v4

# Install dependencies
pip install -r requirements.txt

# Verify GPU visibility
python -c "import cupy; print(cupy.cuda.Device())"
```

### Directory Structure

```
oracle-v4/
├── World_woe_main_adaptive.py     # Main simulation driver
├── core_solver.py                 # Pseudo-spectral solver
├── data_interface.py              # ERA5 / HURDAT2 integration
├── oracle_logger.py               # Logging and diagnostics
├── storm_tracker.py               # Vortex tracking & classification
├── oracle_adaptive.py             # Phase-aware navigation
├── boundary_conditions.py         # Ocean-atmosphere physics
├── amr_handler.py                 # Adaptive Mesh Refinement
├── kalman_filter.py               # Bayesian data assimilation
├── Train_oracle.py                # Learning pipeline
├── oracle_memory.py               # Recording system
├── visualiser.py                  # 3D VTK and 2D frame visualization
├── run_training_campaign.py       # Multi-run campaign manager
├── hurdat2.txt                    # Historical track database
├── oracle_learned_params_v4.json  # Calibrated parameters
└── oracle_memory_db/              # Run history and memory files
```

### Quick Start Examples

**Single Hurricane Simulation**
```bash
python World_woe_main_adaptive.py \
    --storm KATRINA \
    --year 2005 \
    --initial-wind 50.0
```

**Training Campaign (20 Runs)**
```bash
python run_training_campaign.py \
    --storm WILMA \
    --year 2005 \
    --runs 20 \
    --vary
```

**Custom Storm**
```bash
python World_woe_main_adaptive.py \
    --storm "HURRICANE_NAME" \
    --year YYYY \
    --initial-wind 50.0
```


---

## Quick Start

```bash
# Verify GPU
nvidia-smi

# Check CuPy
python -c "import cupy; print(cupy.__version__)"

# Launch simulation
python run_ivan_long_haul.py

# Monitor progress
tail -f logs/oracle_v4_IVAN_*.log
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [PATCHES.md](./docs/PATCHES.md) | Complete patch history (V28-V52) |
| [TECHNICAL.md](./docs/TECHNICAL.md) | Architecture deep-dive |
| [CAMPAIGN.md](./docs/CAMPAIGN.md) | Storm roster & status |

---

## Campaign Storms

| # | Storm | Year | Status |
|---|-------|------|--------|
| 1 | Charley | 2004 | ✅ Complete (110 km RMSE) |
| 2 | Sandy | 2012 | ✅ Complete |
| 3 | Ivan | 2004 | 🔄 Running (300K frames) |
| 4-15 | [12 more](./docs/CAMPAIGN.md) | Various | ⏳ Queued |

---

## Key Features

### Safety Stack
```
Genesis Governor    → Prevents early explosion
Fatigue Protocol    → Fuel based on structure
WISDOM Regulation   → Intensity capping
Lazarus Protocol    → Thermodynamic survival
```

### Tracking Stack
```
Dual Lock          → Separates health from navigation
Bullseye Protocol  → Pressure-based center detection
Cooldown Mode      → Recovers from edge locks
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3060 (8GB) | RTX 4080/5070+ (16GB) |
| VRAM | 8 GB | 16 GB |
| RAM | 16 GB | 32 GB |
| Storage | 50 GB SSD | 100 GB NVMe |

---

## AI Ensemble

Developed through human-AI collaboration:

| Contributor | Role |
|-------------|------|
| **Justin** | Lead Developer, Ensemble Orchestration |
| **Gemini** | Architecture, Physics Theory |
| **Claude** | Implementation, Documentation |
| **GPT** | Code Auditing, Semantic Analysis |
| **Grok** | Parameter Optimization |

---

## Data Sources

- **ERA5:** Copernicus Climate Change Service
- **HURDAT2:** NOAA National Hurricane Center

---

## License

Copyright 2025 Justin Watford

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

**Last Updated:** December 2025  
**Target Completion:** June 2026

