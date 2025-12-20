# Oracle (Woe Solver Suite)

Oracle is an experimental, research-oriented hurricane lifecycle simulator and tracking system. It explores whether a local CFD-style core plus an adaptive guidance layer can maintain realistic storm structure and track under clearly defined modes.

**Important:** Oracle is a research prototype. It is **not** an operational forecasting system and has **not** been validated for predictive skill.

## What’s in this repo
Oracle combines:
- A GPU-accelerated simulation core (NumPy/CuPy-style array compute)
- Storm tracking (center finding, intensity classification, track metrics)
- Adaptive “oracle” logic for phase-aware guidance / interventions
- Optional data assimilation hooks (ERA5 reanalysis workflow)

## Key capabilities (current)
- Long-haul historical storm replays (multi-day runs, high frame counts)
- Track evaluation using geographic distance metrics (e.g., Haversine/RMSE-style)
- Diagnostics aimed at stability + debuggability (logs, run artifacts, metrics)

## Quick start (minimal)
> If you’re new here, start with `quick_start.py` and then move to the main driver.

### 1) Clone + create an environment
```bash
git clone https://github.com/justingwatford-dev/Woe-Solver-Suite.git
cd Woe-Solver-Suite

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
2) Install dependencies

bash
Copy code
pip install numpy scipy matplotlib
3) Run a basic test
bash
Copy code
python quick_start.py
Running a storm simulation
Primary entry point:

World_woe_main_adaptive.py — main simulation driver

Example (as documented):

bash
Copy code
python World_woe_main_adaptive.py --storm KATRINA --year 2005 --initial-wind 50.0
Training campaigns
Campaign runner:

run_training_campaign.py

Example:

bash
Copy code
python run_training_campaign.py --storm WILMA --year 2005 --runs 20 --vary
Data sources
Oracle can integrate:

ERA5 reanalysis via the Copernicus Climate Data Store (CDS) API 
GitHub

HURDAT2 historical track data 
GitHub

ERA5 setup notes (high level):

Create a CDS account

Install/configure your .cdsapirc

Ensure the CDS API workflow is functional before long runs 
GitHub

Documentation
TECHNICAL.md — architecture and deeper details

PATCHES.md — patch history / version notes

CAMPAIGN.md — storms and run status

Performance notes
Performance depends heavily on GPU, resolution, and whether ERA5 operations are active. The repo contains example frame-count / wall-time benchmarks; treat them as hardware-specific and reproducibility-in-progress.

Contributing / feedback
Issues and PRs are welcome. If reporting a bug, please include:

OS + Python version

GPU/CUDA info (if using CuPy)

exact command/config

a short log snippet + any output artifacts

Disclaimer
Oracle is a public research prototype. Do not use it for real-world safety or forecasting decisions.

License
MIT (see LICENSE).

