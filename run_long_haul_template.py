#!/usr/bin/env python3
"""
ORACLE V4 - LONG HAUL RUNNER V55 (Three-Mode System + Ablation)

Goal:
  One consistent entrypoint for ALL long-haul storms, so you only change storm/year
  (and optionally frames/seed/wind) instead of cloning a new script each time.

Examples:
  # Vanilla long haul (explicit frames)
  python run_long_haul_template.py --storm IVAN --year 2004 --frames 300000 --seed 215955 --wind 40

  # Adaptive Oracle (recommended for production runs)
  python run_long_haul_template.py --storm KATRINA --year 2005 --frames 300000 --seed 215955 --wind 40 \
    --oracle-mode adaptive --oracle-params oracle_learned_params_v4.json

  # If you prefer specifying duration instead of frames (uses 21600 frames/day by default)
  python run_long_haul_template.py --storm CHARLEY --year 2004 --target-days 7 --seed 215955 --wind 35
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime

from World_woe_main_adaptive import Simulation3D


DEFAULT_FRAMES_PER_DAY = 21600  # ~4 s/frame -> 21600 frames/day (based on current sim scaling)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Oracle V4 Long Haul Runner (Template)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mission identity
    p.add_argument("--storm", required=True, type=str, help="Storm name (e.g., IVAN, KATRINA)")
    p.add_argument("--year", required=True, type=int, help="Storm year (e.g., 2004, 2005)")

    # Duration controls (pick ONE)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--frames", type=int, help="Total frames to run")
    g.add_argument("--target-days", type=float, help="Target simulated days (converted to frames)")

    # Repro/physics knobs
    p.add_argument("--seed", type=int, default=215955, help="Random seed (also used as run_id in logs)")
    p.add_argument("--wind", type=float, default=40.0, help="Initial wind speed (knots)")
    p.add_argument("--mu", type=float, default=0.12, help="Viscosity parameter")

    # Grid
    p.add_argument("--nx", type=int, default=128)
    p.add_argument("--ny", type=int, default=128)
    p.add_argument("--nz", type=int, default=64)

    # Oracle
    p.add_argument("--oracle-mode", type=str, default="off", choices=["off", "adaptive", "fixed", "learning"])
    p.add_argument("--oracle-params", type=str, default="oracle_learned_params_v4.json",
                   help="Learned params JSON (used only when --oracle-mode adaptive)")

    # V55: Simulation modes (peer review response)
    p.add_argument("--mode", type=str, default="reconstruction",
                   choices=["free_run", "assisted", "reconstruction"],
                   help="Simulation mode: free_run (Mode A - forecast), "
                        "assisted (Mode B - Oracle guidance), "
                        "reconstruction (Mode C - historical anchoring)")
    
    # V55: Ablation testing
    p.add_argument("--ablation", action="store_true",
                   help="Disable heuristic nudges (Stall Breaker, Ghost Nudge) for ablation testing")


    # Memory + bookkeeping
    p.add_argument("--no-memory", action="store_true", help="Disable Oracle memory collection")
    p.add_argument("--write-manifest", action="store_true",
                   help="Write a small run manifest JSON alongside the logs (recommended)")
    p.add_argument("--manifest-dir", type=str, default="run_manifests")

    # Safety
    p.add_argument("--dry-run", action="store_true", help="Initialize + print computed timing, but do not run")

    return p


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> int:
    args = _build_parser().parse_args()

    storm = args.storm.strip().upper()
    year = int(args.year)

    # Convert target-days to frames (before we init Simulation3D)
    if args.target_days is not None:
        frames = int(round(args.target_days * DEFAULT_FRAMES_PER_DAY))
    else:
        frames = int(args.frames)

    # Oracle params only relevant in adaptive mode
    oracle_params_file = args.oracle_params if (args.oracle_mode == "adaptive" and os.path.exists(args.oracle_params)) else None
    
    if args.oracle_mode == "adaptive" and not os.path.exists(args.oracle_params):
        print(f"WARNING: --oracle-mode adaptive specified but params file not found: {args.oracle_params}")
        print(f"         Continuing with oracle_mode='off' instead")
        oracle_mode_actual = "off"
    else:
        oracle_mode_actual = args.oracle_mode

    run_tag = f"{storm}_{year}_{_now_tag()}_seed{args.seed}"

    print(f"\n--- INITIATING LONG HAUL PROTOCOL ---")
    print(f" Storm: {storm} ({year})")
    print(f" Run tag: {run_tag}")
    print(f" Frames: {frames:,}")
    if args.target_days is not None:
        print(f" Target days requested: {args.target_days:.2f} (using {DEFAULT_FRAMES_PER_DAY} frames/day for conversion)")
    print(f" Seed: {args.seed}")
    print(f" Initial wind: {args.wind:.1f} kts | mu: {args.mu:.3f}")
    print(f" Grid: {args.nx} x {args.ny} x {args.nz}")
    print(f" Oracle mode: {oracle_mode_actual}")
    if oracle_params_file:
        print(f" Oracle params: {oracle_params_file}")
    print(f" Collect memory: {not args.no_memory}")
    print("")

    # Initialize Simulation (this also sets up the Oracle logger inside Simulation3D)
    sim = Simulation3D(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        num_frames=frames,
        mu=args.mu,
        initial_wind_kts=args.wind,
        storm_name=storm,
        storm_year=year,
        oracle_mode=oracle_mode_actual,
        oracle_params_file=oracle_params_file,
        collect_memory=(not args.no_memory),
        random_seed=args.seed,
        simulation_mode=args.mode,
        ablation_mode=args.ablation
    )

    # Compute actual simulated-time scaling (based on current sim constants)
    # dt_solver is dimensionless; T_CHAR converts it to seconds.
    sec_per_frame = float(sim.dt_solver) * float(sim.T_CHAR)
    sim_seconds = frames * sec_per_frame
    sim_days = sim_seconds / 86400.0
    frames_per_day_actual = 86400.0 / sec_per_frame if sec_per_frame > 0 else float("inf")

    print("--- SIM TIME SCALING (from Simulation3D) ---")
    print(f" Seconds per frame: {sec_per_frame:.3f} s")
    print(f" Frames per day (actual): {frames_per_day_actual:,.0f}")
    print(f" Sim duration: {sim_days:.2f} days ({sim_seconds/3600.0:.2f} hours)")
    if args.target_days is not None:
        # Warn if the hard-coded conversion doesn't match actual scaling much
        mismatch = abs(frames_per_day_actual - DEFAULT_FRAMES_PER_DAY) / DEFAULT_FRAMES_PER_DAY
        if mismatch > 0.05:
            print(f" WARNING: frames/day differs from DEFAULT ({DEFAULT_FRAMES_PER_DAY}) by {mismatch*100:.1f}%")
    print("")

    # Optional run manifest (tiny JSON you can diff across storms)
    if args.write_manifest:
        os.makedirs(args.manifest_dir, exist_ok=True)
        manifest_path = os.path.join(args.manifest_dir, f"{run_tag}.json")
        manifest = {
            "run_tag": run_tag,
            "storm_name": storm,
            "storm_year": year,
            "frames": frames,
            "target_days_requested": args.target_days,
            "seed": args.seed,
            "initial_wind_kts": args.wind,
            "mu": args.mu,
            "grid": {"nx": args.nx, "ny": args.ny, "nz": args.nz},
            "oracle": {
                "mode": oracle_mode_actual,
                "params_file": oracle_params_file,
                "collect_memory": (not args.no_memory),
            },
            "simulation": {
                "mode": args.mode,
                "ablation": args.ablation,
            },
            "sim_scaling": {
                "sec_per_frame": sec_per_frame,
                "frames_per_day_actual": frames_per_day_actual,
                "sim_days": sim_days,
            },
            "timestamp_local": datetime.now().isoformat(timespec="seconds"),
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Manifest written: {manifest_path}")
        print("")

    if args.dry_run:
        print("Dry-run requested: exiting before sim.run().")
        return 0

    # Execute
    start = time.time()
    try:
        sim.run()
    finally:
        wall_hrs = (time.time() - start) / 3600.0
        print(f"\n--- LONG HAUL COMPLETE ---")
        print(f" Wall Time: {wall_hrs:.2f} hours")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())