"""
Quick Start Scripts for Oracle V4
Common workflows made easy
"""
import numpy as np
import random
from World_woe_main_adaptive import Simulation3D

# === LOGGING SETUP ===
try:
    from oracle_logger import setup_global_logger, log_info
    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False
    def log_info(msg):
        log_info(msg)
    print("WARNING: oracle_logger.py not found - logging to console only")

# === GPU/CPU SAFEGUARD ===
def to_cpu(val):
    """Safely convert CuPy/NumPy scalars to Python floats"""
    if hasattr(val, 'item'):
        return val.item()
    return float(val)

def run_baseline_v3(storm_name='HUGO', storm_year=1989, init_wind=50.0, seed=1989):
    """
    Run baseline Oracle V3 (fixed thresholds) for comparison
    """
    log_info("\n" + "="*80)
    log_info(f"Running BASELINE: Oracle V3 (Fixed Thresholds) for {storm_name.upper()} ({storm_year})")
    log_info("="*80 + "\n")
    
    random.seed(seed)
    np.random.seed(seed)
    
    sim = Simulation3D(
        nx=128,
        ny=128,
        nz=64,
        num_frames=10000,
        initial_wind_kts=init_wind,
        mu=0.32,
        storm_name=storm_name,
        storm_year=storm_year,
        oracle_mode='fixed',
        collect_memory=False,
        random_seed=seed
    )
    
    sim.run()
    
    # === GPU FIX ===
    acc = sim.storm_tracker.calculate_historical_accuracy(0)
    rmse = to_cpu(acc[0])
    
    log_info("\n✓ Baseline run complete")
    log_info(f"  Track RMSE: {rmse:.2f} km")
    log_info(f"  Oracle triggers: {sim.oracle_trigger_count}")


def run_learning_single(storm_name='HUGO', storm_year=1989, init_wind=50.0, seed=1989):
    """
    Run single simulation in learning mode
    """
    log_info("\n" + "="*80)
    log_info(f"Running LEARNING MODE: Data Collection for {storm_name.upper()} ({storm_year})")
    log_info("="*80 + "\n")
    
    random.seed(seed)
    np.random.seed(seed)
    
    sim = Simulation3D(
        nx=128,
        ny=128,
        nz=64,
        num_frames=10000,
        initial_wind_kts=init_wind,
        mu=0.32,
        storm_name=storm_name,
        storm_year=storm_year,
        oracle_mode='learning',
        collect_memory=True,
        random_seed=seed
    )
    
    sim.run()
    
    log_info("\n✓ Learning run complete")
    log_info("  Memory saved to oracle_memory_db/")


def run_adaptive(storm_name='HUGO', storm_year=1989, init_wind=50.0, seed=1989):
    """
    Run Oracle V4 adaptive mode (requires training first)
    """
    log_info("\n" + "="*80)
    log_info(f"Running ADAPTIVE MODE: Oracle V4 for {storm_name.upper()} ({storm_year})")
    log_info("="*80 + "\n")
    
    import os
    if not os.path.exists('oracle_learned_params_v4.json'):
        log_info("❌ ERROR: No trained parameters found!")
        log_info("   Run training first: python train_oracle.py --phase full")
        return
    
    random.seed(seed)
    np.random.seed(seed)
    
    sim = Simulation3D(
        nx=128,
        ny=128,
        nz=64,
        num_frames=10000,
        initial_wind_kts=init_wind,
        mu=0.32,
        storm_name=storm_name,
        storm_year=storm_year,
        oracle_mode='adaptive',  # ✓ CORRECT MODE
        oracle_params_file='oracle_learned_params_v4.json',  # ✓ USE LEARNED PARAMS
        collect_memory=True,
        random_seed=seed
    )
    
    sim.run()
    
    # === GPU FIX ===
    acc = sim.storm_tracker.calculate_historical_accuracy(0)
    rmse = to_cpu(acc[0])
    
    log_info("\n✓ Adaptive run complete")
    log_info(f"  Track RMSE: {rmse:.2f} km")
    log_info(f"  Oracle triggers: {sim.oracle_trigger_count}")
    
    if sim.oracle:
        stats = sim.oracle.get_statistics()
        log_info(f"  Oracle checks: {stats['total_checks']}")
        log_info(f"  Efficiency: {(sim.oracle_trigger_count/max(stats['total_checks'], 1))*100:.1f}%")


def quick_train(num_runs=5):
    """
    Quick training with minimal runs (for testing)
    """
    log_info("\n" + "="*80)
    log_info(f"QUICK TRAIN: {num_runs} runs")
    log_info("="*80 + "\n")
    
    from train_oracle import run_training_phase, analyze_and_train
    
    # Collect data
    log_info("\n### Collecting Data ###\n")
    run_training_phase(num_runs=num_runs, vary_params=False)
    
    # Train
    log_info("\n### Training ###\n")
    analyze_and_train(min_runs=3)  # Lower threshold for quick train
    
    log_info("\n✓ Quick training complete!")
    log_info("  Run adaptive mode: python quick_start.py adaptive")


if __name__ == "__main__":
    import sys
    
    # Parse optional storm parameters
    storm_name = 'HUGO'
    storm_year = 1989
    init_wind = 50.0
    seed = 1989
    
    # Look for --storm, --year, --wind, --seed flags
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--storm' and i + 1 < len(args):
            storm_name = args[i + 1].upper()
            i += 2
        elif args[i] == '--year' and i + 1 < len(args):
            storm_year = int(args[i + 1])
            i += 2
        elif args[i] == '--wind' and i + 1 < len(args):
            init_wind = float(args[i + 1])
            i += 2
        elif args[i] == '--seed' and i + 1 < len(args):
            seed = int(args[i + 1])
            i += 2
        else:
            i += 1
    
    # Extract command (non-flag argument)
    command = None
    for arg in sys.argv[1:]:
        if not arg.startswith('--') and command is None:
            # Check if previous arg was a flag
            prev_idx = sys.argv.index(arg) - 1
            if prev_idx > 0 and sys.argv[prev_idx] in ['--storm', '--year', '--wind', '--seed']:
                continue
            command = arg
            break
    
    commands = {
        'baseline': lambda: run_baseline_v3(storm_name, storm_year, init_wind, seed),
        'learning': lambda: run_learning_single(storm_name, storm_year, init_wind, seed),
        'adaptive': lambda: run_adaptive(storm_name, storm_year, init_wind, seed),
        'quick-train': lambda: quick_train(num_runs=5)
    }
    
    if command is None:
        log_info("\nOracle V4 Quick Start")
        log_info("="*80)
        log_info("\nUsage: python quick_start.py <command> [options]")
        log_info("\nCommands:")
        log_info("  baseline     - Run Oracle V3 (fixed thresholds)")
        log_info("  learning     - Run single learning mode simulation")
        log_info("  adaptive     - Run Oracle V4 (adaptive mode)")
        log_info("  quick-train  - Quick training with 5 runs")
        log_info("\nOptions:")
        log_info("  --storm NAME   - Hurricane name (default: HUGO)")
        log_info("  --year YEAR    - Hurricane year (default: 1989)")
        log_info("  --wind SPEED   - Initial wind speed in kts (default: 50.0)")
        log_info("  --seed SEED    - Random seed (default: 1989)")
        log_info("\nExamples:")
        log_info("  python quick_start.py baseline")
        log_info("  python quick_start.py adaptive --storm ANDREW --year 1992")
        log_info("  python quick_start.py learning --seed 2024")
        log_info("  python quick_start.py quick-train")
        log_info("")  # Empty line
        sys.exit(0)
    
    if command in commands:
        commands[command]()
    else:
        log_info(f"❌ Unknown command: {command}")
        log_info(f"Available commands: {', '.join(commands.keys())}")