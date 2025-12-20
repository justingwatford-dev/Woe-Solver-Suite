"""
Oracle V4 Training Campaign
Runs multiple simulations with varied parameters
"""
import numpy as np
import random
from World_woe_main_adaptive import Simulation3D
from datetime import datetime

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

def run_training_campaign(num_runs=10, vary_params=False, run_index=None,
                          storm_name='HUGO', storm_year=1989):
    """
    Execute training campaign with multiple runs
    Args:
        num_runs: Number of training runs (default: 10)
        vary_params: Whether to vary physics parameters slightly
        run_index: If set, run only this index
        storm_name: Hurricane name (default: HUGO)
        storm_year: Hurricane year (default: 1989)
    """
    # === SETUP CAMPAIGN LOGGER ===
    if LOGGING_AVAILABLE:
        campaign_logger = setup_global_logger(
            storm_name=storm_name,
            storm_year=storm_year,
            run_id=f"campaign_{num_runs}runs"
        )
    else:
        campaign_logger = None
    
    log_info("="*80)
    log_info("ORACLE V4 TRAINING CAMPAIGN")
    log_info("="*80)
    log_info(f"\nStorm: {storm_name.upper()} ({storm_year})")
    log_info(f"Executing {num_runs} training runs...")
    log_info(f"Parameter variation: {'ENABLED' if vary_params else 'DISABLED'}")
    log_info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_info("")  # Empty line

    results = []

    # Flexible run index logic
    if run_index is not None:
        run_indices = [run_index]
    else:
        run_indices = range(num_runs)

    for run_idx in run_indices:
        log_info(f"\n{'='*80}")
        log_info(f"TRAINING RUN {run_idx + 1}/{num_runs}")
        log_info(f"{'='*80}\n")
        
        # FIXED: Always calculate seed based on run index for reproducibility
        seed = 1989 + run_idx * 100
        
        # COMPATIBILITY FIX: Force legacy MT19937 generator
        legacy_rng = np.random.RandomState(seed)
        np.random.set_state(legacy_rng.get_state())
        
        # CRITICAL FIX: Set Python's random module state too
        random.seed(seed)
        
        # Fixed: Always vary parameters deterministically based on seed
        if vary_params:
            mu = 0.32 + np.random.uniform(-0.02, 0.02)
            initial_wind = 50.00 + np.random.uniform(-5, 5)
        else:
            mu = 0.32 + np.random.uniform(-0.02, 0.02)
            initial_wind = 50.00 + np.random.uniform(-3, 3)
            
        log_info("Run parameters:")
        log_info(f" Storm: {storm_name.upper()} ({storm_year})")
        log_info(f" Seed: {seed}")
        log_info(f" Initial wind: {initial_wind:.1f} kts")
        log_info(f" Viscosity: {mu:.4f}")
        log_info("")  # Empty line
        try:
            # Run simulation
            sim = Simulation3D(
                nx=128,
                ny=128,
                nz=64,
                num_frames=10000,
                mu=mu,
                initial_wind_kts=initial_wind,
                storm_name=storm_name,
                storm_year=storm_year,
                oracle_mode='learning',  # Fixed thresholds for data collection
                collect_memory=True,
                random_seed=seed  # Use run-specific seed for true reproducibility
            )
            sim.run()
            
            # Get results
            acc = sim.storm_tracker.calculate_historical_accuracy(0)
            
            # === INTEROP FIX: Convert GPU results to CPU floats ===
            # acc is likely a tuple of CuPy scalars. We must convert them
            # before storing them in the results dict or printing them.
            track_rmse = to_cpu(acc[0])
            nav_conf   = to_cpu(acc[6])
            
            results.append({
                'run': run_idx + 1,
                'seed': seed,
                'status': 'SUCCESS',
                'track_rmse': track_rmse,
                'nav_confidence': nav_conf,
                'oracle_triggers': sim.oracle_trigger_count
            })
            log_info(f"\n✓ Run {run_idx + 1} complete")
            log_info(f" Track RMSE: {track_rmse:.2f} km")
            log_info(f" Oracle triggers: {sim.oracle_trigger_count}")
            
        except Exception as e:
            log_info(f"\n✗ Run {run_idx + 1} FAILED")
            log_info(f" Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'run': run_idx + 1,
                'seed': seed,
                'status': 'FAILED',
                'error': str(e)
            })
            continue

    # Summary
    log_info("\n" + "="*80)
    log_info("TRAINING CAMPAIGN COMPLETE")
    log_info("="*80)
    successful = [r for r in results if r['status'] == 'SUCCESS']
    failed = [r for r in results if r['status'] == 'FAILED']
    log_info(f"\nResults:")
    log_info(f" Successful runs: {len(successful)}/{num_runs}")
    log_info(f" Failed runs: {len(failed)}/{num_runs}")
    if successful:
        avg_rmse = np.mean([r['track_rmse'] for r in successful])
        avg_triggers = np.mean([r['oracle_triggers'] for r in successful])
        log_info(f"\nAverage performance:")
        log_info(f" Track RMSE: {avg_rmse:.2f} km")
        log_info(f" Oracle triggers: {avg_triggers:.1f}")
        log_info(f"\nTraining data saved to: oracle_memory_db/")
        log_info(f"Total memory files: {len(successful)}")
        log_info(f"\n✓ Ready for Phase 2: Training")
        log_info(f" Run: python train_oracle.py --phase train")
    else:
        log_info("\n✗ No successful runs - cannot proceed to training")
    log_info(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # === CLOSE CAMPAIGN LOGGER ===
    if campaign_logger:
        campaign_logger.close()
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Oracle V4 Training Campaign")
    parser.add_argument('--runs', type=int, default=10,
                        help='Number of training runs (default: 10)')
    parser.add_argument('--vary', action='store_true',
                        help='Vary parameters for diversity')
    parser.add_argument('--run-index', type=int, default=None,
                        help='Run only the specified index')
    parser.add_argument('--storm', type=str, default='HUGO',
                        help='Storm name to simulate (default: HUGO, options: ANDREW, KATRINA, etc.)')
    parser.add_argument('--year', type=int, default=1989,
                        help='Storm year (default: 1989, e.g., 1992 for Andrew, 2005 for Katrina)')
    args = parser.parse_args()
    run_training_campaign(
        num_runs=args.runs,
        vary_params=args.vary,
        run_index=args.run_index,
        storm_name=args.storm,
        storm_year=args.year
    )