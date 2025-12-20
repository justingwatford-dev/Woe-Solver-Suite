"""
Oracle V4 Training Pipeline
Run multiple simulations to collect data, then train adaptive parameters
"""

import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

from World_woe_main_adaptive import Simulation3D
from oracle_learner import OracleLearner

# === GPU/CPU SAFEGUARD ===
def to_cpu(val):
    """Safely convert CuPy/NumPy scalars to Python floats"""
    if hasattr(val, 'item'):
        return val.item()
    return float(val)

def run_training_phase(num_runs=10, base_seed=1989, vary_params=True):
    """
    Run multiple simulations in learning mode to collect training data
    """
    print("="*80)
    print("ORACLE V4 TRAINING PHASE")
    print("="*80)
    print(f"\nCollecting training data from {num_runs} simulation runs...")
    print(f"Base seed: {base_seed}")
    print(f"Parameter variation: {'ENABLED' if vary_params else 'DISABLED'}")
    print()
    
    training_runs = []
    
    for run_idx in range(num_runs):
        print(f"\n{'='*80}")
        print(f"TRAINING RUN {run_idx + 1}/{num_runs}")
        print(f"{'='*80}\n")
        
        # Vary random seed for diversity
        run_seed = base_seed + run_idx * 100
        np.random.seed(run_seed)
        
        # Optionally vary parameters slightly for robustness
        if vary_params:
            # Small variations in initial conditions
            initial_wind = 50.0 + np.random.uniform(-5, 5)
            mu = 0.32 + np.random.uniform(-0.03, 0.03)
        else:
            initial_wind = 50.0
            mu = 0.32
            
        print(f"Run parameters:")
        print(f"  Seed: {run_seed}")
        print(f"  Initial wind: {initial_wind:.1f} kts")
        print(f"  Viscosity (mu): {mu:.3f}")
        print()
        
        try:
            # Create simulation in LEARNING mode
            sim = Simulation3D(
                nx=128,
                ny=128,
                nz=64,
                num_frames=10000,
                mu=mu,
                initial_wind_kts=initial_wind,
                oracle_mode='learning',  # Fixed thresholds for data collection
                collect_memory=True
            )
            
            # Run simulation
            sim.run()
            
            # Record success
            training_runs.append({
                'run': run_idx + 1,
                'seed': run_seed,
                'status': 'SUCCESS',
                'triggers': sim.oracle_trigger_count
            })
            
        except Exception as e:
            print(f"\n!!! RUN {run_idx + 1} FAILED !!!")
            print(f"Error: {e}")
            training_runs.append({
                'run': run_idx + 1,
                'seed': run_seed,
                'status': 'FAILED',
                'error': str(e)
            })
            continue
            
    # Summary
    print("\n" + "="*80)
    print("TRAINING PHASE COMPLETE")
    print("="*80)
    
    successful = sum(1 for r in training_runs if r['status'] == 'SUCCESS')
    failed = sum(1 for r in training_runs if r['status'] == 'FAILED')
    
    print(f"\nResults:")
    print(f"  Successful runs: {successful}/{num_runs}")
    print(f"  Failed runs: {failed}/{num_runs}")
    
    if successful > 0:
        total_triggers = sum(r.get('triggers', 0) for r in training_runs if r['status'] == 'SUCCESS')
        avg_triggers = total_triggers / successful
        print(f"  Total Oracle triggers: {total_triggers}")
        print(f"  Average triggers per run: {avg_triggers:.1f}")
    
    print(f"\nTraining data saved to: oracle_memory_db/")
    
    return training_runs

def analyze_and_train(min_runs=5):
    """
    Analyze collected data and train Oracle parameters
    """
    print("\n" + "="*80)
    print("ORACLE V4 ANALYSIS & TRAINING")
    print("="*80)
    
    # Check if we have enough data
    memory_dir = Path("oracle_memory_db")
    if not memory_dir.exists():
        print("\n❌ ERROR: No training data found!")
        print("   Run training phase first: python train_oracle.py --phase collect")
        return False
        
    memory_files = list(memory_dir.glob("*.json"))
    num_files = len(memory_files)
    
    print(f"\nFound {num_files} training run(s) in oracle_memory_db/")
    
    if num_files < min_runs:
        print(f"\n⚠ WARNING: Only {num_files} run(s) found, recommended minimum is {min_runs}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Training aborted.")
            return False
    
    # Load and analyze
    print("\nLoading training data...")
    learner = OracleLearner()
    learner.load_database("oracle_memory_db")
    
    print("\nAnalyzing intervention patterns...")
    learned_params = learner.analyze_interventions()
    
    print("\nGenerating statistics...")
    stats = learner.get_phase_statistics()
    
    # Display statistics
    print("\n" + "="*80)
    print("PHASE STATISTICS")
    print("="*80)
    
    for phase, phase_stats in stats.items():
        print(f"\n{phase.upper()}:")
        print(f"  Occurrences: {phase_stats['occurrences']}")
        print(f"  Interventions: {phase_stats['interventions']}")
        print(f"  Helpful: {phase_stats['helpful_interventions']}")
        print(f"  Success rate: {phase_stats['success_rate']:.1%}")
        print(f"  Mean drift: {phase_stats['mean_drift']:.0f} km")
        print(f"  Mean LOCK: {phase_stats['mean_lock']:.2f}")
        print(f"  Mean intensity: {phase_stats['mean_intensity']:.0f} kts")
    
    # Save learned parameters
    output_file = "oracle_learned_params_v4.json"
    learner.save_learned_params(output_file)
    
    print("\n" + "="*80)
    print("✓ TRAINING COMPLETE")
    print("="*80)
    print(f"\nLearned parameters saved to: {output_file}")
    print("\nYou can now run simulations with Oracle V4 Adaptive mode:")
    print("  python world_woe_main.py --oracle-mode adaptive")
    
    return True

def test_adaptive_oracle(num_test_runs=3):
    """
    Test the trained Oracle on new simulations
    """
    print("\n" + "="*80)
    print("ORACLE V4 TESTING PHASE")
    print("="*80)
    
    # Check if trained parameters exist
    params_file = "oracle_learned_params_v4.json"
    if not os.path.exists(params_file):
        print(f"\n❌ ERROR: Trained parameters not found!")
        print(f"   Expected file: {params_file}")
        print("   Run training phase first: python train_oracle.py --phase train")
        return False
    
    print(f"\nTesting adaptive Oracle with {num_test_runs} run(s)...")
    print(f"Using parameters from: {params_file}\n")
    
    test_results = []
    
    for run_idx in range(num_test_runs):
        print(f"\n{'='*80}")
        print(f"TEST RUN {run_idx + 1}/{num_test_runs}")
        print(f"{'='*80}\n")
        
        # Use different seed than training
        test_seed = 2025 + run_idx * 100
        np.random.seed(test_seed)
        
        try:
            # Create simulation in ADAPTIVE mode
            sim = Simulation3D(
                nx=128,
                ny=128,
                nz=64,
                num_frames=10000,
                mu=0.32,
                initial_wind_kts=50.0,
                oracle_mode='adaptive',
                oracle_params_file=params_file,
                collect_memory=True  # Continue learning
            )
            
            # Run simulation
            sim.run()
            
            # Get final accuracy
            acc = sim.storm_tracker.calculate_historical_accuracy(0)
            
            # === GPU FIX: Ensure scalar float ===
            track_rmse = to_cpu(acc[0])
            
            # Get Oracle stats
            oracle_stats = sim.oracle.get_statistics() if sim.oracle else {}
            
            test_results.append({
                'run': run_idx + 1,
                'seed': test_seed,
                'status': 'SUCCESS',
                'track_rmse': track_rmse,
                'oracle_triggers': sim.oracle_trigger_count,
                'oracle_checks': oracle_stats.get('total_checks', 0)
            })
            
        except Exception as e:
            print(f"\n!!! TEST RUN {run_idx + 1} FAILED !!!")
            print(f"Error: {e}")
            test_results.append({
                'run': run_idx + 1,
                'seed': test_seed,
                'status': 'FAILED',
                'error': str(e)
            })
            continue
    
    # Summary
    print("\n" + "="*80)
    print("TESTING PHASE COMPLETE")
    print("="*80)
    
    successful = [r for r in test_results if r['status'] == 'SUCCESS']
    
    if successful:
        avg_rmse = np.mean([r['track_rmse'] for r in successful])
        avg_triggers = np.mean([r['oracle_triggers'] for r in successful])
        avg_checks = np.mean([r['oracle_checks'] for r in successful])
        
        print(f"\nResults ({len(successful)} successful run(s)):")
        print(f"  Average track RMSE: {avg_rmse:.2f} km") # Changed from deg to km to match new units
        print(f"  Average Oracle triggers: {avg_triggers:.1f}")
        print(f"  Average Oracle checks: {avg_checks:.1f}")
        if avg_checks > 0:
            print(f"  Trigger efficiency: {(avg_triggers/avg_checks)*100:.1f}%")
        
        print("\nDetailed results:")
        for r in successful:
            print(f"  Run {r['run']}: RMSE={r['track_rmse']:.2f} km, Triggers={r['oracle_triggers']}")
    else:
        print("\n❌ All test runs failed!")
    
    return len(successful) > 0

def compare_oracle_versions():
    """
    Run comparison between V3 (fixed) and V4 (adaptive)
    """
    print("\n" + "="*80)
    print("ORACLE VERSION COMPARISON: V3 vs V4")
    print("="*80)
    
    results = {}
    
    for version, mode in [('V3', 'fixed'), ('V4', 'adaptive')]:
        print(f"\n{'='*80}")
        print(f"Running {version} ({mode.upper()} mode)")
        print(f"{'='*80}\n")
        
        np.random.seed(1989)  # Same seed for fair comparison
        
        try:
            sim = Simulation3D(
                nx=128,
                ny=128,
                nz=64,
                num_frames=10000,
                mu=0.32,
                initial_wind_kts=50.0,
                oracle_mode=mode,
                oracle_params_file='oracle_learned_params_v4.json' if mode == 'adaptive' else None,
                collect_memory=False  # Don't need memory for comparison
            )
            
            sim.run()
            
            acc = sim.storm_tracker.calculate_historical_accuracy(0)
            
            # === GPU FIX: Ensure scalar floats ===
            results[version] = {
                'status': 'SUCCESS',
                'track_rmse': to_cpu(acc[0]),
                'nav_confidence': to_cpu(acc[6]),
                'oracle_triggers': sim.oracle_trigger_count
            }
            
            if mode == 'adaptive' and sim.oracle:
                oracle_stats = sim.oracle.get_statistics()
                results[version]['oracle_checks'] = oracle_stats['total_checks']
                results[version]['trigger_rate'] = oracle_stats['trigger_rate']
            
        except Exception as e:
            print(f"\n!!! {version} FAILED !!!")
            print(f"Error: {e}")
            results[version] = {'status': 'FAILED', 'error': str(e)}
    
    # Display comparison
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    if all(r['status'] == 'SUCCESS' for r in results.values()):
        print("\n{:<20} {:<15} {:<15}".format("Metric", "V3 (Fixed)", "V4 (Adaptive)"))
        print("-" * 50)
        
        v3, v4 = results['V3'], results['V4']
        
        print("{:<20} {:<15.2f} {:<15.2f}".format("Track RMSE (km)", v3['track_rmse'], v4['track_rmse']))
        print("{:<20} {:<15.1f} {:<15.1f}".format("Nav Confidence (%)", v3['nav_confidence'], v4['nav_confidence']))
        print("{:<20} {:<15} {:<15}".format("Oracle Triggers", v3['oracle_triggers'], v4['oracle_triggers']))
        
        if 'oracle_checks' in v4:
            print("{:<20} {:<15} {:<15}".format("Oracle Checks", "N/A", v4['oracle_checks']))
            print("{:<20} {:<15} {:<15.1%}".format("Trigger Rate", "N/A", v4['trigger_rate']))
        
        # Calculate improvements
        rmse_improvement = ((v3['track_rmse'] - v4['track_rmse']) / v3['track_rmse']) * 100
        trigger_reduction = ((v3['oracle_triggers'] - v4['oracle_triggers']) / max(v3['oracle_triggers'], 1)) * 100
        
        print("\n{:<20} {:<15}".format("RMSE Improvement", f"{rmse_improvement:+.1f}%"))
        print("{:<20} {:<15}".format("Trigger Reduction", f"{trigger_reduction:+.1f}%"))
        
        print("\n" + "="*80)
        
        if rmse_improvement > 0 and trigger_reduction > 0:
            print("✓ V4 shows improvement in both accuracy AND efficiency!")
        elif rmse_improvement > 0:
            print("✓ V4 shows improvement in accuracy")
        elif trigger_reduction > 0:
            print("✓ V4 shows improvement in efficiency")
        else:
            print("⚠ V4 results similar to V3 - may need more training data")
    else:
        print("\n❌ Comparison incomplete due to failures")
        for version, result in results.items():
            print(f"{version}: {result['status']}")
    
    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Oracle V4 Training Pipeline")
    parser.add_argument('--phase', type=str, choices=['collect', 'train', 'test', 'compare', 'full'],
                       default='full', help='Training phase to execute')
    parser.add_argument('--runs', type=int, default=10, help='Number of training runs')
    parser.add_argument('--test-runs', type=int, default=3, help='Number of test runs')
    parser.add_argument('--min-runs', type=int, default=5, help='Minimum runs required for training')
    parser.add_argument('--vary-params', action='store_true', help='Vary parameters for robustness')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("ORACLE V4 TRAINING PIPELINE")
    print("="*80)
    print(f"\nPhase: {args.phase.upper()}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    success = True
    
    if args.phase in ['collect', 'full']:
        print("\n### PHASE 1: DATA COLLECTION ###\n")
        training_results = run_training_phase(
            num_runs=args.runs,
            vary_params=args.vary_params
        )
        successful_runs = sum(1 for r in training_results if r['status'] == 'SUCCESS')
        if successful_runs < args.min_runs:
            print(f"\n❌ ERROR: Not enough successful runs ({successful_runs} < {args.min_runs})")
            success = False
    
    if args.phase in ['train', 'full'] and success:
        print("\n### PHASE 2: TRAINING ###\n")
        success = analyze_and_train(min_runs=args.min_runs)
    
    if args.phase in ['test', 'full'] and success:
        print("\n### PHASE 3: TESTING ###\n")
        success = test_adaptive_oracle(num_test_runs=args.test_runs)
    
    if args.phase == 'compare':
        print("\n### COMPARISON MODE ###\n")
        compare_oracle_versions()
    
    # Final summary
    print("\n" + "="*80)
    if success:
        print("✓ PIPELINE COMPLETE")
    else:
        print("❌ PIPELINE FAILED")
    print("="*80)
    print()