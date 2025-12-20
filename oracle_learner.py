"""
Oracle Learner
Analyzes multiple runs to extract optimal intervention strategies
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from oracle_memory import OracleMemory

class OracleLearner:
    """
    Learns optimal Oracle parameters from historical simulations
    """
    def __init__(self):
        self.memory_database = []
        self.learned_params = None
        
    def load_database(self, directory="oracle_memory_db"):
        """
        Load all memory files from directory
        """
        directory = Path(directory)
        memory_files = list(directory.glob("*.json"))
        
        print(f"\n=== Loading Oracle Memory Database ===")
        print(f"  Directory: {directory}")
        print(f"  Files found: {len(memory_files)}")
        
        for filepath in memory_files:
            memory = OracleMemory.load(filepath)
            self.memory_database.append(memory)
            print(f"  ✓ Loaded: {filepath.name}")
            
        print(f"\nTotal runs loaded: {len(self.memory_database)}")
        
    def analyze_interventions(self):
        """
        Analyze all interventions to find patterns
        """
        print("\n=== Analyzing Intervention Patterns ===")
        
        # Collect all interventions
        all_interventions = []
        for memory in self.memory_database:
            all_interventions.extend([
                i for i in memory.interventions 
                if i['intervened']
            ])
            
        print(f"Total interventions across all runs: {len(all_interventions)}")
        
        # Group by phase
        by_phase = defaultdict(list)
        for intervention in all_interventions:
            phase = intervention.get('phase', 'other')
            # Handle None phase values
            if phase is None:
                phase = 'unknown'
            by_phase[phase].append(intervention)
            
        # Analyze each phase
        learned_params = {}
        
        for phase, interventions in by_phase.items():
            params = self._analyze_phase(phase, interventions)
            learned_params[phase] = params
            
            # Handle None phase gracefully
            phase_name = phase.upper() if phase else "UNKNOWN"
            
            print(f"\n{phase_name} Phase:")
            print(f"  Sample size: {len(interventions)}")
            print(f"  Optimal drift threshold: {params['drift_threshold_km']:.0f} km")
            print(f"  Optimal patience: {params['patience_frames']:.0f} frames")
            print(f"  Confidence: {params['confidence']:.2f}")
            
        self.learned_params = learned_params
        return learned_params
        
    def _analyze_phase(self, phase, interventions):
        """
        Find optimal parameters for a specific phase
        """
        helpful = [i for i in interventions if i['was_helpful']]
        unhelpful = [i for i in interventions if not i['was_helpful']]
        
        if len(helpful) == 0:
            # No helpful interventions, be very conservative
            return {
                'drift_threshold_km': 150.0,
                'patience_frames': 400,
                'confidence': 0.0
            }
            
        # Analyze helpful interventions
        helpful_drifts = [i['drift_km'] for i in helpful]
        helpful_confusion = [i['confusion_frames'] for i in helpful]
        
        # Find minimum drift that was helpful
        min_helpful_drift = np.percentile(helpful_drifts, 25)  # 25th percentile
        
        # If we have unhelpful data, ensure threshold is above those
        if unhelpful:
            unhelpful_drifts = [i['drift_km'] for i in unhelpful]
            max_unhelpful_drift = np.percentile(unhelpful_drifts, 75)  # 75th percentile
            
            # Threshold should be between max unhelpful and min helpful
            drift_threshold = (max_unhelpful_drift + min_helpful_drift) / 2.0
        else:
            drift_threshold = min_helpful_drift * 0.8  # Slightly more conservative
            
        # Analyze patience (confusion frames)
        median_confusion = np.median(helpful_confusion)
        
        # Confidence based on sample size and consistency
        confidence = min(1.0, len(helpful) / 10.0)  # Full confidence at 10+ samples
        
        # Adjust for consistency
        drift_std = np.std(helpful_drifts) if len(helpful) > 1 else 100.0
        consistency_factor = 1.0 - min(drift_std / 100.0, 0.5)
        confidence *= consistency_factor
        
        return {
            'drift_threshold_km': float(drift_threshold),
            'patience_frames': int(median_confusion),
            'confidence': float(confidence),
            'sample_size': len(helpful),
            'mean_drift': float(np.mean(helpful_drifts)),
            'std_drift': float(drift_std)
        }
        
    def get_phase_statistics(self):
        """
        Get statistics about each phase across all runs
        """
        stats = defaultdict(lambda: {
            'occurrences': 0,
            'interventions': 0,
            'helpful_interventions': 0,
            'mean_drift': [],
            'mean_lock': [],
            'mean_intensity': []
        })
        
        for memory in self.memory_database:
            for intervention in memory.interventions:
                if not intervention['intervened']:
                    continue
                    
                phase = intervention.get('phase', 'other')
                # Handle None phase values
                if phase is None:
                    phase = 'unknown'
                stats[phase]['occurrences'] += 1
                stats[phase]['interventions'] += 1
                
                if intervention['was_helpful']:
                    stats[phase]['helpful_interventions'] += 1
                    
                stats[phase]['mean_drift'].append(intervention['drift_km'])
                stats[phase]['mean_lock'].append(intervention['lock_score'])
                stats[phase]['mean_intensity'].append(intervention['intensity_kts'])
                
        # Calculate means
        for phase in stats:
            stats[phase]['mean_drift'] = np.mean(stats[phase]['mean_drift'])
            stats[phase]['mean_lock'] = np.mean(stats[phase]['mean_lock'])
            stats[phase]['mean_intensity'] = np.mean(stats[phase]['mean_intensity'])
            stats[phase]['success_rate'] = (
                stats[phase]['helpful_interventions'] / stats[phase]['interventions']
                if stats[phase]['interventions'] > 0 else 0.0
            )
            
        return dict(stats)
        
    def save_learned_params(self, filepath="oracle_learned_params_v4.json"):
        """
        Save learned parameters to file
        """
        if self.learned_params is None:
            raise ValueError("No parameters learned yet. Run analyze_interventions() first.")
            
        data = {
            'version': 'V4_Adaptive',
            'training_runs': len(self.memory_database),
            'timestamp': np.datetime64('now').astype(str),
            'parameters': self.learned_params,
            'statistics': self.get_phase_statistics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"\n=== Learned Parameters Saved ===")
        print(f"  File: {filepath}")
        print(f"  Training runs: {data['training_runs']}")
        print(f"  Phases learned: {len(self.learned_params)}")
        
    @staticmethod
    def load_learned_params(filepath="oracle_learned_params_v4.json"):
        """
        Load pre-trained parameters
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        print(f"\n=== Loading Learned Parameters ===")
        print(f"  Version: {data['version']}")
        print(f"  Trained on: {data['training_runs']} runs")
        print(f"  Phases: {', '.join(data['parameters'].keys())}")
        
        return data['parameters']