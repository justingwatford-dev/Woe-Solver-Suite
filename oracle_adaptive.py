"""
Oracle's Judgment V4: Adaptive Protocol
Uses learned parameters for context-aware intervention
"""

import json
import numpy as np
from datetime import timedelta

class AdaptiveOracle:
    """
    Oracle that adapts thresholds based on storm phase
    """
    def __init__(self, learned_params=None, fallback_drift_km=75.0, fallback_patience=200):
        """
        Args:
            learned_params: Dict of phase-specific parameters (from oracle_learner)
            fallback_drift_km: Default drift threshold if phase not learned
            fallback_patience: Default patience if phase not learned
        """
        self.learned_params = learned_params
        self.fallback_drift_km = fallback_drift_km
        self.fallback_patience = fallback_patience
        
        self.trigger_count = 0
        self.check_count = 0
        
        # Track which phases we've seen
        self.phase_usage = {}
        
    @classmethod
    def from_file(cls, filepath="oracle_learned_params_v4.json"):
        """
        Load Oracle with learned parameters from file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        return cls(learned_params=data['parameters'])
        
    def classify_phase(self, state):
        """
        Determine current storm phase
        
        Args:
            state: Dict with keys: intensity, dW_dt, lock, frame, in_erc
            
        Returns:
            phase name (string)
        """
        intensity = state['intensity']
        dW_dt = state['dW_dt']
        lock = state['lock']
        frame = state['frame']
        in_erc = state.get('in_erc', False)
        
        # Genesis (early, weak)
        if frame < 1500 and intensity < 65:
            return 'genesis'
            
        # Rapid intensification
        if dW_dt > 20 and intensity < 150:
            return 'intensification'
            
        # Eyewall replacement cycle
        if in_erc or (lock < 0.50 and abs(dW_dt) > 30):
            return 'erc'
            
        # Mature (strong, stable)
        if lock > 0.55 and abs(dW_dt) < 15 and intensity > 80:
            return 'mature'
            
        # Weakening
        if dW_dt < -20:
            return 'weakening'
            
        # Weak/disorganized
        if intensity < 50:
            return 'weak'
            
        return 'other'
        
    def get_thresholds(self, phase):
        """
        Get drift threshold and patience for given phase
        
        Returns:
            (drift_threshold_km, patience_frames, confidence)
        """
        if self.learned_params and phase in self.learned_params:
            params = self.learned_params[phase]
            
            # Track usage
            if phase not in self.phase_usage:
                self.phase_usage[phase] = 0
            self.phase_usage[phase] += 1
            
            return (
                params['drift_threshold_km'],
                params['patience_frames'],
                params.get('confidence', 1.0)
            )
        else:
            # Fallback to defaults
            return (self.fallback_drift_km, self.fallback_patience, 0.5)
            
    def should_intervene(self, state, drift_km, confusion_frames):
        """
        Decide whether to intervene based on adaptive thresholds
        
        Args:
            state: Current storm state dict
            drift_km: Distance from historical position
            confusion_frames: How long Oracle has been confused
            
        Returns:
            (should_intervene, reason_dict)
        """
        self.check_count += 1
        
        # Classify phase
        phase = self.classify_phase(state)
        
        # Get thresholds for this phase
        drift_threshold, patience_threshold, confidence = self.get_thresholds(phase)
        
        # Check conditions
        confused_long_enough = confusion_frames >= patience_threshold
        drifted_too_far = drift_km >= drift_threshold
        
        should_intervene = confused_long_enough and drifted_too_far
        
        reason = {
            'phase': phase,
            'drift_threshold_km': drift_threshold,
            'patience_threshold': patience_threshold,
            'confidence': confidence,
            'confused_long_enough': confused_long_enough,
            'drifted_too_far': drifted_too_far,
            'using_learned_params': phase in (self.learned_params or {})
        }
        
        if should_intervene:
            self.trigger_count += 1
            
        return should_intervene, reason
        
    def get_statistics(self):
        """
        Get usage statistics
        """
        return {
            'total_checks': self.check_count,
            'total_triggers': self.trigger_count,
            'trigger_rate': self.trigger_count / max(self.check_count, 1),
            'phase_usage': self.phase_usage
        }