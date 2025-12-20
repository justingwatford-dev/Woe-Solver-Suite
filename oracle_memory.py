"""
Oracle Memory System - Enhanced for Rich Data Collection
Records every decision point with multi-faceted evaluation
"""

import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

class OracleMemory:
    """
    Records Oracle's decision-making process for learning
    Enhanced with multi-window evaluation and intensity shock detection
    """
    def __init__(self, storm_name="HUGO", year=1989):
        self.storm_name = storm_name
        self.year = year
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.interventions = []
        self.track_history = []
        self.intensity_history = []
        self.lock_history = []
        
        # Simulation parameters
        self.start_time = None
        self.dt_solver = None
        self.historical_track = None
        self.T_CHAR = 40000.0  # V17: Fallback value for time characteristic
        
    def set_simulation_params(self, start_time, dt_solver, historical_track, t_char=None):
        """
        Set simulation parameters needed for evaluation.
        """
        self.start_time = start_time
        self.dt_solver = dt_solver
        self.historical_track = historical_track
        if t_char:  # V17: Accept T_CHAR from main script
            self.T_CHAR = t_char
        
    def record_check(self, frame, state_data, decision):
        """
        Record every time Oracle checks (every 200 frames of confusion)
        
        Args:
            frame: Current simulation frame
            state_data: Dict with storm state
            decision: Boolean - did Oracle intervene?
        """
        record = {
            # Timing
            'frame': frame,
            'timestamp': datetime.now().isoformat(),
            
            # State at decision point
            'lock_score': float(state_data['lock']),
            'intensity_kts': float(state_data['intensity']),
            'intensity_trend': float(state_data['dW_dt']),
            'stage': state_data['stage'],
            
            # Drift information
            'drift_km': float(state_data['drift']),
            'simulated_lat': float(state_data['sim_lat']),
            'simulated_lon': float(state_data['sim_lon']),
            'historical_lat': float(state_data['hist_lat']),
            'historical_lon': float(state_data['hist_lon']),
            
            # Context
            'confusion_frames': int(state_data['confusion_frames']),
            'in_erc': bool(state_data['erc_detected']),
            'ohc_at_center': float(state_data.get('ohc', 120.0)),
            
            # Decision
            'intervened': bool(decision),
            
            # === ENHANCED OUTCOMES (filled by evaluate_interventions) ===
            
            # Multi-window track errors
            'track_error_pre': None,
            'track_error_post_250': None,
            'track_error_post_500': None,
            'track_error_post_1000': None,
            'track_improvement_250': None,
            'track_improvement_500': None,
            'track_improvement_1000': None,
            
            # Intensity shock metrics
            'intensity_immediate_drop_pct': None,
            'intensity_shock_100frames': None,
            'intensity_decline_rate': None,
            'intensity_recovered': None,
            
            # Structural integrity
            'lock_pre_intervention': None,
            'lock_immediate_post': None,
            'lock_avg_post_500': None,
            'lock_ever_recovered': None,
            'lock_recovery_frames': None,
            
            # Prime Directive scores
            'track_accuracy_score': None,
            'intensity_realism_score': None,
            'structural_integrity_score': None,
            'helpfulness_score': None,
            'was_helpful': None,
            
            # Phase classification
            'phase': None
        }
        
        self.interventions.append(record)
        
    def record_state(self, frame, lat, lon, intensity, lock):
        """
        Record state every N frames for trajectory analysis
        """
        self.track_history.append({
            'frame': frame,
            'lat': float(lat),
            'lon': float(lon)
        })
        self.intensity_history.append({
            'frame': frame,
            'intensity': float(intensity)
        })
        self.lock_history.append({
            'frame': frame,
            'lock': float(lock)
        })
        
    def evaluate_interventions(self):
        """
        Post-hoc evaluation using enhanced Prime Directive logic
        """
        print("\n=== Evaluating Oracle Interventions (Enhanced Prime Directive) ===")
        
        if self.historical_track is None:
            print("⚠ WARNING: No historical track provided")
            return
        
        for intervention in self.interventions:
            if not intervention['intervened']:
                intervention['was_helpful'] = False
                intervention['helpfulness_score'] = 0.0
                continue

            frame = intervention['frame']
            
            # === ENHANCED TRACK ACCURACY ===
            # Multi-window evaluation
            pre_error = self._get_historical_track_error(frame - 200, frame)
            post_250 = self._get_historical_track_error(frame, frame + 250)
            post_500 = self._get_historical_track_error(frame, frame + 500)
            post_1000 = self._get_historical_track_error(frame, frame + 1000)
            
            intervention['track_error_pre'] = float(pre_error)
            intervention['track_error_post_250'] = float(post_250)
            intervention['track_error_post_500'] = float(post_500)
            intervention['track_error_post_1000'] = float(post_1000)
            
            # Calculate improvements
            improvement_250 = (pre_error - post_250) / max(pre_error, 1.0)
            improvement_500 = (pre_error - post_500) / max(pre_error, 1.0)
            improvement_1000 = (pre_error - post_1000) / max(pre_error, 1.0)
            
            intervention['track_improvement_250'] = float(improvement_250)
            intervention['track_improvement_500'] = float(improvement_500)
            intervention['track_improvement_1000'] = float(improvement_1000)
            
            # Use 500-frame window for score (primary metric)
            track_score = max(0.0, improvement_500)
            intervention['track_accuracy_score'] = float(track_score)
            
            # === ENHANCED INTENSITY REALISM ===
            intensity_metrics = self._get_enhanced_intensity_metrics(frame)
            
            intervention['intensity_immediate_drop_pct'] = intensity_metrics['immediate_drop_pct']
            intervention['intensity_shock_100frames'] = intensity_metrics['shock_100']
            intervention['intensity_decline_rate'] = intensity_metrics['decline_rate']
            intervention['intensity_recovered'] = intensity_metrics['recovered']
            
            intensity_score = intensity_metrics['score']
            intervention['intensity_realism_score'] = float(intensity_score)
            
            # === ENHANCED STRUCTURAL INTEGRITY ===
            structure_metrics = self._get_enhanced_structure_metrics(frame)
            
            intervention['lock_pre_intervention'] = structure_metrics['lock_pre']
            intervention['lock_immediate_post'] = structure_metrics['lock_immediate']
            intervention['lock_avg_post_500'] = structure_metrics['lock_avg_500']
            intervention['lock_ever_recovered'] = structure_metrics['ever_recovered']
            intervention['lock_recovery_frames'] = structure_metrics['recovery_frames']
            
            structure_score = structure_metrics['score']
            intervention['structural_integrity_score'] = float(structure_score)
            
            # === FINAL HELPFULNESS JUDGMENT ===
            # Weights: 60% track, 20% intensity, 20% structure
            final_score = (0.6 * track_score) + (0.2 * intensity_score) + (0.2 * structure_score)
            
            was_helpful = bool(final_score > 0.25)
            intervention['was_helpful'] = was_helpful
            intervention['helpfulness_score'] = float(final_score)
            intervention['phase'] = self._classify_phase(intervention)
            
            # === DETAILED LOGGING ===
            helpful_str = f"✓ HELPFUL (Score: {final_score:.2f})" if was_helpful else f"✗ UNHELPFUL (Score: {final_score:.2f})"
            print(f"\n  Frame {frame}: Drift {intervention['drift_km']:.0f}km → {helpful_str}")
            print(f"    Track:     {track_score:.2f} (250f: {improvement_250:+.2f}, 500f: {improvement_500:+.2f}, 1000f: {improvement_1000:+.2f})")
            print(f"    Intensity: {intensity_score:.2f} (drop: {intensity_metrics['immediate_drop_pct']:.0f}%, rate: {intensity_metrics['decline_rate']:.1f})")
            print(f"    Structure: {structure_score:.2f} (LOCK: {structure_metrics['lock_pre']:.2f}→{structure_metrics['lock_avg_500']:.2f})")
            
        print(f"\nTotal interventions: {sum(1 for i in self.interventions if i['intervened'])}")
        print(f"Helpful: {sum(1 for i in self.interventions if i['was_helpful'])}")
        print(f"Unhelpful: {sum(1 for i in self.interventions if i['intervened'] and not i['was_helpful'])}")
        
    def _get_historical_track_error(self, start_frame, end_frame):
        """
        Calculate average distance from TRUE historical track (HURDAT2)
        """
        tracks = [t for t in self.track_history 
                 if start_frame <= t['frame'] <= end_frame]
        
        if not tracks:
            return 200.0  # Default high error
            
        errors = []
        for sim_track in tracks:
            # Calculate simulation time
            # V17: Fix time-travel bug - use T_CHAR instead of hardcoded 1000
            sim_time = self.start_time + timedelta(
                seconds=(sim_track['frame'] * self.dt_solver * self.T_CHAR)
            )
            
            # Find closest historical point
            hist_point = self.historical_track.iloc[
                (self.historical_track['datetime'] - sim_time).abs().argsort()[:1]
            ]
            
            if hist_point.empty:
                continue
                
            hist_lat = hist_point['latitude'].iloc[0]
            hist_lon = hist_point['longitude'].iloc[0]
            
            # Calculate distance in km
            error_km = np.sqrt(
                (sim_track['lat'] - hist_lat)**2 + 
                (sim_track['lon'] - hist_lon)**2
            ) * 111.0
            errors.append(error_km)
            
        return float(np.mean(errors)) if errors else 200.0
        
    def _get_enhanced_intensity_metrics(self, intervention_frame):
        """
        Calculate comprehensive intensity realism metrics
        Returns dict with multiple intensity indicators
        """
        # Get intensities around intervention
        pre_intensity = None
        immediate_post = None
        post_100 = None
        
        for i in self.intensity_history:
            if i['frame'] == intervention_frame:
                pre_intensity = i['intensity']
            elif pre_intensity and i['frame'] == intervention_frame + 1:
                immediate_post = i['intensity']
            elif pre_intensity and i['frame'] == intervention_frame + 100:
                post_100 = i['intensity']
                
        # Get all post-intervention intensities (500 frame window)
        post_intensities = [
            i for i in self.intensity_history 
            if intervention_frame < i['frame'] <= intervention_frame + 500
        ]
        
        if not post_intensities or not pre_intensity:
            return {
                'immediate_drop_pct': 0.0,
                'shock_100': 0.0,
                'decline_rate': 0.0,
                'recovered': False,
                'score': 0.0
            }
            
        intensities = [p['intensity'] for p in post_intensities]
        
        # 1. Immediate intensity shock (% drop)
        immediate_drop_pct = 0.0
        if immediate_post:
            immediate_drop_pct = ((pre_intensity - immediate_post) / pre_intensity) * 100.0
            
        # 2. Intensity shock at 100 frames
        shock_100 = 0.0
        if post_100:
            shock_100 = ((pre_intensity - post_100) / pre_intensity) * 100.0
            
        # 3. Average decline rate (kts per 100 frames)
        frames = [p['frame'] - intervention_frame for p in post_intensities]
        decline_rate = 0.0
        if len(intensities) >= 2:
            # Linear fit
            coeffs = np.polyfit(frames, intensities, 1)
            decline_rate = coeffs[0] * 100.0  # slope * 100 frames
            
        # 4. Did intensity recover?
        max_post = max(intensities)
        recovered = max_post >= (0.9 * pre_intensity)
        
        # === SCORING LOGIC ===
        # Penalize severe drops, reward recovery
        score = 1.0
        
        # Penalize immediate shock (>40% is catastrophic)
        if immediate_drop_pct > 40:
            score -= 0.5
        elif immediate_drop_pct > 20:
            score -= 0.3
            
        # Penalize steep decline rate
        if decline_rate < -20:  # Losing >20 kts per 100 frames
            score -= 0.4
        elif decline_rate < -10:
            score -= 0.2
            
        # Reward recovery
        if recovered:
            score += 0.3
            
        score = float(np.clip(score, 0.0, 1.0))
        
        return {
            'immediate_drop_pct': float(immediate_drop_pct),
            'shock_100': float(shock_100),
            'decline_rate': float(decline_rate),
            'recovered': bool(recovered),
            'score': score
        }
        
    def _get_enhanced_structure_metrics(self, intervention_frame):
        """
        Calculate comprehensive structural integrity metrics
        Returns dict with LOCK recovery indicators
        """
        # Get LOCK scores around intervention
        lock_pre = None
        lock_immediate = None
        
        for l in self.lock_history:
            if l['frame'] == intervention_frame:
                lock_pre = l['lock']
            elif lock_pre and l['frame'] == intervention_frame + 1:
                lock_immediate = l['lock']
                break
                
        # Get all post-intervention LOCK scores (500 frame window)
        post_locks = [
            l for l in self.lock_history 
            if intervention_frame < l['frame'] <= intervention_frame + 500
        ]
        
        if not post_locks or lock_pre is None:
            return {
                'lock_pre': 0.0,
                'lock_immediate': 0.0,
                'lock_avg_500': 0.0,
                'ever_recovered': False,
                'recovery_frames': None,
                'score': 0.0
            }
            
        locks = [p['lock'] for p in post_locks]
        lock_avg_500 = np.mean(locks)
        
        # Did LOCK ever recover to healthy level (0.65)?
        ever_recovered = any(l > 0.65 for l in locks)
        
        # How long until recovery?
        recovery_frames = None
        for i, l_data in enumerate(post_locks):
            if l_data['lock'] > 0.65:
                recovery_frames = l_data['frame'] - intervention_frame
                break
                
        # === SCORING LOGIC ===
        # Reward recovery, penalize sustained low LOCK
        score = lock_avg_500 / 0.75  # Normalize to 0-1 (0.75 = healthy)
        
        # Bonus for recovery
        if ever_recovered:
            score += 0.2
            
        # Bonus for quick recovery
        if recovery_frames and recovery_frames < 200:
            score += 0.1
            
        score = float(np.clip(score, 0.0, 1.0))
        
        return {
            'lock_pre': float(lock_pre) if lock_pre else 0.0,
            'lock_immediate': float(lock_immediate) if lock_immediate else 0.0,
            'lock_avg_500': float(lock_avg_500),
            'ever_recovered': bool(ever_recovered),
            'recovery_frames': int(recovery_frames) if recovery_frames else None,
            'score': score
        }
        
    def _classify_phase(self, intervention):
        """
        Classify storm phase at intervention
        """
        intensity = intervention['intensity_kts']
        dW_dt = intervention['intensity_trend']
        lock = intervention['lock_score']
        frame = intervention['frame']
        in_erc = intervention['in_erc']
        
        # Genesis
        if frame < 1500 and intensity < 65:
            return 'genesis'
            
        # Rapid intensification
        if dW_dt > 20 and intensity < 150:
            return 'intensification'
            
        # ERC
        if in_erc or (lock < 0.50 and abs(dW_dt) > 30):
            return 'erc'
            
        # Mature/steady
        if lock > 0.55 and abs(dW_dt) < 15 and intensity > 80:
            return 'mature'
            
        # Weakening
        if dW_dt < -20:
            return 'weakening'
            
        # Weak/disorganized
        if intensity < 50:
            return 'weak'
            
        return 'other'
        
    def save(self, directory="oracle_memory_db"):
        """
        Save memory to disk
        """
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        filename = f"{self.storm_name}_{self.year}_run_{self.run_id}.json"
        filepath = Path(directory) / filename
        
        data = {
            'metadata': {
                'storm_name': self.storm_name,
                'year': self.year,
                'run_id': self.run_id,
                'evaluation_version': 'Enhanced_Prime_Directive_v2',
                'total_frames': max([t['frame'] for t in self.track_history]) if self.track_history else 0,
                'num_interventions': sum(1 for i in self.interventions if i['intervened']),
                'num_helpful': sum(1 for i in self.interventions if i['was_helpful'])
            },
            'interventions': self.interventions,
            'track_history': self.track_history,
            'intensity_history': self.intensity_history,
            'lock_history': self.lock_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"\n=== Oracle Memory Saved (Enhanced) ===")
        print(f"  File: {filepath}")
        print(f"  Interventions: {data['metadata']['num_interventions']}")
        print(f"  Helpful: {data['metadata']['num_helpful']}")
        print(f"  Evaluation: Enhanced Prime Directive v2")
        
        return str(filepath)
        
    @classmethod
    def load(cls, filepath):
        """
        Load memory from disk
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        memory = cls(
            storm_name=data['metadata']['storm_name'],
            year=data['metadata']['year']
        )
        memory.run_id = data['metadata']['run_id']
        memory.interventions = data['interventions']
        memory.track_history = data['track_history']
        memory.intensity_history = data['intensity_history']
        memory.lock_history = data['lock_history']
        
        return memory