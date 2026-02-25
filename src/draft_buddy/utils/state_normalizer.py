import numpy as np
from typing import List, Dict, Set

class StateNormalizer:
    """
    Handles scaling and normalization of draft environment state vectors.
    """

    def __init__(self, config):
        """
        Parameters
        ----------
        config : Config
            Application configuration containing enabled features and roster structure.
        """
        self.config = config
        # Mapping of feature names to their estimated maximum bounds for Min-Max normalization
        self._feature_max_bounds = {
            "best_available_qb_points": 450.0,
            "best_available_rb_points": 400.0,
            "best_available_wr_points": 400.0,
            "best_available_te_points": 300.0,
            "best_available_qb_vorp": 150.0,
            "best_available_rb_vorp": 150.0,
            "best_available_wr_vorp": 150.0,
            "best_available_te_vorp": 100.0,
            "current_roster_qb_count": self.config.draft.ROSTER_STRUCTURE['QB'] + self.config.draft.BENCH_MAXES['QB'],
            "current_roster_rb_count": self.config.draft.ROSTER_STRUCTURE['RB'] + self.config.draft.BENCH_MAXES['RB'],
            "current_roster_wr_count": self.config.draft.ROSTER_STRUCTURE['WR'] + self.config.draft.BENCH_MAXES['WR'],
            "current_roster_te_count": self.config.draft.ROSTER_STRUCTURE['TE'] + self.config.draft.BENCH_MAXES['TE'],
            "available_roster_slots_qb": self.config.draft.ROSTER_STRUCTURE['QB'] + self.config.draft.BENCH_MAXES['QB'],
            "available_roster_slots_rb": self.config.draft.ROSTER_STRUCTURE['RB'] + self.config.draft.BENCH_MAXES['RB'],
            "available_roster_slots_wr": self.config.draft.ROSTER_STRUCTURE['WR'] + self.config.draft.BENCH_MAXES['WR'],
            "available_roster_slots_te": self.config.draft.ROSTER_STRUCTURE['TE'] + self.config.draft.BENCH_MAXES['TE'],
            "available_roster_slots_flex": self.config.draft.ROSTER_STRUCTURE['FLEX'],
            "qb_available_flag": 1.0,
            "rb_available_flag": 1.0,
            "wr_available_flag": 1.0,
            "te_available_flag": 1.0,
            "current_pick_number": float(self.config.draft.NUM_TEAMS * (sum(self.config.draft.ROSTER_STRUCTURE.values()) + self.config.draft.TOTAL_BENCH_SIZE)),
            "agent_start_position": float(self.config.draft.NUM_TEAMS),
            "second_best_available_qb_points": 450.0,
            "second_best_available_rb_points": 400.0,
            "second_best_available_wr_points": 400.0,
            "second_best_available_te_points": 300.0,
            "next_pick_opponent_qb_count": self.config.draft.ROSTER_STRUCTURE['QB'] + self.config.draft.BENCH_MAXES['QB'],
            "next_pick_opponent_rb_count": self.config.draft.ROSTER_STRUCTURE['RB'] + self.config.draft.BENCH_MAXES['RB'],
            "next_pick_opponent_wr_count": self.config.draft.ROSTER_STRUCTURE['WR'] + self.config.draft.BENCH_MAXES['WR'],
            "next_pick_opponent_te_count": self.config.draft.ROSTER_STRUCTURE['TE'] + self.config.draft.BENCH_MAXES['TE'],
            "best_qb_bye_week_conflict": 5.0,
            "best_rb_bye_week_conflict": 5.0,
            "best_wr_bye_week_conflict": 5.0,
            "best_te_bye_week_conflict": 5.0,
            "qb_scarcity": 100.0,
            "rb_scarcity": 100.0,
            "wr_scarcity": 100.0,
            "te_scarcity": 100.0,
            "top_3_qb_points_1": 450.0, "top_3_qb_points_2": 450.0, "top_3_qb_points_3": 450.0,
            "top_3_rb_points_1": 400.0, "top_3_rb_points_2": 400.0, "top_3_rb_points_3": 400.0,
            "top_3_wr_points_1": 400.0, "top_3_wr_points_2": 400.0, "top_3_wr_points_3": 400.0,
            "top_3_te_points_1": 300.0, "top_3_te_points_2": 300.0, "top_3_te_points_3": 300.0,
            "qb_imminent_threat": self.config.draft.NUM_TEAMS - 1,
            "rb_imminent_threat": self.config.draft.NUM_TEAMS - 1,
            "wr_imminent_threat": self.config.draft.NUM_TEAMS - 1,
            "te_imminent_threat": self.config.draft.NUM_TEAMS - 1,
            "bye_week_4_count": 5.0, "bye_week_5_count": 5.0, "bye_week_6_count": 5.0,
            "bye_week_7_count": 5.0, "bye_week_8_count": 5.0, "bye_week_9_count": 5.0,
            "bye_week_10_count": 5.0, "bye_week_11_count": 5.0, "bye_week_12_count": 5.0,
            "bye_week_13_count": 5.0, "bye_week_14_count": 5.0,
            "current_stack_count": 10.0,
            "stack_target_available_flag": 1.0,
        }

    def normalize(self, state_values_map: Dict[str, float]) -> np.ndarray:
        """
        Normalizes the state values according to the configured method.
        """
        enabled_features = self.config.training.ENABLED_STATE_FEATURES
        state_array = np.array([state_values_map[f] for f in enabled_features])

        if self.config.training.STATE_NORMALIZATION_METHOD == 'min_max':
            return self._normalize_min_max(state_array, enabled_features)
        elif self.config.training.STATE_NORMALIZATION_METHOD == 'z_score':
            return self._normalize_z_score(state_array)
        
        return state_array

    def _normalize_min_max(self, state_array: np.ndarray, enabled_features: List[str]) -> np.ndarray:
        """
        Scales state features to the range [0, 1] based on predefined maximum bounds.
        """
        normalized_state = []
        for i, feature_name in enumerate(enabled_features):
            val = state_array[i]
            max_val = self._feature_max_bounds.get(feature_name, 1.0)
            min_val = 0.0
            
            if max_val == min_val:
                normalized_state.append(0.0)
            else:
                normalized_state.append((val - min_val) / (max_val - min_val))
        
        return np.array(normalized_state, dtype=np.float32)

    def _normalize_z_score(self, state_array: np.ndarray) -> np.ndarray:
        """
        Performs Z-score normalization (mean=0, std=1) on the state vector.
        """
        mean = np.mean(state_array)
        std = np.std(state_array)
        if std == 0:
            return state_array - mean
        return (state_array - mean) / std
