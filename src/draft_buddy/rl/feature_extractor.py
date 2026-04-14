"""Feature extraction from DraftState to normalized tensors."""

from __future__ import annotations

import numpy as np
from typing import Dict, List

from draft_buddy.core.draft_state import DraftState
from draft_buddy.core.stacking import calculate_stack_count


class FeatureExtractor:
    """Build normalized observation vectors from pure draft state."""

    def __init__(self, config, state_normalizer) -> None:
        """Initialize extractor.

        Parameters
        ----------
        config : Config
            Runtime configuration containing enabled feature names.
        state_normalizer : StateNormalizer
            Normalizer used to convert feature map into array.
        """
        self._config = config
        self._state_normalizer = state_normalizer

    def extract(self, draft_state: DraftState, player_map: Dict[int, object], team_id: int) -> np.ndarray:
        """Extract normalized features for one team perspective.

        Parameters
        ----------
        draft_state : DraftState
            Current mutable draft state.
        player_map : Dict[int, object]
            Player lookup by id.
        team_id : int
            Team perspective for feature construction.

        Returns
        -------
        np.ndarray
            Normalized feature vector.
        """
        global_features = self.compute_global_state_features(draft_state, player_map)
        feature_map = self.build_state_map_for_team(
            draft_state=draft_state,
            player_map=player_map,
            team_id=team_id,
            global_features=global_features,
        )
        return self._state_normalizer.normalize(feature_map)

    def compute_global_state_features(
        self, draft_state: DraftState, player_map: Dict[int, object]
    ) -> Dict[str, float]:
        """Build global features independent of team perspective."""
        enabled = set(self._config.training.ENABLED_STATE_FEATURES)
        sorted_by_position = self._build_sorted_available_by_position(draft_state, player_map)
        global_features: Dict[str, float] = {}

        for position in ["QB", "RB", "WR", "TE"]:
            best_key = f"best_available_{position.lower()}_points"
            second_key = f"second_best_available_{position.lower()}_points"
            vorp_key = f"best_available_{position.lower()}_vorp"
            avail_key = f"{position.lower()}_available_flag"
            scarcity_key = f"{position.lower()}_scarcity"
            if best_key in enabled:
                global_features[best_key] = self._kth_points(sorted_by_position, position, 1)
            if second_key in enabled:
                global_features[second_key] = self._kth_points(sorted_by_position, position, 2)
            if vorp_key in enabled:
                global_features[vorp_key] = self._calculate_vorp(
                    draft_state, sorted_by_position, position
                )
            if avail_key in enabled:
                global_features[avail_key] = 1.0 if sorted_by_position[position] else 0.0
            if scarcity_key in enabled:
                global_features[scarcity_key] = self.calculate_scarcity(
                    sorted_by_position, position
                )

        if "current_pick_number" in enabled:
            global_features["current_pick_number"] = float(draft_state.get_current_pick_number())

        for position in ["QB", "RB", "WR", "TE"]:
            for rank in [1, 2, 3]:
                key = f"top_3_{position.lower()}_points_{rank}"
                if key in enabled:
                    global_features[key] = self._kth_points(sorted_by_position, position, rank)

        return global_features

    def build_state_map_for_team(
        self,
        draft_state: DraftState,
        player_map: Dict[int, object],
        team_id: int,
        global_features: Dict[str, float],
    ) -> Dict[str, float]:
        """Merge global and team-specific features into one map."""
        enabled = set(self._config.training.ENABLED_STATE_FEATURES)
        rosters = draft_state.get_rosters()
        roster_counts = rosters[team_id]
        feature_map = dict(global_features)

        self._add_roster_count_features(feature_map, roster_counts, enabled)
        self._add_available_slot_features(feature_map, roster_counts, enabled)
        self._add_draft_context_features(
            feature_map, draft_state, rosters, team_id, enabled
        )
        self._add_bye_week_features(
            feature_map, draft_state, player_map, rosters, team_id, enabled
        )
        self._add_threat_features(feature_map, draft_state, rosters, team_id, enabled)
        self._add_stack_features(feature_map, draft_state, player_map, rosters, team_id, enabled)

        return feature_map

    def calculate_scarcity(
        self, sorted_by_position: Dict[str, List[object]], position: str, rank: int = 5
    ) -> float:
        """Calculate drop-off from best available to rank-th available."""
        players = sorted_by_position[position]
        if not players:
            return 0.0
        best_points = float(players[0].projected_points)
        if len(players) >= rank:
            return best_points - float(players[rank - 1].projected_points)
        if len(players) > 1:
            return best_points - float(players[-1].projected_points)
        return 0.0

    def calculate_imminent_threat(
        self, draft_state: DraftState, rosters: Dict[int, dict], team_id: int, position: str
    ) -> int:
        """Count opponents between current and next team pick needing a starter."""
        draft_order = draft_state.get_draft_order()
        current_pick_idx = draft_state.get_current_pick_idx()
        try:
            next_pick_idx = draft_order.index(team_id, current_pick_idx + 1)
        except ValueError:
            next_pick_idx = len(draft_order)
        needed_starters = self._config.draft.ROSTER_STRUCTURE.get(position, 0)
        if needed_starters == 0:
            return 0
        opponents = set(draft_order[current_pick_idx + 1 : next_pick_idx])
        threat_count = 0
        for opponent_id in opponents:
            if rosters[opponent_id].get(position, 0) < needed_starters:
                threat_count += 1
        return threat_count

    def get_agent_bye_week_vector(
        self, draft_state: DraftState, rosters: Dict[int, dict]
    ) -> np.ndarray:
        """Return bye-week histogram for agent team."""
        return self._get_team_bye_week_vector(
            rosters[ draft_state.get_agent_team_id() ]["PLAYERS"]
        )

    def _build_sorted_available_by_position(
        self, draft_state: DraftState, player_map: Dict[int, object]
    ) -> Dict[str, List[object]]:
        """Build sorted available player lists by position."""
        grouped: Dict[str, List[object]] = {"QB": [], "RB": [], "WR": [], "TE": []}
        for player_id in draft_state.get_available_player_ids():
            player = player_map[player_id]
            if player.position in grouped:
                grouped[player.position].append(player)
        for position in grouped:
            grouped[position].sort(key=lambda player: player.projected_points, reverse=True)
        return grouped

    def _kth_points(
        self, sorted_by_position: Dict[str, List[object]], position: str, rank: int
    ) -> float:
        """Return projected points for rank-th available player at position."""
        players = sorted_by_position[position]
        if len(players) < rank:
            return 0.0
        return float(players[rank - 1].projected_points)

    def _calculate_vorp(
        self,
        draft_state: DraftState,
        sorted_by_position: Dict[str, List[object]],
        position: str,
    ) -> float:
        """Calculate value over replacement for best available player."""
        players = sorted_by_position[position]
        if not players:
            return 0.0
        needed_starters = 0
        rosters = draft_state.get_rosters()
        required = self._config.draft.ROSTER_STRUCTURE.get(position, 0)
        for team_id in range(1, self._config.draft.NUM_TEAMS + 1):
            missing = max(0, required - rosters[team_id].get(position, 0))
            needed_starters += missing
        replacement_index = min(max(0, needed_starters + 1), len(players) - 1)
        score_before = float(players[max(0, replacement_index - 1)].projected_points)
        replacement_score = float(players[replacement_index].projected_points)
        score_after = float(players[min(len(players) - 1, replacement_index + 1)].projected_points)
        baseline = (score_before + replacement_score + score_after) / 3.0
        return float(players[0].projected_points) - baseline

    def _add_roster_count_features(
        self, feature_map: Dict[str, float], roster_counts: dict, enabled: set
    ) -> None:
        """Add roster count features for team."""
        if "current_roster_qb_count" in enabled:
            feature_map["current_roster_qb_count"] = float(roster_counts["QB"])
        if "current_roster_rb_count" in enabled:
            feature_map["current_roster_rb_count"] = float(roster_counts["RB"])
        if "current_roster_wr_count" in enabled:
            feature_map["current_roster_wr_count"] = float(roster_counts["WR"])
        if "current_roster_te_count" in enabled:
            feature_map["current_roster_te_count"] = float(roster_counts["TE"])

    def _add_available_slot_features(
        self, feature_map: Dict[str, float], roster_counts: dict, enabled: set
    ) -> None:
        """Add available-slot features for team."""
        if "available_roster_slots_qb" in enabled:
            feature_map["available_roster_slots_qb"] = float(
                self._config.draft.ROSTER_STRUCTURE["QB"]
                + self._config.draft.BENCH_MAXES["QB"]
                - roster_counts["QB"]
            )
        if "available_roster_slots_rb" in enabled:
            feature_map["available_roster_slots_rb"] = float(
                self._config.draft.ROSTER_STRUCTURE["RB"]
                + self._config.draft.BENCH_MAXES["RB"]
                - roster_counts["RB"]
            )
        if "available_roster_slots_wr" in enabled:
            feature_map["available_roster_slots_wr"] = float(
                self._config.draft.ROSTER_STRUCTURE["WR"]
                + self._config.draft.BENCH_MAXES["WR"]
                - roster_counts["WR"]
            )
        if "available_roster_slots_te" in enabled:
            feature_map["available_roster_slots_te"] = float(
                self._config.draft.ROSTER_STRUCTURE["TE"]
                + self._config.draft.BENCH_MAXES["TE"]
                - roster_counts["TE"]
            )
        if "available_roster_slots_flex" in enabled:
            feature_map["available_roster_slots_flex"] = float(
                self._config.draft.ROSTER_STRUCTURE["FLEX"] - roster_counts["FLEX"]
            )

    def _add_draft_context_features(
        self,
        feature_map: Dict[str, float],
        draft_state: DraftState,
        rosters: Dict[int, dict],
        team_id: int,
        enabled: set,
    ) -> None:
        """Add pick-order and neighbor roster context features."""
        if "agent_start_position" in enabled:
            feature_map["agent_start_position"] = float(team_id)
        next_team_id = self._get_next_opponent_team_id_for(draft_state, team_id)
        if "next_pick_opponent_qb_count" in enabled:
            feature_map["next_pick_opponent_qb_count"] = float(rosters[next_team_id].get("QB", 0))
        if "next_pick_opponent_rb_count" in enabled:
            feature_map["next_pick_opponent_rb_count"] = float(rosters[next_team_id].get("RB", 0))
        if "next_pick_opponent_wr_count" in enabled:
            feature_map["next_pick_opponent_wr_count"] = float(rosters[next_team_id].get("WR", 0))
        if "next_pick_opponent_te_count" in enabled:
            feature_map["next_pick_opponent_te_count"] = float(rosters[next_team_id].get("TE", 0))

    def _add_bye_week_features(
        self,
        feature_map: Dict[str, float],
        draft_state: DraftState,
        player_map: Dict[int, object],
        rosters: Dict[int, dict],
        team_id: int,
        enabled: set,
    ) -> None:
        """Add bye-week conflict and histogram features."""
        sorted_by_position = self._build_sorted_available_by_position(draft_state, player_map)
        for position in ["QB", "RB", "WR", "TE"]:
            key = f"best_{position.lower()}_bye_week_conflict"
            if key in enabled:
                feature_map[key] = float(
                    self._count_bye_week_conflicts(rosters[team_id]["PLAYERS"], sorted_by_position[position])
                )
        team_bye_vector = self._get_team_bye_week_vector(rosters[team_id]["PLAYERS"])
        for week in range(4, 15):
            key = f"bye_week_{week}_count"
            if key in enabled:
                feature_map[key] = float(team_bye_vector[week - 4])

    def _add_threat_features(
        self,
        feature_map: Dict[str, float],
        draft_state: DraftState,
        rosters: Dict[int, dict],
        team_id: int,
        enabled: set,
    ) -> None:
        """Add imminent-threat features."""
        if "qb_imminent_threat" in enabled:
            feature_map["qb_imminent_threat"] = float(
                self.calculate_imminent_threat(draft_state, rosters, team_id, "QB")
            )
        if "rb_imminent_threat" in enabled:
            feature_map["rb_imminent_threat"] = float(
                self.calculate_imminent_threat(draft_state, rosters, team_id, "RB")
            )
        if "wr_imminent_threat" in enabled:
            feature_map["wr_imminent_threat"] = float(
                self.calculate_imminent_threat(draft_state, rosters, team_id, "WR")
            )
        if "te_imminent_threat" in enabled:
            feature_map["te_imminent_threat"] = float(
                self.calculate_imminent_threat(draft_state, rosters, team_id, "TE")
            )

    def _add_stack_features(
        self,
        feature_map: Dict[str, float],
        draft_state: DraftState,
        player_map: Dict[int, object],
        rosters: Dict[int, dict],
        team_id: int,
        enabled: set,
    ) -> None:
        """Add stacking features."""
        if "current_stack_count" in enabled:
            feature_map["current_stack_count"] = float(calculate_stack_count(rosters[team_id]["PLAYERS"]))
        if "stack_target_available_flag" in enabled:
            feature_map["stack_target_available_flag"] = float(
                self._stack_target_available_flag(
                    roster_players=rosters[team_id]["PLAYERS"],
                    available_ids=draft_state.get_available_player_ids(),
                    player_map=player_map,
                )
            )

    def _get_next_opponent_team_id_for(self, draft_state: DraftState, team_id: int) -> int:
        """Get next drafting team that is not the current perspective team."""
        draft_order = draft_state.get_draft_order()
        current_pick_idx = draft_state.get_current_pick_idx()
        if current_pick_idx + 1 >= len(draft_order):
            return team_id
        next_index = current_pick_idx + 1
        while next_index < len(draft_order):
            next_team_id = draft_order[next_index]
            if next_team_id != team_id:
                return next_team_id
            next_index += 1
        return team_id

    def _count_bye_week_conflicts(
        self, roster_players: List[object], sorted_players_for_position: List[object]
    ) -> int:
        """Count roster players sharing bye with best available positional player."""
        if not sorted_players_for_position:
            return 0
        best_player = sorted_players_for_position[0]
        bye_week = best_player.bye_week
        if not bye_week or np.isnan(bye_week):
            return 0
        return sum(1 for player in roster_players if player.bye_week == bye_week)

    def _get_team_bye_week_vector(self, roster_players: List[object]) -> np.ndarray:
        """Build weeks 4-14 bye-week histogram for a roster."""
        bye_week_vector = np.zeros(11, dtype=np.float32)
        for player in roster_players:
            bye_week = player.bye_week
            if bye_week and not np.isnan(bye_week) and 4 <= bye_week <= 14:
                bye_week_vector[int(bye_week) - 4] += 1
        return bye_week_vector

    def _stack_target_available_flag(
        self, roster_players: List[object], available_ids: set[int], player_map: Dict[int, object]
    ) -> int:
        """Return 1 if roster has a QB with available WR/TE teammate."""
        qb_teams = {player.team for player in roster_players if player.position == "QB" and player.team}
        if not qb_teams:
            return 0
        for player_id in available_ids:
            player = player_map[player_id]
            if player.position in {"WR", "TE"} and player.team and player.team in qb_teams:
                return 1
        return 0
