"""Smart bench-drop search for roster-cap trades."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import combinations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from draft_buddy.core.entities import Player, PlayerCatalog
from draft_buddy.core.roster_utils import categorize_roster_by_slots
from draft_buddy.inseason.probability import RemainingWeekMonteCarlo
from draft_buddy.inseason.projections import ros_totals_from_weekly
from draft_buddy.simulator.team_identity import team_label


@dataclass
class DropCandidateOption:
    """One legal set of players to drop after an unbalanced trade."""

    dropped_player_ids: List[int]
    dropped_player_names: List[str]
    simulated_win_probability: float
    ros_starter_points: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the drop option.

        Returns
        -------
        dict
            JSON-friendly payload.
        """
        return asdict(self)


class SmartDropEvaluator:
    """Choose bench drops that maximize remaining-week playoff odds.

    Parameters
    ----------
    catalog : PlayerCatalog
        Player lookup.
    roster_structure : mapping of str to int
        Starter slots including FLEX.
    bench_maxes : mapping of str to int
        Per-position bench limits used by roster scoring.
    roster_cap : int
        Maximum skill-roster size.
    variations_count : int
        Maximum drop combinations to score.
    monte_carlo : RemainingWeekMonteCarlo
        Deterministic remaining-week scorer (one run per variation).
    """

    def __init__(
        self,
        catalog: PlayerCatalog,
        roster_structure: Mapping[str, int],
        bench_maxes: Mapping[str, int],
        roster_cap: int,
        variations_count: int,
        monte_carlo: RemainingWeekMonteCarlo,
    ) -> None:
        self._catalog = catalog
        self._roster_structure = dict(roster_structure)
        self._bench_maxes = dict(bench_maxes)
        self._roster_cap = int(roster_cap)
        self._variations_count = max(1, int(variations_count))
        self._monte_carlo = monte_carlo

    def apply_simple(
        self,
        roster_player_ids: Sequence[int],
        incoming_ids: Sequence[int],
        outgoing_ids: Sequence[int],
        ir_player_ids: Sequence[int],
        weekly_projections: Mapping[int, Mapping[str, Any]],
        start_week: int,
    ) -> tuple[List[int], Optional[DropCandidateOption]]:
        """Drop the lowest-ROS bench players without running a season sim.

        Parameters
        ----------
        roster_player_ids : sequence of int
            Current roster.
        incoming_ids : sequence of int
            Players received.
        outgoing_ids : sequence of int
            Players sent.
        ir_player_ids : sequence of int
            IR players excluded from drop search.
        weekly_projections : mapping
            Weekly ROS vectors.
        start_week : int
            First remaining week.

        Returns
        -------
        tuple
            Final player ids and the selected drop option (``None`` if none needed).
        """
        outgoing = set(int(player_id) for player_id in outgoing_ids)
        incoming = [int(player_id) for player_id in incoming_ids]
        kept = [player_id for player_id in roster_player_ids if player_id not in outgoing]
        combined = kept + [player_id for player_id in incoming if player_id not in kept]
        overflow = len(combined) - self._roster_cap
        if overflow <= 0:
            return combined, None
        ros_totals = ros_totals_from_weekly(weekly_projections, start_week)
        drop_pool = self._drop_pool(combined, incoming, ir_player_ids, ros_totals)
        dropped = drop_pool[:overflow]
        candidate_roster = [player_id for player_id in combined if player_id not in set(dropped)]
        names = [
            self._catalog.require(player_id).name if player_id in self._catalog else str(player_id)
            for player_id in dropped
        ]
        players = self._players_with_ros(candidate_roster, ros_totals)
        return candidate_roster, DropCandidateOption(
            dropped_player_ids=list(dropped),
            dropped_player_names=names,
            simulated_win_probability=0.0,
            ros_starter_points=_starter_points(players, self._roster_structure, self._bench_maxes),
        )

    def apply(
        self,
        team_id: int,
        roster_player_ids: Sequence[int],
        incoming_ids: Sequence[int],
        outgoing_ids: Sequence[int],
        ir_player_ids: Sequence[int],
        weekly_projections: Mapping[int, Mapping[str, Any]],
        matchups,
        league_rosters: Mapping[str, List[int]],
        start_week: int,
        starting_records: Mapping[str, Mapping[str, Any]],
    ) -> tuple[List[int], Optional[DropCandidateOption]]:
        """Return the post-trade roster and the chosen drop option.

        Parameters
        ----------
        team_id : int
            Team receiving extra players.
        roster_player_ids : sequence of int
            Current roster.
        incoming_ids : sequence of int
            Players received.
        outgoing_ids : sequence of int
            Players sent.
        ir_player_ids : sequence of int
            IR players excluded from drop search.
        weekly_projections : mapping
            Weekly ROS vectors.
        matchups : pandas.DataFrame
            Remaining schedule.
        league_rosters : mapping
            Full-league rosters keyed by team label.
        start_week : int
            First remaining week.
        starting_records : mapping
            Locked standings.

        Returns
        -------
        tuple
            Final player ids and the selected drop option (``None`` if none needed).
        """
        outgoing = set(int(player_id) for player_id in outgoing_ids)
        incoming = [int(player_id) for player_id in incoming_ids]
        kept = [player_id for player_id in roster_player_ids if player_id not in outgoing]
        combined = kept + [player_id for player_id in incoming if player_id not in kept]
        overflow = len(combined) - self._roster_cap
        if overflow <= 0:
            return combined, None
        ros_totals = ros_totals_from_weekly(weekly_projections, start_week)
        drop_pool = self._drop_pool(combined, incoming, ir_player_ids, ros_totals)
        variations = self._variations(drop_pool, overflow)
        best: Optional[DropCandidateOption] = None
        best_roster = combined
        for dropped in variations:
            candidate_roster = [player_id for player_id in combined if player_id not in dropped]
            option = self._score_variation(
                team_id,
                candidate_roster,
                dropped,
                ros_totals,
                weekly_projections,
                matchups,
                league_rosters,
                start_week,
                starting_records,
            )
            if best is None or option.simulated_win_probability > best.simulated_win_probability:
                best = option
                best_roster = candidate_roster
        return best_roster, best

    def _drop_pool(
        self,
        combined_ids: Sequence[int],
        incoming_ids: Sequence[int],
        ir_player_ids: Sequence[int],
        ros_totals: Mapping[int, float],
    ) -> List[int]:
        """Return lowest-ROS droppable players, preferring bench and incoming non-starters."""
        ir_ids = set(int(player_id) for player_id in ir_player_ids)
        players = self._players_with_ros(combined_ids, ros_totals)
        starters, bench, _flex = categorize_roster_by_slots(
            players, self._roster_structure, self._bench_maxes
        )
        starter_ids = {
            player.player_id
            for slot_players in starters.values()
            for player in slot_players
        }
        incoming = set(int(player_id) for player_id in incoming_ids)
        droppable = []
        for player in players:
            if player.player_id in ir_ids:
                continue
            if player.player_id in starter_ids and player.player_id not in incoming:
                continue
            droppable.append(player.player_id)
        droppable.sort(key=lambda player_id: ros_totals.get(player_id, 0.0))
        return droppable

    def _variations(self, drop_pool: Sequence[int], overflow: int) -> List[frozenset[int]]:
        """Generate up to ``variations_count`` drop sets of size ``overflow``."""
        if overflow <= 0:
            return [frozenset()]
        pool = list(drop_pool)
        if len(pool) < overflow:
            pool = list(drop_pool)
        if len(pool) <= overflow:
            return [frozenset(pool[:overflow])] if pool else [frozenset()]
        window = pool[: overflow + 2]
        options = [frozenset(combo) for combo in combinations(window, overflow)]
        return options[: self._variations_count]

    def _score_variation(
        self,
        team_id: int,
        candidate_roster: Sequence[int],
        dropped: Iterable[int],
        ros_totals: Mapping[int, float],
        weekly_projections: Mapping[int, Mapping[str, Any]],
        matchups,
        league_rosters: Mapping[str, List[int]],
        start_week: int,
        starting_records: Mapping[str, Mapping[str, Any]],
    ) -> DropCandidateOption:
        """Score one drop set with a deterministic remaining-week sim."""
        label = team_label(team_id)
        trial_rosters = dict(league_rosters)
        trial_rosters[label] = list(candidate_roster)
        metrics = self._monte_carlo.evaluate_once(
            weekly_projections,
            matchups,
            trial_rosters,
            start_week=start_week,
            starting_records=starting_records,
        )
        dropped_ids = [int(player_id) for player_id in dropped]
        names = [
            self._catalog.require(player_id).name if player_id in self._catalog else str(player_id)
            for player_id in dropped_ids
        ]
        players = self._players_with_ros(candidate_roster, ros_totals)
        starter_points = _starter_points(players, self._roster_structure, self._bench_maxes)
        team_metrics = metrics.get(team_id) or {"p_playoff": 0.0}
        return DropCandidateOption(
            dropped_player_ids=dropped_ids,
            dropped_player_names=names,
            simulated_win_probability=float(team_metrics.get("p_playoff", 0.0)),
            ros_starter_points=starter_points,
        )

    def _players_with_ros(
        self,
        player_ids: Sequence[int],
        ros_totals: Mapping[int, float],
    ) -> List[Player]:
        """Return catalog players with ``projected_points`` set to remaining ROS."""
        players: List[Player] = []
        for player_id in player_ids:
            player = self._catalog.get(int(player_id))
            if player is None:
                continue
            players.append(replace(player, projected_points=float(ros_totals.get(player.player_id, 0.0))))
        return players


def _starter_points(
    players: Sequence[Player],
    roster_structure: Mapping[str, int],
    bench_maxes: Mapping[str, int],
) -> float:
    """Return ROS starter plus flex points for a roster."""
    from draft_buddy.core.roster_utils import calculate_roster_scores

    if not players:
        return 0.0
    scores = calculate_roster_scores(list(players), dict(roster_structure), dict(bench_maxes))
    return float(scores.get("starters_total_points", 0.0))
