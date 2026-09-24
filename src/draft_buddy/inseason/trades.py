"""Inbound trade evaluation and outbound trade search."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from itertools import combinations
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

from draft_buddy.core.entities import Player, PlayerCatalog
from draft_buddy.core.roster_utils import calculate_roster_scores, categorize_roster_by_slots
from draft_buddy.inseason.advisor import TradeAdvisorWriter
from draft_buddy.inseason.drops import DropCandidateOption, SmartDropEvaluator
from draft_buddy.inseason.exceptions import TradeValidationError
from draft_buddy.inseason.probability import RemainingWeekMonteCarlo
from draft_buddy.inseason.projections import ros_totals_from_weekly
from draft_buddy.inseason.schedule import snapshot_matchups_frame
from draft_buddy.inseason.state import LeagueSnapshot
from draft_buddy.simulator.team_identity import team_label

FLEX_ELIGIBLE = frozenset({"RB", "WR", "TE"})
SKILL_POSITIONS = frozenset({"QB", "RB", "WR", "TE"})


@dataclass
class TradeProposal:
    """One scored outbound (or counter) trade."""

    proposal_id: str
    user_team_id: int
    partner_team_id: int
    user_sends_player_ids: List[int]
    partner_sends_player_ids: List[int]
    user_win_prob_delta: float
    partner_win_prob_delta: float
    user_expected_wins_delta: float = 0.0
    partner_expected_wins_delta: float = 0.0
    user_starter_points_delta: float = 0.0
    partner_starter_points_delta: float = 0.0
    optimal_drops: Optional[DropCandidateOption] = None
    trade_pitch: Optional[str] = None
    internal_rationale: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the proposal."""
        payload = asdict(self)
        if self.optimal_drops is not None:
            payload["optimal_drops"] = self.optimal_drops.to_dict()
        return payload


@dataclass
class InboundTradeEvaluationResult:
    """Numeric inbound evaluation plus optional LLM copy."""

    verdict: str
    win_probability_delta: float
    expected_wins_delta: float
    ros_starter_points_delta: float
    user_p_playoff: float
    partner_p_playoff: float
    optimal_bench_drops: Optional[DropCandidateOption] = None
    defensive_breakdown: str = ""
    trade_pitch: str = ""
    suggested_counter_offer: Optional[TradeProposal] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the evaluation."""
        payload = asdict(self)
        if self.optimal_bench_drops is not None:
            payload["optimal_bench_drops"] = self.optimal_bench_drops.to_dict()
        if self.suggested_counter_offer is not None:
            payload["suggested_counter_offer"] = self.suggested_counter_offer.to_dict()
        return payload


class InboundTradeService:
    """Evaluate one inbound N-for-M trade against remaining-week odds."""

    def __init__(
        self,
        catalog: PlayerCatalog,
        roster_structure: Mapping[str, int],
        bench_maxes: Mapping[str, int],
        drop_evaluator: SmartDropEvaluator,
        monte_carlo: RemainingWeekMonteCarlo,
        n_runs: int,
        accept_delta: float,
        counter_delta: float,
        last_regular_week: int,
        advisor: Optional[TradeAdvisorWriter] = None,
    ) -> None:
        self._catalog = catalog
        self._roster_structure = dict(roster_structure)
        self._bench_maxes = dict(bench_maxes)
        self._drop_evaluator = drop_evaluator
        self._monte_carlo = monte_carlo
        self._n_runs = int(n_runs)
        self._accept_delta = float(accept_delta)
        self._counter_delta = float(counter_delta)
        self._last_regular_week = int(last_regular_week)
        self._advisor = advisor or TradeAdvisorWriter()

    def evaluate(
        self,
        snapshot: LeagueSnapshot,
        weekly_projections: Mapping[int, Mapping[str, Any]],
        user_team_id: int,
        offering_team_id: int,
        players_sending: Sequence[int],
        players_receiving: Sequence[int],
        include_writeup: bool = False,
    ) -> InboundTradeEvaluationResult:
        """Score an inbound offer.

        Parameters
        ----------
        snapshot : LeagueSnapshot
            Current league state.
        weekly_projections : mapping
            Weekly ROS vectors.
        user_team_id : int
            Evaluating manager.
        offering_team_id : int
            Offering manager.
        players_sending : sequence of int
            Players the user would give up.
        players_receiving : sequence of int
            Players the user would receive.
        include_writeup : bool, optional
            When true, request an LLM defensive read.

        Returns
        -------
        InboundTradeEvaluationResult
            Numeric verdict and optional copy.
        """
        user = snapshot.team(user_team_id)
        partner = snapshot.team(offering_team_id)
        sending = [int(player_id) for player_id in players_sending]
        receiving = [int(player_id) for player_id in players_receiving]
        _assert_owned(sending, user.player_ids, user_team_id)
        _assert_owned(receiving, partner.player_ids, offering_team_id)
        matchups = snapshot_matchups_frame(snapshot)
        baseline = self._monte_carlo.evaluate(
            weekly_projections,
            matchups,
            snapshot.rosters_by_label(),
            n_runs=self._n_runs,
            start_week=snapshot.current_week,
            starting_records=snapshot.starting_records(),
        )
        user_roster, drops = self._drop_evaluator.apply(
            user_team_id,
            user.player_ids,
            receiving,
            sending,
            user.ir_player_ids,
            weekly_projections,
            matchups,
            snapshot.rosters_by_label(),
            snapshot.current_week,
            snapshot.starting_records(),
        )
        partner_roster, _partner_drops = self._drop_evaluator.apply(
            offering_team_id,
            partner.player_ids,
            sending,
            receiving,
            partner.ir_player_ids,
            weekly_projections,
            matchups,
            snapshot.rosters_by_label(),
            snapshot.current_week,
            snapshot.starting_records(),
        )
        after_rosters = snapshot.rosters_by_label()
        after_rosters[team_label(user_team_id)] = user_roster
        after_rosters[team_label(offering_team_id)] = partner_roster
        after = self._monte_carlo.evaluate(
            weekly_projections,
            matchups,
            after_rosters,
            n_runs=self._n_runs,
            start_week=snapshot.current_week,
            starting_records=snapshot.starting_records(),
        )
        user_delta = _metric_delta(after, baseline, user_team_id, "p_playoff")
        wins_delta = _metric_delta(after, baseline, user_team_id, "expected_wins")
        ros_totals = ros_totals_from_weekly(
            weekly_projections, snapshot.current_week, self._last_regular_week
        )
        starter_delta = _starter_delta(
            user.player_ids,
            user_roster,
            self._catalog,
            ros_totals,
            self._roster_structure,
            self._bench_maxes,
        )
        verdict = verdict_for_delta(user_delta, self._accept_delta, self._counter_delta)
        writeup = {"defensive_breakdown": "", "trade_pitch": ""}
        if include_writeup:
            writeup = self._advisor.describe(
                _inbound_prompt(user_team_id, offering_team_id, sending, receiving, verdict, user_delta)
            )
        return InboundTradeEvaluationResult(
            verdict=verdict,
            win_probability_delta=user_delta,
            expected_wins_delta=wins_delta,
            ros_starter_points_delta=starter_delta,
            user_p_playoff=float((after.get(user_team_id) or {}).get("p_playoff", 0.0)),
            partner_p_playoff=float((after.get(offering_team_id) or {}).get("p_playoff", 0.0)),
            optimal_bench_drops=drops,
            defensive_breakdown=writeup["defensive_breakdown"],
            trade_pitch=writeup["trade_pitch"],
        )


class TradeFinderService:
    """Search 1v1 / 2v1 / 1v2 / 2v2 trades with a partner fairness floor."""

    def __init__(
        self,
        catalog: PlayerCatalog,
        roster_structure: Mapping[str, int],
        bench_maxes: Mapping[str, int],
        drop_evaluator: SmartDropEvaluator,
        monte_carlo: RemainingWeekMonteCarlo,
        n_runs: int,
        last_regular_week: int,
        max_candidates: int,
        player_pool_top_n: int,
        ros_point_floor: float,
        partner_starter_epsilon: float,
        partner_playoff_floor: float,
        partner_wins_floor: float,
        outbound_shapes: Sequence[Sequence[int]],
        advisor: Optional[TradeAdvisorWriter] = None,
    ) -> None:
        self._catalog = catalog
        self._roster_structure = dict(roster_structure)
        self._bench_maxes = dict(bench_maxes)
        self._drop_evaluator = drop_evaluator
        self._monte_carlo = monte_carlo
        self._n_runs = int(n_runs)
        self._last_regular_week = int(last_regular_week)
        self._max_candidates = int(max_candidates)
        self._pool_top_n = int(player_pool_top_n)
        self._ros_floor = float(ros_point_floor)
        self._partner_starter_epsilon = float(partner_starter_epsilon)
        self._partner_playoff_floor = float(partner_playoff_floor)
        self._partner_wins_floor = float(partner_wins_floor)
        self._shapes = [(int(shape[0]), int(shape[1])) for shape in outbound_shapes]
        self._advisor = advisor or TradeAdvisorWriter()

    def search(
        self,
        snapshot: LeagueSnapshot,
        weekly_projections: Mapping[int, Mapping[str, Any]],
        user_team_id: int,
        include_writeup: bool = False,
        exhaustive: bool = False,
        progress: Optional[Callable[[int, int], None]] = None,
        workers: int = 1,
        partner_team_id: Optional[int] = None,
    ) -> List[TradeProposal]:
        """Return ranked outbound trade proposals.

        Parameters
        ----------
        snapshot : LeagueSnapshot
            Current league state.
        weekly_projections : mapping
            Weekly ROS vectors.
        user_team_id : int
            Searching manager.
        include_writeup : bool, optional
            When true, write pitches for the top three proposals.
        exhaustive : bool, optional
            When true, score every configured shape across full skill-player
            rosters. Skips the pool, starter, candidate-cap, and fairness filters.
        progress : callable, optional
            Called with ``(completed, total)`` while Monte Carlo scoring runs.
        workers : int, optional
            Process count for season sims. ``1`` stays on this process.
        partner_team_id : int, optional
            When set, only this opponent is searched.

        Returns
        -------
        list of TradeProposal
            Proposals sorted by the user's playoff-odds gain.
        """
        matchups = snapshot_matchups_frame(snapshot)
        baseline_rosters = snapshot.rosters_by_label()
        baseline = self._monte_carlo.evaluate(
            weekly_projections,
            matchups,
            baseline_rosters,
            n_runs=self._n_runs,
            start_week=snapshot.current_week,
            starting_records=snapshot.starting_records(),
        )
        ros_totals = ros_totals_from_weekly(
            weekly_projections, snapshot.current_week, self._last_regular_week
        )
        survivors = self._heuristic_candidates(
            snapshot,
            user_team_id,
            ros_totals,
            weekly_projections,
            matchups,
            exhaustive=exhaustive,
            partner_team_id=partner_team_id,
        )
        metrics = self._season_metrics(
            survivors,
            weekly_projections,
            matchups,
            snapshot,
            baseline,
            workers=workers,
            progress=progress,
        )
        scored: List[TradeProposal] = []
        for index, (proposal, _rosters) in enumerate(survivors):
            user_delta, partner_delta, user_wins, partner_wins = metrics[index]
            proposal.user_win_prob_delta = user_delta
            proposal.partner_win_prob_delta = partner_delta
            proposal.user_expected_wins_delta = user_wins
            proposal.partner_expected_wins_delta = partner_wins
            scored.append(proposal)
        scored.sort(key=lambda item: item.user_win_prob_delta, reverse=True)
        if include_writeup:
            for proposal in scored[:3]:
                writeup = self._advisor.describe(_outbound_prompt(proposal))
                proposal.trade_pitch = writeup["trade_pitch"]
        return scored

    def _season_metrics(
        self,
        survivors: Sequence[Tuple[TradeProposal, Dict[str, List[int]]]],
        weekly_projections: Mapping[int, Mapping[str, Any]],
        matchups,
        snapshot: LeagueSnapshot,
        baseline: Mapping[int, Mapping[str, float]],
        workers: int,
        progress: Optional[Callable[[int, int], None]],
    ) -> Dict[int, Tuple[float, float, float, float]]:
        """Return playoff and win deltas for each surviving roster.

        Parameters
        ----------
        survivors : sequence
            Proposal and post-trade rosters pairs.
        weekly_projections : mapping
            Weekly ROS vectors.
        matchups : pandas.DataFrame
            Remaining schedule.
        snapshot : LeagueSnapshot
            Current league state.
        baseline : mapping
            Pre-trade Monte Carlo summary.
        workers : int
            Process count. Values above 1 use a process pool.
        progress : callable or None
            Called with ``(completed, total)``.

        Returns
        -------
        dict of int to tuple
            ``(user_playoff, partner_playoff, user_wins, partner_wins)`` by index.
        """
        if workers > 1 and len(survivors) > 1:
            return _parallel_season_metrics(
                survivors,
                weekly_projections,
                matchups,
                snapshot,
                baseline,
                self._monte_carlo.worker_settings(),
                n_runs=self._n_runs,
                workers=workers,
                progress=progress,
            )
        return _serial_season_metrics(
            survivors,
            weekly_projections,
            matchups,
            snapshot,
            baseline,
            self._monte_carlo,
            n_runs=self._n_runs,
            progress=progress,
        )

    def _heuristic_candidates(
        self,
        snapshot: LeagueSnapshot,
        user_team_id: int,
        ros_totals: Mapping[int, float],
        weekly_projections: Mapping[int, Mapping[str, Any]],
        matchups,
        exhaustive: bool = False,
        partner_team_id: Optional[int] = None,
    ) -> List[Tuple[TradeProposal, Dict[str, List[int]]]]:
        """Generate trade shapes, then keep the top candidates before Monte Carlo."""
        user = snapshot.team(user_team_id)
        if partner_team_id is not None and int(partner_team_id) == int(user_team_id):
            raise ValueError("Search partner must be a different team.")
        user_pool = self._search_pool(user.player_ids, ros_totals, "sell", exhaustive)
        partners = (
            [snapshot.team(int(partner_team_id))]
            if partner_team_id is not None
            else list(snapshot.teams.values())
        )
        ranked: List[Tuple[float, TradeProposal, Dict[str, List[int]]]] = []
        for partner in partners:
            if partner.team_id == user_team_id:
                continue
            partner_pool = self._search_pool(partner.player_ids, ros_totals, "buy", exhaustive)
            for user_n, partner_n in self._shapes:
                for user_send in _choose(user_pool, user_n):
                    for partner_send in _choose(partner_pool, partner_n):
                        scored = self._heuristic_score(
                            snapshot,
                            user,
                            partner,
                            list(user_send),
                            list(partner_send),
                            ros_totals,
                            weekly_projections,
                            matchups,
                            apply_filters=not exhaustive,
                        )
                        if scored is None:
                            continue
                        ranked.append(scored)
        ranked.sort(key=lambda item: item[0], reverse=True)
        if exhaustive:
            return [(proposal, rosters) for _score, proposal, rosters in ranked]
        return [(proposal, rosters) for _score, proposal, rosters in ranked[: self._max_candidates]]

    def _heuristic_score(
        self,
        snapshot: LeagueSnapshot,
        user,
        partner,
        user_send: List[int],
        partner_send: List[int],
        ros_totals: Mapping[int, float],
        weekly_projections: Mapping[int, Mapping[str, Any]],
        matchups,
        apply_filters: bool = True,
    ) -> Optional[Tuple[float, TradeProposal, Dict[str, List[int]]]]:
        """Return a ranked candidate or ``None`` when the heuristic rejects it."""
        user_roster, user_drops = self._drop_evaluator.apply_simple(
            user.player_ids,
            partner_send,
            user_send,
            user.ir_player_ids,
            weekly_projections,
            snapshot.current_week,
        )
        partner_roster, _partner_drops = self._drop_evaluator.apply_simple(
            partner.player_ids,
            user_send,
            partner_send,
            partner.ir_player_ids,
            weekly_projections,
            snapshot.current_week,
        )
        user_delta = _starter_delta(
            user.player_ids,
            user_roster,
            self._catalog,
            ros_totals,
            self._roster_structure,
            self._bench_maxes,
        )
        partner_delta = _starter_delta(
            partner.player_ids,
            partner_roster,
            self._catalog,
            ros_totals,
            self._roster_structure,
            self._bench_maxes,
        )
        hole_improved = _hole_improved(
            user.player_ids,
            user_roster,
            partner_send,
            self._catalog,
            self._roster_structure,
            self._bench_maxes,
        )
        if apply_filters and user_delta <= 0 and not hole_improved:
            return None
        if apply_filters and partner_delta < self._partner_starter_epsilon:
            return None
        after = snapshot.rosters_by_label()
        after[team_label(user.team_id)] = user_roster
        after[team_label(partner.team_id)] = partner_roster
        proposal = TradeProposal(
            proposal_id=str(uuid4()),
            user_team_id=user.team_id,
            partner_team_id=partner.team_id,
            user_sends_player_ids=list(user_send),
            partner_sends_player_ids=list(partner_send),
            user_win_prob_delta=0.0,
            partner_win_prob_delta=0.0,
            user_starter_points_delta=user_delta,
            partner_starter_points_delta=partner_delta,
            optimal_drops=user_drops,
        )
        return user_delta, proposal, after

    def _search_pool(
        self,
        player_ids: Sequence[int],
        ros_totals: Mapping[int, float],
        form: str,
        exhaustive: bool,
    ) -> List[int]:
        """Return the players eligible to be sent in one search mode.

        Parameters
        ----------
        player_ids : sequence of int
            Roster player ids.
        ros_totals : mapping of int to float
            Remaining ROS totals.
        form : str
            ``sell`` or ``buy``. Used only by the heuristic pool.
        exhaustive : bool
            When true, return every skill player on the roster.

        Returns
        -------
        list of int
            Candidate player ids.
        """
        if exhaustive:
            return self._skill_player_ids(player_ids)
        return self._player_pool(player_ids, ros_totals, form=form)

    def _skill_player_ids(self, player_ids: Sequence[int]) -> List[int]:
        """Return roster ids whose catalog position is QB, RB, WR, or TE."""
        skill_ids: List[int] = []
        for player_id in player_ids:
            player = self._catalog.get(int(player_id))
            if player is None or player.position not in SKILL_POSITIONS:
                continue
            skill_ids.append(int(player_id))
        return skill_ids

    def _player_pool(
        self,
        player_ids: Sequence[int],
        ros_totals: Mapping[int, float],
        form: str = "sell",
    ) -> List[int]:
        """Return top ROS players, hole fillers, and form-tagged players.

        Parameters
        ----------
        player_ids : sequence of int
            Roster player ids.
        ros_totals : mapping of int to float
            Remaining ROS totals.
        form : str
            ``sell`` adds sell-high players. ``buy`` adds buy-low players.

        Returns
        -------
        list of int
            Candidate player ids.
        """
        ranked = sorted(
            (player_id for player_id in player_ids if ros_totals.get(player_id, 0.0) >= self._ros_floor),
            key=lambda player_id: ros_totals.get(player_id, 0.0),
            reverse=True,
        )
        pool = ranked[: self._pool_top_n]
        hole_ids = _positional_hole_ids(
            player_ids, self._catalog, ros_totals, self._roster_structure, self._bench_maxes
        )
        for player_id in hole_ids:
            if player_id not in pool and ros_totals.get(player_id, 0.0) >= self._ros_floor:
                pool.append(player_id)
        for player_id in player_ids:
            if player_id in pool:
                continue
            player = self._catalog.get(int(player_id))
            if player is None:
                continue
            if form == "sell" and is_sell_high(player):
                pool.append(player_id)
            elif form == "buy" and is_buy_low(player):
                pool.append(player_id)
        return pool


FORM_GAP = 1.25
MIN_CURRENT_GAMES_FOR_FORM = 3


def partner_within_fairness(
    playoff_delta: float,
    wins_delta: float,
    playoff_floor: float,
    wins_floor: float,
) -> bool:
    """Return whether a partner result clears the loss floor.

    A deal is kept unless playoff odds and expected wins both fall past
    their floors.

    Parameters
    ----------
    playoff_delta : float
        Partner change in playoff probability.
    wins_delta : float
        Partner change in expected wins.
    playoff_floor : float
        Minimum allowed playoff-probability change.
    wins_floor : float
        Minimum allowed expected-wins change.

    Returns
    -------
    bool
        True when the offer should stay in the printed list.
    """
    return not (playoff_delta < playoff_floor and wins_delta < wins_floor)


def is_promoted_running_back(player: Player) -> bool:
    """Return whether an RB just became the listed starter.

    Parameters
    ----------
    player : Player
        Catalog player.

    Returns
    -------
    bool
        True for depth-chart order 1 with fewer than three current games.
    """
    return (
        player.position == "RB"
        and player.sleeper_depth_chart_order == 1
        and int(player.current_season_games or 0) < MIN_CURRENT_GAMES_FOR_FORM
    )


def is_sell_high(player: Player) -> bool:
    """Return whether current scoring is well above the two-year median.

    Parameters
    ----------
    player : Player
        Catalog player.

    Returns
    -------
    bool
        True when the player is a sell-high candidate.
    """
    if is_promoted_running_back(player):
        return False
    median = player.weekly_points_median
    current = player.current_season_ppg
    if int(player.current_season_games or 0) < MIN_CURRENT_GAMES_FOR_FORM:
        return False
    if median is None or current is None or median <= 0:
        return False
    return float(current) >= float(median) * FORM_GAP


def is_buy_low(player: Player) -> bool:
    """Return whether the two-year median or ceiling is well above this season.

    Parameters
    ----------
    player : Player
        Catalog player.

    Returns
    -------
    bool
        True when the player is a buy-low candidate.
    """
    current = player.current_season_ppg
    if int(player.current_season_games or 0) < MIN_CURRENT_GAMES_FOR_FORM or current is None:
        return False
    current_value = float(current)
    if current_value <= 0:
        return False
    median = player.weekly_points_median
    ceiling = player.weekly_points_ceiling
    if median is not None and float(median) >= current_value * FORM_GAP:
        return True
    return ceiling is not None and float(ceiling) >= current_value * FORM_GAP


def verdict_for_delta(delta: float, accept_threshold: float, counter_threshold: float) -> str:
    """Map a playoff-odds delta onto ACCEPT / COUNTER / REJECT.

    Parameters
    ----------
    delta : float
        User ΔP(playoff).
    accept_threshold : float
        Minimum delta for ACCEPT.
    counter_threshold : float
        Minimum delta for COUNTER.

    Returns
    -------
    str
        Verdict token.
    """
    if delta >= accept_threshold:
        return "ACCEPT"
    if delta >= counter_threshold:
        return "COUNTER"
    return "REJECT"


def _assert_owned(player_ids: Sequence[int], roster: Sequence[int], team_id: int) -> None:
    """Raise when a listed player is not on the expected roster."""
    owned = set(int(player_id) for player_id in roster)
    missing = [player_id for player_id in player_ids if int(player_id) not in owned]
    if missing:
        raise TradeValidationError(
            f"Players {missing} are not on team {team_id}"
        )


_SEARCH_WORKER: Dict[str, Any] = {}
_SIM_CHUNK_SIZE = 8


def _serial_season_metrics(
    survivors: Sequence[Tuple[TradeProposal, Dict[str, List[int]]]],
    weekly_projections: Mapping[int, Mapping[str, Any]],
    matchups,
    snapshot: LeagueSnapshot,
    baseline: Mapping[int, Mapping[str, float]],
    monte_carlo: RemainingWeekMonteCarlo,
    n_runs: int,
    progress: Optional[Callable[[int, int], None]],
) -> Dict[int, Tuple[float, float, float, float]]:
    """Score each post-trade roster on this process."""
    metrics: Dict[int, Tuple[float, float, float, float]] = {}
    total = len(survivors)
    starting_records = snapshot.starting_records()
    for index, (proposal, after_rosters) in enumerate(survivors):
        after = monte_carlo.evaluate(
            weekly_projections,
            matchups,
            after_rosters,
            n_runs=n_runs,
            start_week=snapshot.current_week,
            starting_records=starting_records,
        )
        metrics[index] = _trade_deltas(after, baseline, proposal)
        completed = index + 1
        if progress is not None and (completed == total or completed % 25 == 0):
            progress(completed, total)
    return metrics


def _parallel_season_metrics(
    survivors: Sequence[Tuple[TradeProposal, Dict[str, List[int]]]],
    weekly_projections: Mapping[int, Mapping[str, Any]],
    matchups,
    snapshot: LeagueSnapshot,
    baseline: Mapping[int, Mapping[str, float]],
    worker_settings: Mapping[str, Any],
    n_runs: int,
    workers: int,
    progress: Optional[Callable[[int, int], None]],
) -> Dict[int, Tuple[float, float, float, float]]:
    """Score post-trade rosters across worker processes."""
    items = [
        (
            index,
            proposal.user_team_id,
            proposal.partner_team_id,
            {label: [int(player_id) for player_id in roster] for label, roster in rosters.items()},
        )
        for index, (proposal, rosters) in enumerate(survivors)
    ]
    chunks = [items[offset : offset + _SIM_CHUNK_SIZE] for offset in range(0, len(items), _SIM_CHUNK_SIZE)]
    state = {
        "weekly": weekly_projections,
        "matchups": matchups,
        "starting_records": snapshot.starting_records(),
        "start_week": int(snapshot.current_week),
        "n_runs": int(n_runs),
        "baseline": baseline,
        "worker_settings": dict(worker_settings),
    }
    metrics: Dict[int, Tuple[float, float, float, float]] = {}
    completed = 0
    total = len(items)
    if progress is not None:
        progress(0, total)
    with ProcessPoolExecutor(
        max_workers=min(int(workers), len(chunks)),
        initializer=_init_search_worker,
        initargs=(state,),
    ) as pool:
        futures = [pool.submit(_score_trade_chunk, chunk) for chunk in chunks]
        for future in as_completed(futures):
            for index, deltas in future.result():
                metrics[index] = deltas
                completed += 1
            if progress is not None and (completed == total or completed % 25 == 0):
                progress(completed, total)
    return metrics


def _init_search_worker(state: Mapping[str, Any]) -> None:
    """Store shared season inputs and a fresh Monte Carlo estimator."""
    settings = state["worker_settings"]
    _SEARCH_WORKER["weekly"] = state["weekly"]
    _SEARCH_WORKER["matchups"] = state["matchups"]
    _SEARCH_WORKER["starting_records"] = state["starting_records"]
    _SEARCH_WORKER["start_week"] = state["start_week"]
    _SEARCH_WORKER["n_runs"] = state["n_runs"]
    _SEARCH_WORKER["baseline"] = state["baseline"]
    _SEARCH_WORKER["monte_carlo"] = RemainingWeekMonteCarlo(
        roster_structure=settings["roster_structure"],
        num_playoff_teams=settings["num_playoff_teams"],
        noise_sigma=settings["noise_sigma"],
    )


def _score_trade_chunk(
    chunk: Sequence[Tuple[int, int, int, Dict[str, List[int]]]],
) -> List[Tuple[int, Tuple[float, float, float, float]]]:
    """Score one batch of post-trade rosters inside a worker."""
    monte_carlo = _SEARCH_WORKER["monte_carlo"]
    scored: List[Tuple[int, Tuple[float, float, float, float]]] = []
    for index, user_team_id, partner_team_id, rosters in chunk:
        after = monte_carlo.evaluate(
            _SEARCH_WORKER["weekly"],
            _SEARCH_WORKER["matchups"],
            rosters,
            n_runs=_SEARCH_WORKER["n_runs"],
            start_week=_SEARCH_WORKER["start_week"],
            starting_records=_SEARCH_WORKER["starting_records"],
        )
        scored.append(
            (
                index,
                (
                    _metric_delta(after, _SEARCH_WORKER["baseline"], user_team_id, "p_playoff"),
                    _metric_delta(after, _SEARCH_WORKER["baseline"], partner_team_id, "p_playoff"),
                    _metric_delta(after, _SEARCH_WORKER["baseline"], user_team_id, "expected_wins"),
                    _metric_delta(after, _SEARCH_WORKER["baseline"], partner_team_id, "expected_wins"),
                ),
            )
        )
    return scored


def _trade_deltas(
    after: Mapping[int, Mapping[str, float]],
    baseline: Mapping[int, Mapping[str, float]],
    proposal: TradeProposal,
) -> Tuple[float, float, float, float]:
    """Return user and partner playoff and win deltas for one offer."""
    return (
        _metric_delta(after, baseline, proposal.user_team_id, "p_playoff"),
        _metric_delta(after, baseline, proposal.partner_team_id, "p_playoff"),
        _metric_delta(after, baseline, proposal.user_team_id, "expected_wins"),
        _metric_delta(after, baseline, proposal.partner_team_id, "expected_wins"),
    )


def _metric_delta(
    after: Mapping[int, Mapping[str, float]],
    baseline: Mapping[int, Mapping[str, float]],
    team_id: int,
    key: str,
) -> float:
    """Return after minus baseline for one team metric."""
    after_value = float((after.get(team_id) or {}).get(key, 0.0))
    before_value = float((baseline.get(team_id) or {}).get(key, 0.0))
    return after_value - before_value


def _players_with_ros(
    player_ids: Sequence[int],
    catalog: PlayerCatalog,
    ros_totals: Mapping[int, float],
) -> List[Player]:
    """Overlay remaining ROS totals onto catalog players."""
    players: List[Player] = []
    for player_id in player_ids:
        player = catalog.get(int(player_id))
        if player is None:
            continue
        players.append(replace(player, projected_points=float(ros_totals.get(player.player_id, 0.0))))
    return players


def _starter_delta(
    before_ids: Sequence[int],
    after_ids: Sequence[int],
    catalog: PlayerCatalog,
    ros_totals: Mapping[int, float],
    roster_structure: Mapping[str, int],
    bench_maxes: Mapping[str, int],
) -> float:
    """Return ROS starter-point change from before to after."""
    before = calculate_roster_scores(
        _players_with_ros(before_ids, catalog, ros_totals),
        dict(roster_structure),
        dict(bench_maxes),
    )
    after = calculate_roster_scores(
        _players_with_ros(after_ids, catalog, ros_totals),
        dict(roster_structure),
        dict(bench_maxes),
    )
    return float(after.get("starters_total_points", 0.0)) - float(before.get("starters_total_points", 0.0))


def _hole_improved(
    before_ids: Sequence[int],
    after_ids: Sequence[int],
    received_ids: Sequence[int],
    catalog: PlayerCatalog,
    roster_structure: Mapping[str, int],
    bench_maxes: Mapping[str, int],
) -> bool:
    """Return whether a received player fills a previously empty starter slot."""
    before_players = [player for player_id in before_ids if (player := catalog.get(int(player_id)))]
    after_players = [player for player_id in after_ids if (player := catalog.get(int(player_id)))]
    before_starters, _bench, _flex = categorize_roster_by_slots(
        before_players, dict(roster_structure), dict(bench_maxes)
    )
    after_starters, _after_bench, _after_flex = categorize_roster_by_slots(
        after_players, dict(roster_structure), dict(bench_maxes)
    )
    received = set(int(player_id) for player_id in received_ids)
    for slot, players in after_starters.items():
        after_ids_in_slot = {player.player_id for player in players}
        before_ids_in_slot = {player.player_id for player in before_starters.get(slot, [])}
        if after_ids_in_slot & received and len(after_ids_in_slot) > len(before_ids_in_slot):
            return True
        if not before_ids_in_slot and after_ids_in_slot:
            return True
    return False


def _positional_hole_ids(
    player_ids: Sequence[int],
    catalog: PlayerCatalog,
    ros_totals: Mapping[int, float],
    roster_structure: Mapping[str, int],
    bench_maxes: Mapping[str, int],
) -> List[int]:
    """Return player ids at positions with fewer than required starters."""
    players = _players_with_ros(player_ids, catalog, ros_totals)
    starters, bench, _flex = categorize_roster_by_slots(
        players, dict(roster_structure), dict(bench_maxes)
    )
    hole_positions = set()
    for position, needed in roster_structure.items():
        if position == "FLEX":
            continue
        if len(starters.get(position, [])) < int(needed):
            hole_positions.add(position)
    return [
        player.player_id
        for player in list(bench)
        if player.position in hole_positions
    ]


def _choose(player_ids: Sequence[int], count: int) -> Iterable[Tuple[int, ...]]:
    """Yield combinations of ``count`` player ids."""
    if count <= 0:
        return []
    if len(player_ids) < count:
        return []
    if count == 1:
        return [(player_id,) for player_id in player_ids]
    return combinations(player_ids, count)


def _inbound_prompt(
    user_team_id: int,
    partner_team_id: int,
    sending: Sequence[int],
    receiving: Sequence[int],
    verdict: str,
    delta: float,
) -> str:
    """Build a compact inbound-eval prompt."""
    return (
        f"User team {user_team_id} sends {list(sending)} and receives {list(receiving)} "
        f"from team {partner_team_id}. Numeric verdict={verdict}, "
        f"delta_p_playoff={delta:.4f}."
    )


def _outbound_prompt(proposal: TradeProposal) -> str:
    """Build a compact outbound-search prompt."""
    return (
        f"Propose a pitch for user team {proposal.user_team_id} sending "
        f"{proposal.user_sends_player_ids} to team {proposal.partner_team_id} for "
        f"{proposal.partner_sends_player_ids}. User delta_p_playoff="
        f"{proposal.user_win_prob_delta:.4f}, partner delta_p_playoff="
        f"{proposal.partner_win_prob_delta:.4f}."
    )
