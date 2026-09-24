"""Composition root for in-season snapshot, projections, and trades."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from draft_buddy.core.entities import PlayerCatalog
from draft_buddy.data.sleeper_client import SleeperGateway, SleeperHttpGateway
from draft_buddy.inseason.advisor import TradeAdvisorWriter
from draft_buddy.inseason.drops import SmartDropEvaluator
from draft_buddy.inseason.espn_adapter import EspnSnapshotAdapter, matchups_from_rows
from draft_buddy.inseason.schedule import overlay_matchups
from draft_buddy.inseason.exceptions import SnapshotNotFoundError
from draft_buddy.inseason.probability import RemainingWeekMonteCarlo
from draft_buddy.inseason.projections import (
    SlidingScaleProjectionEngine,
    inputs_from_catalog,
    ros_totals_from_weekly,
)
from draft_buddy.inseason.sleeper_adapter import SleeperSnapshotAdapter
from draft_buddy.inseason.state import LeagueSnapshot, roster_cap
from draft_buddy.inseason.store import InSeasonStore
from draft_buddy.inseason.trades import InboundTradeService, TradeFinderService


class InSeasonService:
    """League-scoped facade used by the Compose trades CLI.

    Parameters
    ----------
    config : Config
        Runtime configuration.
    catalog : PlayerCatalog
        Shared player catalog.
    store : InSeasonStore
        Snapshot and projection persistence.
    sleeper_gateway : SleeperGateway, optional
        Injected Sleeper client.
    advisor : TradeAdvisorWriter, optional
        Optional LLM writeup helper.
    rng : numpy.random.Generator, optional
        Monte Carlo RNG for tests.
    """

    def __init__(
        self,
        config,
        catalog: PlayerCatalog,
        store: InSeasonStore,
        sleeper_gateway: Optional[SleeperGateway] = None,
        advisor: Optional[TradeAdvisorWriter] = None,
        rng=None,
    ) -> None:
        self._config = config
        self._catalog = catalog
        self._store = store
        self._sleeper_gateway = sleeper_gateway
        self._advisor = advisor or TradeAdvisorWriter()
        self._espn_adapter = EspnSnapshotAdapter()
        self._engine = SlidingScaleProjectionEngine(
            weights=config.projections.sliding_scale_weights,
            min_current_games_for_full_weight=config.projections.min_current_games_for_full_weight,
            depth_chart_curr_weight_cap=config.projections.depth_chart_curr_weight_cap,
            depth_chart_starter_games=config.projections.depth_chart_starter_games,
        )
        self._monte_carlo = RemainingWeekMonteCarlo(
            roster_structure=config.draft.ROSTER_STRUCTURE,
            num_playoff_teams=int(config.reward.REGULAR_SEASON_REWARD.get("NUM_PLAYOFF_TEAMS", 6)),
            noise_sigma=config.inseason.monte_carlo_noise_sigma,
            rng=rng,
        )
        cap = roster_cap(config.draft.ROSTER_STRUCTURE, config.draft.TOTAL_BENCH_SIZE)
        self._drop_evaluator = SmartDropEvaluator(
            catalog=catalog,
            roster_structure=config.draft.ROSTER_STRUCTURE,
            bench_maxes=config.draft.BENCH_MAXES,
            roster_cap=cap,
            variations_count=config.trade.smart_drop_variations_count,
            monte_carlo=self._monte_carlo,
        )
        last_week = int(config.reward.NUM_REGULAR_SEASON_WEEKS)
        self._inbound = InboundTradeService(
            catalog=catalog,
            roster_structure=config.draft.ROSTER_STRUCTURE,
            bench_maxes=config.draft.BENCH_MAXES,
            drop_evaluator=self._drop_evaluator,
            monte_carlo=self._monte_carlo,
            n_runs=config.inseason.monte_carlo_runs_eval,
            accept_delta=config.trade.accept_playoff_delta,
            counter_delta=config.trade.counter_playoff_delta,
            last_regular_week=last_week,
            advisor=self._advisor,
        )
        self._finder = TradeFinderService(
            catalog=catalog,
            roster_structure=config.draft.ROSTER_STRUCTURE,
            bench_maxes=config.draft.BENCH_MAXES,
            drop_evaluator=self._drop_evaluator,
            monte_carlo=self._monte_carlo,
            n_runs=config.inseason.monte_carlo_runs_search,
            last_regular_week=last_week,
            max_candidates=config.trade.max_monte_carlo_candidates,
            player_pool_top_n=config.trade.player_pool_top_n,
            ros_point_floor=config.trade.ros_point_floor,
            partner_starter_epsilon=config.trade.partner_starter_delta_epsilon,
            partner_playoff_floor=config.trade.partner_playoff_delta_floor,
            partner_wins_floor=config.trade.partner_expected_wins_floor,
            outbound_shapes=config.trade.outbound_shapes,
            advisor=self._advisor,
        )

    def refresh_sleeper_snapshot(self, current_week: Optional[int] = None) -> LeagueSnapshot:
        """Fetch live Sleeper rosters into the persisted snapshot.

        Parameters
        ----------
        current_week : int, optional
            Override for the NFL week.

        Returns
        -------
        LeagueSnapshot
            Stored snapshot.
        """
        gateway = self._sleeper_gateway or SleeperHttpGateway()
        adapter = SleeperSnapshotAdapter(gateway)
        snapshot = adapter.build_snapshot(
            league_id=self._config.league.league_id,
            sleeper_league_id=self._config.draft.SLEEPER_LEAGUE_ID,
            roster_id_to_team_id=self._config.draft.SLEEPER_ROSTER_ID_TO_TEAM_ID,
            catalog=self._catalog,
            team_names=self._config.draft.TEAM_MANAGER_MAPPING,
            last_regular_week=int(self._config.reward.NUM_REGULAR_SEASON_WEEKS),
            current_week=current_week,
        )
        self._store.save_snapshot(snapshot)
        return snapshot

    def refresh_espn_snapshot(self, payload: Mapping[str, Any]) -> LeagueSnapshot:
        """Build and persist a snapshot from an ESPN paste payload.

        Parameters
        ----------
        payload : mapping
            Pasted week, records, and player names or ids.

        Returns
        -------
        LeagueSnapshot
            Stored snapshot.
        """
        team_ids = list(range(1, int(self._config.draft.NUM_TEAMS) + 1))
        snapshot = self._espn_adapter.build_snapshot(
            league_id=self._config.league.league_id,
            payload=payload,
            catalog=self._catalog,
            team_names=self._config.draft.TEAM_MANAGER_MAPPING,
            last_regular_week=int(self._config.reward.NUM_REGULAR_SEASON_WEEKS),
            team_ids=team_ids,
        )
        self._store.save_snapshot(snapshot)
        return snapshot

    def apply_schedule(self, payload: Mapping[str, Any]) -> LeagueSnapshot:
        """Merge an optional matchup file onto the stored snapshot.

        Parameters
        ----------
        payload : mapping
            Object with a ``matchups`` list of week, home, and away rows.

        Returns
        -------
        LeagueSnapshot
            Snapshot with the merged remaining slate.

        Raises
        ------
        SnapshotNotFoundError
            If no snapshot has been saved.
        """
        snapshot = self.get_snapshot()
        current_week = int(snapshot.current_week)
        last_week = int(self._config.reward.NUM_REGULAR_SEASON_WEEKS)
        provided = matchups_from_rows(
            list(payload.get("matchups") or []),
            self._config.draft.TEAM_MANAGER_MAPPING,
            current_week,
        )
        team_ids = list(range(1, int(self._config.draft.NUM_TEAMS) + 1))
        snapshot.remaining_matchups = overlay_matchups(
            snapshot.remaining_matchups,
            provided,
            team_ids,
            current_week,
            last_week,
        )
        self._store.save_snapshot(snapshot)
        return snapshot

    def get_snapshot(self) -> LeagueSnapshot:
        """Return the persisted snapshot.

        Returns
        -------
        LeagueSnapshot
            Stored state.

        Raises
        ------
        SnapshotNotFoundError
            If no snapshot has been saved.
        """
        snapshot = self._store.load_snapshot()
        if snapshot is None:
            raise SnapshotNotFoundError("No in-season snapshot is loaded.")
        return snapshot

    def refresh_projections(self, current_week: Optional[int] = None) -> Dict[int, Dict[str, object]]:
        """Rebuild weekly ROS vectors from the catalog and persist them.

        Parameters
        ----------
        current_week : int, optional
            Week used for remaining-season fill. Snapshot week when omitted.

        Returns
        -------
        dict
            Weekly projections keyed by player id.
        """
        week = int(current_week) if current_week is not None else self._current_week()
        inputs = inputs_from_catalog(self._catalog, week)
        weekly = self._engine.project_players(inputs, week)
        self._store.save_projections(weekly, week)
        return weekly

    def weekly_projections(self, snapshot: LeagueSnapshot) -> Dict[int, Dict[str, object]]:
        """Return stored ROS vectors, generating catalog fallbacks when missing.

        Parameters
        ----------
        snapshot : LeagueSnapshot
            Current league state.

        Returns
        -------
        dict
            Weekly projections.
        """
        stored = self._store.load_projections()
        if stored:
            return stored
        return self.refresh_projections(snapshot.current_week)

    def evaluate_trade(
        self,
        user_team_id: int,
        offering_team_id: int,
        players_sending: Sequence[int],
        players_receiving: Sequence[int],
        include_writeup: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate an inbound trade against the stored snapshot.

        Parameters
        ----------
        user_team_id : int
            Evaluating manager.
        offering_team_id : int
            Offering manager.
        players_sending : sequence of int
            Players the user gives up.
        players_receiving : sequence of int
            Players the user receives.
        include_writeup : bool, optional
            Request an LLM writeup after the numbers.

        Returns
        -------
        dict
            Inbound evaluation payload.
        """
        snapshot = self.get_snapshot()
        weekly = self.weekly_projections(snapshot)
        result = self._inbound.evaluate(
            snapshot,
            weekly,
            user_team_id,
            offering_team_id,
            players_sending,
            players_receiving,
            include_writeup=include_writeup,
        )
        return result.to_dict()

    def search_trades(
        self,
        user_team_id: int,
        include_writeup: bool = False,
        exhaustive: bool = False,
        progress: Optional[Callable[[int, int], None]] = None,
        workers: int = 1,
        partner_team_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Search outbound trades for one manager.

        Parameters
        ----------
        user_team_id : int
            Searching manager.
        include_writeup : bool, optional
            Request pitches for the top three proposals.
        exhaustive : bool, optional
            Score every configured shape on full skill-player rosters.
        progress : callable, optional
            Called with ``(completed, total)`` during Monte Carlo scoring.
        workers : int, optional
            Process count for season sims.
        partner_team_id : int, optional
            When set, only this opponent is searched.

        Returns
        -------
        dict
            Ranked proposals.
        """
        snapshot = self.get_snapshot()
        weekly = self.weekly_projections(snapshot)
        proposals = self._finder.search(
            snapshot,
            weekly,
            user_team_id,
            include_writeup=include_writeup,
            exhaustive=exhaustive,
            progress=progress,
            workers=workers,
            partner_team_id=partner_team_id,
        )
        return {
            "proposals": [proposal.to_dict() for proposal in proposals],
            "exhaustive": exhaustive,
        }

    def state_payload(self) -> Dict[str, Any]:
        """Return snapshot plus ROS totals for CLI show.

        Returns
        -------
        dict
            UI state payload.
        """
        snapshot = self.get_snapshot()
        weekly = self.weekly_projections(snapshot)
        ros_totals = ros_totals_from_weekly(
            weekly, snapshot.current_week, int(self._config.reward.NUM_REGULAR_SEASON_WEEKS)
        )
        teams = []
        for team in snapshot.teams.values():
            players = []
            for player_id in team.player_ids:
                player = self._catalog.get(player_id)
                if player is None:
                    continue
                players.append(
                    {
                        "player_id": player.player_id,
                        "name": player.name,
                        "position": player.position,
                        "ros_points": round(float(ros_totals.get(player.player_id, 0.0)), 1),
                    }
                )
            teams.append(
                {
                    "team_id": team.team_id,
                    "display_name": team.display_name,
                    "wins": team.record.wins,
                    "losses": team.record.losses,
                    "ties": team.record.ties,
                    "players": players,
                }
            )
        payload = snapshot.to_dict()
        payload["teams"] = teams
        return payload

    def _current_week(self) -> int:
        """Return snapshot week or 1 when none is stored."""
        snapshot = self._store.load_snapshot()
        if snapshot is None:
            return 1
        return int(snapshot.current_week)
