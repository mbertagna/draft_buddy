import os
from typing import Optional

import pandas as pd

from draft_buddy.data.scoring import ScoringService
from draft_buddy.data.scoring.presets import ESPN_FULL_PPR_TRACKABLE

from .adp_matcher import AdpMatcher
from .cache_paths import nflverse_cache_dir, sleeper_cache_dir
from .nflverse_client import NflverseCsvDownloader
from .nflverse_crosswalk import NflverseCrosswalkBuilder
from .nflverse_ids import normalize_gsis_id, normalize_sleeper_id
from .pipeline_diagnostics import (
    ProcessDraftResult,
    StageCounts,
    build_stage_counts,
    empty_stage_counts,
)
from .rookie_projector import RookieProjector
from .sleeper_catalog import (
    DEFAULT_SEARCH_RANK_SCAN_DEPTH,
    DEFAULT_TOP_SEARCH_RANK_REPORT_SIZE,
    SearchRankMatchReport,
    SleeperCatalogBuilder,
    build_search_rank_nflverse_match_report,
    format_search_rank_match_summary,
)
from .sleeper_client import SleeperGateway, SleeperHttpGateway

DEFAULT_SCORING_RULES = dict(ESPN_FULL_PPR_TRACKABLE)

class FantasyDataProcessor:
    """
    Orchestrates the fantasy football data pipeline.

    Delegates network I/O, scoring, and rookie projection to injected services.
    """

    def __init__(
        self,
        scoring_rules: dict = None,
        positions: list = None,
        cache_dir: str = "./data",
        bye_weeks_override: dict = None,
        project_rookies: bool = True,
        rookie_projection_method: str = "draft",
        rookie_projection_params: dict = None,
        start_year: int = 1999,
        data_downloader=None,
        scoring_service: Optional[ScoringService] = None,
        rookie_projector: Optional[RookieProjector] = None,
        adp_matcher: Optional[AdpMatcher] = None,
        sleeper_gateway: Optional[SleeperGateway] = None,
        sleeper_catalog_builder: Optional[SleeperCatalogBuilder] = None,
        crosswalk_builder: Optional[NflverseCrosswalkBuilder] = None,
    ):
        """
        Initialize the processor with configuration and optional service injections.

        Parameters
        ----------
        scoring_rules : dict, optional
            Fantasy scoring rules.
        positions : list, optional
            Positions to include.
        cache_dir : str, optional
            Root data directory. Raw downloads are cached under
            source-specific subdirectories beneath it (see
            :mod:`draft_buddy.data.cache_paths`).
        bye_weeks_override : dict, optional
            Bye week data {week: [teams]}.
        project_rookies : bool, optional
            Whether to project rookies.
        rookie_projection_method : str, optional
            'draft', 'adp', or 'hybrid'.
        rookie_projection_params : dict, optional
            Parameters for rookie projection.
        start_year : int, optional
            First year of historical data.
        data_downloader : DataDownloader, optional
            Injected downloader. Defaults to NflverseCsvDownloader.
        scoring_service : ScoringService, optional
            Injected scoring service. Defaults to ScoringService(scoring_rules).
        rookie_projector : RookieProjector, optional
            Injected rookie projector. Defaults to RookieProjector from params.
        adp_matcher : AdpMatcher, optional
            Injected ADP matcher for fuzzy matching. Defaults to AdpMatcher().
        sleeper_gateway : SleeperGateway, optional
            Injected Sleeper data source. Defaults to a SleeperHttpGateway
            cached under the Sleeper-specific subdirectory of cache_dir.
        sleeper_catalog_builder : SleeperCatalogBuilder, optional
            Injected catalog builder. Defaults to a new SleeperCatalogBuilder().
        crosswalk_builder : NflverseCrosswalkBuilder, optional
            Injected builder for sleeper_id to GSIS crosswalk rows.
        """
        self.scoring_rules = scoring_rules if scoring_rules is not None else DEFAULT_SCORING_RULES
        self.positions = positions if positions is not None else ["QB", "RB", "WR", "TE", "K"]
        self.cache_dir = cache_dir
        self.bye_weeks_override = bye_weeks_override
        self.project_rookies = project_rookies
        self.rookie_projection_method = rookie_projection_method
        self.rookie_projection_params = rookie_projection_params or {
            "scale_min": 5,
            "scale_max": 80,
            "udfa_percentile": 75,
        }
        self.start_year = start_year

        nflverse_cache = nflverse_cache_dir(cache_dir)
        self._downloader = data_downloader or NflverseCsvDownloader(nflverse_cache)
        self._scoring_service = scoring_service or ScoringService(self.scoring_rules)
        self._adp_matcher = adp_matcher or AdpMatcher()
        self._sleeper_gateway = sleeper_gateway or SleeperHttpGateway(sleeper_cache_dir(cache_dir))
        self._sleeper_catalog_builder = sleeper_catalog_builder or SleeperCatalogBuilder()
        self._crosswalk_builder = crosswalk_builder or NflverseCrosswalkBuilder(self._downloader)
        rp_params = self.rookie_projection_params
        self._rookie_projector = rookie_projector or RookieProjector(
            scale_min=rp_params.get("scale_min", 5),
            scale_max=rp_params.get("scale_max", 80),
            udfa_percentile=rp_params.get("udfa_percentile", 75),
            adp_matcher=self._adp_matcher,
        )

        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_team_bye_weeks(self) -> dict:
        """
        Processes the user-provided bye week dictionary. This is the only source for bye weeks.
        Returns a dictionary mapping a team abbreviation to its bye week.
        """
        if not self.bye_weeks_override:
            print("Warning: No 'bye_weeks_override' data provided. 'bye_week' column will be empty.")
            return {}

        print("Processing provided bye week data.")
        # Invert the dictionary from {week: [teams]} to {team: week} for easy mapping
        inverted_byes = {team: week for week, teams in self.bye_weeks_override.items() for team in teams}
        return inverted_byes

    def _resolve_nflverse_player_ids(
        self, catalog_df: pd.DataFrame, crosswalk_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add ``nflverse_player_id`` to the catalog from GSIS ids.

        Parameters
        ----------
        catalog_df : pd.DataFrame
            Sleeper catalog with ``gsis_id`` and ``sleeper_id``.
        crosswalk_df : pd.DataFrame
            Optional sleeper_id to GSIS crosswalk from nflverse rosters.

        Returns
        -------
        pd.DataFrame
            Catalog copy with ``nflverse_player_id`` and ``draft_number`` columns.
        """
        resolved_df = catalog_df.copy()
        if "gsis_id" in resolved_df.columns:
            resolved_df["nflverse_player_id"] = resolved_df["gsis_id"].apply(normalize_gsis_id)
        else:
            resolved_df["nflverse_player_id"] = pd.NA

        if not crosswalk_df.empty and "sleeper_id" in crosswalk_df.columns:
            crosswalk_lookup = crosswalk_df.copy()
            crosswalk_lookup["sleeper_id"] = crosswalk_lookup["sleeper_id"].apply(normalize_sleeper_id)
            if "gsis_id" in crosswalk_lookup.columns:
                crosswalk_lookup["crosswalk_nflverse_player_id"] = crosswalk_lookup["gsis_id"].apply(
                    normalize_gsis_id
                )
            resolved_df["sleeper_id"] = resolved_df["sleeper_id"].apply(normalize_sleeper_id)
            resolved_df = resolved_df.merge(
                crosswalk_lookup[
                    [
                        column
                        for column in ["sleeper_id", "crosswalk_nflverse_player_id", "draft_number"]
                        if column in crosswalk_lookup.columns
                    ]
                ],
                on="sleeper_id",
                how="left",
            )
            resolved_df["nflverse_player_id"] = resolved_df["nflverse_player_id"].fillna(
                resolved_df.get("crosswalk_nflverse_player_id")
            )
            if "crosswalk_nflverse_player_id" in resolved_df.columns:
                resolved_df = resolved_df.drop(columns=["crosswalk_nflverse_player_id"])

        if "draft_number" not in resolved_df.columns:
            resolved_df["draft_number"] = pd.NA

        return resolved_df

    def _attach_legacy_stats_to_catalog(
        self,
        resolved_catalog_df: pd.DataFrame,
        legacy_stats_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Attach aggregated nflverse stats onto a resolved Sleeper catalog.

        Parameters
        ----------
        resolved_catalog_df : pd.DataFrame
            Catalog with ``nflverse_player_id`` already resolved.
        legacy_stats_df : pd.DataFrame
            Aggregated legacy stats keyed by nflverse ``player_id``.

        Returns
        -------
        pd.DataFrame
            Catalog with ``total_pts``, ``games_played_frac``, and
            ``is_rookie_original`` columns added.
        """
        return self._scoring_service.attach_legacy_stats_by_player_id(
            resolved_catalog_df, legacy_stats_df
        )

    @staticmethod
    def _nflverse_player_id_allowlist(resolved_df: pd.DataFrame) -> set[int]:
        """Return non-null normalized nflverse player ids from a resolved frame.

        Parameters
        ----------
        resolved_df : pd.DataFrame
            Frame with an ``nflverse_player_id`` column.

        Returns
        -------
        set[int]
            Allowlist of integer nflverse/GSIS player ids.
        """
        if "nflverse_player_id" not in resolved_df.columns:
            return set()
        ids = resolved_df["nflverse_player_id"].dropna()
        return {int(player_id) for player_id in ids}

    def process_draft_data(self,
                           draft_year: int,
                           measure_of_center: str = 'median',
                           adp_filepath: str | None = None,
                           adp_match_threshold: int = 85,
                           adp_col_map: dict | None = None) -> ProcessDraftResult:
        """
        Orchestrate the data pipeline: fetch, score, aggregate, merge, project rookies.

        Parameters
        ----------
        draft_year : int
            The draft year.
        measure_of_center : str, optional
            'median' or 'mean' for legacy stats aggregation.
        adp_filepath : str, optional
            Path to ADP CSV for adp/hybrid rookie projection.
        adp_match_threshold : int, optional
            Fuzzy match threshold for ADP.
        adp_col_map : dict, optional
            ADP column mapping.

        Returns
        -------
        ProcessDraftResult
            Draft players, weekly projections, missing-veteran frame, stage
            counts, and optional Sleeper rank-scan report.
        """
        missing_nflverse_stats_df = pd.DataFrame()
        search_rank_report: SearchRankMatchReport | None = None
        stage_counts = empty_stage_counts()

        if self.project_rookies:
            print("Building Sleeper-anchored player catalog...")
            all_sleeper_players_df = self._sleeper_gateway.fetch_all_players()
            catalog_df = self._sleeper_catalog_builder.build_base_catalog(
                all_sleeper_players_df, self.positions
            )
            crosswalk_df = self._crosswalk_builder.build(self.cache_dir, draft_year)
            resolved_catalog_df = self._resolve_nflverse_player_ids(catalog_df, crosswalk_df)
            allowlist = self._nflverse_player_id_allowlist(resolved_catalog_df)

            print("Fetching player pool and historical data...")
            legacy_raw_df, _draft_year_stats_df = self._downloader.fetch_legacy_stats(
                draft_year=draft_year,
                start_year=self.start_year,
                end_year=draft_year,
                nflverse_player_ids=allowlist,
            )
            scored_historical = self._scoring_service.apply_scoring(legacy_raw_df)
            legacy_stats_df = self._scoring_service.aggregate_legacy_stats(
                scored_historical, measure_of_center
            )
            draft_players_df = self._attach_legacy_stats_to_catalog(
                resolved_catalog_df, legacy_stats_df
            )
            search_rank_report = self._build_search_rank_nflverse_coverage(
                all_sleeper_players_df, draft_players_df
            )
            if search_rank_report is not None:
                print(format_search_rank_match_summary(search_rank_report))
            missing_nflverse_stats_df = self._find_likely_veterans_missing_stats(draft_players_df)
            rookies_df = draft_players_df[draft_players_df['is_rookie_original'] == True]
            stage_counts = build_stage_counts(
                sleeper_directory=len(all_sleeper_players_df),
                catalog=len(catalog_df),
                gsis_resolved=int(resolved_catalog_df["nflverse_player_id"].notna().sum()),
                nflverse_matched=int((~draft_players_df["is_rookie_original"]).sum()),
                rookie_projected=int(draft_players_df["is_rookie_original"].sum()),
            )
            if not rookies_df.empty:
                print(f"Estimating points for {len(rookies_df)} rookies using method='{self.rookie_projection_method}'...")
                draft_players_df = self._rookie_projector.project_rookies(
                    draft_players_df,
                    method=self.rookie_projection_method,
                    adp_filepath=adp_filepath,
                    match_threshold=adp_match_threshold,
                    adp_col_map=adp_col_map,
                )
        else:
            print("Building nflverse draft-year roster pool...")
            roster_df = self._downloader.fetch_draft_year_roster(draft_year, self.positions)
            roster_allowlist_df = roster_df.rename(columns={"player_id": "nflverse_player_id"})
            allowlist = self._nflverse_player_id_allowlist(roster_allowlist_df)

            print("Fetching player pool and historical data...")
            legacy_raw_df, draft_year_stats_df = self._downloader.fetch_legacy_stats(
                draft_year=draft_year,
                start_year=self.start_year,
                end_year=draft_year,
                nflverse_player_ids=allowlist,
            )
            scored_historical = self._scoring_service.apply_scoring(legacy_raw_df)
            legacy_stats_df = self._scoring_service.aggregate_legacy_stats(
                scored_historical, measure_of_center
            )
            draft_year_scored = self._scoring_service.apply_scoring(draft_year_stats_df)
            draft_players_df = self._scoring_service.merge_draft_year_with_legacy(
                draft_year_scored, legacy_stats_df
            )
            stage_counts = StageCounts(
                sleeper_directory=0,
                catalog=len(roster_df),
                gsis_resolved=len(allowlist),
                nflverse_matched=len(draft_players_df),
                rookie_projected=0,
            )

        team_bye_weeks = self._get_team_bye_weeks()
        draft_players_df = draft_players_df.copy()
        draft_players_df['bye_week'] = draft_players_df['recent_team'].map(team_bye_weeks)

        draft_players_df = self._scoring_service.apply_rookie_metadata(draft_players_df)
        weekly_projections = self._scoring_service.generate_weekly_projections(draft_players_df)
        draft_players_df = self._scoring_service.finalize_draft_players(draft_players_df)

        print("Processing complete.")
        return ProcessDraftResult(
            draft_players_df=draft_players_df,
            weekly_projections=weekly_projections,
            missing_nflverse_stats_df=missing_nflverse_stats_df,
            stage_counts=stage_counts,
            search_rank_report=search_rank_report,
        )

    def _build_search_rank_nflverse_coverage(
        self,
        all_sleeper_players_df: pd.DataFrame,
        catalog_with_stats_df: pd.DataFrame,
        top_n: int = DEFAULT_TOP_SEARCH_RANK_REPORT_SIZE,
    ) -> SearchRankMatchReport | None:
        """Build Sleeper rank-order nflverse match coverage without printing the scan.

        Parameters
        ----------
        all_sleeper_players_df : pd.DataFrame
            Full Sleeper player directory.
        catalog_with_stats_df : pd.DataFrame
            Catalog immediately after nflverse stats attach.
        top_n : int, optional
            Number of top ``search_rank`` players to evaluate.

        Returns
        -------
        SearchRankMatchReport or None
            Structured report when ``search_rank`` is available.
        """
        if "search_rank" not in all_sleeper_players_df.columns:
            print("Skipping Sleeper top-player match report: search_rank unavailable.")
            return None

        return build_search_rank_nflverse_match_report(
            all_sleeper_players_df,
            catalog_with_stats_df,
            self.positions,
            eligible_top_n=top_n,
            scan_depth=DEFAULT_SEARCH_RANK_SCAN_DEPTH,
        )

    @staticmethod
    def _find_likely_veterans_missing_stats(draft_players_df: pd.DataFrame) -> pd.DataFrame:
        """Flag Sleeper players with no nflverse stats match who are unlikely rookies.

        A genuine rookie (``years_exp == 0``) with no legacy stats is
        expected and routed to rookie projection. A player with
        ``years_exp > 0`` and no legacy stats match likely indicates a
        broken nflverse crosswalk or name mismatch worth a manual look.

        Parameters
        ----------
        draft_players_df : pd.DataFrame
            Catalog after stats attach, with ``is_rookie_original`` and
            ``years_exp`` columns.

        Returns
        -------
        pd.DataFrame
            Rows likely to be veterans with a missing stats match.
        """
        if "years_exp" not in draft_players_df.columns:
            return pd.DataFrame()
        is_likely_veteran = draft_players_df["years_exp"].fillna(0) > 0
        return draft_players_df[draft_players_df["is_rookie_original"] & is_likely_veteran].reset_index(drop=True)

    def merge_adp_data(self,
                       computed_df: pd.DataFrame,
                       adp_filepath: str,
                       match_threshold: int = 85,
                       adp_col_map: dict = None) -> tuple:
        """
        Merges external ADP data with the computed player data using a weighted fuzzy matching score.

        Delegates to the injected AdpMatcher service.

        Parameters
        ----------
        computed_df : pd.DataFrame
            Player roster with columns player_id, player_display_name, position, recent_team, etc.
        adp_filepath : str
            Path to the ADP CSV file.
        match_threshold : int, optional
            Minimum fuzzy match score (0-100) to accept a match. Default 85.
        adp_col_map : dict, optional
            Column mapping for ADP file: {'Player': str, 'Team': str, 'POS': str}.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            (merged_df, unmatched_df, borderline_df).
        """
        return self._adp_matcher.merge_adp_data(
            computed_df=computed_df,
            adp_filepath=adp_filepath,
            match_threshold=match_threshold,
            adp_col_map=adp_col_map,
        )
