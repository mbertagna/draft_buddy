"""Tests for data processor orchestration."""

from __future__ import annotations

import pandas as pd

from draft_buddy.data.data_processor import FantasyDataProcessor


class FakeDownloader:
    """Downloader returning deterministic frames."""

    def fetch_legacy_stats(self, draft_year: int, positions: list, start_year: int, end_year: int):
        _ = (draft_year, positions, start_year, end_year)
        historical = pd.DataFrame(
            [{"player_id": 1, "player_display_name": "Vet", "position": "QB", "recent_team": "BUF", "season": 2024, "week": 1, "total_pts": 20.0}]
        )
        draft_year_stats = pd.DataFrame(
            [{"player_id": 1, "player_display_name": "Vet", "position": "QB", "recent_team": "BUF", "total_pts": 25.0}]
        )
        return historical, draft_year_stats

    def fetch_draft_year_roster(self, draft_year: int, positions: list) -> pd.DataFrame:
        _ = (draft_year, positions)
        return pd.DataFrame()


class FakeSleeperGateway:
    """In-memory stand-in for SleeperGateway used in tests."""

    def fetch_all_players(self) -> pd.DataFrame:
        """Return a small Sleeper directory for processor tests."""
        return pd.DataFrame(
            [
                {
                    "sleeper_id": "1",
                    "full_name": "Vet",
                    "position": "QB",
                    "team": "BUF",
                    "status": "Active",
                    "injury_status": None,
                    "depth_chart_position": "QB",
                    "years_exp": 5,
                    "gsis_id": "00-0000001",
                },
                {
                    "sleeper_id": "2",
                    "full_name": "Rookie",
                    "position": "QB",
                    "team": "KC",
                    "status": "Active",
                    "injury_status": None,
                    "depth_chart_position": "QB2",
                    "years_exp": 0,
                    "gsis_id": None,
                },
            ]
        )

    def fetch_league_rosters(self, _league_id: str) -> pd.DataFrame:
        """Return an empty roster frame; unused by process_draft_data."""
        return pd.DataFrame(columns=["roster_id", "sleeper_id"])


class FakeCrosswalkBuilder:
    """Crosswalk builder returning deterministic draft slot metadata."""

    def build(self, data_root: str, draft_year: int) -> pd.DataFrame:
        _ = (data_root, draft_year)
        return pd.DataFrame(
            [
                {"sleeper_id": "2", "gsis_id": "00-0000002", "draft_number": 2},
            ]
        )


class FakeScoringService:
    """Scoring service with call tracking."""

    def __init__(self):
        self.calls = []

    def apply_scoring(self, df: pd.DataFrame) -> pd.DataFrame:
        self.calls.append("apply_scoring")
        return df.copy()

    def aggregate_legacy_stats(self, df: pd.DataFrame, measure_of_center: str) -> pd.DataFrame:
        self.calls.append(("aggregate_legacy_stats", measure_of_center))
        return pd.DataFrame(
            [{"player_id": 1, "player_display_name": "Vet", "position": "QB", "recent_team": "BUF", "total_pts": 20.0, "games_played_frac": 1.0}]
        )

    def attach_legacy_stats_by_player_id(self, catalog_df: pd.DataFrame, legacy_stats_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        _ = (legacy_stats_df, kwargs)
        self.calls.append("attach_legacy_stats_by_player_id")
        return pd.DataFrame(
            [
                {"player_id": 1, "player_display_name": "Vet", "position": "QB", "recent_team": "BUF", "total_pts": 20.0, "games_played_frac": 1.0, "draft_number": 1, "is_rookie_original": False},
                {"player_id": 2, "player_display_name": "Rookie", "position": "QB", "recent_team": "KC", "total_pts": None, "games_played_frac": None, "draft_number": 2, "is_rookie_original": True},
            ]
        )

    def merge_draft_year_with_legacy(self, draft_year_scored_df: pd.DataFrame, legacy_stats_df: pd.DataFrame) -> pd.DataFrame:
        self.calls.append("merge_draft_year_with_legacy")
        return draft_year_scored_df.assign(games_played_frac=1.0)

    def apply_rookie_metadata(self, draft_players_df: pd.DataFrame) -> pd.DataFrame:
        self.calls.append("apply_rookie_metadata")
        return draft_players_df.assign(games_played_frac=draft_players_df["games_played_frac"].fillna("R"))

    def generate_weekly_projections(self, draft_players_df: pd.DataFrame):
        self.calls.append("generate_weekly_projections")
        return {row.player_id: {"position": row.position, 1: row.total_pts} for row in draft_players_df.itertuples()}

    def finalize_draft_players(self, draft_players_df: pd.DataFrame) -> pd.DataFrame:
        self.calls.append("finalize_draft_players")
        return draft_players_df


class FakeRookieProjector:
    """Rookie projector filling missing total points."""

    def project_rookies(self, draft_players_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        _ = kwargs
        return draft_players_df.assign(total_pts=draft_players_df["total_pts"].fillna(15.0))


class FakeAdpMatcher:
    """Matcher used to validate delegation."""

    def __init__(self):
        self.last_kwargs = None

    def merge_adp_data(self, **kwargs):
        self.last_kwargs = kwargs
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def test_default_downloader_and_gateway_use_source_specific_cache_subdirs(tmp_path) -> None:
    """Verify nflverse and Sleeper raw data are cached under distinct subdirectories."""
    processor = FantasyDataProcessor(cache_dir=str(tmp_path))

    assert processor._downloader._cache_dir == str(tmp_path / "cache" / "nflverse")
    assert processor._sleeper_gateway._cache_dir == str(tmp_path / "cache" / "sleeper")


def test_get_team_bye_weeks_returns_empty_dict_without_override() -> None:
    """Verify missing bye-week overrides yield an empty mapping."""
    processor = FantasyDataProcessor(bye_weeks_override=None)

    assert processor._get_team_bye_weeks() == {}


def test_get_team_bye_weeks_inverts_week_mapping() -> None:
    """Verify bye-week overrides invert from week-to-teams into team-to-week."""
    processor = FantasyDataProcessor(bye_weeks_override={7: ["BUF"], 10: ["KC"]})

    assert processor._get_team_bye_weeks() == {"BUF": 7, "KC": 10}


def test_process_draft_data_projects_rookies_and_applies_bye_week_mapping() -> None:
    """Verify rookie-enabled processing applies rookie projection and bye weeks."""
    scoring_service = FakeScoringService()
    processor = FantasyDataProcessor(
        bye_weeks_override={7: ["BUF"], 10: ["KC"]},
        data_downloader=FakeDownloader(),
        scoring_service=scoring_service,
        rookie_projector=FakeRookieProjector(),
        adp_matcher=FakeAdpMatcher(),
        sleeper_gateway=FakeSleeperGateway(),
        crosswalk_builder=FakeCrosswalkBuilder(),
    )
    draft_players_df, weekly_projections, _ = processor.process_draft_data(draft_year=2025)

    assert set(draft_players_df["bye_week"]) == {7, 10} and weekly_projections[2][1] == 15.0
    assert "attach_legacy_stats_by_player_id" in scoring_service.calls


def test_find_likely_veterans_missing_stats_flags_non_rookie_gaps() -> None:
    """Verify a non-rookie with no stats match (years_exp > 0) is flagged."""
    draft_players_df = pd.DataFrame(
        [
            {"player_id": 1, "is_rookie_original": False, "years_exp": 5},
            {"player_id": 2, "is_rookie_original": True, "years_exp": 4},
            {"player_id": 3, "is_rookie_original": True, "years_exp": 0},
        ]
    )

    result = FantasyDataProcessor._find_likely_veterans_missing_stats(draft_players_df)

    assert list(result["player_id"]) == [2]


def test_find_likely_veterans_missing_stats_returns_empty_without_years_exp_column() -> None:
    """Verify catalogs without a years_exp column produce no false positives."""
    draft_players_df = pd.DataFrame([{"player_id": 1, "is_rookie_original": True}])

    result = FantasyDataProcessor._find_likely_veterans_missing_stats(draft_players_df)

    assert result.empty


def test_process_draft_data_uses_draft_year_merge_when_rookies_disabled() -> None:
    """Verify non-rookie processing uses the draft-year merge path."""
    scoring_service = FakeScoringService()
    processor = FantasyDataProcessor(
        project_rookies=False,
        data_downloader=FakeDownloader(),
        scoring_service=scoring_service,
        rookie_projector=FakeRookieProjector(),
        adp_matcher=FakeAdpMatcher(),
        sleeper_gateway=FakeSleeperGateway(),
        crosswalk_builder=FakeCrosswalkBuilder(),
    )
    processor.process_draft_data(draft_year=2025)

    assert "merge_draft_year_with_legacy" in scoring_service.calls


def test_merge_adp_data_delegates_to_matcher() -> None:
    """Verify merge_adp_data forwards its arguments to the matcher dependency."""
    matcher = FakeAdpMatcher()
    processor = FantasyDataProcessor(adp_matcher=matcher)
    computed_df = pd.DataFrame([{"player_id": 1}])
    processor.merge_adp_data(computed_df, "adp.csv", match_threshold=90, adp_col_map={"Player": "Player"})

    assert matcher.last_kwargs["match_threshold"] == 90 and matcher.last_kwargs["computed_df"].equals(computed_df)


def test_attach_legacy_stats_to_catalog_matches_veteran_by_gsis_id() -> None:
    """Verify a veteran attaches stats via Sleeper gsis_id without name matching."""
    processor = FantasyDataProcessor()
    catalog_df = pd.DataFrame(
        [
            {
                "player_id": 4983,
                "player_display_name": "DJ Moore",
                "position": "WR",
                "recent_team": "BUF",
                "sleeper_id": "4983",
                "gsis_id": "00-0034827",
                "years_exp": 8,
            }
        ]
    )
    legacy_stats_df = pd.DataFrame(
        [{"player_id": 34827, "total_pts": 12.2, "games_played_frac": 0.9}]
    )

    result = processor._attach_legacy_stats_to_catalog(catalog_df, legacy_stats_df, pd.DataFrame())

    assert float(result.iloc[0]["total_pts"]) == 12.2
    assert bool(result.iloc[0]["is_rookie_original"]) is False


def test_attach_legacy_stats_to_catalog_uses_crosswalk_when_sleeper_gsis_missing() -> None:
    """Verify crosswalk gsis_id is used when Sleeper does not publish one."""
    processor = FantasyDataProcessor()
    catalog_df = pd.DataFrame(
        [
            {
                "player_id": 9226,
                "player_display_name": "De'Von Achane",
                "position": "RB",
                "recent_team": "MIA",
                "sleeper_id": "9226",
                "gsis_id": None,
                "years_exp": 3,
            }
        ]
    )
    crosswalk_df = pd.DataFrame(
        [{"sleeper_id": "9226", "gsis_id": "00-0039040", "draft_number": 84}]
    )
    legacy_stats_df = pd.DataFrame(
        [{"player_id": 39040, "total_pts": 20.5, "games_played_frac": 1.0}]
    )

    result = processor._attach_legacy_stats_to_catalog(catalog_df, legacy_stats_df, crosswalk_df)

    assert float(result.iloc[0]["total_pts"]) == 20.5
    assert bool(result.iloc[0]["is_rookie_original"]) is False


def test_attach_legacy_stats_to_catalog_marks_unmatched_rows_as_rookies() -> None:
    """Verify unmatched catalog rows are flagged for rookie projection."""
    processor = FantasyDataProcessor()
    catalog_df = pd.DataFrame(
        [
            {
                "player_id": 9999,
                "player_display_name": "Brand New",
                "position": "WR",
                "recent_team": "DAL",
                "sleeper_id": "9999",
                "gsis_id": None,
                "years_exp": 0,
            }
        ]
    )

    result = processor._attach_legacy_stats_to_catalog(catalog_df, pd.DataFrame(), pd.DataFrame())

    assert bool(result.iloc[0]["is_rookie_original"]) is True
