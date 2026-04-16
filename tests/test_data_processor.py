"""Tests for data processor orchestration."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from draft_buddy.data.data_processor import FantasyDataProcessor


class FakeDownloader:
    """Downloader returning deterministic frames."""

    def fetch_player_pool(self, draft_year: int, positions: list, start_year: int, end_year: int):
        _ = (draft_year, positions, start_year, end_year)
        draft_pool = pd.DataFrame(
            [
                {"player_id": 1, "player_display_name": "Vet", "position": "QB", "recent_team": "BUF", "draft_number": 1},
                {"player_id": 2, "player_display_name": "Rookie", "position": "QB", "recent_team": "KC", "draft_number": 2},
            ]
        )
        historical = pd.DataFrame(
            [{"player_id": 1, "player_display_name": "Vet", "position": "QB", "recent_team": "BUF", "season": 2024, "week": 1, "total_pts": 20.0}]
        )
        draft_year_stats = pd.DataFrame(
            [{"player_id": 1, "player_display_name": "Vet", "position": "QB", "recent_team": "BUF", "total_pts": 25.0}]
        )
        return draft_pool, historical, draft_year_stats


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

    def merge_roster_with_legacy(self, draft_pool_df: pd.DataFrame, legacy_stats_df: pd.DataFrame) -> pd.DataFrame:
        self.calls.append("merge_roster_with_legacy")
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
    )
    draft_players_df, weekly_projections = processor.process_draft_data(draft_year=2025)

    assert set(draft_players_df["bye_week"]) == {7, 10} and weekly_projections[2][1] == 15.0


def test_process_draft_data_uses_draft_year_merge_when_rookies_disabled() -> None:
    """Verify non-rookie processing uses the draft-year merge path."""
    scoring_service = FakeScoringService()
    processor = FantasyDataProcessor(
        project_rookies=False,
        data_downloader=FakeDownloader(),
        scoring_service=scoring_service,
        rookie_projector=FakeRookieProjector(),
        adp_matcher=FakeAdpMatcher(),
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
