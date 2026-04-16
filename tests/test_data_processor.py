"""Tests for fantasy data processor orchestration."""

from unittest.mock import Mock

import pandas as pd

from draft_buddy.data_pipeline.data_processor import FantasyDataProcessor


def test_get_team_bye_weeks_inverts_week_to_team_mapping():
    """Verify bye-week config is inverted into team-to-week map."""
    processor = FantasyDataProcessor(
        bye_weeks_override={7: ["KC", "BUF"], 8: ["MIA"]},
        data_downloader=Mock(),
        scoring_service=Mock(),
        rookie_projector=Mock(),
        adp_matcher=Mock(),
    )

    assert processor._get_team_bye_weeks() == {"KC": 7, "BUF": 7, "MIA": 8}


def test_get_team_bye_weeks_returns_empty_dict_when_override_is_none():
    """Verify missing bye-week override returns empty mapping."""
    processor = FantasyDataProcessor(
        bye_weeks_override=None,
        data_downloader=Mock(),
        scoring_service=Mock(),
        rookie_projector=Mock(),
        adp_matcher=Mock(),
    )

    assert processor._get_team_bye_weeks() == {}


def test_process_draft_data_calls_dependencies_in_expected_sequence():
    """Verify orchestration order across downloader, scorer, projector, and finalization."""
    calls = []
    draft_pool_df = pd.DataFrame(
        [{"player_id": 1, "recent_team": "KC", "is_rookie_original": True, "position": "QB"}]
    )
    legacy_stats_df = pd.DataFrame([{"player_id": 99, "position": "QB"}])
    draft_year_stats_df = pd.DataFrame([{"player_id": 100, "position": "QB"}])
    merged_df = pd.DataFrame(
        [{"player_id": 1, "recent_team": "KC", "is_rookie_original": True, "position": "QB", "total_pts": None}]
    )
    projected_df = pd.DataFrame(
        [{"player_id": 1, "recent_team": "KC", "is_rookie_original": True, "position": "QB", "total_pts": 200.0}]
    )
    metadata_df = projected_df.copy()
    finalized_df = projected_df.copy()
    weekly_projections = {1: {"position": "QB", 1: 15.0}}

    downloader = Mock()
    scorer = Mock()
    projector = Mock()

    downloader.fetch_player_pool.side_effect = lambda **kwargs: (
        calls.append("fetch_player_pool") or (draft_pool_df, legacy_stats_df, draft_year_stats_df)
    )
    scorer.apply_scoring.side_effect = lambda dataframe: calls.append("apply_scoring") or dataframe
    scorer.aggregate_legacy_stats.side_effect = (
        lambda dataframe, moc: calls.append("aggregate_legacy_stats") or dataframe
    )
    scorer.merge_roster_with_legacy.side_effect = (
        lambda roster, legacy: calls.append("merge_roster_with_legacy") or merged_df
    )
    projector.project_rookies.side_effect = (
        lambda *args, **kwargs: calls.append("project_rookies") or projected_df
    )
    scorer.apply_rookie_metadata.side_effect = (
        lambda dataframe: calls.append("apply_rookie_metadata") or metadata_df
    )
    scorer.generate_weekly_projections.side_effect = (
        lambda dataframe: calls.append("generate_weekly_projections") or weekly_projections
    )
    scorer.finalize_draft_players.side_effect = (
        lambda dataframe: calls.append("finalize_draft_players") or finalized_df
    )

    processor = FantasyDataProcessor(
        bye_weeks_override={7: ["KC"]},
        data_downloader=downloader,
        scoring_service=scorer,
        rookie_projector=projector,
        adp_matcher=Mock(),
    )
    processor.process_draft_data(draft_year=2025)

    assert calls == [
        "fetch_player_pool",
        "apply_scoring",
        "aggregate_legacy_stats",
        "merge_roster_with_legacy",
        "project_rookies",
        "apply_rookie_metadata",
        "generate_weekly_projections",
        "finalize_draft_players",
    ]


def test_process_draft_data_returns_finalized_players_and_weekly_projections():
    """Verify process_draft_data returns expected tuple outputs."""
    draft_pool_df = pd.DataFrame(
        [{"player_id": 1, "recent_team": "KC", "is_rookie_original": False, "position": "QB"}]
    )
    legacy_stats_df = pd.DataFrame([{"player_id": 1, "position": "QB"}])
    draft_year_stats_df = pd.DataFrame([{"player_id": 1, "position": "QB"}])
    merged_df = pd.DataFrame(
        [{"player_id": 1, "recent_team": "KC", "is_rookie_original": False, "position": "QB", "total_pts": 100.0}]
    )
    weekly_projections = {1: {"position": "QB", 1: 15.0}}
    finalized_df = merged_df.copy()

    downloader = Mock()
    scorer = Mock()
    projector = Mock()
    downloader.fetch_player_pool.return_value = (draft_pool_df, legacy_stats_df, draft_year_stats_df)
    scorer.apply_scoring.return_value = legacy_stats_df
    scorer.aggregate_legacy_stats.return_value = legacy_stats_df
    scorer.merge_roster_with_legacy.return_value = merged_df
    scorer.apply_rookie_metadata.return_value = merged_df
    scorer.generate_weekly_projections.return_value = weekly_projections
    scorer.finalize_draft_players.return_value = finalized_df
    projector.project_rookies.return_value = merged_df

    processor = FantasyDataProcessor(
        bye_weeks_override={7: ["KC"]},
        data_downloader=downloader,
        scoring_service=scorer,
        rookie_projector=projector,
        adp_matcher=Mock(),
    )
    result = processor.process_draft_data(draft_year=2025)

    assert result == (finalized_df, weekly_projections)


def test_process_draft_data_uses_draft_year_merge_when_rookie_projection_disabled():
    """Verify disabled rookie projection path merges draft-year with legacy stats."""
    draft_pool_df = pd.DataFrame([{"player_id": 1, "recent_team": "KC", "position": "QB"}])
    legacy_stats_df = pd.DataFrame([{"player_id": 1, "position": "QB"}])
    draft_year_stats_df = pd.DataFrame([{"player_id": 1, "position": "QB"}])
    merged_df = pd.DataFrame([{"player_id": 1, "recent_team": "KC", "position": "QB", "total_pts": 120.0}])

    downloader = Mock()
    scorer = Mock()
    downloader.fetch_player_pool.return_value = (draft_pool_df, legacy_stats_df, draft_year_stats_df)
    scorer.apply_scoring.side_effect = [legacy_stats_df, draft_year_stats_df]
    scorer.aggregate_legacy_stats.return_value = legacy_stats_df
    scorer.merge_draft_year_with_legacy.return_value = merged_df
    scorer.apply_rookie_metadata.return_value = merged_df
    scorer.generate_weekly_projections.return_value = {1: {"position": "QB", 1: 10.0}}
    scorer.finalize_draft_players.return_value = merged_df

    processor = FantasyDataProcessor(
        project_rookies=False,
        bye_weeks_override={7: ["KC"]},
        data_downloader=downloader,
        scoring_service=scorer,
        rookie_projector=Mock(),
        adp_matcher=Mock(),
    )
    processor.process_draft_data(draft_year=2025)

    assert scorer.merge_draft_year_with_legacy.called


def test_merge_adp_data_delegates_to_injected_adp_matcher():
    """Verify merge_adp_data forwards call to matcher dependency."""
    matcher = Mock()
    matcher.merge_adp_data.return_value = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    processor = FantasyDataProcessor(
        data_downloader=Mock(),
        scoring_service=Mock(),
        rookie_projector=Mock(),
        adp_matcher=matcher,
    )
    processor.merge_adp_data(pd.DataFrame(), "adp.csv", 85)

    assert matcher.merge_adp_data.called
