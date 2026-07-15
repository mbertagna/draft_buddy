"""Tests for pipeline diagnostics builders."""

from __future__ import annotations

import pandas as pd

from draft_buddy.data.pipeline_diagnostics import (
    build_adp_match_section,
    build_draft_relevant_missing_veterans,
    build_google_career_timeline_url,
    build_nflverse_match_section,
    build_pipeline_diagnostics,
    build_stage_counts,
    diagnostics_to_dict,
    format_gap_copy_line,
)
from draft_buddy.data.sleeper_catalog import SearchRankMatchReport, SearchRankMatchRow


def test_build_google_career_timeline_url_includes_suffix() -> None:
    """Verify Google links request an NFL career timeline."""
    url = build_google_career_timeline_url("Ben Roethlisberger")

    assert "Ben+Roethlisberger" in url or "Ben%20Roethlisberger" in url
    assert "nfl+career+timeline" in url or "nfl%20career%20timeline" in url


def test_build_stage_counts_preserves_funnel_values() -> None:
    """Verify stage counts store the provided funnel integers."""
    counts = build_stage_counts(
        sleeper_directory=1000,
        catalog=200,
        gsis_resolved=180,
        nflverse_matched=150,
        rookie_projected=50,
    )

    assert counts.catalog == 200 and counts.nflverse_matched == 150


def test_build_nflverse_match_section_extracts_gap_players() -> None:
    """Verify gap scan rows become copyable unmatched players."""
    report = SearchRankMatchReport(
        eligible_top_n=3,
        matched_count=2,
        gap_count=1,
        skipped_rookie_count=0,
        skipped_retired_count=0,
        pool_ranks_scanned=3,
        scan_rows=(
            SearchRankMatchRow(1, "1", "Josh Allen", "QB", "matched"),
            SearchRankMatchRow(176, "176", "Ben Roethlisberger", "QB", "gap"),
            SearchRankMatchRow(15, "15", "Rookie", "RB", "rookie"),
        ),
    )

    section = build_nflverse_match_section(report, "summary")

    assert len(section.gap_players) == 1
    assert section.gap_players[0].name == "Ben Roethlisberger"
    assert section.gap_players[0].copy_line == format_gap_copy_line(
        "Ben Roethlisberger", "QB", 176
    )


def test_build_draft_relevant_missing_veterans_filters_by_search_rank() -> None:
    """Verify only missing veterans under the rank threshold are kept."""
    missing_df = pd.DataFrame(
        [
            {
                "player_display_name": "Near Rank",
                "position": "WR",
                "search_rank": 100,
            },
            {
                "player_display_name": "Far Rank",
                "position": "RB",
                "search_rank": 900,
            },
            {
                "player_display_name": "Sentinel",
                "position": "TE",
                "search_rank": 9_999_999,
            },
        ]
    )

    section = build_draft_relevant_missing_veterans(missing_df, max_search_rank=400)

    assert section.total_missing_veterans == 3
    assert [player.name for player in section.players] == ["Near Rank"]


def test_build_adp_match_section_keeps_skill_and_counts_dst() -> None:
    """Verify skill unmatched rows are listed and DST rows are counted separately."""
    unmatched_df = pd.DataFrame(
        [
            {"Player": "Hollywood Brown", "Pos": "WR", "adp": 306.5},
            {"Player": "Chiefs", "Pos": "DST", "adp": 120.0},
        ]
    )

    section = build_adp_match_section(
        unmatched_df,
        pd.DataFrame(),
        total_adp_players=10,
        matched_count=8,
    )

    assert section.unmatched_dst_count == 1
    assert len(section.skill_unmatched) == 1
    assert section.skill_unmatched[0].name == "Hollywood Brown"
    assert "unmatched ADP" in section.skill_unmatched[0].copy_line


def test_diagnostics_to_dict_includes_top_level_keys() -> None:
    """Verify diagnostics serialize with the expected top-level keys."""
    diagnostics = build_pipeline_diagnostics(
        league_id="red_league_10",
        league_name="Red League",
        draft_year=2026,
        lookback_seasons=2,
        stage_counts=build_stage_counts(
            sleeper_directory=1,
            catalog=1,
            gsis_resolved=1,
            nflverse_matched=1,
            rookie_projected=0,
        ),
        search_rank_report=None,
        search_rank_summary="none",
        missing_nflverse_stats_df=pd.DataFrame(),
        unmatched_adp_df=pd.DataFrame(),
        borderline_adp_df=pd.DataFrame(),
        total_adp_players=0,
        matched_adp_count=0,
    )

    payload = diagnostics_to_dict(diagnostics)

    assert set(payload) >= {
        "league_id",
        "stage_counts",
        "nflverse_match",
        "draft_relevant_missing",
        "adp_match",
    }
