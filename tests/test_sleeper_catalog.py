"""Tests for building the Sleeper-anchored draft catalog."""

from __future__ import annotations

import pandas as pd

from draft_buddy.data.sleeper_catalog import (
    SleeperCatalogBuilder,
    build_search_rank_nflverse_match_report,
    classify_search_rank_player,
    format_search_rank_match_summary,
    format_top_search_rank_match_summary,
    is_sleeper_retired_or_inactive,
    is_sleeper_rookie,
    print_search_rank_nflverse_match_report,
    select_top_search_rank_players,
    summarize_nflverse_match_for_top_search_rank,
)


def _sleeper_players_df() -> pd.DataFrame:
    """Return a small, unfiltered Sleeper player directory fixture."""
    return pd.DataFrame(
        [
            {
                "sleeper_id": "1",
                "full_name": "Josh Allen",
                "position": "QB",
                "team": "BUF",
                "status": "Active",
                "injury_status": None,
                "depth_chart_position": "QB",
                "years_exp": 7,
            },
            {
                "sleeper_id": "2",
                "full_name": "Free Agent Guy",
                "position": "WR",
                "team": None,
                "status": "Active",
                "injury_status": None,
                "depth_chart_position": None,
                "years_exp": 3,
            },
            {
                "sleeper_id": "3",
                "full_name": "Some Kicker",
                "position": "K",
                "team": "KC",
                "status": "Active",
                "injury_status": None,
                "depth_chart_position": "K",
                "years_exp": 2,
            },
        ]
    )


def test_build_base_catalog_keeps_rostered_and_active_free_agent_skill_players() -> None:
    """Verify active free agents at skill positions are retained in the catalog."""
    catalog_df = SleeperCatalogBuilder().build_base_catalog(
        _sleeper_players_df(), positions=["QB", "RB", "WR", "TE"]
    )

    assert list(catalog_df["sleeper_id"]) == ["1", "2"]


def test_build_base_catalog_derives_player_id_from_sleeper_id() -> None:
    """Verify player_id is an int cast of sleeper_id."""
    catalog_df = SleeperCatalogBuilder().build_base_catalog(
        _sleeper_players_df(), positions=["QB", "RB", "WR", "TE"]
    )

    assert int(catalog_df.iloc[0]["player_id"]) == 1


def test_build_base_catalog_renames_sleeper_columns_for_catalog_use() -> None:
    """Verify Sleeper fields are renamed to catalog and sleeper_* column names."""
    catalog_df = SleeperCatalogBuilder().build_base_catalog(
        _sleeper_players_df(), positions=["QB", "RB", "WR", "TE"]
    )

    row = catalog_df.iloc[0]
    assert row["player_display_name"] == "Josh Allen"
    assert row["recent_team"] == "BUF"
    assert row["sleeper_status"] == "Active"
    assert row["sleeper_depth_chart_position"] == "QB"


def test_find_rostered_players_excluded_by_filter_reports_gaps() -> None:
    """Verify a league-rostered player excluded by the base filter is reported."""
    builder = SleeperCatalogBuilder()
    all_players_df = _sleeper_players_df()
    catalog_df = builder.build_base_catalog(all_players_df, positions=["QB", "RB", "WR", "TE"])
    rostered_df = pd.DataFrame([{"roster_id": 1, "sleeper_id": "1"}, {"roster_id": 1, "sleeper_id": "3"}])

    excluded_df = builder.find_rostered_players_excluded_by_filter(all_players_df, rostered_df, catalog_df)

    assert list(excluded_df["sleeper_id"]) == ["3"]


def test_find_rostered_players_excluded_by_filter_empty_when_fully_covered() -> None:
    """Verify no gaps are reported when every rostered player is in the catalog."""
    builder = SleeperCatalogBuilder()
    all_players_df = _sleeper_players_df()
    catalog_df = builder.build_base_catalog(all_players_df, positions=["QB", "RB", "WR", "TE"])
    rostered_df = pd.DataFrame([{"roster_id": 1, "sleeper_id": "1"}])

    excluded_df = builder.find_rostered_players_excluded_by_filter(all_players_df, rostered_df, catalog_df)

    assert excluded_df.empty


def _ranked_sleeper_players_df() -> pd.DataFrame:
    """Return a Sleeper directory with search_rank values for coverage tests."""
    return pd.DataFrame(
        [
            {
                "sleeper_id": "10",
                "full_name": "Star RB",
                "position": "RB",
                "team": "BUF",
                "status": "Active",
                "search_rank": 1,
                "years_exp": 5,
            },
            {
                "sleeper_id": "20",
                "full_name": "Star WR",
                "position": "WR",
                "team": "KC",
                "status": "Active",
                "search_rank": 2,
                "years_exp": 4,
            },
            {
                "sleeper_id": "30",
                "full_name": "Rookie QB",
                "position": "QB",
                "team": "NE",
                "status": "Active",
                "search_rank": 3,
                "years_exp": 0,
            },
            {
                "sleeper_id": "40",
                "full_name": "Retired Ghost",
                "position": "RB",
                "team": None,
                "status": "Active",
                "search_rank": 4,
                "years_exp": 8,
            },
            {
                "sleeper_id": "50",
                "full_name": "Gap TE",
                "position": "TE",
                "team": "DAL",
                "status": "Active",
                "search_rank": 5,
                "years_exp": 3,
            },
            {
                "sleeper_id": "60",
                "full_name": "Deep Bench",
                "position": "TE",
                "team": "DAL",
                "status": "Active",
                "search_rank": 9999999,
                "years_exp": 2,
            },
        ]
    )


def test_select_top_search_rank_players_returns_lowest_ranks_first() -> None:
    """Verify top-N selection sorts by ascending search_rank."""
    top_df = select_top_search_rank_players(
        _ranked_sleeper_players_df(), positions=["QB", "RB", "WR", "TE"], top_n=2
    )

    assert list(top_df["sleeper_id"]) == ["10", "20"]


def test_summarize_nflverse_match_for_top_search_rank_counts_stats_matches() -> None:
    """Verify only catalog rows with legacy stats count as matched."""
    top_df = select_top_search_rank_players(
        _ranked_sleeper_players_df(), positions=["QB", "RB", "WR", "TE"], top_n=3
    )
    catalog_df = pd.DataFrame(
        [
            {"sleeper_id": "10", "is_rookie_original": False},
            {"sleeper_id": "20", "is_rookie_original": True},
            {"sleeper_id": "30", "is_rookie_original": True},
        ]
    )

    matched_count, total_count, unmatched_df = summarize_nflverse_match_for_top_search_rank(
        top_df, catalog_df
    )

    assert matched_count == 1
    assert total_count == 3
    assert list(unmatched_df["sleeper_id"]) == ["20", "30"]


def test_format_top_search_rank_match_summary() -> None:
    """Verify the summary string reports matched and total counts."""
    summary = format_top_search_rank_match_summary(87, 100)

    assert summary == "Sleeper top 100 players: 87/100 matched to nflverse legacy stats."


def test_is_sleeper_rookie_detects_zero_years_exp() -> None:
    """Verify years_exp == 0 is treated as a rookie."""
    row = pd.Series({"years_exp": 0})

    assert is_sleeper_rookie(row) is True


def test_is_sleeper_retired_or_inactive_detects_teamless_veteran_ghost() -> None:
    """Verify active teamless veterans are excluded as retired/inactive."""
    row = pd.Series({"sleeper_id": "40", "team": None, "status": "Active", "years_exp": 8})

    assert is_sleeper_retired_or_inactive(row, {"40"}) is True


def test_build_search_rank_nflverse_match_report_skips_rookies_and_retired_for_top_n() -> None:
    """Verify eligible top-N excludes rookies and retired ghosts before counting gaps."""
    catalog_df = pd.DataFrame(
        [
            {"sleeper_id": "10", "is_rookie_original": False},
            {"sleeper_id": "20", "is_rookie_original": False},
            {"sleeper_id": "50", "is_rookie_original": True},
        ]
    )

    report = build_search_rank_nflverse_match_report(
        _ranked_sleeper_players_df(),
        catalog_df,
        positions=["QB", "RB", "WR", "TE"],
        eligible_top_n=3,
        scan_depth=10,
    )

    assert report.matched_count == 2
    assert report.gap_count == 1
    assert report.skipped_rookie_count == 1
    assert report.skipped_retired_count == 1
    assert report.pool_ranks_scanned == 5


def test_build_search_rank_nflverse_match_report_counts_veteran_gaps() -> None:
    """Verify eligible veterans without stats are counted as gaps."""
    catalog_df = pd.DataFrame(
        [
            {"sleeper_id": "10", "is_rookie_original": False},
            {"sleeper_id": "20", "is_rookie_original": False},
            {"sleeper_id": "50", "is_rookie_original": True},
        ]
    )

    report = build_search_rank_nflverse_match_report(
        _ranked_sleeper_players_df(),
        catalog_df,
        positions=["QB", "RB", "WR", "TE"],
        eligible_top_n=3,
        scan_depth=10,
    )

    assert report.matched_count == 2
    assert report.gap_count == 1


def test_format_search_rank_match_summary_includes_skip_counts() -> None:
    """Verify the headline summary reports eligible matches and skipped counts."""
    report = build_search_rank_nflverse_match_report(
        _ranked_sleeper_players_df(),
        pd.DataFrame(
            [
                {"sleeper_id": "10", "is_rookie_original": False},
                {"sleeper_id": "20", "is_rookie_original": False},
                {"sleeper_id": "50", "is_rookie_original": True},
            ]
        ),
        positions=["QB", "RB", "WR", "TE"],
        eligible_top_n=3,
        scan_depth=5,
    )
    summary = format_search_rank_match_summary(report)

    assert "Sleeper nflverse match: 2/3 eligible top-ranked players matched" in summary
    assert "skipped 1 rookies" in summary
    assert "and 1 retired/inactive" in summary


def test_print_search_rank_nflverse_match_report_collapses_matched_runs(capsys) -> None:
    """Verify long matched runs collapse after a few sample names."""
    matched_rows = [
        {
            "sleeper_id": str(index),
            "full_name": f"Matched {index}",
            "position": "RB",
            "team": "BUF",
            "status": "Active",
            "search_rank": index,
            "years_exp": 4,
        }
        for index in range(1, 8)
    ]
    sleeper_df = pd.DataFrame(matched_rows)
    catalog_df = pd.DataFrame(
        [{"sleeper_id": str(index), "is_rookie_original": False} for index in range(1, 8)]
    )
    report = build_search_rank_nflverse_match_report(
        sleeper_df,
        catalog_df,
        positions=["RB"],
        eligible_top_n=7,
        scan_depth=7,
    )

    print_search_rank_nflverse_match_report(report, max_matched_names_shown=2)
    output = capsys.readouterr().out

    assert "✓ Matched 1 (RB)" in output
    assert "(5 more matched players)" in output
    assert output.count("✓ Matched") == 2


def test_classify_search_rank_player_marks_veteran_without_stats_as_gap() -> None:
    """Verify a rostered veteran without stats is classified as a gap."""
    row = pd.Series(
        {
            "sleeper_id": "50",
            "full_name": "Gap TE",
            "position": "TE",
            "team": "DAL",
            "status": "Active",
            "years_exp": 3,
        }
    )

    category = classify_search_rank_player(row, matched_ids=set(), catalog_ids={"50"})

    assert category == "gap"
