"""Tests for pipeline HTML/JSON report rendering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from draft_buddy.data.pipeline_diagnostics import (
    build_pipeline_diagnostics,
    build_stage_counts,
)
from draft_buddy.data.pipeline_report import (
    export_pipeline_report,
    render_pipeline_report_html,
)
from draft_buddy.data.sleeper_catalog import SearchRankMatchReport, SearchRankMatchRow


def _sample_diagnostics():
    """Build a small diagnostics fixture for report tests."""
    report = SearchRankMatchReport(
        eligible_top_n=2,
        matched_count=1,
        gap_count=1,
        skipped_rookie_count=0,
        skipped_retired_count=0,
        pool_ranks_scanned=2,
        scan_rows=(
            SearchRankMatchRow(1, "1", "Josh Allen", "QB", "matched"),
            SearchRankMatchRow(176, "176", "Ben Roethlisberger", "QB", "gap"),
        ),
    )
    return build_pipeline_diagnostics(
        league_id="red_league_10",
        league_name="Red League",
        draft_year=2026,
        lookback_seasons=2,
        stage_counts=build_stage_counts(
            sleeper_directory=100,
            catalog=50,
            gsis_resolved=40,
            nflverse_matched=30,
            rookie_projected=20,
        ),
        search_rank_report=report,
        search_rank_summary="Sleeper nflverse match: 1/2 eligible top-ranked players matched",
        missing_nflverse_stats_df=pd.DataFrame(
            [
                {
                    "player_display_name": "Near Gap",
                    "position": "WR",
                    "search_rank": 50,
                }
            ]
        ),
        unmatched_adp_df=pd.DataFrame(
            [{"Player": "Hollywood Brown", "Pos": "WR", "adp": 306.5}]
        ),
        borderline_adp_df=pd.DataFrame(),
        total_adp_players=10,
        matched_adp_count=9,
    )


def test_render_pipeline_report_html_includes_gap_and_google_timeline() -> None:
    """Verify HTML includes funnel labels, gap names, and timeline search URLs."""
    html = render_pipeline_report_html(_sample_diagnostics())

    assert "Stage funnel" in html
    assert "Ben Roethlisberger" in html
    assert "nfl+career+timeline" in html or "nfl%20career%20timeline" in html
    assert "Ben Roethlisberger (QB) rank 176 — no nflverse stats" in html
    assert "Hollywood Brown" in html


def test_export_pipeline_report_writes_json_and_html(tmp_path: Path) -> None:
    """Verify export writes both diagnostic artifacts."""
    json_path, html_path = export_pipeline_report(_sample_diagnostics(), str(tmp_path))

    assert Path(json_path).exists()
    assert Path(html_path).exists()
    assert "pipeline_diagnostics.json" in json_path
    assert "pipeline_report.html" in html_path
    assert "Ben Roethlisberger" in Path(html_path).read_text(encoding="utf-8")
