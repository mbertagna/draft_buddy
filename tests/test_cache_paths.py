"""Tests for raw data cache directory conventions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from draft_buddy.data.cache_paths import (
    adp_cache_dir,
    insights_run_current_path,
    insights_runs_dir,
    insights_search_cache_dir,
    insights_search_runs_dir,
    insights_synthesis_cache_dir,
    insights_synthesis_runs_dir,
    model_adp_by_position_output_path,
    model_adp_output_path,
    new_insights_run_id,
    nflverse_cache_dir,
    player_insights_exports_dir,
    player_insights_output_path,
    position_guide_output_path,
    resolve_latest_player_insights_path,
    resolve_latest_position_guide_path,
    sleeper_cache_dir,
)
from draft_buddy.data.insights.schemas import (
    Confidence,
    DepthRole,
    PlayerInsight,
    PlayerInsightsFile,
    PlayingTimeTier,
    RecoveryStatus,
    RiskLevel,
)


def _sample_insight() -> PlayerInsight:
    """Return a minimal valid player insight."""
    return PlayerInsight(
        outlook_phrase="Starter role expected",
        summary="Should lead the backfield.",
        depth_role=DepthRole.STARTER,
        playing_time_tier=PlayingTimeTier.HIGH,
        injury_risk=RiskLevel.LOW,
        upside=RiskLevel.HIGH,
        recovery_status=RecoveryStatus.NA,
        overall_confidence=Confidence.HIGH,
    )


def test_nflverse_cache_dir_is_nested_under_data_root() -> None:
    """Verify the nflverse cache path is scoped under a 'cache/nflverse' subdirectory."""
    assert nflverse_cache_dir("./data") == "./data/cache/nflverse"


def test_sleeper_cache_dir_is_nested_under_data_root() -> None:
    """Verify the Sleeper cache path is scoped under a 'cache/sleeper' subdirectory."""
    assert sleeper_cache_dir("./data") == "./data/cache/sleeper"


def test_adp_cache_dir_is_nested_under_data_root() -> None:
    """Verify the ADP cache path is scoped under a 'cache/adp' subdirectory."""
    assert adp_cache_dir("./data") == "./data/cache/adp"


def test_insights_search_cache_dir_is_nested_under_data_root() -> None:
    """Verify the insights search cache path is under cache/insights/search."""
    assert insights_search_cache_dir("./data") == "./data/cache/insights/search"


def test_insights_synthesis_cache_dir_is_nested_under_data_root() -> None:
    """Verify the insights synthesis cache path is under cache/insights/synthesis."""
    assert insights_synthesis_cache_dir("./data") == "./data/cache/insights/synthesis"


def test_insights_search_runs_dir_is_nested_under_search_cache() -> None:
    """Verify search runs live under cache/insights/search/runs."""
    assert insights_search_runs_dir("./data") == "./data/cache/insights/search/runs"


def test_insights_synthesis_runs_dir_is_nested_under_synthesis_cache() -> None:
    """Verify synthesis runs live under cache/insights/synthesis/runs."""
    assert (
        insights_synthesis_runs_dir("./data") == "./data/cache/insights/synthesis/runs"
    )


def test_insights_runs_dir_appends_runs_segment() -> None:
    """Verify insights_runs_dir appends runs under a cache root."""
    assert insights_runs_dir("./data/cache/insights/search") == (
        "./data/cache/insights/search/runs"
    )


def test_insights_run_current_path_points_at_current_json() -> None:
    """Verify the current pointer path is cache_root/current.json."""
    assert insights_run_current_path("./data/cache/insights/search") == (
        "./data/cache/insights/search/current.json"
    )


def test_new_insights_run_id_uses_utc_timestamp_format() -> None:
    """Verify run ids match the export timestamp format."""
    generated_at = datetime(2026, 8, 2, 3, 45, 0, tzinfo=timezone.utc)

    assert new_insights_run_id(generated_at) == "20260802T034500Z"


def test_player_insights_exports_dir_is_nested_under_data_root() -> None:
    """Verify insights exports live under data/insights/exports."""
    assert player_insights_exports_dir("./data") == "./data/insights/exports"


def test_player_insights_output_path_includes_year_and_timestamp() -> None:
    """Verify merged insights output path includes year and UTC timestamp."""
    generated_at = datetime(2026, 7, 4, 17, 7, 47, tzinfo=timezone.utc)
    path = player_insights_output_path("./data", 2026, generated_at)

    assert path == "./data/insights/exports/player_insights_2026_20260704T170747Z.json"


def test_resolve_latest_player_insights_path_picks_newest_timestamped_export(
    tmp_path: Path,
) -> None:
    """Verify resolve_latest selects the newest timestamped export by filename."""
    exports_dir = tmp_path / "insights" / "exports"
    exports_dir.mkdir(parents=True)
    older = exports_dir / "player_insights_2026_20260704T120000Z.json"
    newer = exports_dir / "player_insights_2026_20260704T170747Z.json"
    older.write_text("{}", encoding="utf-8")
    newer.write_text("{}", encoding="utf-8")

    resolved = resolve_latest_player_insights_path(str(tmp_path))

    assert resolved == str(newer)


def test_resolve_latest_player_insights_path_falls_back_to_legacy_export(
    tmp_path: Path,
) -> None:
    """Verify resolve_latest falls back to legacy undated exports when needed."""
    legacy = tmp_path / "player_insights_2026.json"
    payload = PlayerInsightsFile(
        draft_year=2026,
        generated_at=datetime.now(timezone.utc),
        model="gemini-2.0-flash",
        players={"1": _sample_insight()},
    )
    legacy.write_text(json.dumps(payload.model_dump(mode="json")), encoding="utf-8")

    resolved = resolve_latest_player_insights_path(str(tmp_path))

    assert resolved == str(legacy)


def test_position_guide_output_path_includes_league_id(tmp_path: Path) -> None:
    """Verify position-guide filenames include league id and team count."""
    generated_at = datetime(2026, 7, 5, 22, 45, tzinfo=timezone.utc)
    path = position_guide_output_path(
        str(tmp_path),
        "red_league_10",
        10,
        2,
        2026,
        generated_at,
        ext="json",
    )

    assert Path(path).name.startswith("position_guide_red_league_10_10teams_slot2_")


def test_model_adp_output_paths_include_league_id(tmp_path: Path) -> None:
    """Verify model-ADP filenames include league id so same-size leagues do not collide."""
    generated_at = datetime(2026, 7, 5, 22, 45, tzinfo=timezone.utc)
    adp_path = model_adp_output_path(
        str(tmp_path), "redraft_nbfl_12", 12, 2026, generated_at, ext="json"
    )
    by_pos_path = model_adp_by_position_output_path(
        str(tmp_path), "redraft_nbfl_12", 12, 2026, generated_at, ext="json"
    )

    assert "redraft_nbfl_12" in Path(adp_path).name
    assert "redraft_nbfl_12" in Path(by_pos_path).name


def test_resolve_latest_position_guide_path_disambiguates_leagues(tmp_path: Path) -> None:
    """Verify latest-guide lookup is keyed by league id, not only team count."""
    generated_at = datetime(2026, 7, 5, 22, 45, tzinfo=timezone.utc)
    red_path = position_guide_output_path(
        str(tmp_path), "red_league_10", 12, 5, 2026, generated_at, ext="json"
    )
    nbfl_path = position_guide_output_path(
        str(tmp_path),
        "redraft_nbfl_12",
        12,
        5,
        2026,
        datetime(2026, 7, 6, 22, 45, tzinfo=timezone.utc),
        ext="json",
    )
    Path(red_path).write_text("{}", encoding="utf-8")
    Path(nbfl_path).write_text("{}", encoding="utf-8")

    resolved = resolve_latest_position_guide_path(
        str(tmp_path), "red_league_10", 12, 5, 2026, ext="json"
    )

    assert resolved == red_path
