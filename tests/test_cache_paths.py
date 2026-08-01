"""Tests for raw data cache directory conventions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from draft_buddy.data.cache_paths import (
    adp_cache_dir,
    insights_league_search_cache_dir,
    insights_search_cache_dir,
    insights_synthesis_cache_dir,
    insights_team_search_cache_dir,
    insights_team_synthesis_cache_dir,
    nflverse_cache_dir,
    player_insights_exports_dir,
    player_insights_output_path,
    resolve_latest_player_insights_path,
    resolve_latest_team_insights_path,
    sleeper_cache_dir,
    team_insights_exports_dir,
    team_insights_output_path,
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


def test_insights_league_search_cache_dir_is_separate_from_team_search() -> None:
    """Verify the league-wide search cache path is under cache/insights/league_search."""
    assert insights_league_search_cache_dir("./data") == "./data/cache/insights/league_search"


def test_insights_team_search_cache_dir_is_separate_from_player_search() -> None:
    """Verify the team search cache path is under cache/insights/team_search."""
    assert insights_team_search_cache_dir("./data") == "./data/cache/insights/team_search"


def test_insights_team_synthesis_cache_dir_is_separate_from_player_synthesis() -> None:
    """Verify the team synthesis cache path is under cache/insights/team_synthesis."""
    assert insights_team_synthesis_cache_dir("./data") == "./data/cache/insights/team_synthesis"


def test_team_insights_exports_dir_is_separate_from_player_exports() -> None:
    """Verify team insights exports live under data/insights/team_exports."""
    assert team_insights_exports_dir("./data") == "./data/insights/team_exports"


def test_team_insights_output_path_includes_year_and_timestamp() -> None:
    """Verify merged team insights output path includes year and UTC timestamp."""
    generated_at = datetime(2026, 7, 4, 17, 7, 47, tzinfo=timezone.utc)
    path = team_insights_output_path("./data", 2026, generated_at)

    assert path == "./data/insights/team_exports/team_insights_2026_20260704T170747Z.json"


def test_resolve_latest_team_insights_path_picks_newest_timestamped_export(
    tmp_path: Path,
) -> None:
    """Verify resolve_latest_team_insights_path selects the newest export by filename."""
    exports_dir = tmp_path / "insights" / "team_exports"
    exports_dir.mkdir(parents=True)
    older = exports_dir / "team_insights_2026_20260704T120000Z.json"
    newer = exports_dir / "team_insights_2026_20260704T170747Z.json"
    older.write_text("{}", encoding="utf-8")
    newer.write_text("{}", encoding="utf-8")

    resolved = resolve_latest_team_insights_path(str(tmp_path))

    assert resolved == str(newer)


def test_resolve_latest_team_insights_path_returns_none_when_missing(tmp_path: Path) -> None:
    """Verify resolve_latest_team_insights_path returns None when no exports exist."""
    resolved = resolve_latest_team_insights_path(str(tmp_path))

    assert resolved is None
