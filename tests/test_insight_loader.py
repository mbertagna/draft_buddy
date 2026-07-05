"""Tests for player insights JSON loader."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from draft_buddy.data.insights.loader import (
    LoadedPlayerInsights,
    load_latest_player_insights,
    load_player_insights,
    load_player_insights_file,
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


def test_load_player_insights_returns_empty_dict_when_missing(tmp_path: Path) -> None:
    """Verify missing insights file returns an empty mapping."""
    result = load_player_insights(str(tmp_path / "missing.json"))
    assert result == {}


def test_load_player_insights_parses_file(tmp_path: Path) -> None:
    """Verify insights file loads keyed by integer sleeper id."""
    payload = PlayerInsightsFile(
        draft_year=2026,
        generated_at=datetime.now(timezone.utc),
        model="gemini-2.0-flash",
        players={"4034": _sample_insight()},
    )
    path = tmp_path / "player_insights_2026.json"
    path.write_text(json.dumps(payload.model_dump(mode="json")), encoding="utf-8")

    loaded = load_player_insights(str(path))

    assert 4034 in loaded
    assert loaded[4034].outlook_phrase == "Starter role expected"


def test_load_player_insights_file_returns_metadata(tmp_path: Path) -> None:
    """Verify load_player_insights_file returns players and file metadata."""
    payload = PlayerInsightsFile(
        draft_year=2026,
        generated_at=datetime(2026, 7, 4, 17, 7, 47, tzinfo=timezone.utc),
        model="gemini-2.0-flash",
        players={"4034": _sample_insight()},
    )
    path = tmp_path / "player_insights_2026.json"
    path.write_text(json.dumps(payload.model_dump(mode="json")), encoding="utf-8")

    loaded = load_player_insights_file(str(path))

    assert loaded.source_path == str(path)
    assert loaded.meta is not None
    assert loaded.meta.draft_year == 2026
    assert 4034 in loaded.players


def test_load_latest_player_insights_returns_empty_when_missing(tmp_path: Path) -> None:
    """Verify load_latest_player_insights returns empty values when no exports exist."""
    loaded = load_latest_player_insights(str(tmp_path))

    assert loaded.players == {}
    assert loaded.meta is None
    assert loaded.source_path is None


def test_load_latest_player_insights_loads_newest_export(tmp_path: Path) -> None:
    """Verify load_latest_player_insights loads the newest timestamped export."""
    exports_dir = tmp_path / "insights" / "exports"
    exports_dir.mkdir(parents=True)
    payload = PlayerInsightsFile(
        draft_year=2026,
        generated_at=datetime(2026, 7, 4, 17, 7, 47, tzinfo=timezone.utc),
        model="gemini-2.0-flash",
        players={"4034": _sample_insight()},
    )
    export_path = exports_dir / "player_insights_2026_20260704T170747Z.json"
    export_path.write_text(json.dumps(payload.model_dump(mode="json")), encoding="utf-8")

    loaded = load_latest_player_insights(str(tmp_path))

    assert loaded.source_path == str(export_path)
    assert loaded.meta is not None
    assert 4034 in loaded.players
