"""Tests for durable draft state persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from draft_buddy.core.draft_state_store import (
    archive_draft_state,
    load_draft_state_payload,
    prune_archives,
    save_draft_state,
    write_json_atomic,
)


def test_write_json_atomic_replaces_target(tmp_path: Path) -> None:
    """Verify atomic writes create a valid JSON target file."""
    target = tmp_path / "draft_state.json"

    write_json_atomic(str(target), {"ok": True})

    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}


def test_save_draft_state_copies_readable_primary_to_prev(tmp_path: Path) -> None:
    """Verify saving preserves the previous readable primary copy."""
    primary = tmp_path / "draft_state.json"
    prev = tmp_path / "draft_state.prev.json"
    primary.write_text(json.dumps({"version": 1}), encoding="utf-8")

    save_draft_state(str(primary), {"version": 2}, prev_path=str(prev))

    assert json.loads(prev.read_text(encoding="utf-8")) == {"version": 1}
    assert json.loads(primary.read_text(encoding="utf-8")) == {"version": 2}


def test_load_draft_state_payload_recovers_from_prev(tmp_path: Path) -> None:
    """Verify corrupt primary falls back to the rolling previous file."""
    primary = tmp_path / "draft_state.json"
    prev = tmp_path / "draft_state.prev.json"
    primary.write_text("{not-json", encoding="utf-8")
    prev.write_text(json.dumps({"recovered": True}), encoding="utf-8")

    result = load_draft_state_payload(str(primary), prev_path=str(prev))

    assert result is not None
    assert result.recovered_from_fallback is True
    assert result.payload == {"recovered": True}
    assert result.warning is not None and "recovered from" in result.warning


def test_archive_draft_state_prunes_to_retention(tmp_path: Path) -> None:
    """Verify archive retention deletes older timestamped copies."""
    primary = tmp_path / "draft_state.json"
    archives = tmp_path / "saved_states"
    primary.write_text(json.dumps({"keep": True}), encoding="utf-8")
    archives.mkdir()
    for index in range(3):
        (archives / f"draft_state_2020-01-0{index + 1}_00-00-00.json").write_text(
            json.dumps({"index": index}), encoding="utf-8"
        )

    archive_path = archive_draft_state(str(primary), str(archives), retention=2)

    assert archive_path is not None
    remaining = sorted(path.name for path in archives.glob("draft_state_*.json"))
    assert len(remaining) == 2


def test_prune_archives_keeps_newest_only(tmp_path: Path) -> None:
    """Verify prune retains only the newest archives."""
    archives = tmp_path / "saved_states"
    archives.mkdir()
    older = archives / "draft_state_2020-01-01_00-00-00.json"
    newer = archives / "draft_state_2020-01-02_00-00-00.json"
    older.write_text("{}", encoding="utf-8")
    newer.write_text("{}", encoding="utf-8")

    prune_archives(str(archives), retention=1)

    assert newer.exists() and not older.exists()


def test_load_draft_state_payload_raises_when_all_candidates_fail(tmp_path: Path) -> None:
    """Verify load fails clearly when every candidate is unreadable."""
    primary = tmp_path / "draft_state.json"
    prev = tmp_path / "draft_state.prev.json"
    primary.write_text("{bad", encoding="utf-8")
    prev.write_text("{also-bad", encoding="utf-8")

    with pytest.raises(ValueError, match="Could not load draft state"):
        load_draft_state_payload(str(primary), prev_path=str(prev))
