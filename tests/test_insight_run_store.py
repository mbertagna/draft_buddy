"""Tests for timestamped insights cache run resolution."""

from __future__ import annotations

import json
from pathlib import Path

from draft_buddy.data.cache_paths import (
    read_insights_run_current,
    write_insights_run_current,
)
from draft_buddy.data.insights.run_store import (
    has_legacy_search_content,
    has_legacy_synthesis_content,
    resolve_or_create_run_root,
)


def test_resolve_creates_first_run_when_cache_empty(tmp_path: Path) -> None:
    """Empty cache creates a timestamped run and current pointer."""
    cache_root = tmp_path / "search"
    resolved = resolve_or_create_run_root(str(cache_root), force=False, kind="search")

    assert resolved.run_id is not None
    assert resolved.path == str(cache_root / "runs" / resolved.run_id)
    assert Path(resolved.path).is_dir()
    assert read_insights_run_current(str(cache_root)) == resolved.run_id


def test_resolve_reuses_current_run_without_force(tmp_path: Path) -> None:
    """Existing current.json is reused when force is false."""
    cache_root = tmp_path / "search"
    first = resolve_or_create_run_root(str(cache_root), force=False, kind="search")
    second = resolve_or_create_run_root(str(cache_root), force=False, kind="search")

    assert second.path == first.path
    assert second.run_id == first.run_id


def test_resolve_force_creates_new_empty_run(tmp_path: Path) -> None:
    """Force allocates a new run and updates current.json."""
    cache_root = tmp_path / "search"
    first = resolve_or_create_run_root(str(cache_root), force=False, kind="search")
    (Path(first.path) / "marker.txt").write_text("keep", encoding="utf-8")

    forced = resolve_or_create_run_root(str(cache_root), force=True, kind="search")

    assert forced.run_id != first.run_id
    assert forced.path != first.path
    assert Path(forced.path).is_dir()
    assert not (Path(forced.path) / "marker.txt").exists()
    assert (Path(first.path) / "marker.txt").exists()
    assert read_insights_run_current(str(cache_root)) == forced.run_id


def test_resolve_uses_legacy_search_root_when_no_current(tmp_path: Path) -> None:
    """Flat search player dirs are used when current.json is absent."""
    cache_root = tmp_path / "search"
    player_dir = cache_root / "4034"
    player_dir.mkdir(parents=True)
    (player_dir / "manifest.json").write_text("{}", encoding="utf-8")

    resolved = resolve_or_create_run_root(str(cache_root), force=False, kind="search")

    assert resolved.path == str(cache_root)
    assert resolved.run_id is None


def test_resolve_uses_legacy_synthesis_root_when_no_current(tmp_path: Path) -> None:
    """Flat synthesis json files are used when current.json is absent."""
    cache_root = tmp_path / "synthesis"
    cache_root.mkdir(parents=True)
    (cache_root / "4034.json").write_text("{}", encoding="utf-8")

    resolved = resolve_or_create_run_root(
        str(cache_root), force=False, kind="synthesis"
    )

    assert resolved.path == str(cache_root)
    assert resolved.run_id is None


def test_resolve_without_create_returns_cache_root_when_empty(tmp_path: Path) -> None:
    """Read-only resolve leaves an empty cache without creating a run."""
    cache_root = tmp_path / "search"
    resolved = resolve_or_create_run_root(
        str(cache_root),
        force=False,
        kind="search",
        create_if_missing=False,
    )

    assert resolved.path == str(cache_root)
    assert resolved.run_id is None
    assert read_insights_run_current(str(cache_root)) is None


def test_has_legacy_search_content_detects_player_manifest(tmp_path: Path) -> None:
    """Legacy search detection requires a player dir with manifest.json."""
    cache_root = tmp_path / "search"
    player_dir = cache_root / "4034"
    player_dir.mkdir(parents=True)
    (player_dir / "manifest.json").write_text("{}", encoding="utf-8")

    assert has_legacy_search_content(str(cache_root)) is True


def test_has_legacy_synthesis_content_detects_player_json(tmp_path: Path) -> None:
    """Legacy synthesis detection requires a root-level player json file."""
    cache_root = tmp_path / "synthesis"
    cache_root.mkdir(parents=True)
    (cache_root / "4034.json").write_text("{}", encoding="utf-8")

    assert has_legacy_synthesis_content(str(cache_root)) is True


def test_write_and_read_insights_run_current_round_trip(tmp_path: Path) -> None:
    """current.json round-trips the run_id."""
    cache_root = tmp_path / "search"
    write_insights_run_current(str(cache_root), "20260802T034500Z")

    assert read_insights_run_current(str(cache_root)) == "20260802T034500Z"
    payload = json.loads((cache_root / "current.json").read_text(encoding="utf-8"))
    assert payload == {"run_id": "20260802T034500Z"}
