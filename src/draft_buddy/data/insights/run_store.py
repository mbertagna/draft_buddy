"""Resolve timestamped insights cache run directories."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from draft_buddy.data.cache_paths import (
    insights_runs_dir,
    new_insights_run_id,
    read_insights_run_current,
    write_insights_run_current,
)

InsightsCacheKind = Literal["search", "synthesis"]


@dataclass(frozen=True, slots=True)
class InsightsRunRoot:
    """Resolved filesystem root for one insights cache run.

    Parameters
    ----------
    path : str
        Directory that stores per-player cache files for this run.
    run_id : str or None
        Timestamped run id when under ``runs/``, or ``None`` for legacy roots.
    """

    path: str
    run_id: str | None


def has_legacy_search_content(cache_root: str) -> bool:
    """Return whether a flat (pre-run) search cache exists at ``cache_root``.

    Parameters
    ----------
    cache_root : str
        Insights search cache root.

    Returns
    -------
    bool
        True when at least one player directory with a manifest is present.
    """
    if not os.path.isdir(cache_root):
        return False
    for name in os.listdir(cache_root):
        if name in ("runs", "current.json"):
            continue
        player_dir = os.path.join(cache_root, name)
        if os.path.isdir(player_dir) and os.path.isfile(
            os.path.join(player_dir, "manifest.json")
        ):
            return True
    return False


def has_legacy_synthesis_content(cache_root: str) -> bool:
    """Return whether a flat (pre-run) synthesis cache exists at ``cache_root``.

    Parameters
    ----------
    cache_root : str
        Insights synthesis cache root.

    Returns
    -------
    bool
        True when at least one per-player ``.json`` file is present at the root.
    """
    if not os.path.isdir(cache_root):
        return False
    for name in os.listdir(cache_root):
        if name in ("runs", "current.json") or not name.endswith(".json"):
            continue
        path = os.path.join(cache_root, name)
        if os.path.isfile(path):
            return True
    return False


def _has_legacy_content(cache_root: str, kind: InsightsCacheKind) -> bool:
    """Dispatch legacy detection for search or synthesis caches."""
    if kind == "search":
        return has_legacy_search_content(cache_root)
    return has_legacy_synthesis_content(cache_root)


def _create_run_root(cache_root: str) -> InsightsRunRoot:
    """Create an empty timestamped run and point ``current.json`` at it."""
    runs_dir = insights_runs_dir(cache_root)
    os.makedirs(runs_dir, exist_ok=True)

    base_id = new_insights_run_id()
    run_id = base_id
    run_path = os.path.join(runs_dir, run_id)
    suffix = 1
    while os.path.exists(run_path):
        run_id = f"{base_id}_{suffix}"
        run_path = os.path.join(runs_dir, run_id)
        suffix += 1

    os.makedirs(run_path)
    write_insights_run_current(cache_root, run_id)
    return InsightsRunRoot(path=run_path, run_id=run_id)


def resolve_or_create_run_root(
    cache_root: str,
    *,
    force: bool,
    kind: InsightsCacheKind,
    create_if_missing: bool = True,
) -> InsightsRunRoot:
    """Resolve the active insights cache run root, creating one when needed.

    Resolution order:

    1. When ``force`` is true, always create a new empty timestamped run.
    2. When ``current.json`` exists, use ``runs/{run_id}/``.
    3. When legacy flat content exists at ``cache_root``, use that root.
    4. Otherwise create the first timestamped run when ``create_if_missing``
       is true; else return ``cache_root`` with no run id.

    Parameters
    ----------
    cache_root : str
        Search or synthesis cache parent directory.
    force : bool
        When true, allocate a new empty run and update the current pointer.
    kind : {"search", "synthesis"}
        Cache kind used for legacy layout detection.
    create_if_missing : bool, optional
        When false, do not create a first run for empty caches (useful for
        read-only resolution such as loading search results during synthesis).

    Returns
    -------
    InsightsRunRoot
        Active run path and optional run id.
    """
    os.makedirs(cache_root, exist_ok=True)

    if force:
        return _create_run_root(cache_root)

    run_id = read_insights_run_current(cache_root)
    if run_id is not None:
        run_path = os.path.join(insights_runs_dir(cache_root), run_id)
        os.makedirs(run_path, exist_ok=True)
        return InsightsRunRoot(path=run_path, run_id=run_id)

    if _has_legacy_content(cache_root, kind):
        return InsightsRunRoot(path=cache_root, run_id=None)

    if create_if_missing:
        return _create_run_root(cache_root)

    return InsightsRunRoot(path=cache_root, run_id=None)
