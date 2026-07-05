"""Load merged player insights JSON for runtime consumption."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from draft_buddy.data.cache_paths import resolve_latest_player_insights_path
from draft_buddy.data.insights.schemas import PlayerInsight, PlayerInsightsFile


@dataclass(frozen=True)
class LoadedPlayerInsights:
    """Result of loading a player insights export file.

    Parameters
    ----------
    players : dict[int, PlayerInsight]
        Insights keyed by sleeper id / player id.
    meta : PlayerInsightsFile | None
        File-level metadata, or ``None`` when no file was found.
    source_path : str | None
        Filesystem path that was loaded, or ``None`` when no file was found.
    """

    players: dict[int, PlayerInsight]
    meta: PlayerInsightsFile | None
    source_path: str | None


def load_player_insights(path: str) -> dict[int, PlayerInsight]:
    """Load player insights from a JSON file keyed by sleeper id.

    Parameters
    ----------
    path : str
        Path to a player insights JSON export file.

    Returns
    -------
    dict[int, PlayerInsight]
        Mapping of sleeper id to insight. Returns an empty dict when the file
        does not exist.
    """
    file_path = Path(path)
    if not file_path.exists():
        return {}

    with file_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    insights_file = PlayerInsightsFile.model_validate(payload)
    return {int(sleeper_id): insight for sleeper_id, insight in insights_file.players.items()}


def load_player_insights_file(path: str) -> LoadedPlayerInsights:
    """Load a player insights export file and its metadata.

    Parameters
    ----------
    path : str
        Path to a player insights JSON export file.

    Returns
    -------
    LoadedPlayerInsights
        Parsed insights, metadata, and source path. Returns empty players and
        ``meta=None`` when the file does not exist.
    """
    file_path = Path(path)
    if not file_path.exists():
        return LoadedPlayerInsights(players={}, meta=None, source_path=None)

    with file_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    insights_file = PlayerInsightsFile.model_validate(payload)
    players = {
        int(sleeper_id): insight for sleeper_id, insight in insights_file.players.items()
    }
    return LoadedPlayerInsights(
        players=players,
        meta=insights_file,
        source_path=str(file_path),
    )


def load_latest_player_insights(data_root: str) -> LoadedPlayerInsights:
    """Load the newest player insights export under a data root.

    Parameters
    ----------
    data_root : str
        Root data directory (e.g. ``./data``).

    Returns
    -------
    LoadedPlayerInsights
        Parsed insights from the newest export, or empty values when none
        exist.
    """
    latest_path = resolve_latest_player_insights_path(data_root)
    if latest_path is None:
        return LoadedPlayerInsights(players={}, meta=None, source_path=None)
    return load_player_insights_file(latest_path)
