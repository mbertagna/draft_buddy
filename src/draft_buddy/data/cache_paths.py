"""Shared conventions for where raw external data sources are cached.

Every data source that downloads (or is manually placed as) raw input data
gets its own subdirectory under a common data root, keeping large,
re-downloadable inputs separate from generated outputs and from each other.
"""

from __future__ import annotations

import glob
import os
import re
from datetime import datetime, timezone

PLAYER_INSIGHTS_EXPORT_PATTERN = re.compile(
    r"^player_insights_(?P<year>\d+)_(?P<timestamp>\d{8}T\d{6}Z)\.json$"
)

TEAM_INSIGHTS_EXPORT_PATTERN = re.compile(
    r"^team_insights_(?P<year>\d+)_(?P<timestamp>\d{8}T\d{6}Z)\.json$"
)

POSITION_GUIDE_EXPORT_PATTERN = re.compile(
    r"^position_guide_(?P<num_teams>\d+)teams_slot(?P<slot>\d+)_"
    r"(?P<year>\d+)_(?P<timestamp>\d{8}T\d{6}Z)\.(?P<ext>json|html)$"
)


def nflverse_cache_dir(data_root: str) -> str:
    """Return the nflverse raw-data cache directory under a data root.

    Parameters
    ----------
    data_root : str
        Root data directory (e.g. ``./data``).

    Returns
    -------
    str
        Path to the nflverse cache directory.
    """
    return os.path.join(data_root, "cache", "nflverse")


def sleeper_cache_dir(data_root: str) -> str:
    """Return the Sleeper raw-data cache directory under a data root.

    Parameters
    ----------
    data_root : str
        Root data directory (e.g. ``./data``).

    Returns
    -------
    str
        Path to the Sleeper cache directory.
    """
    return os.path.join(data_root, "cache", "sleeper")


def adp_cache_dir(data_root: str) -> str:
    """Return the directory for FantasyPros ADP HTML table snapshots.

    Expected filename pattern:
    ``fantasypros-{year}-overall-adp-rankings.html``.

    Parameters
    ----------
    data_root : str
        Root data directory (e.g. ``./data``).

    Returns
    -------
    str
        Path to the ADP cache directory.
    """
    return os.path.join(data_root, "cache", "adp")


def insights_search_cache_dir(data_root: str) -> str:
    """Return the Google CSE search cache directory for player insights.

    Parameters
    ----------
    data_root : str
        Root data directory (e.g. ``./data``).

    Returns
    -------
    str
        Path to the insights search cache directory.
    """
    return os.path.join(data_root, "cache", "insights", "search")


def insights_synthesis_cache_dir(data_root: str) -> str:
    """Return the Gemini synthesis cache directory for player insights.

    Parameters
    ----------
    data_root : str
        Root data directory (e.g. ``./data``).

    Returns
    -------
    str
        Path to the insights synthesis cache directory.
    """
    return os.path.join(data_root, "cache", "insights", "synthesis")


def player_insights_exports_dir(data_root: str) -> str:
    """Return the directory for timestamped merged player insights exports.

    Parameters
    ----------
    data_root : str
        Root data directory (e.g. ``./data``).

    Returns
    -------
    str
        Path to the insights exports directory.
    """
    return os.path.join(data_root, "insights", "exports")


def _format_insights_timestamp(generated_at: datetime) -> str:
    """Format a UTC datetime as a filesystem-safe insights export suffix.

    Parameters
    ----------
    generated_at : datetime
        UTC generation timestamp.

    Returns
    -------
    str
        Timestamp suffix such as ``20260704T170747Z``.
    """
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=timezone.utc)
    else:
        generated_at = generated_at.astimezone(timezone.utc)
    return generated_at.strftime("%Y%m%dT%H%M%SZ")


def player_insights_output_path(
    data_root: str,
    year: int,
    generated_at: datetime,
) -> str:
    """Return the path for a new timestamped merged player insights export.

    Parameters
    ----------
    data_root : str
        Root data directory (e.g. ``./data``).
    year : int
        Draft year (e.g. ``2026``).
    generated_at : datetime
        UTC generation timestamp embedded in the filename.

    Returns
    -------
    str
        Path to ``player_insights_{year}_{timestamp}.json`` under exports.
    """
    timestamp = _format_insights_timestamp(generated_at)
    filename = f"player_insights_{year}_{timestamp}.json"
    return os.path.join(player_insights_exports_dir(data_root), filename)


def _resolve_latest_timestamped_export(exports_dir: str, pattern: re.Pattern[str]) -> str | None:
    """Return the newest timestamped export in ``exports_dir`` matching ``pattern``.

    Parameters
    ----------
    exports_dir : str
        Directory containing timestamped export files.
    pattern : re.Pattern[str]
        Compiled filename pattern with a ``timestamp`` capture group.

    Returns
    -------
    str | None
        Full path to the newest export, or ``None`` when none exist.
    """
    if not os.path.isdir(exports_dir):
        return None

    newest_path: str | None = None
    newest_timestamp: str | None = None
    for filename in os.listdir(exports_dir):
        match = pattern.match(filename)
        if match is None:
            continue
        timestamp = match.group("timestamp")
        if newest_timestamp is None or timestamp > newest_timestamp:
            newest_timestamp = timestamp
            newest_path = os.path.join(exports_dir, filename)

    return newest_path


def _resolve_latest_legacy_export(data_root: str) -> str | None:
    """Return the newest legacy undated insights export under ``data_root``.

    Parameters
    ----------
    data_root : str
        Root data directory (e.g. ``./data``).

    Returns
    -------
    str | None
        Full path to the newest legacy export, or ``None`` when none exist.
    """
    pattern = os.path.join(data_root, "player_insights_*.json")
    candidates = [
        path
        for path in glob.glob(pattern)
        if PLAYER_INSIGHTS_EXPORT_PATTERN.match(os.path.basename(path)) is None
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def position_guide_exports_dir(data_root: str) -> str:
    """Return the directory for timestamped position guide exports.

    Parameters
    ----------
    data_root : str
        Root data directory (e.g. ``./data``).

    Returns
    -------
    str
        Path to the position guide exports directory.
    """
    return os.path.join(data_root, "guides", "exports")


def position_guide_output_path(
    data_root: str,
    num_teams: int,
    slot: int,
    year: int,
    generated_at: datetime,
    ext: str = "json",
) -> str:
    """Return the path for a new timestamped position guide export.

    Parameters
    ----------
    data_root : str
        Root data directory (e.g. ``./data``).
    num_teams : int
        League size (e.g. ``12``).
    slot : int
        User draft slot (1-based).
    year : int
        Draft year (e.g. ``2026``).
    generated_at : datetime
        UTC generation timestamp embedded in the filename.
    ext : str, optional
        File extension without dot (``json`` or ``html``).

    Returns
    -------
    str
        Path to ``position_guide_{num_teams}teams_slot{slot}_{year}_{timestamp}.{ext}``.
    """
    timestamp = _format_insights_timestamp(generated_at)
    filename = f"position_guide_{num_teams}teams_slot{slot}_{year}_{timestamp}.{ext}"
    exports_dir = position_guide_exports_dir(data_root)
    os.makedirs(exports_dir, exist_ok=True)
    return os.path.join(exports_dir, filename)


def resolve_latest_position_guide_path(
    data_root: str,
    num_teams: int,
    slot: int,
    year: int,
    ext: str = "json",
) -> str | None:
    """Return the newest position guide export for league size, slot, and year.

    Parameters
    ----------
    data_root : str
        Root data directory (e.g. ``./data``).
    num_teams : int
        League size filter.
    slot : int
        Draft slot filter.
    year : int
        Draft year filter.
    ext : str, optional
        File extension without dot.

    Returns
    -------
    str | None
        Full path to the newest matching export, or ``None`` when none exist.
    """
    exports_dir = position_guide_exports_dir(data_root)
    if not os.path.isdir(exports_dir):
        return None

    newest_path: str | None = None
    newest_timestamp: str | None = None
    for filename in os.listdir(exports_dir):
        match = POSITION_GUIDE_EXPORT_PATTERN.match(filename)
        if match is None:
            continue
        if (
            int(match.group("num_teams")) != num_teams
            or int(match.group("slot")) != slot
            or int(match.group("year")) != year
            or match.group("ext") != ext
        ):
            continue
        timestamp = match.group("timestamp")
        if newest_timestamp is None or timestamp > newest_timestamp:
            newest_timestamp = timestamp
            newest_path = os.path.join(exports_dir, filename)

    return newest_path


def resolve_latest_player_insights_path(data_root: str) -> str | None:
    """Return the path to the newest player insights export file.

    Prefers timestamped exports under ``data/insights/exports``. Falls back
    to legacy undated ``data/player_insights_{year}.json`` files when the
    exports directory is empty.

    Parameters
    ----------
    data_root : str
        Root data directory (e.g. ``./data``).

    Returns
    -------
    str | None
        Full path to the newest insights file, or ``None`` when none exist.
    """
    latest_export = _resolve_latest_timestamped_export(
        player_insights_exports_dir(data_root), PLAYER_INSIGHTS_EXPORT_PATTERN
    )
    if latest_export is not None:
        return latest_export
    return _resolve_latest_legacy_export(data_root)


def insights_league_search_cache_dir(data_root: str) -> str:
    """Return the search cache directory for league-wide (all-32-teams) queries.

    League-wide roundup articles (best/worst-case previews, division
    rankings, strength-of-schedule breakdowns) are fetched once per run
    here instead of once per team, then matched against each team by name
    at synthesis time.

    Parameters
    ----------
    data_root : str
        Root data directory (e.g. ``./data``).

    Returns
    -------
    str
        Path to the league-wide search cache directory.
    """
    return os.path.join(data_root, "cache", "insights", "league_search")


def insights_team_search_cache_dir(data_root: str) -> str:
    """Return the search cache directory for team insight enrichment.

    Parameters
    ----------
    data_root : str
        Root data directory (e.g. ``./data``).

    Returns
    -------
    str
        Path to the team search cache directory, kept separate from the
        player search cache so team data never interferes with player
        loading.
    """
    return os.path.join(data_root, "cache", "insights", "team_search")


def insights_team_synthesis_cache_dir(data_root: str) -> str:
    """Return the synthesis cache directory for team insight enrichment.

    Parameters
    ----------
    data_root : str
        Root data directory (e.g. ``./data``).

    Returns
    -------
    str
        Path to the team synthesis cache directory, kept separate from the
        player synthesis cache.
    """
    return os.path.join(data_root, "cache", "insights", "team_synthesis")


def team_insights_exports_dir(data_root: str) -> str:
    """Return the directory for timestamped merged team insights exports.

    Parameters
    ----------
    data_root : str
        Root data directory (e.g. ``./data``).

    Returns
    -------
    str
        Path to the team insights exports directory, kept separate from
        ``data/insights/exports`` so team files never interfere with
        ``resolve_latest_player_insights_path``.
    """
    return os.path.join(data_root, "insights", "team_exports")


def team_insights_output_path(
    data_root: str,
    year: int,
    generated_at: datetime,
) -> str:
    """Return the path for a new timestamped merged team insights export.

    Parameters
    ----------
    data_root : str
        Root data directory (e.g. ``./data``).
    year : int
        Draft year (e.g. ``2026``).
    generated_at : datetime
        UTC generation timestamp embedded in the filename.

    Returns
    -------
    str
        Path to ``team_insights_{year}_{timestamp}.json`` under team exports.
    """
    timestamp = _format_insights_timestamp(generated_at)
    filename = f"team_insights_{year}_{timestamp}.json"
    return os.path.join(team_insights_exports_dir(data_root), filename)


def resolve_latest_team_insights_path(data_root: str) -> str | None:
    """Return the path to the newest team insights export file.

    Parameters
    ----------
    data_root : str
        Root data directory (e.g. ``./data``).

    Returns
    -------
    str | None
        Full path to the newest team insights file, or ``None`` when none
        exist.
    """
    return _resolve_latest_timestamped_export(
        team_insights_exports_dir(data_root), TEAM_INSIGHTS_EXPORT_PATTERN
    )
