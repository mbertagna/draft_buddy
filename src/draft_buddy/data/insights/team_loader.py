"""Load merged team outlooks JSON for runtime consumption."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from draft_buddy.data.cache_paths import resolve_latest_team_insights_path
from draft_buddy.data.insights.team_schemas import TeamOutlook, TeamOutlooksFile


@dataclass(frozen=True)
class LoadedTeamOutlooks:
    """Result of loading a team outlooks export file.

    Parameters
    ----------
    teams : dict[str, TeamOutlook]
        Outlooks keyed by team abbreviation.
    meta : TeamOutlooksFile | None
        File-level metadata, or ``None`` when no file was found.
    source_path : str | None
        Filesystem path that was loaded, or ``None`` when no file was found.
    """

    teams: dict[str, TeamOutlook]
    meta: TeamOutlooksFile | None
    source_path: str | None


def load_team_outlooks(path: str) -> dict[str, TeamOutlook]:
    """Load team outlooks from a JSON file keyed by team abbreviation.

    Parameters
    ----------
    path : str
        Path to a team outlooks JSON export file.

    Returns
    -------
    dict[str, TeamOutlook]
        Mapping of team abbreviation to outlook. Returns an empty dict when
        the file does not exist.
    """
    file_path = Path(path)
    if not file_path.exists():
        return {}

    with file_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    outlooks_file = TeamOutlooksFile.model_validate(payload)
    return dict(outlooks_file.teams)


def load_team_outlooks_file(path: str) -> LoadedTeamOutlooks:
    """Load a team outlooks export file and its metadata.

    Parameters
    ----------
    path : str
        Path to a team outlooks JSON export file.

    Returns
    -------
    LoadedTeamOutlooks
        Parsed outlooks, metadata, and source path. Returns empty teams and
        ``meta=None`` when the file does not exist.
    """
    file_path = Path(path)
    if not file_path.exists():
        return LoadedTeamOutlooks(teams={}, meta=None, source_path=None)

    with file_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    outlooks_file = TeamOutlooksFile.model_validate(payload)
    return LoadedTeamOutlooks(
        teams=dict(outlooks_file.teams),
        meta=outlooks_file,
        source_path=str(file_path),
    )


def load_latest_team_outlooks(data_root: str) -> LoadedTeamOutlooks:
    """Load the newest team outlooks export under a data root.

    Parameters
    ----------
    data_root : str
        Root data directory (e.g. ``./data``).

    Returns
    -------
    LoadedTeamOutlooks
        Parsed outlooks from the newest export, or empty values when none
        exist.
    """
    latest_path = resolve_latest_team_insights_path(data_root)
    if latest_path is None:
        return LoadedTeamOutlooks(teams={}, meta=None, source_path=None)
    return load_team_outlooks_file(latest_path)
