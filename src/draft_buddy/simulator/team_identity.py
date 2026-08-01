"""Canonical team identity helpers for season simulation.

Compute paths key schedules and standings by ``Team {id}`` labels derived from
integer team ids. Display names from ``TEAM_MANAGER_MAPPING`` are cosmetic and
must not be used as simulation keys.
"""

from __future__ import annotations

import re
from typing import Dict, Optional

_TEAM_LABEL_PATTERN = re.compile(r"^Team (\d+)$")


def team_label(team_id: int) -> str:
    """Return the canonical compute label for a team id.

    Parameters
    ----------
    team_id : int
        One-based team number.

    Returns
    -------
    str
        Label of the form ``Team {id}``.
    """
    return f"Team {team_id}"


def parse_team_id(label: str | None) -> Optional[int]:
    """Parse a team id from a ``Team {n}`` label.

    Parameters
    ----------
    label : str or None
        Candidate label text.

    Returns
    -------
    int or None
        Parsed team id when the label matches, otherwise ``None``.
    """
    if not label:
        return None
    match = _TEAM_LABEL_PATTERN.match(str(label).strip())
    if not match:
        return None
    return int(match.group(1))


def display_name_to_team_label_map(team_display_names: Dict[int, str]) -> Dict[str, str]:
    """Build display-name to ``Team {id}`` lookup for schedule CSV ingestion.

    Parameters
    ----------
    team_display_names : Dict[int, str]
        Cosmetic team id to display-name mapping.

    Returns
    -------
    Dict[str, str]
        Reverse map used only when translating legacy name-keyed schedules.
    """
    return {
        display_name: team_label(team_id)
        for team_id, display_name in team_display_names.items()
        if display_name
    }


def team_labels_for_league(num_teams: int) -> list[str]:
    """Return ordered compute labels for every team in a league.

    Parameters
    ----------
    num_teams : int
        Number of teams in the league.

    Returns
    -------
    list[str]
        ``Team 1`` … ``Team {num_teams}``.
    """
    return [team_label(team_id) for team_id in range(1, num_teams + 1)]
