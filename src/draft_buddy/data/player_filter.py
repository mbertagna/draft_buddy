"""Filters for excluding unavailable players from a draftable catalog."""

from __future__ import annotations

from typing import Iterable

from draft_buddy.core import Player, PlayerCatalog


def exclude_inactive_players(
    catalog: PlayerCatalog,
    roster_statuses: Iterable[str],
    injury_statuses: Iterable[str],
) -> PlayerCatalog:
    """Return a catalog with roster- or injury-inactive players removed.

    Parameters
    ----------
    catalog : PlayerCatalog
        Source player catalog.
    roster_statuses : Iterable[str]
        ``Player.sleeper_status`` values treated as unavailable (e.g.
        ``"Inactive"``).
    injury_statuses : Iterable[str]
        ``Player.sleeper_injury_status`` values treated as unavailable (e.g.
        ``"IR"``).

    Returns
    -------
    PlayerCatalog
        Catalog containing only players that do not match either exclusion
        set. Players missing status data are always kept.
    """
    excluded_roster_statuses = set(roster_statuses)
    excluded_injury_statuses = set(injury_statuses)
    return PlayerCatalog(
        player
        for player in catalog
        if not _is_excluded(player, excluded_roster_statuses, excluded_injury_statuses)
    )


def _is_excluded(
    player: Player, excluded_roster_statuses: set[str], excluded_injury_statuses: set[str]
) -> bool:
    """Return whether a single player matches an exclusion status."""
    if player.sleeper_status in excluded_roster_statuses:
        return True
    return player.sleeper_injury_status in excluded_injury_statuses
