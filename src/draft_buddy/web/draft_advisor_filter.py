"""Shared filter logic for the draft assistant."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from draft_buddy.core.entities import Player


def passes_gp_filter(games_played_frac: Any, gp_min: Optional[float]) -> bool:
    """Return whether a player passes the minimum games-played fraction filter.

    Parameters
    ----------
    games_played_frac : Any
        Player games-played fraction or ``"R"`` for rookies.
    gp_min : Optional[float]
        Minimum fraction threshold. When ``None``, all players pass.

    Returns
    -------
    bool
        True when the player should remain in the candidate pool.
    """
    if games_played_frac == "R":
        return True
    if gp_min is None:
        return True
    if games_played_frac is None:
        return False
    try:
        numeric = float(games_played_frac)
    except (TypeError, ValueError):
        return False
    if numeric != numeric:  # NaN check without importing math
        return False
    return numeric >= gp_min


def exclude_ignored_players(
    players: Sequence[Player],
    ignore_player_ids: Sequence[int] | None,
) -> list[Player]:
    """Remove user-blinded players from an available-player sequence.

    Parameters
    ----------
    players : Sequence[Player]
        Available players before blinding.
    ignore_player_ids : Sequence[int], optional
        Player ids to exclude from AI suggestions.

    Returns
    -------
    list[Player]
        Players that are not in the ignore set.
    """
    if not ignore_player_ids:
        return list(players)
    ignore_set = set(ignore_player_ids)
    return [player for player in players if player.player_id not in ignore_set]
