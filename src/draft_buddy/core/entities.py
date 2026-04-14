"""Core draft entities and value objects."""

from dataclasses import dataclass, field
from typing import List

from draft_buddy.domain.entities import Player


@dataclass
class Pick:
    """Represents a single draft pick event.

    Parameters
    ----------
    pick_number : int
        One-indexed global draft pick number.
    team_id : int
        Team making the pick.
    player_id : int
        Identifier of the drafted player.
    is_manual_pick : bool
        Whether the pick came from a user/manual interaction.
    """

    pick_number: int
    team_id: int
    player_id: int
    is_manual_pick: bool = False


@dataclass
class DraftHistory:
    """Draft pick history container."""

    picks: List[Pick] = field(default_factory=list)


@dataclass
class Roster:
    """Team roster grouped with simple counts.

    Parameters
    ----------
    players : List[Player]
        Players currently on the roster.
    qb_count : int
        Number of quarterback slots occupied.
    rb_count : int
        Number of running back slots occupied.
    wr_count : int
        Number of wide receiver slots occupied.
    te_count : int
        Number of tight end slots occupied.
    flex_count : int
        Number of flex slots occupied.
    """

    players: List[Player] = field(default_factory=list)
    qb_count: int = 0
    rb_count: int = 0
    wr_count: int = 0
    te_count: int = 0
    flex_count: int = 0
