"""Core domain entities for fantasy draft modeling."""

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np


@dataclass
class Player:
    """
    Fantasy-relevant player record for draft and simulation.

    Parameters
    ----------
    player_id : int
        Stable player identifier.
    name : str
        Display name.
    position : str
        Position code (QB, RB, WR, TE).
    projected_points : float
        Season projection used for value.
    games_played_frac : float or str
        Fraction of games played, or 'R' for rookie marker when applicable.
    adp : float
        Average draft position; infinity when unknown.
    bye_week : int, optional
        NFL bye week.
    team : str, optional
        Team abbreviation for stacking and display.
    """

    player_id: int
    name: str
    position: str  # QB, RB, WR, TE
    projected_points: float
    games_played_frac: Union[float, str] = 1.0
    adp: float = field(default=np.inf)  # Default to infinity if no ADP
    bye_week: Optional[int] = None
    team: Optional[str] = None

    def to_dict(self):
        """
        Serialize the player to a JSON-friendly dict.

        Returns
        -------
        dict
            Field names mapped to primitive values.
        """
        return {
            'player_id': self.player_id,
            'name': self.name,
            'position': self.position,
            'projected_points': self.projected_points,
            'games_played_frac': self.games_played_frac,
            'adp': self.adp if np.isfinite(self.adp) else None,
            'bye_week': self.bye_week,
            'team': self.team
        }
