"""Player context record for insight enrichment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union


@dataclass(frozen=True, slots=True)
class InsightPlayerContext:
    """Fantasy player row used as context for search and synthesis.

    Parameters
    ----------
    sleeper_id : str
        Sleeper player identifier (same as internal ``player_id``).
    name : str
        Display name.
    position : str
        Position code (QB, RB, WR, TE).
    team : str
        NFL team abbreviation.
    adp : float
        Average draft position.
    projected_points : float
        Season projection.
    games_played_frac : float or str
        Fraction of games played, or ``"R"`` for rookies.
    sleeper_injury_status : str, optional
        Sleeper injury designation when present.
    sleeper_depth_chart_position : str, optional
        Sleeper depth chart position when present.
    draft_year : int
        Draft season year used in search queries.
    """

    sleeper_id: str
    name: str
    position: str
    team: str
    adp: float
    projected_points: float
    games_played_frac: Union[float, str]
    draft_year: int
    sleeper_injury_status: Optional[str] = None
    sleeper_depth_chart_position: Optional[str] = None

    def is_rookie(self) -> bool:
        """Return whether the player is flagged as a rookie."""
        return self.games_played_frac == "R"

    def numeric_games_played_frac(self) -> Optional[float]:
        """Return games played fraction as a float when available."""
        if self.is_rookie():
            return None
        try:
            return float(self.games_played_frac)
        except (TypeError, ValueError):
            return None
