"""Pydantic schemas for offline position guide exports."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal

from pydantic import BaseModel, Field

POSITION_CODES = ("QB", "RB", "WR", "TE")


class PositionProbabilities(BaseModel):
    """Position probability distribution for one pick."""

    QB: float = 0.0
    RB: float = 0.0
    WR: float = 0.0
    TE: float = 0.0


class PositionGuidePick(BaseModel):
    """Aggregated position guidance for one user pick.

    Parameters
    ----------
    user_pick_index : int
        One-based index of the user's pick.
    overall_pick_number : int
        Overall pick number in the draft.
    round : int
        Draft round (1-based).
    positions : PositionProbabilities
        Averaged position probabilities.
    top_position : str
        Position with highest probability.
    sample_count : int
        Number of simulations contributing to this row.
    """

    user_pick_index: int
    overall_pick_number: int
    round: int
    positions: PositionProbabilities
    top_position: Literal["QB", "RB", "WR", "TE"]
    sample_count: int


class PositionGuideFile(BaseModel):
    """Top-level export for a static position guide cheat sheet."""

    schema_version: int = 1
    generated_at: datetime
    draft_year: int
    draft_slot: int
    num_teams: int
    simulations: int
    checkpoint_path: str
    checkpoint_episode: int
    player_data_csv: str
    enabled_state_features: List[str]
    roster_structure: Dict[str, int]
    total_user_picks: int
    picks: List[PositionGuidePick] = Field(default_factory=list)
