"""Pydantic schemas for offline position guide exports."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

POSITION_CODES = ("QB", "RB", "WR", "TE")


class PositionProbabilities(BaseModel):
    """Position probability distribution for one pick."""

    QB: float = 0.0
    RB: float = 0.0
    WR: float = 0.0
    TE: float = 0.0


class TopPlayerEntry(BaseModel):
    """Frequency of one player being drafted at a specific pick and position.

    Parameters
    ----------
    player_id : int
        Drafted player identifier.
    name : str
        Player display name.
    times_drafted : int
        Number of simulations where this player was taken at this pick.
    share : float
        Fraction of that position's picks (at this slot/pick) taken by this
        player, in ``[0, 1]``.
    """

    player_id: int
    name: str
    times_drafted: int
    share: float


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
    top_players : Dict[str, List[TopPlayerEntry]]
        Up to five most-frequently-drafted players per position, restricted
        to positions with at least one recorded pick.
    """

    user_pick_index: int
    overall_pick_number: int
    round: int
    positions: PositionProbabilities
    top_position: Literal["QB", "RB", "WR", "TE"]
    sample_count: int
    top_players: Dict[str, List[TopPlayerEntry]] = Field(default_factory=dict)


class PositionGuideFile(BaseModel):
    """Top-level export for a static position guide cheat sheet."""

    schema_version: int = 2
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
    temperature: float = 1.0
    generation_mode: Literal["self_play"] = "self_play"
    prune_inactive: bool = False
    limit_adp: Optional[int] = None
    draft_pool_size: Optional[int] = None
    picks: List[PositionGuidePick] = Field(default_factory=list)


class ModelAdpEntry(BaseModel):
    """Model-derived average draft position for one player.

    Parameters
    ----------
    player_id : int
        Player identifier.
    name : str
        Player display name.
    position : str
        Position code.
    model_adp : float
        Mean overall pick number across simulations in which the player was
        drafted by the self-play policy.
    std_dev : float
        Standard deviation of the overall pick number.
    times_drafted : int
        Number of simulations in which the player was drafted.
    draft_rate : float
        Fraction of simulations in which the player was drafted, in
        ``[0, 1]``.
    market_adp : Optional[float]
        Market ADP from the input player catalog, for comparison. ``None``
        when unavailable.
    """

    player_id: int
    name: str
    position: str
    model_adp: float
    std_dev: float
    times_drafted: int
    draft_rate: float
    market_adp: Optional[float] = None


class ModelAdpFile(BaseModel):
    """Top-level export for model-derived ADP across all simulated slots."""

    schema_version: int = 1
    generated_at: datetime
    draft_year: int
    num_teams: int
    simulations: int
    checkpoint_path: str
    checkpoint_episode: int
    player_data_csv: str
    temperature: float = 1.0
    prune_inactive: bool = False
    limit_adp: Optional[int] = None
    draft_pool_size: Optional[int] = None
    players: List[ModelAdpEntry] = Field(default_factory=list)
