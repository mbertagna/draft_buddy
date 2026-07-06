"""Pydantic schemas for the live draft assistant."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AdvisorScope(str, Enum):
    """Controls auto-fire eligibility on the client and server."""

    AGENT_ONLY = "agent_only"
    ALL_TEAMS = "all_teams"


class AdvisorTrigger(str, Enum):
    """How the assistant request was initiated."""

    AUTO = "auto"
    MANUAL = "manual"


class AdvisorRequest(BaseModel):
    """Request body for the draft assistant endpoint."""

    team_id: Optional[int] = None
    gp_min: Optional[float] = None
    scope: AdvisorScope = AdvisorScope.AGENT_ONLY
    trigger: AdvisorTrigger = AdvisorTrigger.MANUAL


class AlternatePick(BaseModel):
    """Alternate player recommendation."""

    player_id: int
    name: str
    reason: str


class PickRecommendation(BaseModel):
    """Structured assistant recommendation returned to the client."""

    advising_team_id: int
    is_agent_team: bool
    recommended_player_id: int
    recommended_name: str
    confidence: str
    rationale_bullets: list[str] = Field(default_factory=list, max_length=5)
    alternates: list[AlternatePick] = Field(default_factory=list, max_length=3)
    flags: list[str] = Field(default_factory=list)
    unknown_factors: list[str] = Field(default_factory=list)
