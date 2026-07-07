"""Pydantic schemas for the live draft assistant."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from draft_buddy.llm.model_registry import (
    default_advisor_agent_model,
    default_advisor_other_teams_model,
)


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
    agent_model: str = Field(default_factory=default_advisor_agent_model)
    other_teams_model: str = Field(default_factory=default_advisor_other_teams_model)


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


class AdvisorShortlistError(Exception):
    """Raised when the model recommends a player outside the candidate shortlist."""


class AdvisorResult(BaseModel):
    """Advisor API response, including optional degraded fallback fields."""

    degraded: bool = False
    parse_error: Optional[str] = None
    raw_content: Optional[str] = None
    advising_team_id: int
    is_agent_team: bool
    recommended_player_id: Optional[int] = None
    recommended_name: Optional[str] = None
    confidence: Optional[str] = None
    rationale_bullets: list[str] = Field(default_factory=list, max_length=5)
    alternates: list[AlternatePick] = Field(default_factory=list, max_length=3)
    flags: list[str] = Field(default_factory=list)
    unknown_factors: list[str] = Field(default_factory=list)

    @classmethod
    def from_recommendation(cls, recommendation: PickRecommendation) -> AdvisorResult:
        """Build a successful advisor result from a validated recommendation."""
        return cls(
            degraded=False,
            advising_team_id=recommendation.advising_team_id,
            is_agent_team=recommendation.is_agent_team,
            recommended_player_id=recommendation.recommended_player_id,
            recommended_name=recommendation.recommended_name,
            confidence=recommendation.confidence,
            rationale_bullets=recommendation.rationale_bullets,
            alternates=recommendation.alternates,
            flags=recommendation.flags,
            unknown_factors=recommendation.unknown_factors,
        )


def sanitize_advisor_payload(payload: dict) -> dict:
    """Normalize raw LLM JSON so it satisfies ``PickRecommendation`` validators.

    Parameters
    ----------
    payload : dict
        Parsed JSON from structured model output.

    Returns
    -------
    dict
        Payload safe to pass to ``PickRecommendation.model_validate``.
    """
    sanitized = dict(payload)

    recommended_id = sanitized.get("recommended_player_id")
    if recommended_id is not None:
        sanitized["recommended_player_id"] = _coerce_int(recommended_id)

    recommended_name = sanitized.get("recommended_name")
    if not recommended_name:
        fallback_name = sanitized.get("recommended_player_name") or sanitized.get("player_name")
        if fallback_name:
            sanitized["recommended_name"] = str(fallback_name)

    advising_team_id = sanitized.get("advising_team_id")
    if advising_team_id is not None:
        sanitized["advising_team_id"] = _coerce_int(advising_team_id)

    alternates = sanitized.get("alternates")
    if isinstance(alternates, list):
        cleaned_alternates = []
        for alternate in alternates[:3]:
            normalized = _sanitize_alternate(alternate)
            if normalized is not None:
                cleaned_alternates.append(normalized)
        sanitized["alternates"] = cleaned_alternates

    rationale_bullets = sanitized.get("rationale_bullets")
    if isinstance(rationale_bullets, list):
        sanitized["rationale_bullets"] = [str(item) for item in rationale_bullets[:5]]

    return sanitized


def _sanitize_alternate(alternate: object) -> dict | None:
    """Normalize one alternate pick entry from model output."""
    if not isinstance(alternate, dict):
        return None

    name = alternate.get("name") or alternate.get("player_name") or alternate.get("recommended_name")
    reason = alternate.get("reason") or alternate.get("rationale") or alternate.get("summary")
    player_id = _coerce_int(alternate.get("player_id"))
    if player_id is None or not name:
        return None

    return {
        "player_id": player_id,
        "name": str(name).strip(),
        "reason": str(reason or "").strip(),
    }


def _coerce_int(value: object) -> int | None:
    """Convert a scalar value to int when possible."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_pick_recommendation(payload: dict) -> PickRecommendation:
    """Validate a raw recommendation payload from any advisor gateway."""
    return PickRecommendation.model_validate(sanitize_advisor_payload(payload))


def extract_partial_from_payload(payload: dict | None) -> dict[str, object]:
    """Best-effort extraction of display fields from invalid model output.

    Parameters
    ----------
    payload : dict, optional
        Parsed JSON from the model, if available.

    Returns
    -------
    dict[str, object]
        Partial fields safe for degraded UI rendering.
    """
    if not isinstance(payload, dict):
        return {}

    sanitized = sanitize_advisor_payload(payload)
    partial: dict[str, object] = {}

    recommended_id = sanitized.get("recommended_player_id")
    if recommended_id is not None:
        partial["recommended_player_id"] = recommended_id

    recommended_name = sanitized.get("recommended_name")
    if recommended_name:
        partial["recommended_name"] = recommended_name

    confidence = sanitized.get("confidence")
    if confidence:
        partial["confidence"] = str(confidence)

    rationale_bullets = sanitized.get("rationale_bullets")
    if isinstance(rationale_bullets, list):
        partial["rationale_bullets"] = [str(item) for item in rationale_bullets[:5]]

    alternates = sanitized.get("alternates")
    if isinstance(alternates, list):
        partial["alternates"] = _parse_partial_alternates(alternates)

    flags = sanitized.get("flags")
    if isinstance(flags, list):
        partial["flags"] = [str(item) for item in flags]

    unknown_factors = sanitized.get("unknown_factors")
    if isinstance(unknown_factors, list):
        partial["unknown_factors"] = [str(item) for item in unknown_factors]

    return partial


def _parse_partial_alternates(alternates: list[object]) -> list[AlternatePick]:
    """Convert sanitized alternate dicts into partial AlternatePick models."""
    parsed: list[AlternatePick] = []
    for alternate in alternates[:3]:
        if isinstance(alternate, AlternatePick):
            parsed.append(alternate)
            continue
        normalized = _sanitize_alternate(alternate)
        if normalized is None:
            continue
        try:
            parsed.append(AlternatePick.model_validate(normalized))
        except ValidationError:
            continue
    return parsed


def build_degraded_advisor_result(
    *,
    parse_error: str,
    raw_content: str,
    payload: dict | None,
    advising_team_id: int,
    is_agent_team: bool,
) -> AdvisorResult:
    """Build a degraded advisor response from the last failed model output.

    Parameters
    ----------
    parse_error : str
        Validation or parse error message.
    raw_content : str
        Raw model text for manual review.
    payload : dict, optional
        Parsed JSON object when available.
    advising_team_id : int
        Team receiving advice.
    is_agent_team : bool
        Whether the advising team is the user agent team.

    Returns
    -------
    AdvisorResult
        Degraded response with partial fields when extractable.
    """
    partial = extract_partial_from_payload(payload)
    return AdvisorResult(
        degraded=True,
        parse_error=parse_error,
        raw_content=raw_content or None,
        advising_team_id=advising_team_id,
        is_agent_team=is_agent_team,
        recommended_player_id=partial.get("recommended_player_id"),  # type: ignore[arg-type]
        recommended_name=partial.get("recommended_name"),  # type: ignore[arg-type]
        confidence=partial.get("confidence"),  # type: ignore[arg-type]
        rationale_bullets=partial.get("rationale_bullets", []),  # type: ignore[arg-type]
        alternates=partial.get("alternates", []),  # type: ignore[arg-type]
        flags=partial.get("flags", []),  # type: ignore[arg-type]
        unknown_factors=partial.get("unknown_factors", []),  # type: ignore[arg-type]
    )
