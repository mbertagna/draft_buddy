"""Retry and degraded fallback handling for advisor structured output."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Set

from pydantic import ValidationError

from draft_buddy.web.draft_advisor_schemas import (
    AdvisorResult,
    AdvisorShortlistError,
    build_degraded_advisor_result,
    parse_pick_recommendation,
)

ADVISOR_MAX_ATTEMPTS = 2

REPAIR_SUFFIX = (
    "\n\n---\nYour previous response failed validation: {error}\n"
    "Return ONLY valid JSON matching PickRecommendation. "
    "Required fields, in this order: reasoning (string), evidence (array of strings, up to 4), "
    "recommended_player_id (int), recommended_name (string), confidence (string), "
    "quick_take (one short sentence), pros (string, optional — blank if none), "
    "cons (string, optional — blank if none), risks (array of strings, up to 3), "
    "alternates (array of objects with player_id, name, reason), flags, unknown_factors, "
    "advising_team_id, is_agent_team."
)


def is_retriable_advisor_error(error: Exception) -> bool:
    """Return whether an advisor parse error should trigger one retry.

    Parameters
    ----------
    error : Exception
        Error raised while fetching or validating model output.

    Returns
    -------
    bool
        True when a repair retry may succeed.
    """
    if isinstance(error, (ValidationError, AdvisorShortlistError)):
        return True
    if isinstance(error, RuntimeError):
        message = str(error).lower()
        if any(token in message for token in ("402", "401", "payment required", "unauthorized")):
            return False
        return any(
            phrase in message
            for phrase in ("invalid json", "empty content", "validation", "outside the candidate")
        )
    return False


def recommend_with_resilience(
    *,
    system_prompt: str,
    user_prompt: str,
    fetch_payload: Callable[[str], tuple[Dict[str, Any], str]],
    advising_team_id: int,
    is_agent_team: bool,
    valid_player_ids: Set[int],
) -> AdvisorResult:
    """Fetch and validate an advisor recommendation with one repair retry.

    Parameters
    ----------
    system_prompt : str
        System instructions for the model.
    user_prompt : str
        Markdown draft context payload.
    fetch_payload : Callable[[str], tuple[Dict[str, Any], str]]
        Callable that returns parsed JSON and raw response text for a user prompt.
    advising_team_id : int
        Team receiving advice.
    is_agent_team : bool
        Whether the advising team is the user agent team.
    valid_player_ids : Set[int]
        Candidate shortlist ids allowed for recommendations.

    Returns
    -------
    AdvisorResult
        Structured recommendation or a degraded partial response.

    Raises
    ------
    Exception
        When a non-retriable error occurs on the first attempt.
    """
    last_error = "Unknown validation failure."
    last_payload: Dict[str, Any] | None = None
    last_raw = ""

    for attempt in range(ADVISOR_MAX_ATTEMPTS):
        prompt = user_prompt if attempt == 0 else user_prompt + REPAIR_SUFFIX.format(error=last_error)
        try:
            payload, raw_content = fetch_payload(prompt)
            last_payload = payload
            last_raw = raw_content
            recommendation = parse_pick_recommendation(payload)
            if recommendation.recommended_player_id not in valid_player_ids:
                raise AdvisorShortlistError(
                    f"recommended_player_id {recommendation.recommended_player_id} "
                    "is outside the candidate shortlist"
                )
            return AdvisorResult.from_recommendation(
                recommendation.model_copy(
                    update={
                        "advising_team_id": advising_team_id,
                        "is_agent_team": is_agent_team,
                    }
                )
            )
        except Exception as error:
            last_error = str(error)
            if hasattr(error, "raw_content") and getattr(error, "raw_content"):
                last_raw = str(getattr(error, "raw_content"))
            if hasattr(error, "payload") and getattr(error, "payload") is not None:
                last_payload = getattr(error, "payload")
            if not is_retriable_advisor_error(error):
                raise
            if attempt >= ADVISOR_MAX_ATTEMPTS - 1:
                return build_degraded_advisor_result(
                    parse_error=last_error,
                    raw_content=last_raw or _payload_as_raw(last_payload),
                    payload=last_payload,
                    advising_team_id=advising_team_id,
                    is_agent_team=is_agent_team,
                )

    return build_degraded_advisor_result(
        parse_error=last_error,
        raw_content=last_raw or _payload_as_raw(last_payload),
        payload=last_payload,
        advising_team_id=advising_team_id,
        is_agent_team=is_agent_team,
    )


def _payload_as_raw(payload: Dict[str, Any] | None) -> str:
    """Serialize a payload dict for degraded display."""
    if not payload:
        return ""
    try:
        return json.dumps(payload, indent=2)
    except TypeError:
        return str(payload)
