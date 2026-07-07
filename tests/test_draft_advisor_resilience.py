"""Tests for advisor retry and degraded fallback handling."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from draft_buddy.web.draft_advisor_resilience import (
    is_retriable_advisor_error,
    recommend_with_resilience,
)
from draft_buddy.web.draft_advisor_schemas import AdvisorShortlistError, build_degraded_advisor_result


def test_is_retriable_advisor_error_for_validation_and_shortlist() -> None:
    """Verify validation and shortlist errors are retriable."""
    from draft_buddy.web.draft_advisor_schemas import PickRecommendation

    with pytest.raises(ValidationError):
        PickRecommendation.model_validate({})

    assert is_retriable_advisor_error(AdvisorShortlistError("outside shortlist")) is True
    assert is_retriable_advisor_error(RuntimeError("OpenRouter returned invalid JSON: {}")) is True


def test_is_retriable_advisor_error_rejects_payment_failures() -> None:
    """Verify payment failures are not retried."""
    assert is_retriable_advisor_error(RuntimeError("OpenRouter request failed (402): Payment Required")) is False


def test_recommend_with_resilience_retries_then_succeeds() -> None:
    """Verify one repair retry can recover from a validation failure."""
    calls: list[str] = []

    def fetch_payload(user_prompt: str) -> tuple[dict, str]:
        calls.append(user_prompt)
        if len(calls) == 1:
            payload = {
                "advising_team_id": 2,
                "is_agent_team": True,
                "confidence": "high",
            }
            return payload, '{"partial": true}'

        payload = {
            "advising_team_id": 2,
            "is_agent_team": True,
            "recommended_player_id": 123,
            "recommended_name": "RB One",
            "confidence": "high",
            "rationale_bullets": ["Strong value."],
            "alternates": [],
            "flags": [],
            "unknown_factors": [],
        }
        return payload, '{"ok": true}'

    result = recommend_with_resilience(
        system_prompt="system",
        user_prompt="context",
        fetch_payload=fetch_payload,
        advising_team_id=2,
        is_agent_team=True,
        valid_player_ids={123},
    )

    assert result.degraded is False
    assert result.recommended_player_id == 123
    assert len(calls) == 2
    assert "failed validation" in calls[1]


def test_recommend_with_resilience_returns_degraded_after_exhausted_retries() -> None:
    """Verify repeated validation failures return a degraded response."""
    payload = {
        "recommended_player_id": "999",
        "recommended_name": "WR One",
        "confidence": "medium",
        "rationale_bullets": ["Good fit."],
        "alternates": [{"player_id": "9493", "player_name": "Alt WR", "reason": "Fallback"}],
    }

    def fetch_payload(_user_prompt: str) -> tuple[dict, str]:
        return payload, '{"invalid": true}'

    result = recommend_with_resilience(
        system_prompt="system",
        user_prompt="context",
        fetch_payload=fetch_payload,
        advising_team_id=2,
        is_agent_team=True,
        valid_player_ids={123},
    )

    assert result.degraded is True
    assert result.recommended_name == "WR One"
    assert result.rationale_bullets == ["Good fit."]
    assert result.parse_error
    assert result.raw_content


def test_build_degraded_advisor_result_extracts_partial_alternates() -> None:
    """Verify degraded responses preserve sanitized alternate picks."""
    result = build_degraded_advisor_result(
        parse_error="missing field",
        raw_content='{"recommended_name":"QB One"}',
        payload={
            "recommended_name": "QB One",
            "alternates": [{"player_id": "1", "player_name": "Alt QB", "rationale": "Safe floor"}],
        },
        advising_team_id=2,
        is_agent_team=True,
    )

    assert result.degraded is True
    assert len(result.alternates) == 1
    assert result.alternates[0].name == "Alt QB"


def test_recommend_with_resilience_raises_non_retriable_errors() -> None:
    """Verify payment failures bubble up without degraded fallback."""

    def fetch_payload(_user_prompt: str) -> tuple[dict, str]:
        raise RuntimeError("OpenRouter request failed (402): Payment Required")

    with pytest.raises(RuntimeError, match="402"):
        recommend_with_resilience(
            system_prompt="system",
            user_prompt="context",
            fetch_payload=fetch_payload,
            advising_team_id=2,
            is_agent_team=True,
            valid_player_ids={123},
        )
