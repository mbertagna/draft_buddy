"""Tests for draft advisor schema sanitization."""

from __future__ import annotations

from draft_buddy.web.draft_advisor_schemas import parse_pick_recommendation, sanitize_advisor_payload


def test_sanitize_advisor_payload_maps_alternate_player_name() -> None:
    """Verify player_name is mapped to name for alternate picks."""
    payload = {
        "advising_team_id": 2,
        "is_agent_team": True,
        "recommended_player_id": "123",
        "recommended_name": "RB One",
        "confidence": "high",
        "alternates": [
            {
                "player_id": "9493",
                "player_name": "WR Two",
                "rationale": "Strong volume.",
            }
        ],
    }

    sanitized = sanitize_advisor_payload(payload)
    recommendation = parse_pick_recommendation(payload)

    assert sanitized["alternates"][0]["name"] == "WR Two"
    assert sanitized["alternates"][0]["reason"] == "Strong volume."
    assert recommendation.alternates[0].player_id == 9493


def test_sanitize_advisor_payload_drops_invalid_alternates() -> None:
    """Verify alternates missing required fields are removed."""
    payload = {
        "advising_team_id": 2,
        "is_agent_team": True,
        "recommended_player_id": 123,
        "recommended_name": "RB One",
        "confidence": "high",
        "alternates": [{"player_id": "9493"}],
    }

    recommendation = parse_pick_recommendation(payload)

    assert recommendation.alternates == []
