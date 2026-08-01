"""Tests for draft advisor schema sanitization."""

from __future__ import annotations

from draft_buddy.web.draft_advisor_schemas import (
    build_degraded_advisor_result,
    parse_pick_recommendation,
    sanitize_advisor_payload,
)


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


def test_sanitize_advisor_payload_defaults_missing_teaching_fields() -> None:
    """Verify missing recap and risks sanitize to safe defaults."""
    payload = {
        "advising_team_id": 2,
        "is_agent_team": True,
        "recommended_player_id": 123,
        "recommended_name": "RB One",
        "confidence": "high",
    }

    recommendation = parse_pick_recommendation(payload)

    assert recommendation.plain_english_recap == ""
    assert recommendation.risks == []


def test_sanitize_advisor_payload_truncates_risks_and_keeps_recap() -> None:
    """Verify teaching fields round-trip through sanitization."""
    payload = {
        "advising_team_id": 2,
        "is_agent_team": True,
        "recommended_player_id": 123,
        "recommended_name": "RB One",
        "confidence": "high",
        "plain_english_recap": "  Take the top remaining RB.  ",
        "risks": ["Injury concern", "Bye week pile-up", "Committee risk", "Extra"],
        "rationale_bullets": ["Highest VORP among need fills."],
    }

    recommendation = parse_pick_recommendation(payload)

    assert recommendation.plain_english_recap == "Take the top remaining RB."
    assert recommendation.risks == [
        "Injury concern",
        "Bye week pile-up",
        "Committee risk",
    ]


def test_build_degraded_advisor_result_extracts_teaching_fields() -> None:
    """Verify degraded responses preserve recap and risks when present."""
    result = build_degraded_advisor_result(
        parse_error="missing field",
        raw_content="{}",
        payload={
            "recommended_name": "QB One",
            "plain_english_recap": "Safe QB floor.",
            "risks": ["Late-round QB run"],
        },
        advising_team_id=2,
        is_agent_team=True,
    )

    assert result.degraded is True
    assert result.plain_english_recap == "Safe QB floor."
    assert result.risks == ["Late-round QB run"]
