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
    """Verify missing reasoning/quick_take/pros/cons/risks sanitize to safe defaults."""
    payload = {
        "advising_team_id": 2,
        "is_agent_team": True,
        "recommended_player_id": 123,
        "recommended_name": "RB One",
        "confidence": "high",
    }

    recommendation = parse_pick_recommendation(payload)

    assert recommendation.reasoning == ""
    assert recommendation.quick_take == ""
    assert recommendation.pros == ""
    assert recommendation.cons == ""
    assert recommendation.evidence == []
    assert recommendation.risks == []


def test_sanitize_advisor_payload_allows_pros_without_cons() -> None:
    """Verify pros and cons are independently optional, not forced in pairs.

    A clear best-player-available pick may have a pro with no genuine con.
    """
    payload = {
        "advising_team_id": 2,
        "is_agent_team": True,
        "recommended_player_id": 123,
        "recommended_name": "RB One",
        "confidence": "high",
        "quick_take": "Best remaining RB by a wide VORP margin.",
        "pros": "highest VORP on the board; fills your RB2 need",
    }

    recommendation = parse_pick_recommendation(payload)

    assert recommendation.pros == "highest VORP on the board; fills your RB2 need"
    assert recommendation.cons == ""


def test_sanitize_advisor_payload_keeps_teaching_fields_untruncated() -> None:
    """Verify reasoning-first teaching fields round-trip without length truncation."""
    long_quick_take = "word " * 60
    payload = {
        "advising_team_id": 2,
        "is_agent_team": True,
        "recommended_player_id": 123,
        "recommended_name": "RB One",
        "confidence": "high",
        "reasoning": "Best overall value beats a shallow need-fill at this pick.",
        "evidence": ["Highest VORP among need fills.", "Top-3 ADP value at position."],
        "quick_take": long_quick_take,
        "cons": "  tough Week 7 bye cluster  ",
        "risks": ["Injury concern", "Bye week pile-up", "Committee risk", "Extra"],
    }

    recommendation = parse_pick_recommendation(payload)

    assert recommendation.reasoning == "Best overall value beats a shallow need-fill at this pick."
    assert recommendation.evidence == [
        "Highest VORP among need fills.",
        "Top-3 ADP value at position.",
    ]
    assert recommendation.quick_take == long_quick_take.strip()
    assert recommendation.cons == "tough Week 7 bye cluster"
    assert recommendation.risks == [
        "Injury concern",
        "Bye week pile-up",
        "Committee risk",
    ]


def test_build_degraded_advisor_result_extracts_teaching_fields() -> None:
    """Verify degraded responses preserve quick_take and risks when present."""
    result = build_degraded_advisor_result(
        parse_error="missing field",
        raw_content="{}",
        payload={
            "recommended_name": "QB One",
            "quick_take": "Safe QB floor with a clean bye week.",
            "risks": ["Late-round QB run"],
        },
        advising_team_id=2,
        is_agent_team=True,
    )

    assert result.degraded is True
    assert result.quick_take == "Safe QB floor with a clean bye week."
    assert result.risks == ["Late-round QB run"]
