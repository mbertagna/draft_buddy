"""Tests for team outlook Pydantic schemas and post-validation."""

from __future__ import annotations

import pytest

from draft_buddy.data.insights.schemas import Confidence, EvidenceBullet
from draft_buddy.data.insights.team_schemas import (
    ScheduleHardness,
    TeamOutlook,
    TeamTier,
    apply_team_outlook_post_validation,
    default_unknown_team_outlook,
    sanitize_team_synthesis_payload,
)


def _sample_bullet(
    url: str = "https://espn.com/article",
    published_date: str | None = "2026-06-01",
) -> EvidenceBullet:
    """Return a minimal evidence bullet for tests."""
    return EvidenceBullet(
        text="New offensive coordinator installs up-tempo scheme.",
        source_domain="espn.com",
        source_title="Team Outlook",
        source_url=url,
        published_date=published_date,
    )


def _outlook(**overrides) -> TeamOutlook:
    """Build a default valid team outlook with optional overrides."""
    defaults = dict(
        team_abbr="SF",
        season=2026,
        outlook_summary="Contender with a retooled offensive line.",
        offense_tier=TeamTier.HIGH,
        offense_notes="Strong receiving corps.",
        defense_tier=TeamTier.MEDIUM,
        defense_notes="Pass rush questions.",
        schedule_hardness=ScheduleHardness.AVERAGE,
        schedule_notes="Middling strength of schedule.",
        key_storylines=["New OC installs uptempo scheme"],
        overall_confidence=Confidence.MEDIUM,
        bullets=[_sample_bullet()],
    )
    defaults.update(overrides)
    return TeamOutlook(**defaults)


def test_outlook_summary_rejects_more_than_two_sentences() -> None:
    """Verify outlook_summary enforces the two-sentence limit."""
    with pytest.raises(ValueError, match="2 sentences"):
        _outlook(outlook_summary="First. Second. Third.")


def test_key_storylines_rejects_more_than_four_items() -> None:
    """Verify key_storylines enforces the four-item limit."""
    with pytest.raises(ValueError, match="4 items"):
        _outlook(key_storylines=["a", "b", "c", "d", "e"])


def test_bullets_rejects_more_than_four_items() -> None:
    """Verify bullets enforces the four-item limit."""
    with pytest.raises(ValueError, match="4 items"):
        _outlook(bullets=[_sample_bullet(f"https://espn.com/{i}") for i in range(5)])


def test_apply_team_outlook_post_validation_downgrades_without_bullets() -> None:
    """Verify judgment fields downgrade to unknown when no bullets support them."""
    outlook = _outlook(bullets=[])

    validated = apply_team_outlook_post_validation(outlook)

    assert validated.offense_tier == TeamTier.UNKNOWN
    assert validated.defense_tier == TeamTier.UNKNOWN
    assert validated.schedule_hardness == ScheduleHardness.UNKNOWN
    assert "offense_tier" in validated.fields_unknown
    assert "defense_tier" in validated.fields_unknown
    assert "schedule_hardness" in validated.fields_unknown


def test_apply_team_outlook_post_validation_filters_bullet_urls() -> None:
    """Verify bullets outside allowed URLs are removed."""
    outlook = _outlook(
        bullets=[
            _sample_bullet("https://espn.com/allowed"),
            _sample_bullet("https://blocked.com/article"),
        ]
    )

    validated = apply_team_outlook_post_validation(
        outlook,
        allowed_urls={"https://espn.com/allowed"},
    )

    assert len(validated.bullets) == 1
    assert validated.bullets[0].source_url == "https://espn.com/allowed"


def test_apply_team_outlook_post_validation_fills_evidence_as_of() -> None:
    """Verify evidence_as_of is derived from the newest bullet date."""
    outlook = _outlook(
        bullets=[
            _sample_bullet(published_date="2026-04-01"),
            _sample_bullet(url="https://espn.com/b", published_date="2026-06-15"),
        ]
    )

    validated = apply_team_outlook_post_validation(outlook)

    assert validated.evidence_as_of == "2026-06-15"


def test_default_unknown_team_outlook_has_safe_fallback() -> None:
    """Verify default unknown team outlook sets all judgment fields to unknown."""
    outlook = default_unknown_team_outlook("SF", 2026, ["query a"])

    assert outlook.offense_tier == TeamTier.UNKNOWN
    assert outlook.defense_tier == TeamTier.UNKNOWN
    assert outlook.schedule_hardness == ScheduleHardness.UNKNOWN
    assert outlook.overall_confidence == Confidence.UNKNOWN
    assert "offense_tier" in outlook.fields_unknown
    assert "schedule_hardness" in outlook.fields_unknown


def test_sanitize_team_synthesis_payload_truncates_and_caps_lists() -> None:
    """Verify sanitize truncates outlook_summary and caps list lengths."""
    payload = sanitize_team_synthesis_payload(
        {
            "team_abbr": "SF",
            "season": 2026,
            "outlook_summary": "First sentence. Second sentence. Third sentence.",
            "offense_tier": "high",
            "offense_notes": "n",
            "defense_tier": "medium",
            "defense_notes": "n",
            "schedule_hardness": "average",
            "schedule_notes": "n",
            "key_storylines": ["a", "b", "c", "d", "e"],
            "overall_confidence": "medium",
            "bullets": [dict(_sample_bullet(f"https://espn.com/{i}")) for i in range(6)],
        }
    )

    outlook = TeamOutlook.model_validate(payload)

    assert outlook.outlook_summary == "First sentence. Second sentence."
    assert len(outlook.key_storylines) == 4
    assert len(outlook.bullets) == 4
