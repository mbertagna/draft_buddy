"""Tests for player insight Pydantic schemas and post-validation."""

from __future__ import annotations

import pytest

from draft_buddy.data.insights.schemas import (
    Confidence,
    DepthRole,
    DraftLean,
    EvidenceBullet,
    PlayerInsight,
    PlayingTimeTier,
    RecoveryStatus,
    RiskLevel,
    apply_insight_post_validation,
    default_unknown_insight,
    sanitize_synthesis_payload,
)


def _sample_bullet(
    url: str = "https://espn.com/article",
    published_date: str | None = "2026-06-01",
) -> EvidenceBullet:
    """Return a minimal evidence bullet for tests."""
    return EvidenceBullet(
        text="Expected to handle a workhorse role.",
        source_domain="espn.com",
        source_title="Outlook",
        source_url=url,
        published_date=published_date,
    )


def test_outlook_phrase_rejects_more_than_twelve_words() -> None:
    """Verify outlook_phrase enforces the twelve-word limit."""
    with pytest.raises(ValueError, match="12 words"):
        PlayerInsight(
            outlook_phrase="one two three four five six seven eight nine ten eleven twelve thirteen",
            summary="Short summary.",
            depth_role=DepthRole.STARTER,
            playing_time_tier=PlayingTimeTier.HIGH,
            injury_risk=RiskLevel.LOW,
            upside=RiskLevel.HIGH,
            recovery_status=RecoveryStatus.NA,
            overall_confidence=Confidence.HIGH,
            bullets=[_sample_bullet()],
        )


def test_summary_rejects_more_than_two_sentences() -> None:
    """Verify summary enforces the two-sentence limit."""
    with pytest.raises(ValueError, match="2 sentences"):
        PlayerInsight(
            outlook_phrase="Workhorse role expected",
            summary="First. Second. Third.",
            depth_role=DepthRole.STARTER,
            playing_time_tier=PlayingTimeTier.HIGH,
            injury_risk=RiskLevel.LOW,
            upside=RiskLevel.HIGH,
            recovery_status=RecoveryStatus.NA,
            overall_confidence=Confidence.HIGH,
            bullets=[_sample_bullet()],
        )


def test_apply_insight_post_validation_downgrades_without_bullets() -> None:
    """Verify non-unknown fields downgrade when no bullets support them."""
    insight = PlayerInsight(
        outlook_phrase="Starter role expected",
        summary="Should lead the backfield.",
        depth_role=DepthRole.STARTER,
        playing_time_tier=PlayingTimeTier.HIGH,
        injury_risk=RiskLevel.LOW,
        upside=RiskLevel.HIGH,
        floor=RiskLevel.MEDIUM,
        draft_lean=DraftLean.BUY,
        recovery_status=RecoveryStatus.RECOVERED,
        overall_confidence=Confidence.MEDIUM,
        bullets=[],
    )

    validated = apply_insight_post_validation(insight)

    assert validated.depth_role == DepthRole.UNKNOWN
    assert validated.playing_time_tier == PlayingTimeTier.UNKNOWN
    assert validated.floor == RiskLevel.UNKNOWN
    assert validated.draft_lean == DraftLean.UNKNOWN
    assert "depth_role" in validated.fields_unknown
    assert "floor" in validated.fields_unknown
    assert "draft_lean" in validated.fields_unknown
    assert "recovery_status" in validated.fields_unknown


def test_apply_insight_post_validation_filters_bullet_urls() -> None:
    """Verify bullets outside allowed URLs are removed."""
    insight = PlayerInsight(
        outlook_phrase="Starter role expected",
        summary="Should lead the backfield.",
        depth_role=DepthRole.STARTER,
        playing_time_tier=PlayingTimeTier.HIGH,
        injury_risk=RiskLevel.LOW,
        upside=RiskLevel.HIGH,
        recovery_status=RecoveryStatus.NA,
        overall_confidence=Confidence.HIGH,
        bullets=[
            _sample_bullet("https://espn.com/article"),
            _sample_bullet("https://blocked.com/article"),
        ],
    )

    validated = apply_insight_post_validation(
        insight,
        allowed_urls={"https://espn.com/article"},
    )

    assert len(validated.bullets) == 1
    assert validated.bullets[0].source_url == "https://espn.com/article"


def test_apply_insight_post_validation_fills_evidence_as_of() -> None:
    """Verify evidence_as_of is derived from the newest bullet date."""
    insight = PlayerInsight(
        outlook_phrase="Starter role expected",
        summary="Should lead the backfield.",
        depth_role=DepthRole.STARTER,
        playing_time_tier=PlayingTimeTier.HIGH,
        injury_risk=RiskLevel.LOW,
        upside=RiskLevel.HIGH,
        recovery_status=RecoveryStatus.NA,
        overall_confidence=Confidence.HIGH,
        bullets=[
            _sample_bullet(published_date="2026-04-01"),
            _sample_bullet(
                url="https://espn.com/b",
                published_date="2026-06-15",
            ),
        ],
    )

    validated = apply_insight_post_validation(insight)

    assert validated.evidence_as_of == "2026-06-15"


def test_default_unknown_insight_has_safe_fallback_phrase() -> None:
    """Verify default unknown insight uses the limited-data outlook phrase."""
    insight = default_unknown_insight(["query a"])

    assert insight.outlook_phrase == "Limited data available"
    assert insight.overall_confidence == Confidence.UNKNOWN
    assert insight.draft_lean == DraftLean.UNKNOWN
    assert insight.floor == RiskLevel.UNKNOWN
    assert "depth_role" in insight.fields_unknown
    assert "draft_lean" in insight.fields_unknown


def test_sanitize_synthesis_payload_truncates_long_bullets() -> None:
    """Verify long bullet text is trimmed before validation."""
    long_text = "x" * 200
    payload = sanitize_synthesis_payload(
        {
            "outlook_phrase": "Elite upside with workload concerns after heavy usage",
            "summary": "First sentence. Second sentence. Third sentence.",
            "depth_role": "starter",
            "playing_time_tier": "high",
            "injury_risk": "medium",
            "upside": "high",
            "floor": "medium",
            "draft_lean": "fade",
            "recovery_status": "na",
            "tags": [],
            "overall_confidence": "medium",
            "fields_unknown": [],
            "bullets": [
                {
                    "text": long_text,
                    "source_domain": "espn.com",
                    "source_title": "Outlook",
                    "source_url": "https://espn.com/article",
                }
            ],
        }
    )

    insight = PlayerInsight.model_validate(payload)

    assert len(insight.bullets[0].text) == 160
    assert insight.summary == "First sentence. Second sentence."
    assert insight.outlook_phrase == "Elite upside with workload concerns after heavy usage"
    assert insight.draft_lean == DraftLean.FADE
