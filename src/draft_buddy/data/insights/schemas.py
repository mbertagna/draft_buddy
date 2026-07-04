"""Pydantic schemas for offline player insight enrichment."""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Confidence(str, Enum):
    """Confidence level for synthesized insight fields."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class DepthRole(str, Enum):
    """Expected depth-chart role for the upcoming season."""

    STARTER = "starter"
    CO_STARTER = "co_starter"
    COMMITTEE = "committee"
    BACKUP = "backup"
    UNKNOWN = "unknown"


class PlayingTimeTier(str, Enum):
    """Expected playing-time volume tier."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    """Risk or upside tier."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class RecoveryStatus(str, Enum):
    """Injury recovery status relative to the upcoming season."""

    NA = "na"
    RECOVERING = "recovering"
    RECOVERED = "recovered"
    UNKNOWN = "unknown"


class InsightTag(str, Enum):
    """Controlled vocabulary tags for fantasy-relevant signals."""

    INJURY_RECOVERY = "injury_recovery"
    INJURY_RISK = "injury_risk"
    SUSPENSION_RISK = "suspension_risk"
    ROLE_EXPANSION = "role_expansion"
    ROLE_REDUCTION = "role_reduction"
    QB_CHANGE = "qb_change"
    COACHING_CHANGE = "coaching_change"
    COMPETITION_ADDED = "competition_added"
    BREAKOUT_CANDIDATE = "breakout_candidate"
    AGING_DECLINE = "aging_decline"
    HOLDOUT = "holdout"
    TRADE_RUMOR = "trade_rumor"


class EvidenceBullet(BaseModel):
    """Source-grounded evidence bullet for a player insight."""

    text: str = Field(max_length=120)
    source_domain: str
    source_title: str
    source_url: str
    published_date: Optional[str] = None


class PlayerInsight(BaseModel):
    """Structured fantasy insight record for one player."""

    outlook_phrase: str
    summary: str
    depth_role: DepthRole
    playing_time_tier: PlayingTimeTier
    injury_risk: RiskLevel
    upside: RiskLevel
    recovery_status: RecoveryStatus
    tags: list[InsightTag] = Field(default_factory=list, max_length=4)
    overall_confidence: Confidence
    fields_unknown: list[str] = Field(default_factory=list)
    bullets: list[EvidenceBullet] = Field(default_factory=list, max_length=4)
    search_queries_used: list[str] = Field(default_factory=list)
    snippet_count: int = 0

    @field_validator("outlook_phrase")
    @classmethod
    def validate_outlook_phrase_word_count(cls, value: str) -> str:
        """Enforce a maximum of eight words in the outlook phrase."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("outlook_phrase must not be empty")
        word_count = len(stripped.split())
        if word_count > 8:
            raise ValueError(f"outlook_phrase must be at most 8 words, got {word_count}")
        return stripped

    @field_validator("summary")
    @classmethod
    def validate_summary_sentence_count(cls, value: str) -> str:
        """Enforce at most two sentences in the summary."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("summary must not be empty")
        sentences = [part for part in re.split(r"[.!?]+", stripped) if part.strip()]
        if len(sentences) > 2:
            raise ValueError(f"summary must be at most 2 sentences, got {len(sentences)}")
        return stripped

    @field_validator("tags")
    @classmethod
    def validate_tag_count(cls, value: list[InsightTag]) -> list[InsightTag]:
        """Enforce at most four tags."""
        if len(value) > 4:
            raise ValueError(f"tags must contain at most 4 items, got {len(value)}")
        return value

    @field_validator("bullets")
    @classmethod
    def validate_bullet_count(cls, value: list[EvidenceBullet]) -> list[EvidenceBullet]:
        """Enforce at most four evidence bullets."""
        if len(value) > 4:
            raise ValueError(f"bullets must contain at most 4 items, got {len(value)}")
        return value


class PlayerInsightsFile(BaseModel):
    """Top-level container for all player insights for one draft year."""

    schema_version: str = "1.0"
    draft_year: int
    generated_at: datetime
    model: str
    players: dict[str, PlayerInsight]


JUDGMENT_FIELDS = (
    "depth_role",
    "playing_time_tier",
    "injury_risk",
    "upside",
    "recovery_status",
)

MAX_OUTLOOK_WORDS = 8
MAX_SUMMARY_SENTENCES = 2
MAX_BULLET_TEXT_CHARS = 120
MAX_BULLETS = 4
MAX_TAGS = 4


def _truncate_to_sentences(text: str, max_sentences: int) -> str:
    """Return at most ``max_sentences`` sentences from ``text``."""
    stripped = text.strip()
    if not stripped:
        return stripped

    parts = re.split(r"([.!?]+)", stripped)
    sentences: list[str] = []
    current = ""
    for part in parts:
        if not part:
            continue
        if re.fullmatch(r"[.!?]+", part):
            current += part
            if current.strip():
                sentences.append(current.strip())
            current = ""
            if len(sentences) >= max_sentences:
                break
        else:
            current += part

    if len(sentences) < max_sentences and current.strip():
        sentences.append(current.strip())

    return " ".join(sentences[:max_sentences])


def sanitize_synthesis_payload(payload: dict) -> dict:
    """Normalize raw Gemini JSON so it satisfies ``PlayerInsight`` validators.

    Parameters
    ----------
    payload : dict
        Parsed JSON from Gemini structured output.

    Returns
    -------
    dict
        Payload safe to pass to ``PlayerInsight.model_validate``.
    """
    sanitized = dict(payload)

    outlook = str(sanitized.get("outlook_phrase", "")).strip()
    if outlook:
        words = outlook.split()
        sanitized["outlook_phrase"] = " ".join(words[:MAX_OUTLOOK_WORDS])

    summary = str(sanitized.get("summary", "")).strip()
    if summary:
        sanitized["summary"] = _truncate_to_sentences(summary, MAX_SUMMARY_SENTENCES)

    bullets = sanitized.get("bullets")
    if isinstance(bullets, list):
        trimmed_bullets = []
        for bullet in bullets[:MAX_BULLETS]:
            if not isinstance(bullet, dict):
                continue
            bullet_copy = dict(bullet)
            text = str(bullet_copy.get("text", "")).strip()
            if len(text) > MAX_BULLET_TEXT_CHARS:
                bullet_copy["text"] = text[:MAX_BULLET_TEXT_CHARS].rstrip()
            trimmed_bullets.append(bullet_copy)
        sanitized["bullets"] = trimmed_bullets

    tags = sanitized.get("tags")
    if isinstance(tags, list):
        sanitized["tags"] = tags[:MAX_TAGS]

    return sanitized


def apply_insight_post_validation(
    insight: PlayerInsight,
    allowed_urls: Optional[set[str]] = None,
) -> PlayerInsight:
    """Apply post-validation rules to a synthesized player insight.

    Parameters
    ----------
    insight : PlayerInsight
        Raw insight from the LLM.
    allowed_urls : set[str], optional
        Snippet URLs that bullets must reference. When provided, bullets with
        URLs outside this set are removed.

    Returns
    -------
    PlayerInsight
        Validated and possibly downgraded insight.
    """
    fields_unknown = list(insight.fields_unknown)
    bullets = list(insight.bullets)

    if allowed_urls is not None:
        bullets = [bullet for bullet in bullets if bullet.source_url in allowed_urls]

    if insight.overall_confidence == Confidence.UNKNOWN and not insight.outlook_phrase:
        insight = insight.model_copy(update={"outlook_phrase": "Limited data available"})

    depth_role = insight.depth_role
    playing_time_tier = insight.playing_time_tier
    injury_risk = insight.injury_risk
    upside = insight.upside
    recovery_status = insight.recovery_status

    if depth_role != DepthRole.UNKNOWN and not bullets:
        depth_role = DepthRole.UNKNOWN
        if "depth_role" not in fields_unknown:
            fields_unknown.append("depth_role")

    if playing_time_tier != PlayingTimeTier.UNKNOWN and not bullets:
        playing_time_tier = PlayingTimeTier.UNKNOWN
        if "playing_time_tier" not in fields_unknown:
            fields_unknown.append("playing_time_tier")

    if injury_risk != RiskLevel.UNKNOWN and not bullets:
        injury_risk = RiskLevel.UNKNOWN
        if "injury_risk" not in fields_unknown:
            fields_unknown.append("injury_risk")

    if upside != RiskLevel.UNKNOWN and not bullets:
        upside = RiskLevel.UNKNOWN
        if "upside" not in fields_unknown:
            fields_unknown.append("upside")

    if recovery_status not in (RecoveryStatus.NA, RecoveryStatus.UNKNOWN) and not bullets:
        recovery_status = RecoveryStatus.UNKNOWN
        if "recovery_status" not in fields_unknown:
            fields_unknown.append("recovery_status")

    return insight.model_copy(
        update={
            "depth_role": depth_role,
            "playing_time_tier": playing_time_tier,
            "injury_risk": injury_risk,
            "upside": upside,
            "recovery_status": recovery_status,
            "fields_unknown": fields_unknown,
            "bullets": bullets,
        }
    )


def default_unknown_insight(search_queries_used: list[str]) -> PlayerInsight:
    """Build a fallback insight when synthesis cannot ground any fields.

    Parameters
    ----------
    search_queries_used : list[str]
        Queries that were attempted.

    Returns
    -------
    PlayerInsight
        Insight with all judgment fields set to unknown.
    """
    return PlayerInsight(
        outlook_phrase="Limited data available",
        summary="Insufficient recent reporting to form a grounded outlook.",
        depth_role=DepthRole.UNKNOWN,
        playing_time_tier=PlayingTimeTier.UNKNOWN,
        injury_risk=RiskLevel.UNKNOWN,
        upside=RiskLevel.UNKNOWN,
        recovery_status=RecoveryStatus.UNKNOWN,
        tags=[],
        overall_confidence=Confidence.UNKNOWN,
        fields_unknown=list(JUDGMENT_FIELDS),
        bullets=[],
        search_queries_used=search_queries_used,
        snippet_count=0,
    )
