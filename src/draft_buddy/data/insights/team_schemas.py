"""Pydantic schemas for offline NFL team outlook enrichment."""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from draft_buddy.data.insights.schemas import Confidence, EvidenceBullet


class TeamTier(str, Enum):
    """Relative ability tier for one side of the ball."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class ScheduleHardness(str, Enum):
    """Relative strength-of-schedule tier."""

    HARD = "hard"
    AVERAGE = "average"
    EASY = "easy"
    UNKNOWN = "unknown"


class TeamOutlook(BaseModel):
    """Structured fantasy-relevant outlook record for one NFL team."""

    team_abbr: str
    season: int
    outlook_summary: str
    offense_tier: TeamTier
    offense_notes: str
    defense_tier: TeamTier
    defense_notes: str
    schedule_hardness: ScheduleHardness
    schedule_notes: str
    key_storylines: list[str] = Field(default_factory=list, max_length=4)
    evidence_as_of: Optional[str] = None
    overall_confidence: Confidence
    fields_unknown: list[str] = Field(default_factory=list)
    bullets: list[EvidenceBullet] = Field(default_factory=list, max_length=4)
    search_queries_used: list[str] = Field(default_factory=list)
    snippet_count: int = 0

    @field_validator("outlook_summary")
    @classmethod
    def validate_outlook_summary_sentence_count(cls, value: str) -> str:
        """Enforce at most two sentences in the outlook summary."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("outlook_summary must not be empty")
        sentences = [part for part in re.split(r"[.!?]+", stripped) if part.strip()]
        if len(sentences) > MAX_OUTLOOK_SENTENCES:
            raise ValueError(
                f"outlook_summary must be at most {MAX_OUTLOOK_SENTENCES} sentences, "
                f"got {len(sentences)}"
            )
        return stripped

    @field_validator("key_storylines")
    @classmethod
    def validate_key_storyline_count(cls, value: list[str]) -> list[str]:
        """Enforce at most four key storylines."""
        if len(value) > MAX_KEY_STORYLINES:
            raise ValueError(
                f"key_storylines must contain at most {MAX_KEY_STORYLINES} items, got {len(value)}"
            )
        return value

    @field_validator("bullets")
    @classmethod
    def validate_bullet_count(cls, value: list[EvidenceBullet]) -> list[EvidenceBullet]:
        """Enforce at most four evidence bullets."""
        if len(value) > MAX_BULLETS:
            raise ValueError(f"bullets must contain at most {MAX_BULLETS} items, got {len(value)}")
        return value


class TeamOutlooksFile(BaseModel):
    """Top-level container for all team outlooks for one draft year."""

    schema_version: str = "1.0"
    draft_year: int
    generated_at: datetime
    model: str
    teams: dict[str, TeamOutlook]


JUDGMENT_FIELDS = (
    "offense_tier",
    "defense_tier",
    "schedule_hardness",
)

MAX_OUTLOOK_SENTENCES = 2
MAX_KEY_STORYLINES = 4
MAX_BULLETS = 4


def sanitize_team_synthesis_payload(payload: dict) -> dict:
    """Normalize raw LLM JSON so it satisfies ``TeamOutlook`` validators.

    Parameters
    ----------
    payload : dict
        Parsed JSON from structured model output.

    Returns
    -------
    dict
        Payload safe to pass to ``TeamOutlook.model_validate``.
    """
    sanitized = dict(payload)

    outlook_summary = str(sanitized.get("outlook_summary", "")).strip()
    if outlook_summary:
        sanitized["outlook_summary"] = _truncate_to_sentences(
            outlook_summary, MAX_OUTLOOK_SENTENCES
        )

    key_storylines = sanitized.get("key_storylines")
    if isinstance(key_storylines, list):
        sanitized["key_storylines"] = [
            str(item).strip() for item in key_storylines[:MAX_KEY_STORYLINES] if str(item).strip()
        ]

    bullets = sanitized.get("bullets")
    if isinstance(bullets, list):
        sanitized["bullets"] = [
            dict(bullet) for bullet in bullets[:MAX_BULLETS] if isinstance(bullet, dict)
        ]

    return sanitized


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


def apply_team_outlook_post_validation(
    outlook: TeamOutlook,
    allowed_urls: Optional[set[str]] = None,
) -> TeamOutlook:
    """Apply post-validation rules to a synthesized team outlook.

    Parameters
    ----------
    outlook : TeamOutlook
        Raw outlook from the LLM.
    allowed_urls : set[str], optional
        Snippet URLs that bullets must reference. When provided, bullets with
        URLs outside this set are removed.

    Returns
    -------
    TeamOutlook
        Validated and possibly downgraded outlook.
    """
    fields_unknown = list(outlook.fields_unknown)
    bullets = list(outlook.bullets)

    if allowed_urls is not None:
        bullets = [bullet for bullet in bullets if bullet.source_url in allowed_urls]

    offense_tier = outlook.offense_tier
    defense_tier = outlook.defense_tier
    schedule_hardness = outlook.schedule_hardness

    if offense_tier != TeamTier.UNKNOWN and not bullets:
        offense_tier = TeamTier.UNKNOWN
        if "offense_tier" not in fields_unknown:
            fields_unknown.append("offense_tier")

    if defense_tier != TeamTier.UNKNOWN and not bullets:
        defense_tier = TeamTier.UNKNOWN
        if "defense_tier" not in fields_unknown:
            fields_unknown.append("defense_tier")

    if schedule_hardness != ScheduleHardness.UNKNOWN and not bullets:
        schedule_hardness = ScheduleHardness.UNKNOWN
        if "schedule_hardness" not in fields_unknown:
            fields_unknown.append("schedule_hardness")

    evidence_as_of = outlook.evidence_as_of
    if not evidence_as_of:
        evidence_as_of = _newest_bullet_date(bullets)

    return outlook.model_copy(
        update={
            "offense_tier": offense_tier,
            "defense_tier": defense_tier,
            "schedule_hardness": schedule_hardness,
            "fields_unknown": fields_unknown,
            "bullets": bullets,
            "evidence_as_of": evidence_as_of,
        }
    )


def _newest_bullet_date(bullets: list[EvidenceBullet]) -> Optional[str]:
    """Return the newest ISO date among evidence bullets, if any."""
    dates = [str(bullet.published_date)[:10] for bullet in bullets if bullet.published_date]
    if not dates:
        return None
    return max(dates)


def default_unknown_team_outlook(
    team_abbr: str,
    season: int,
    search_queries_used: list[str],
) -> TeamOutlook:
    """Build a fallback team outlook when synthesis cannot ground any fields.

    Parameters
    ----------
    team_abbr : str
        NFL team abbreviation.
    season : int
        Draft season year.
    search_queries_used : list[str]
        Queries that were attempted.

    Returns
    -------
    TeamOutlook
        Outlook with all judgment fields set to unknown.
    """
    return TeamOutlook(
        team_abbr=team_abbr,
        season=season,
        outlook_summary="Limited data available",
        offense_tier=TeamTier.UNKNOWN,
        offense_notes="",
        defense_tier=TeamTier.UNKNOWN,
        defense_notes="",
        schedule_hardness=ScheduleHardness.UNKNOWN,
        schedule_notes="",
        key_storylines=[],
        evidence_as_of=None,
        overall_confidence=Confidence.UNKNOWN,
        fields_unknown=list(JUDGMENT_FIELDS),
        bullets=[],
        search_queries_used=search_queries_used,
        snippet_count=0,
    )
