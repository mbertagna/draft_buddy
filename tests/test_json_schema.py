"""Tests for strict JSON schema conversion."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from draft_buddy.llm.json_schema import pydantic_model_to_strict_json_schema


class SampleRole(str, Enum):
    """Sample enum for schema conversion."""

    STARTER = "starter"
    BACKUP = "backup"


class SampleNested(BaseModel):
    """Nested object for schema conversion."""

    name: str
    role: SampleRole


class SampleModel(BaseModel):
    """Top-level schema for conversion tests."""

    title: str
    tags: list[str] = Field(default_factory=list, max_length=2)
    nested: SampleNested


def test_pydantic_model_to_strict_json_schema_sets_additional_properties_false() -> None:
    """Verify strict schema conversion marks objects as closed."""
    schema = pydantic_model_to_strict_json_schema(SampleModel)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["nested"]["additionalProperties"] is False
