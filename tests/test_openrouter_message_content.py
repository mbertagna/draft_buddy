"""Tests for OpenRouter message content extraction."""

from __future__ import annotations

from draft_buddy.llm.openrouter_client import _extract_json_from_text, _extract_message_content


def test_extract_message_content_prefers_content_field() -> None:
    """Verify primary content field is returned when present."""
    message = {"content": '{"answer": "yes"}'}
    assert _extract_message_content(message) == '{"answer": "yes"}'


def test_extract_message_content_falls_back_to_reasoning_json() -> None:
    """Verify JSON can be recovered from reasoning text when content is empty."""
    message = {
        "content": "",
        "reasoning": 'Analysis...\n{"recommended_name": "RB One", "recommended_player_id": 2}',
    }
    extracted = _extract_message_content(message)
    assert '"recommended_name": "RB One"' in extracted


def test_extract_json_from_text_finds_embedded_object() -> None:
    """Verify JSON object extraction from surrounding prose."""
    text = "Here is the result: {\"confidence\": \"high\"} done."
    extracted = _extract_json_from_text(text)
    assert extracted == '{"confidence": "high"}'
