"""Tests for OpenRouter structured output client."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel

from draft_buddy.llm.openrouter_client import OpenRouterClient


class SampleResponse(BaseModel):
    """Sample structured response model."""

    answer: str


def test_openrouter_client_posts_structured_request(monkeypatch) -> None:
    """Verify OpenRouter client sends strict json_schema request payload."""
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [
                    {"message": {"content": json.dumps({"answer": "yes"})}},
                ]
            }

    def fake_post(self, url: str, *, headers: dict, json: dict) -> FakeResponse:
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr(httpx.Client, "post", fake_post)

    client = OpenRouterClient(api_key="test-key", model="deepseek/deepseek-v4-flash")
    payload = client.generate_structured(
        system_prompt="system",
        user_prompt="user",
        response_model=SampleResponse,
        schema_name="sample_response",
    )

    assert payload == {"answer": "yes"}
    assert captured["url"].endswith("/chat/completions")
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    request_json = captured["json"]
    assert request_json["model"] == "deepseek/deepseek-v4-flash"
    assert request_json["max_tokens"] == 2048
    assert request_json["response_format"]["type"] == "json_schema"
    assert "plugins" in request_json
    assert request_json["response_format"]["json_schema"]["strict"] is True
    assert request_json["provider"] == {"require_parameters": True}


def test_openrouter_client_raises_on_invalid_json(monkeypatch) -> None:
    """Verify invalid JSON content raises a runtime error."""

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "not-json"}}]}

    monkeypatch.setattr(httpx.Client, "post", lambda *args, **kwargs: FakeResponse())

    client = OpenRouterClient(api_key="test-key", model="deepseek/deepseek-v4-flash")
    with pytest.raises(RuntimeError, match="invalid JSON"):
        client.generate_structured(
            system_prompt="system",
            user_prompt="user",
            response_model=SampleResponse,
            schema_name="sample_response",
        )
