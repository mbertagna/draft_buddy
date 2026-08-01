"""OpenRouter chat-completions client for strict structured JSON output."""

from __future__ import annotations

import json
from typing import Any, Type

import httpx
from pydantic import BaseModel

from draft_buddy.llm.json_schema import pydantic_model_to_strict_json_schema

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 2048
ADVISOR_MAX_TOKENS = 2048
SYNTHESIS_MAX_TOKENS = 2048


class OpenRouterClient:
    """Call OpenRouter with strict JSON-schema constrained responses."""

    def __init__(self, api_key: str, model: str) -> None:
        """
        Parameters
        ----------
        api_key : str
            OpenRouter API key.
        model : str
            OpenRouter model slug.
        """
        self._api_key = api_key
        self._model = model

    @property
    def model(self) -> str:
        """Return the configured model slug."""
        return self._model

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
        schema_name: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        reasoning_effort: str | None = None,
        use_response_healing: bool = True,
    ) -> dict[str, Any]:
        """Generate JSON matching a Pydantic schema via OpenRouter.

        Parameters
        ----------
        system_prompt : str
            System instructions.
        user_prompt : str
            User message content.
        response_model : Type[BaseModel]
            Pydantic model defining the response shape.
        schema_name : str
            Schema name for OpenRouter ``json_schema``.
        temperature : float, optional
            Sampling temperature.
        max_tokens : int, optional
            Upper bound on generated tokens (lower values reduce OpenRouter credit holds).
        reasoning_effort : str, optional
            OpenRouter reasoning effort (``none`` disables thinking on DeepSeek V4).
        use_response_healing : bool, optional
            Whether to enable OpenRouter response-healing plugin.

        Returns
        -------
        dict[str, Any]
            Parsed JSON object from the model response.

        Raises
        ------
        RuntimeError
            When the HTTP request fails or the response is not valid JSON.
        """
        payload, _raw = self.generate_structured_with_raw(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=response_model,
            schema_name=schema_name,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            use_response_healing=use_response_healing,
        )
        return payload

    def generate_structured_with_raw(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
        schema_name: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        reasoning_effort: str | None = None,
        use_response_healing: bool = True,
    ) -> tuple[dict[str, Any], str]:
        """Generate JSON and return both parsed payload and raw response text.

        Parameters
        ----------
        system_prompt : str
            System instructions.
        user_prompt : str
            User message content.
        response_model : Type[BaseModel]
            Pydantic model defining the response shape.
        schema_name : str
            Schema name for OpenRouter ``json_schema``.
        temperature : float, optional
            Sampling temperature.
        max_tokens : int, optional
            Upper bound on generated tokens.
        reasoning_effort : str, optional
            OpenRouter reasoning effort.
        use_response_healing : bool, optional
            Whether to enable OpenRouter response-healing plugin.

        Returns
        -------
        tuple[dict[str, Any], str]
            Parsed JSON object and raw assistant content.

        Raises
        ------
        RuntimeError
            When the HTTP request fails or the response is not valid JSON.
        """
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": pydantic_model_to_strict_json_schema(response_model),
                },
            },
            "provider": {"require_parameters": True},
        }
        if reasoning_effort is not None:
            payload["reasoning"] = {"effort": reasoning_effort}
        if use_response_healing:
            payload["plugins"] = [{"id": "response-healing"}]
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=120.0) as client:
            response = client.post(OPENROUTER_CHAT_URL, headers=headers, json=payload)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as error:
                detail = _extract_error_detail(response)
                raise RuntimeError(
                    f"OpenRouter request failed ({response.status_code}): {detail}"
                ) from error
            body = response.json()

        try:
            message = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(f"Unexpected OpenRouter response shape: {body}") from error

        content = _extract_message_content(message)
        if not content:
            raise RuntimeError("OpenRouter returned empty content.")

        try:
            return json.loads(content), content
        except json.JSONDecodeError as error:
            raise RuntimeError(f"OpenRouter returned invalid JSON: {content}") from error


def _extract_message_content(message: dict[str, Any]) -> str:
    """Return JSON text from an OpenRouter assistant message."""
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()

    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        extracted = _extract_json_from_text(reasoning)
        if extracted:
            return extracted

    reasoning_details = message.get("reasoning_details")
    if isinstance(reasoning_details, list):
        combined = "\n".join(
            str(item.get("text") or item.get("content") or "")
            for item in reasoning_details
            if isinstance(item, dict)
        ).strip()
        extracted = _extract_json_from_text(combined)
        if extracted:
            return extracted

    return ""


def _extract_json_from_text(text: str) -> str:
    """Extract the first JSON object from free-form model text."""
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return ""


def _extract_error_detail(response: httpx.Response) -> str:
    """Return a concise error message from an OpenRouter error response."""
    try:
        payload = response.json()
    except json.JSONDecodeError:
        return response.text[:500] or response.reason_phrase
    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if message:
            return str(message)
    return response.text[:500] or response.reason_phrase
