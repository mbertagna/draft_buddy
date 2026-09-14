"""Tests for insight search provider factory."""

from __future__ import annotations

import pytest

from draft_buddy.data.insights.search_factory import build_search_gateway
from draft_buddy.data.insights.tavily_search_gateway import TavilySearchGateway
from draft_buddy.data.insights.valyu_search_gateway import ValyuSearchGateway


def test_build_search_gateway_defaults_to_tavily(monkeypatch) -> None:
    """Verify Tavily is used when no search provider is configured."""
    monkeypatch.delenv("INSIGHTS_SEARCH_PROVIDER", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")

    gateway = build_search_gateway()

    assert isinstance(gateway, TavilySearchGateway)


def test_build_search_gateway_uses_tavily_env(monkeypatch) -> None:
    """Verify INSIGHTS_SEARCH_PROVIDER=tavily builds the Tavily gateway."""
    monkeypatch.setenv("INSIGHTS_SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")

    gateway = build_search_gateway()

    assert isinstance(gateway, TavilySearchGateway)


def test_build_search_gateway_uses_valyu_override(monkeypatch) -> None:
    """Verify an explicit valyu provider builds the Valyu gateway."""
    monkeypatch.setenv("INSIGHTS_SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("VALYU_API_KEY", "valyu-test")

    gateway = build_search_gateway("valyu")

    assert isinstance(gateway, ValyuSearchGateway)


def test_build_search_gateway_rejects_unknown_provider(monkeypatch) -> None:
    """Verify unknown search providers raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported search provider"):
        build_search_gateway("serper")


def test_build_search_gateway_requires_tavily_api_key(monkeypatch) -> None:
    """Verify missing Tavily keys raise a clear error."""
    monkeypatch.delenv("INSIGHTS_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with pytest.raises(ValueError, match="TAVILY_API_KEY"):
        build_search_gateway()


def test_build_search_gateway_requires_valyu_api_key(monkeypatch) -> None:
    """Verify missing Valyu keys raise a clear error."""
    monkeypatch.delenv("VALYU_API_KEY", raising=False)

    with pytest.raises(ValueError, match="VALYU_API_KEY"):
        build_search_gateway("valyu")
