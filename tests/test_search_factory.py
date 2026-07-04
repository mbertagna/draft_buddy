"""Tests for search provider factory."""

from __future__ import annotations

import pytest

from draft_buddy.data.insights.cse_gateway import GoogleCseGateway
from draft_buddy.data.insights.search_factory import build_search_gateway, resolve_search_provider
from draft_buddy.data.insights.valyu_search_gateway import ValyuSearchGateway


def test_resolve_search_provider_defaults_to_valyu(monkeypatch) -> None:
    """Verify default provider is valyu when unset."""
    monkeypatch.delenv("INSIGHTS_SEARCH_PROVIDER", raising=False)
    assert resolve_search_provider() == "valyu"


def test_resolve_search_provider_honors_env(monkeypatch) -> None:
    """Verify INSIGHTS_SEARCH_PROVIDER overrides the default."""
    monkeypatch.setenv("INSIGHTS_SEARCH_PROVIDER", "google")
    assert resolve_search_provider() == "google"


def test_build_search_gateway_valyu(monkeypatch) -> None:
    """Verify Valyu gateway is built when API key is present."""
    monkeypatch.setenv("VALYU_API_KEY", "valyu-test-key")
    gateway = build_search_gateway("valyu")
    assert isinstance(gateway, ValyuSearchGateway)


def test_build_search_gateway_google(monkeypatch) -> None:
    """Verify Google gateway is built when CSE env vars are present."""
    monkeypatch.setenv("GOOGLE_CSE_API_KEY", "google-key")
    monkeypatch.setenv("GOOGLE_CSE_ID", "cx-id")
    gateway = build_search_gateway("google")
    assert isinstance(gateway, GoogleCseGateway)


def test_build_search_gateway_valyu_requires_api_key(monkeypatch) -> None:
    """Verify missing Valyu API key raises a clear error."""
    monkeypatch.delenv("VALYU_API_KEY", raising=False)
    with pytest.raises(ValueError, match="VALYU_API_KEY"):
        build_search_gateway("valyu")
