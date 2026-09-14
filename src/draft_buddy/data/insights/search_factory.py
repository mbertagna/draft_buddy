"""Factory for insight search gateways."""

from __future__ import annotations

import os
from enum import Enum
from typing import Optional

from draft_buddy.data.insights.search_gateway import SearchGateway
from draft_buddy.data.insights.tavily_search_gateway import build_tavily_search_gateway
from draft_buddy.data.insights.valyu_search_gateway import build_valyu_search_gateway


class SearchProvider(str, Enum):
    """Supported insight search backends."""

    TAVILY = "tavily"
    VALYU = "valyu"


DEFAULT_SEARCH_PROVIDER = SearchProvider.TAVILY
_PROVIDER_ENV = "INSIGHTS_SEARCH_PROVIDER"


def resolve_search_provider(cli_provider: Optional[str] = None) -> SearchProvider:
    """Resolve the search provider from CLI flag or environment.

    Parameters
    ----------
    cli_provider : str, optional
        Explicit provider from ``--provider``.

    Returns
    -------
    SearchProvider
        Resolved search backend.

    Raises
    ------
    ValueError
        When the provider name is unsupported.
    """
    raw = (cli_provider or os.environ.get(_PROVIDER_ENV) or DEFAULT_SEARCH_PROVIDER.value)
    name = raw.strip().lower()
    try:
        return SearchProvider(name)
    except ValueError as error:
        supported = ", ".join(provider.value for provider in SearchProvider)
        raise ValueError(
            f"Unsupported search provider '{raw}'. Choose from: {supported}"
        ) from error


def build_search_gateway(provider: Optional[str] = None) -> SearchGateway:
    """Build a search gateway for the requested provider.

    Parameters
    ----------
    provider : str, optional
        Explicit provider override (``tavily`` or ``valyu``).

    Returns
    -------
    SearchGateway
        Configured search gateway.

    Raises
    ------
    ValueError
        When the provider is unsupported or the API key is missing.
    """
    resolved = resolve_search_provider(provider)
    if resolved == SearchProvider.TAVILY:
        return build_tavily_search_gateway()
    return build_valyu_search_gateway()
