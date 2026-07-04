"""Factory for insight search gateway providers."""

from __future__ import annotations

import os

from draft_buddy.data.insights.cse_gateway import GoogleCseGateway, SearchGateway
from draft_buddy.data.insights.valyu_search_gateway import ValyuSearchGateway

SUPPORTED_SEARCH_PROVIDERS = ("valyu", "google")


def resolve_search_provider(cli_provider: str | None = None) -> str:
    """Resolve the search provider from CLI flag or environment.

    Parameters
    ----------
    cli_provider : str, optional
        Explicit provider from ``--search-provider``.

    Returns
    -------
    str
        Provider name (``valyu`` or ``google``).
    """
    provider = (cli_provider or os.environ.get("INSIGHTS_SEARCH_PROVIDER") or "valyu").lower()
    if provider not in SUPPORTED_SEARCH_PROVIDERS:
        raise ValueError(
            f"Unsupported search provider '{provider}'. "
            f"Choose from: {', '.join(SUPPORTED_SEARCH_PROVIDERS)}"
        )
    return provider


def build_search_gateway(provider: str) -> SearchGateway:
    """Build a search gateway for the requested provider.

    Parameters
    ----------
    provider : str
        Provider name (``valyu`` or ``google``).

    Returns
    -------
    SearchGateway
        Configured search gateway.

    Raises
    ------
    ValueError
        When required environment variables are missing.
    """
    if provider == "valyu":
        api_key = os.environ.get("VALYU_API_KEY")
        if not api_key:
            raise ValueError("VALYU_API_KEY environment variable is required for Valyu search.")
        return ValyuSearchGateway(api_key=api_key)

    api_key = os.environ.get("GOOGLE_CSE_API_KEY")
    search_engine_id = os.environ.get("GOOGLE_CSE_ID")
    if not api_key or not search_engine_id:
        raise ValueError(
            "GOOGLE_CSE_API_KEY and GOOGLE_CSE_ID environment variables are required "
            "for Google CSE search."
        )
    return GoogleCseGateway(api_key=api_key, search_engine_id=search_engine_id)
