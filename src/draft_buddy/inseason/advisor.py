"""Optional LLM writeup for already-scored trades."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

TRADE_SYSTEM_PROMPT = """You are a fantasy football trade advisor.
The numeric verdict is already decided and must not be changed.
Return JSON only with keys defensive_breakdown and trade_pitch.
Keep each field to at most 4 sentences.
"""


class TradeAdvisorWriter:
    """Generate a defensive read and pitch after numeric evaluation.

    Parameters
    ----------
    gateway : DraftAdvisorGateway or None
        LLM gateway exposing ``fetch_payload``. Disabled when omitted.
    """

    def __init__(self, gateway: Optional[Any] = None) -> None:
        self._gateway = gateway

    def describe(self, context: str) -> Dict[str, str]:
        """Return LLM copy, or empty strings when the gateway is unavailable.

        Parameters
        ----------
        context : str
            User prompt describing the scored trade.

        Returns
        -------
        dict
            ``defensive_breakdown`` and ``trade_pitch``.
        """
        empty = {"defensive_breakdown": "", "trade_pitch": ""}
        if self._gateway is None:
            return empty
        try:
            payload, _raw = self._gateway.fetch_payload(TRADE_SYSTEM_PROMPT, context)
        except Exception:
            logger.exception("Trade advisor writeup failed")
            return empty
        if not isinstance(payload, dict):
            return empty
        return {
            "defensive_breakdown": str(payload.get("defensive_breakdown") or ""),
            "trade_pitch": str(payload.get("trade_pitch") or ""),
        }
