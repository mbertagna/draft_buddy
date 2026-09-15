"""Market data gateways for ADP, ECR, and expert-ranking metadata."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd
import requests

FANTASYPROS_BASE_URL = "https://api.fantasypros.com/public/v2/json"
FANTASYPROS_DEFAULT_TIMEOUT_SECONDS = 20
FANTASYPROS_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


class MarketDataUnavailable(RuntimeError):
    """Raised when market data cannot be fetched or loaded from cache."""


@dataclass(frozen=True, slots=True)
class MarketRankingRequest:
    """FantasyPros ranking request metadata.

    Parameters
    ----------
    season : int
        NFL season year.
    scoring : str
        FantasyPros scoring code (``STD``, ``HALF``, or ``PPR``).
    ranking_type : str
        FantasyPros ranking type such as ``ADP`` or ``draft``.
    position : str
        Position filter such as ``ALL``, ``QB``, ``RB``, ``WR``, or ``TE``.
    week : int
        Week value sent to FantasyPros. Preseason rankings use ``0``.
    """

    season: int
    scoring: str
    ranking_type: str = "ADP"
    position: str = "ALL"
    week: int = 0

    @property
    def cache_key(self) -> str:
        """Return a filesystem-safe cache key for this request."""
        parts = [
            str(self.season),
            self.scoring.upper(),
            self.ranking_type.upper(),
            self.position.upper(),
            str(self.week),
        ]
        return "_".join(parts)


@dataclass(frozen=True, slots=True)
class MarketRankingRow:
    """Normalized player ranking row from a market source.

    Parameters
    ----------
    source : str
        Market source identifier.
    source_player_id : str or None
        Provider-specific player id.
    player_name : str
        Player display name.
    position : str
        Position code.
    team : str or None
        NFL team abbreviation.
    adp : float or None
        Average draft position when this row is an ADP feed.
    ecr : float or None
        Expert consensus rank when this row is an ECR feed.
    rank_min : float or None
        Best expert rank when present.
    rank_max : float or None
        Worst expert rank when present.
    rank_ave : float or None
        Average expert rank when present.
    rank_std : float or None
        Expert ranking standard deviation when present.
    tier : int or None
        Provider tier.
    total_experts : int or None
        Number of contributing experts.
    last_updated : str or None
        Provider update marker.
    fetched_at : str
        UTC ISO timestamp of the API response cache.
    """

    source: str
    source_player_id: Optional[str]
    player_name: str
    position: str
    team: Optional[str]
    adp: Optional[float]
    ecr: Optional[float]
    rank_min: Optional[float]
    rank_max: Optional[float]
    rank_ave: Optional[float]
    rank_std: Optional[float]
    tier: Optional[int]
    total_experts: Optional[int]
    last_updated: Optional[str]
    fetched_at: str


class FantasyProsGateway:
    """Fetch and normalize FantasyPros consensus ranking data with cache fallback."""

    def __init__(
        self,
        cache_dir: str,
        api_key: Optional[str] = None,
        session: Optional[requests.Session] = None,
        base_url: str = FANTASYPROS_BASE_URL,
        timeout_seconds: int = FANTASYPROS_DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = 2,
    ) -> None:
        """Initialize the gateway.

        Parameters
        ----------
        cache_dir : str
            Directory where raw FantasyPros API responses are cached.
        api_key : str, optional
            FantasyPros API key. Defaults to ``FANTASYPROS_API_KEY``.
        session : requests.Session, optional
            Injected HTTP session for tests.
        base_url : str, optional
            FantasyPros API base URL.
        timeout_seconds : int, optional
            Request timeout.
        max_retries : int, optional
            Number of retry attempts for transient failures.
        """
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._api_key = api_key if api_key is not None else os.environ.get("FANTASYPROS_API_KEY", "")
        self._session = session or requests.Session()
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries

    def fetch_consensus_rankings(
        self,
        request: MarketRankingRequest,
        *,
        allow_stale_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch FantasyPros consensus rankings as normalized records.

        Parameters
        ----------
        request : MarketRankingRequest
            Request metadata.
        allow_stale_cache : bool, optional
            When ``True``, latest cached response is used if the API is
            unavailable or the key is missing.

        Returns
        -------
        pd.DataFrame
            Normalized ranking rows with provenance columns.
        """
        fetched_at = datetime.now(timezone.utc).isoformat()
        try:
            payload = self._fetch_json(request)
            self._write_cache(request, payload, fetched_at)
        except Exception as error:
            if not allow_stale_cache:
                raise MarketDataUnavailable(str(error)) from error
            payload, fetched_at = self._read_latest_cache(request, original_error=error)
        return rankings_payload_to_dataframe(payload, request.ranking_type, fetched_at)

    def _fetch_json(self, request: MarketRankingRequest) -> Dict[str, Any]:
        """Fetch one FantasyPros JSON payload."""
        if not self._api_key:
            raise MarketDataUnavailable("FANTASYPROS_API_KEY is not configured.")
        url = f"{self._base_url}/nfl/{request.season}/consensus-rankings"
        params = {
            "scoring": request.scoring.upper(),
            "type": request.ranking_type,
            "position": request.position.upper(),
            "week": request.week,
        }
        headers = {"x-api-key": self._api_key}
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self._timeout_seconds,
                )
                if response.status_code in FANTASYPROS_RETRY_STATUS_CODES:
                    response.raise_for_status()
                response.raise_for_status()
                return response.json()
            except requests.RequestException as error:
                last_error = error
                if attempt >= self._max_retries:
                    break
                time.sleep(0.25 * (2**attempt))
        raise MarketDataUnavailable("FantasyPros rankings request failed.") from last_error

    def _write_cache(
        self,
        request: MarketRankingRequest,
        payload: Dict[str, Any],
        fetched_at: str,
    ) -> None:
        """Write immutable payload and latest pointer files."""
        cache_key = request.cache_key.lower()
        timestamp = datetime.fromisoformat(fetched_at).strftime("%Y%m%dT%H%M%SZ")
        snapshot_path = self._cache_dir / f"{cache_key}_{timestamp}.json"
        latest_path = self._cache_dir / f"{cache_key}_latest.json"
        envelope = {"fetched_at": fetched_at, "request": asdict(request), "payload": payload}
        snapshot_path.write_text(json.dumps(envelope, indent=2, sort_keys=True), encoding="utf-8")
        latest_path.write_text(json.dumps(envelope, indent=2, sort_keys=True), encoding="utf-8")

    def _read_latest_cache(
        self,
        request: MarketRankingRequest,
        original_error: Exception,
    ) -> tuple[Dict[str, Any], str]:
        """Load latest cached payload for a request or raise a wrapped error."""
        latest_path = self._cache_dir / f"{request.cache_key.lower()}_latest.json"
        if not latest_path.is_file():
            raise MarketDataUnavailable(
                f"No cached FantasyPros response for {request.cache_key}."
            ) from original_error
        envelope = json.loads(latest_path.read_text(encoding="utf-8"))
        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            raise MarketDataUnavailable(f"Invalid FantasyPros cache: {latest_path}") from original_error
        return payload, str(envelope.get("fetched_at", ""))


def rankings_payload_to_dataframe(
    payload: Dict[str, Any],
    ranking_type: str,
    fetched_at: str,
) -> pd.DataFrame:
    """Normalize a FantasyPros consensus ranking payload.

    Parameters
    ----------
    payload : dict
        Raw response body.
    ranking_type : str
        Source ranking type used to distinguish ADP from ECR.
    fetched_at : str
        UTC timestamp associated with the payload.

    Returns
    -------
    pd.DataFrame
        One row per player with normalized market fields.
    """
    players = payload.get("players", [])
    if not isinstance(players, list):
        raise MarketDataUnavailable("FantasyPros payload has no players list.")

    rows = [
        _normalize_player_row(
            player,
            ranking_type=ranking_type,
            total_experts=_coerce_int(payload.get("total_experts")),
            last_updated=_optional_str(payload.get("last_updated")),
            fetched_at=fetched_at,
        )
        for player in players
        if isinstance(player, dict)
    ]
    return pd.DataFrame([asdict(row) for row in rows])


def _normalize_player_row(
    player: Dict[str, Any],
    *,
    ranking_type: str,
    total_experts: Optional[int],
    last_updated: Optional[str],
    fetched_at: str,
) -> MarketRankingRow:
    """Normalize one FantasyPros player ranking object."""
    rank_ecr = _coerce_float(player.get("rank_ecr"))
    rank_ave = _coerce_float(player.get("rank_ave"))
    rank_value = rank_ave if rank_ave is not None else rank_ecr
    is_adp = ranking_type.upper() == "ADP"
    return MarketRankingRow(
        source=f"fantasypros_{ranking_type.lower()}",
        source_player_id=_optional_str(player.get("player_id")),
        player_name=str(player.get("player_name", "")).strip(),
        position=str(player.get("player_position_id") or player.get("position") or "").upper(),
        team=_optional_str(player.get("player_team_id") or player.get("team")),
        adp=rank_value if is_adp else None,
        ecr=None if is_adp else rank_ecr,
        rank_min=_coerce_float(player.get("rank_min")),
        rank_max=_coerce_float(player.get("rank_max")),
        rank_ave=rank_ave,
        rank_std=_coerce_float(player.get("rank_std") or player.get("sd")),
        tier=_coerce_int(player.get("tier")),
        total_experts=total_experts,
        last_updated=last_updated,
        fetched_at=fetched_at,
    )


def consensus_adp_from_sources(source_frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Combine source ADP rows into a conservative player-level consensus.

    Parameters
    ----------
    source_frames : Iterable[pandas.DataFrame]
        Normalized market source frames containing ``player_name``,
        ``position``, ``team``, ``adp``, and ``source``.

    Returns
    -------
    pd.DataFrame
        Player-level consensus rows using median ADP and source dispersion.
    """
    frames = [frame.copy() for frame in source_frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame(
            columns=[
                "player_name",
                "position",
                "team",
                "consensus_adp",
                "market_source_count",
                "market_adp_dispersion",
                "market_sources",
            ]
        )

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined[pd.to_numeric(combined["adp"], errors="coerce").notna()].copy()
    combined["adp"] = pd.to_numeric(combined["adp"], errors="coerce")
    if combined.empty:
        return pd.DataFrame()

    group_keys = ["player_name", "position", "team"]
    return (
        combined.groupby(group_keys, dropna=False)
        .agg(
            consensus_adp=("adp", "median"),
            market_source_count=("source", "nunique"),
            market_adp_dispersion=("adp", _median_absolute_deviation),
            market_sources=("source", lambda values: ",".join(sorted(set(map(str, values))))),
        )
        .reset_index()
    )


def _median_absolute_deviation(values: pd.Series) -> float:
    """Return median absolute deviation for a numeric series."""
    median_value = float(values.median())
    return float((values - median_value).abs().median())


def _coerce_float(value: Any) -> Optional[float]:
    """Return a finite float or ``None``."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric


def _coerce_int(value: Any) -> Optional[int]:
    """Return an integer or ``None``."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> Optional[str]:
    """Return a stripped string or ``None`` for empty values."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None
