"""Static player valuation metrics derived from history and market price."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

DRAFTABLE_POSITIONS = ("QB", "RB", "WR", "TE")
DEFAULT_RESIDUAL_SCALE_POINTS = 4.0
LOOKBACK_GAMES_ESTIMATE = 34.0


def add_static_player_value_metrics(players_df: pd.DataFrame) -> pd.DataFrame:
    """Add transparent static valuation metrics to player rows.

    Parameters
    ----------
    players_df : pandas.DataFrame
        Player rows with ``position``, ``projected_points``, ``adp``, and
        optionally ``games_played_frac``.

    Returns
    -------
    pandas.DataFrame
        Copy of ``players_df`` with valuation columns appended.
    """
    if players_df.empty:
        return _with_empty_metric_columns(players_df.copy())

    df = players_df.copy()
    df["historical_weekly_median"] = pd.to_numeric(
        df.get("projected_points"), errors="coerce"
    ).fillna(0.0)
    df["projected_points_p50"] = df["historical_weekly_median"]
    df["effective_sample_size"] = df.get("games_played_frac", pd.Series(index=df.index)).apply(
        estimate_effective_sample_size
    )
    df["data_confidence"] = df.get("games_played_frac", pd.Series(index=df.index)).apply(
        estimate_data_confidence
    )
    df["market_implied_points"] = _market_implied_points(df)
    df["projection_edge_points"] = df["historical_weekly_median"] - df["market_implied_points"]
    df["market_premium_points"] = df["market_implied_points"] - df["historical_weekly_median"]

    residual_scale = _position_residual_scales(df)
    df["projection_edge_z"] = df.apply(
        lambda row: _safe_divide(
            row["projection_edge_points"],
            residual_scale.get(str(row.get("position", "")).upper(), DEFAULT_RESIDUAL_SCALE_POINTS),
        ),
        axis=1,
    )
    df["agreement_score"] = df["projection_edge_z"].apply(agreement_score_from_z)
    interval_width = df.apply(_projection_interval_width, axis=1)
    df["projected_points_p20"] = (df["projected_points_p50"] - interval_width).clip(lower=0.0)
    df["projected_points_p80"] = df["projected_points_p50"] + interval_width
    df["projection_basis"] = df.apply(classify_projection_basis, axis=1)
    df["stability_score"] = df.apply(calculate_stability_score, axis=1)
    df["risk_tier"] = df["stability_score"].apply(classify_risk_tier)
    df["player_archetype"] = df.apply(classify_player_archetype, axis=1)
    return df


def estimate_effective_sample_size(games_played_frac: Any) -> float:
    """Estimate observed games from a games-played fraction.

    Parameters
    ----------
    games_played_frac : Any
        Numeric fraction or ``"R"`` for rookies.

    Returns
    -------
    float
        Approximate historical game sample size.
    """
    fraction = _numeric_games_played_fraction(games_played_frac)
    if fraction is None:
        return 0.0
    return float(max(0.0, min(LOOKBACK_GAMES_ESTIMATE, fraction * LOOKBACK_GAMES_ESTIMATE)))


def estimate_data_confidence(games_played_frac: Any) -> float:
    """Estimate confidence in the independent historical signal.

    Parameters
    ----------
    games_played_frac : Any
        Numeric fraction or ``"R"`` for rookies.

    Returns
    -------
    float
        Confidence from 0 to 1.
    """
    fraction = _numeric_games_played_fraction(games_played_frac)
    if fraction is None:
        return 0.35
    return float(max(0.15, min(1.0, fraction)))


def agreement_score_from_z(z_score: float) -> float:
    """Return a 0-100 agreement score from a standardized model-market edge."""
    if not np.isfinite(z_score):
        return 0.0
    return float(100.0 * math.exp(-0.5 * z_score**2))


def classify_projection_basis(row: pd.Series) -> str:
    """Classify which signal is carrying a player projection."""
    games_played_frac = row.get("games_played_frac")
    has_adp = np.isfinite(_float_or_inf(row.get("adp")))
    data_confidence = float(row.get("data_confidence", 0.0))
    if games_played_frac == "R":
        return "rookie_market_imputed" if has_adp else "rookie_unknown"
    if data_confidence < 0.5 and has_adp:
        return "market_supported"
    if has_adp:
        return "blended"
    return "history_supported"


def calculate_stability_score(row: pd.Series) -> float:
    """Calculate a bounded stability score from agreement and data confidence."""
    agreement = float(row.get("agreement_score", 0.0))
    data_confidence = float(row.get("data_confidence", 0.0)) * 100.0
    market_dispersion = _float_or_none(row.get("market_adp_dispersion"))
    market_confidence = 75.0
    if market_dispersion is not None:
        market_confidence = max(0.0, min(100.0, 100.0 - market_dispersion * 5.0))
    score = 0.4 * agreement + 0.4 * data_confidence + 0.2 * market_confidence
    return float(max(0.0, min(100.0, score)))


def classify_risk_tier(stability_score: float) -> str:
    """Return a simple risk tier from a stability score."""
    if stability_score >= 75.0:
        return "low"
    if stability_score >= 50.0:
        return "medium"
    return "high"


def classify_player_archetype(row: pd.Series) -> str:
    """Classify a player into a transparent valuation archetype."""
    if row.get("games_played_frac") == "R":
        return "market_backed_upside" if np.isfinite(_float_or_inf(row.get("adp"))) else "uncertain_upside"

    confidence = float(row.get("data_confidence", 0.0))
    edge = float(row.get("projection_edge_points", 0.0))
    market_premium = float(row.get("market_premium_points", 0.0))
    agreement = float(row.get("agreement_score", 0.0))
    stability = float(row.get("stability_score", 0.0))

    if stability < 40.0:
        return "confirmed_risk"
    if agreement >= 80.0 and confidence >= 0.7:
        return "proven_consensus"
    if market_premium >= 2.0:
        return "market_backed_upside"
    if edge >= 2.0:
        return "model_backed_value"
    return "uncertain_upside"


def _market_implied_points(df: pd.DataFrame) -> pd.Series:
    """Infer market-implied weekly points from position ADP rank."""
    implied = df["historical_weekly_median"].astype(float).copy()
    if "adp" not in df.columns:
        return implied

    for position in DRAFTABLE_POSITIONS:
        mask = df["position"].astype(str).str.upper().eq(position)
        position_df = df[mask].copy()
        if position_df.empty:
            continue
        projection_curve = (
            position_df["historical_weekly_median"]
            .sort_values(ascending=False)
            .reset_index(drop=True)
        )
        adp_ranked = position_df[pd.to_numeric(position_df["adp"], errors="coerce").notna()].copy()
        adp_ranked["position_adp_rank"] = (
            pd.to_numeric(adp_ranked["adp"], errors="coerce")
            .rank(method="first", ascending=True)
            .astype(int)
        )
        for index, row in adp_ranked.iterrows():
            rank_index = min(int(row["position_adp_rank"]) - 1, len(projection_curve) - 1)
            implied.at[index] = float(_smoothed_curve_value(projection_curve, rank_index))
    return implied


def _position_residual_scales(df: pd.DataFrame) -> dict[str, float]:
    """Return robust residual scales by position for edge standardization."""
    scales: dict[str, float] = {}
    for position, position_df in df.groupby(df["position"].astype(str).str.upper()):
        residuals = pd.to_numeric(position_df["projection_edge_points"], errors="coerce").dropna()
        if residuals.empty:
            scales[str(position)] = DEFAULT_RESIDUAL_SCALE_POINTS
            continue
        median = float(residuals.median())
        mad = float((residuals - median).abs().median())
        robust_sigma = mad * 1.4826
        scales[str(position)] = max(DEFAULT_RESIDUAL_SCALE_POINTS, robust_sigma)
    return scales


def _projection_interval_width(row: pd.Series) -> float:
    """Estimate P20/P80 half-width from confidence and market disagreement."""
    points = float(row.get("projected_points_p50", 0.0))
    confidence = float(row.get("data_confidence", 0.0))
    disagreement = abs(float(row.get("projection_edge_points", 0.0)))
    base_width = max(1.5, points * 0.15)
    sparse_penalty = (1.0 - confidence) * max(2.0, points * 0.2)
    return float(base_width + sparse_penalty + 0.25 * disagreement)


def _smoothed_curve_value(values: pd.Series, index: int) -> float:
    """Return a local smoothed value from a projection curve."""
    start = max(0, index - 1)
    end = min(len(values), index + 2)
    return float(values.iloc[start:end].mean())


def _with_empty_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return an empty frame with valuation columns present."""
    for column in STATIC_VALUE_COLUMNS:
        if column not in df.columns:
            df[column] = pd.Series(dtype=float)
    return df


def _numeric_games_played_fraction(value: Any) -> float | None:
    """Return a bounded numeric games-played fraction when available."""
    if value == "R":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return float(max(0.0, min(1.0, numeric)))


def _float_or_none(value: Any) -> float | None:
    """Return finite float or ``None``."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _float_or_inf(value: Any) -> float:
    """Return finite float or positive infinity."""
    numeric = _float_or_none(value)
    return float("inf") if numeric is None else numeric


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return a finite division result."""
    if denominator == 0 or not np.isfinite(denominator):
        return 0.0
    return float(numerator / denominator)


STATIC_VALUE_COLUMNS = (
    "historical_weekly_median",
    "projected_points_p20",
    "projected_points_p50",
    "projected_points_p80",
    "effective_sample_size",
    "data_confidence",
    "market_implied_points",
    "projection_edge_points",
    "market_premium_points",
    "projection_edge_z",
    "agreement_score",
    "projection_basis",
    "stability_score",
    "risk_tier",
    "player_archetype",
)
