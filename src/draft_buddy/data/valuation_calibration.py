"""Historical calibration utilities for player valuation metrics."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class RollingOriginSplit:
    """One leakage-safe train/test split.

    Parameters
    ----------
    train_seasons : tuple[int, ...]
        Seasons available to fit before the test season.
    test_season : int
        Held-out season being evaluated.
    """

    train_seasons: tuple[int, ...]
    test_season: int


@dataclass(frozen=True, slots=True)
class BlendWeights:
    """Calibrated history/market blend weights."""

    history_weight: float
    market_weight: float


def build_rolling_origin_splits(
    seasons: Iterable[int],
    min_train_seasons: int = 3,
) -> list[RollingOriginSplit]:
    """Build chronological train/test splits with no future leakage.

    Parameters
    ----------
    seasons : Iterable[int]
        Candidate seasons.
    min_train_seasons : int, optional
        Minimum number of prior seasons required before a test split.

    Returns
    -------
    list[RollingOriginSplit]
        Ordered rolling-origin splits.
    """
    ordered = tuple(sorted(set(int(season) for season in seasons)))
    splits: list[RollingOriginSplit] = []
    for index in range(min_train_seasons, len(ordered)):
        train_seasons = ordered[:index]
        test_season = ordered[index]
        if max(train_seasons) >= test_season:
            raise ValueError("Rolling split would leak future seasons into training.")
        splits.append(RollingOriginSplit(train_seasons=train_seasons, test_season=test_season))
    return splits


def calculate_realized_player_values(
    scored_weekly_df: pd.DataFrame,
    replacement_quantile: float = 0.25,
) -> pd.DataFrame:
    """Calculate realized season points and rough season VORP.

    Parameters
    ----------
    scored_weekly_df : pandas.DataFrame
        Weekly scored rows with ``season``, ``player_id``, ``position``, and
        ``total_pts``.
    replacement_quantile : float, optional
        Within-position season-points quantile used as a rough historical
        replacement baseline when draft-state demand is unavailable.

    Returns
    -------
    pandas.DataFrame
        Per player-season realized values.
    """
    required = {"season", "player_id", "position", "total_pts"}
    missing = required - set(scored_weekly_df.columns)
    if missing:
        raise ValueError(f"Missing required realized-value columns: {sorted(missing)}")
    if scored_weekly_df.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "player_id",
                "position",
                "realized_points",
                "realized_games",
                "realized_ppg",
                "replacement_points",
                "realized_vorp",
            ]
        )

    grouped = (
        scored_weekly_df.assign(total_pts=pd.to_numeric(scored_weekly_df["total_pts"], errors="coerce").fillna(0.0))
        .groupby(["season", "player_id", "position"], as_index=False)
        .agg(realized_points=("total_pts", "sum"), realized_games=("total_pts", "size"))
    )
    grouped["realized_ppg"] = grouped["realized_points"] / grouped["realized_games"].clip(lower=1)
    baselines = (
        grouped.groupby(["season", "position"])["realized_points"]
        .quantile(replacement_quantile)
        .reset_index(name="replacement_points")
    )
    valued = grouped.merge(baselines, on=["season", "position"], how="left")
    valued["realized_vorp"] = valued["realized_points"] - valued["replacement_points"]
    return valued


def evaluate_forecast_columns(
    validation_df: pd.DataFrame,
    forecast_columns: Sequence[str],
    target_column: str,
) -> pd.DataFrame:
    """Evaluate forecast columns against a realized target."""
    rows = []
    for column in forecast_columns:
        if column not in validation_df.columns:
            continue
        paired = validation_df[[column, target_column]].dropna()
        if paired.empty:
            continue
        error = pd.to_numeric(paired[column], errors="coerce") - pd.to_numeric(
            paired[target_column], errors="coerce"
        )
        rows.append(
            {
                "forecast": column,
                "count": int(error.count()),
                "mae": float(error.abs().mean()),
                "rmse": float(math.sqrt(float((error**2).mean()))),
                "spearman": _spearman_without_scipy(paired[column], paired[target_column]),
            }
        )
    return pd.DataFrame(rows)


def fit_blend_weights(
    validation_df: pd.DataFrame,
    history_column: str,
    market_column: str,
    target_column: str,
) -> BlendWeights:
    """Fit simple inverse-RMSE history/market blend weights."""
    metrics = evaluate_forecast_columns(validation_df, [history_column, market_column], target_column)
    if len(metrics) < 2:
        return BlendWeights(history_weight=0.5, market_weight=0.5)
    rmse_by_forecast = dict(zip(metrics["forecast"], metrics["rmse"]))
    history_precision = _inverse_square(rmse_by_forecast.get(history_column))
    market_precision = _inverse_square(rmse_by_forecast.get(market_column))
    total = history_precision + market_precision
    if total <= 0.0:
        return BlendWeights(history_weight=0.5, market_weight=0.5)
    return BlendWeights(
        history_weight=float(history_precision / total),
        market_weight=float(market_precision / total),
    )


def apply_blend_weights(
    frame: pd.DataFrame,
    history_column: str,
    market_column: str,
    weights: BlendWeights,
    output_column: str = "calibrated_value",
) -> pd.DataFrame:
    """Apply calibrated history/market weights to a DataFrame."""
    out = frame.copy()
    out[output_column] = (
        pd.to_numeric(out[history_column], errors="coerce").fillna(0.0) * weights.history_weight
        + pd.to_numeric(out[market_column], errors="coerce").fillna(0.0) * weights.market_weight
    )
    return out


def write_validation_report(
    metrics: pd.DataFrame,
    weights: BlendWeights,
    output_path: str,
) -> str:
    """Write calibration metrics and weights to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "weights": asdict(weights),
        "metrics": metrics.to_dict(orient="records"),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def _inverse_square(value: float | None) -> float:
    """Return inverse squared error with robust zero handling."""
    if value is None or not np.isfinite(value) or value <= 0:
        return 0.0
    return float(1.0 / (value**2))


def _spearman_without_scipy(left: pd.Series, right: pd.Series) -> float:
    """Return Spearman rank correlation without requiring SciPy."""
    ranked_left = pd.to_numeric(left, errors="coerce").rank(method="average")
    ranked_right = pd.to_numeric(right, errors="coerce").rank(method="average")
    if len(ranked_left.dropna()) < 2 or len(ranked_right.dropna()) < 2:
        return 0.0
    correlation = ranked_left.corr(ranked_right)
    if pd.isna(correlation):
        return 0.0
    return float(correlation)
