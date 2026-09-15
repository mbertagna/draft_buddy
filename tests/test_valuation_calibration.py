"""Tests for valuation calibration utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from draft_buddy.data.valuation_calibration import (
    BlendWeights,
    apply_blend_weights,
    build_rolling_origin_splits,
    calculate_realized_player_values,
    evaluate_forecast_columns,
    fit_blend_weights,
    write_validation_report,
)


def test_build_rolling_origin_splits_uses_only_prior_seasons() -> None:
    """Verify rolling splits never include the test season in training."""
    splits = build_rolling_origin_splits([2020, 2021, 2022, 2023], min_train_seasons=2)

    assert splits[0].train_seasons == (2020, 2021)
    assert splits[0].test_season == 2022
    assert max(splits[1].train_seasons) < splits[1].test_season


def test_calculate_realized_player_values_scores_season_vorp() -> None:
    """Verify realized values aggregate weekly scoring and replacement points."""
    weekly = pd.DataFrame(
        [
            {"season": 2025, "player_id": 1, "position": "RB", "total_pts": 10.0},
            {"season": 2025, "player_id": 1, "position": "RB", "total_pts": 12.0},
            {"season": 2025, "player_id": 2, "position": "RB", "total_pts": 6.0},
            {"season": 2025, "player_id": 2, "position": "RB", "total_pts": 6.0},
        ]
    )

    values = calculate_realized_player_values(weekly, replacement_quantile=0.0)
    top = values[values["player_id"] == 1].iloc[0]

    assert top["realized_points"] == pytest.approx(22.0)
    assert top["replacement_points"] == pytest.approx(12.0)
    assert top["realized_vorp"] == pytest.approx(10.0)


def test_fit_blend_weights_prefers_lower_rmse_forecast() -> None:
    """Verify inverse-RMSE weights favor the better validated signal."""
    validation = pd.DataFrame(
        [
            {"history": 9.0, "market": 5.0, "actual": 10.0},
            {"history": 18.0, "market": 8.0, "actual": 20.0},
            {"history": 27.0, "market": 20.0, "actual": 30.0},
        ]
    )

    weights = fit_blend_weights(validation, "history", "market", "actual")

    assert weights.history_weight > weights.market_weight


def test_apply_blend_weights_creates_calibrated_value() -> None:
    """Verify calibrated values use the supplied history/market weights."""
    frame = pd.DataFrame([{"history": 10.0, "market": 20.0}])

    blended = apply_blend_weights(frame, "history", "market", BlendWeights(0.25, 0.75))

    assert blended.iloc[0]["calibrated_value"] == pytest.approx(17.5)


def test_evaluate_forecast_columns_returns_error_metrics() -> None:
    """Verify forecast evaluation returns MAE, RMSE, and rank correlation."""
    frame = pd.DataFrame(
        [
            {"forecast": 10.0, "actual": 11.0},
            {"forecast": 20.0, "actual": 19.0},
        ]
    )

    metrics = evaluate_forecast_columns(frame, ["forecast"], "actual")

    assert metrics.iloc[0]["mae"] == pytest.approx(1.0)
    assert metrics.iloc[0]["count"] == 2


def test_write_validation_report_persists_metrics_and_weights(tmp_path: Path) -> None:
    """Verify calibration reports are written as JSON."""
    metrics = pd.DataFrame([{"forecast": "history", "mae": 1.0}])
    path = write_validation_report(metrics, BlendWeights(0.6, 0.4), str(tmp_path / "report.json"))

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert payload["weights"]["history_weight"] == pytest.approx(0.6)
