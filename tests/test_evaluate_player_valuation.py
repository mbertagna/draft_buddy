"""Tests for the valuation evaluation CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import evaluate_player_valuation


def test_evaluate_player_valuation_writes_report(tmp_path: Path) -> None:
    """Verify the CLI joins forecast/realized CSVs and writes metrics."""
    forecasts = tmp_path / "forecasts.csv"
    realized = tmp_path / "realized.csv"
    output = tmp_path / "valuation_report.json"
    pd.DataFrame(
        [
            {
                "season": 2025,
                "player_id": 1,
                "history_value_p50": 10.0,
                "market_implied_value": 9.0,
            }
        ]
    ).to_csv(forecasts, index=False)
    pd.DataFrame(
        [{"season": 2025, "player_id": 1, "realized_vorp": 11.0}]
    ).to_csv(realized, index=False)

    path = evaluate_player_valuation.main(
        ["--forecasts", str(forecasts), "--realized", str(realized), "--output", str(output)]
    )

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert payload["metrics"]
