"""Evaluate player valuation forecasts against realized historical values."""

from __future__ import annotations

import argparse

import pandas as pd

from draft_buddy.data.valuation_calibration import (
    apply_blend_weights,
    evaluate_forecast_columns,
    fit_blend_weights,
    write_validation_report,
)


def parse_cli_args(argv=None) -> argparse.Namespace:
    """Parse valuation evaluation CLI arguments.

    Parameters
    ----------
    argv : list[str], optional
        Argument vector excluding program name.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate Draft Buddy valuation forecasts.")
    parser.add_argument("--forecasts", required=True, help="CSV with forecast columns.")
    parser.add_argument("--realized", required=True, help="CSV with realized target columns.")
    parser.add_argument("--output", required=True, help="JSON report path to write.")
    parser.add_argument("--history-column", default="history_value_p50")
    parser.add_argument("--market-column", default="market_implied_value")
    parser.add_argument("--target-column", default="realized_vorp")
    return parser.parse_args(argv)


def main(argv=None) -> str:
    """Run valuation evaluation and write a JSON report.

    Parameters
    ----------
    argv : list[str], optional
        Argument vector excluding program name.

    Returns
    -------
    str
        Written report path.
    """
    args = parse_cli_args(argv)
    forecasts_df = pd.read_csv(args.forecasts)
    realized_df = pd.read_csv(args.realized)
    validation_df = forecasts_df.merge(realized_df, on=["season", "player_id"], how="inner")
    weights = fit_blend_weights(
        validation_df,
        history_column=args.history_column,
        market_column=args.market_column,
        target_column=args.target_column,
    )
    blended_df = apply_blend_weights(
        validation_df,
        history_column=args.history_column,
        market_column=args.market_column,
        weights=weights,
    )
    metrics = evaluate_forecast_columns(
        blended_df,
        [args.history_column, args.market_column, "calibrated_value"],
        args.target_column,
    )
    return write_validation_report(metrics, weights, args.output)


if __name__ == "__main__":
    print(main())
