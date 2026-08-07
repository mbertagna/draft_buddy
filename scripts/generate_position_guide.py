"""Generate static Monte Carlo position guide cheat sheets for every draft slot."""

from __future__ import annotations

import argparse
import os
import sys

from draft_buddy.config import load_runtime_config
from draft_buddy.rl.position_guide.exporter import export_model_adp, export_position_guide
from draft_buddy.rl.position_guide.simulator import PositionGuideSimulator
from draft_buddy.rl.run_utils import resolve_checkpoint_path


def parse_args(runtime_defaults: argparse.Namespace) -> argparse.Namespace:
    """Parse CLI arguments for position guide generation."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate static position probability cheat sheets for every draft slot via "
            "all-slots self-play Monte Carlo simulation."
        )
    )
    parser.add_argument(
        "--num-teams",
        type=int,
        default=runtime_defaults.num_teams,
        help="League size.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=runtime_defaults.year,
        help="Draft year for export metadata.",
    )
    parser.add_argument("--simulations", type=int, default=5000, help="Monte Carlo rollouts.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint file or directory (overrides config.training.MODEL_PATH_TO_LOAD).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.5,
        help=(
            "Softmax temperature applied to every self-play suggestion. Values above 1.0 "
            "soften an overconfident policy while preserving its ranking."
        ),
    )
    parser.add_argument(
        "--prune-inactive",
        action="store_true",
        help="Exclude inactive/injured players from the draftable pool for this run.",
    )
    parser.add_argument(
        "--player-csv",
        type=str,
        default=None,
        help="Player data CSV path (default: Config.paths.PLAYER_DATA_CSV).",
    )
    parser.add_argument("--data-root", type=str, default="./data", help="Data root directory.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the simulation progress bar.",
    )
    return parser.parse_args()


def main() -> int:
    """Run all-slots position guide generation."""
    config = load_runtime_config()
    runtime_defaults = argparse.Namespace(
        num_teams=config.draft.NUM_TEAMS,
        year=config.season.season,
    )
    args = parse_args(runtime_defaults)
    player_csv = args.player_csv or config.paths.PLAYER_DATA_CSV

    if not os.path.isfile(player_csv):
        print(f"Player data CSV not found: {player_csv}", file=sys.stderr)
        print("Run generate_projections.py or docker compose run --rm data first.", file=sys.stderr)
        return 1

    checkpoint_path = resolve_checkpoint_path(args.checkpoint or config.training.MODEL_PATH_TO_LOAD)
    if not checkpoint_path:
        print("Checkpoint not found.", file=sys.stderr)
        return 1

    if args.prune_inactive:
        config.data.EXCLUDE_INACTIVE_PLAYERS = True

    print(
        f"Generating position guides for all {args.num_teams} slots "
        f"({args.simulations} self-play simulations, temperature={args.temperature:g})..."
    )
    simulator = PositionGuideSimulator(
        config=config,
        num_teams=args.num_teams,
        checkpoint_path=checkpoint_path,
        draft_year=args.year,
        simulations=args.simulations,
        seed=args.seed,
        temperature=args.temperature,
        player_data_csv=player_csv,
        show_progress=not args.no_progress,
    )
    try:
        guides, model_adp = simulator.run()
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 1

    for slot in sorted(guides):
        json_path, html_path = export_position_guide(guides[slot], args.data_root)
        print(f"Slot {slot}: JSON={json_path} HTML={html_path}")

    adp_json_path, adp_html_path = export_model_adp(model_adp, args.data_root)
    print(f"Model ADP: JSON={adp_json_path} HTML={adp_html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
