"""Generate a static Monte Carlo position guide cheat sheet."""

from __future__ import annotations

import argparse
import os
import sys

from draft_buddy.config import Config
from draft_buddy.rl.position_guide.exporter import export_position_guide
from draft_buddy.rl.position_guide.simulator import PositionGuideSimulator
from draft_buddy.rl.run_utils import find_latest_checkpoint_in_dir


DEFAULT_CHECKPOINT_DIR = "models/12_teams_random_start/v3"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for position guide generation."""
    parser = argparse.ArgumentParser(
        description="Generate a static position probability cheat sheet via Monte Carlo simulation."
    )
    parser.add_argument("--num-teams", type=int, default=12, help="League size.")
    parser.add_argument("--slot", type=int, default=5, help="Your draft slot (1-based).")
    parser.add_argument("--year", type=int, default=2026, help="Draft year for export metadata.")
    parser.add_argument("--simulations", type=int, default=5000, help="Monte Carlo rollouts.")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory containing checkpoint_episode_*.pth files.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Explicit checkpoint path (overrides --checkpoint-dir).",
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


def resolve_checkpoint_path(args: argparse.Namespace) -> str | None:
    """Resolve checkpoint path from CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    str | None
        Resolved checkpoint path, or ``None`` when not found.
    """
    if args.checkpoint:
        return args.checkpoint
    return find_latest_checkpoint_in_dir(args.checkpoint_dir)


def main() -> int:
    """Run position guide generation."""
    args = parse_args()
    config = Config()
    player_csv = args.player_csv or config.paths.PLAYER_DATA_CSV

    if not os.path.isfile(player_csv):
        print(f"Player data CSV not found: {player_csv}", file=sys.stderr)
        print("Run generate_projections.py or docker compose run --rm data first.", file=sys.stderr)
        return 1

    checkpoint_path = resolve_checkpoint_path(args)
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        print("Checkpoint not found.", file=sys.stderr)
        return 1

    if not (1 <= args.slot <= args.num_teams):
        print(f"Invalid slot {args.slot} for {args.num_teams}-team league.", file=sys.stderr)
        return 1

    print(
        f"Generating position guide: {args.num_teams} teams, slot {args.slot}, "
        f"{args.simulations} simulations..."
    )
    simulator = PositionGuideSimulator(
        config=config,
        draft_slot=args.slot,
        num_teams=args.num_teams,
        checkpoint_path=checkpoint_path,
        draft_year=args.year,
        simulations=args.simulations,
        seed=args.seed,
        player_data_csv=player_csv,
        show_progress=not args.no_progress,
    )
    try:
        guide = simulator.run()
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 1

    json_path, html_path = export_position_guide(guide, args.data_root)
    print(f"JSON guide saved to: {json_path}")
    print(f"HTML guide saved to: {html_path}")
    print(f"Total user picks: {guide.total_user_picks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
