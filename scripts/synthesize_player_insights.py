"""Synthesize Gemini Flash player insights from cached search results."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from draft_buddy.config import load_runtime_config
from draft_buddy.data.cache_paths import (
    insights_search_cache_dir,
    insights_synthesis_cache_dir,
    player_insights_exports_dir,
    player_insights_output_path,
)
from draft_buddy.data.insights.cse_gateway import SearchCacheStore
from draft_buddy.data.insights.gemini_flash_gateway import DEFAULT_GEMINI_MODEL, GeminiFlashGateway
from draft_buddy.data.insights.player_selector import InsightPlayerSelector
from draft_buddy.data.insights.schemas import PlayerInsight
from draft_buddy.data.insights.synthesizer import InsightSynthesizer, SynthesisCacheStore, merge_insights_file


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the synthesis script."""
    parser = argparse.ArgumentParser(
        description="Synthesize structured player insights from cached CSE search results."
    )
    parser.add_argument("--year", type=int, default=2026, help="Draft year.")
    parser.add_argument("--top-n", type=int, default=150, help="Number of top ADP players.")
    parser.add_argument("--data-root", type=str, default="./data", help="Data root directory.")
    parser.add_argument("--concurrency", type=int, default=10, help="Parallel synthesis workers.")
    parser.add_argument("--start-index", type=int, default=0, help="Start index into ADP list.")
    parser.add_argument("--max-players", type=int, default=None, help="Optional player cap.")
    parser.add_argument("--force", action="store_true", help="Re-synthesize even when cache exists.")
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("INSIGHTS_GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        help="Gemini model name.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the synthesis pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY environment variable is required.", file=sys.stderr)
        return 1

    config = load_runtime_config()
    selector = InsightPlayerSelector(config.paths.PLAYER_DATA_CSV, draft_year=args.year)
    players = selector.select(
        top_n=args.top_n,
        start_index=args.start_index,
        max_players=args.max_players,
    )

    search_cache = SearchCacheStore(insights_search_cache_dir(args.data_root))
    synthesis_cache = SynthesisCacheStore(insights_synthesis_cache_dir(args.data_root))
    gateway = GeminiFlashGateway(api_key=api_key, model=args.model)
    synthesizer = InsightSynthesizer(gateway, search_cache, synthesis_cache)

    missing_search: list[str] = [
        player.sleeper_id
        for player in players
        if not search_cache.has_manifest(player.sleeper_id)
    ]
    if missing_search:
        print(
            "Search cache missing for sleeper_ids: "
            + ", ".join(missing_search[:20])
            + (" ..." if len(missing_search) > 20 else ""),
            file=sys.stderr,
        )
        print("Run fetch_player_insight_search.py first.", file=sys.stderr)
        return 1

    counts = {"synthesized": 0, "cached": 0, "failed": 0}
    merged: dict[str, PlayerInsight] = {}

    def _run(player):
        if synthesis_cache.has_cache(player.sleeper_id) and not args.force:
            return player.sleeper_id, synthesis_cache.load(player.sleeper_id), "cached"
        insight = synthesizer.synthesize_player(player, force=args.force)
        return player.sleeper_id, insight, "synthesized"

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(_run, player): player for player in players}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Synthesizing insights"):
            player = futures[future]
            try:
                sleeper_id, insight, status = future.result()
                merged[sleeper_id] = insight
                counts[status] += 1
            except Exception as error:
                counts["failed"] += 1
                print(
                    f"\nFailed for {player.name} ({player.sleeper_id}): {error}",
                    file=sys.stderr,
                )

    insights_file = merge_insights_file(args.year, args.model, merged)
    exports_dir = player_insights_exports_dir(args.data_root)
    os.makedirs(exports_dir, exist_ok=True)
    output_path = player_insights_output_path(
        args.data_root, args.year, insights_file.generated_at
    )
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(insights_file.model_dump(mode="json"), handle, indent=2)

    print(
        f"Done. synthesized={counts['synthesized']} cached={counts['cached']} "
        f"failed={counts['failed']} output={output_path}"
    )
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
