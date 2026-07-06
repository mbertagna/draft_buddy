"""Fetch and cache web search results for top ADP players."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from draft_buddy.config import load_runtime_config
from draft_buddy.data.cache_paths import insights_search_cache_dir
from draft_buddy.data.insights.cse_gateway import (
    QuotaExceededError,
    SearchCacheStore,
    SearchGateway,
    execute_search_with_cache,
    sleep_between_batches,
)
from draft_buddy.data.insights.player_selector import InsightPlayerSelector
from draft_buddy.data.insights.query_builder import InsightQueryBuilder
from draft_buddy.data.insights.search_factory import (
    SUPPORTED_SEARCH_PROVIDERS,
    build_search_gateway,
    resolve_search_provider,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the search fetch script."""
    parser = argparse.ArgumentParser(
        description="Fetch web search results for top ADP players (Valyu or Google CSE)."
    )
    parser.add_argument("--year", type=int, default=2026, help="Draft year for queries.")
    parser.add_argument("--top-n", type=int, default=150, help="Number of top ADP players.")
    parser.add_argument("--data-root", type=str, default="./data", help="Data root directory.")
    parser.add_argument(
        "--search-provider",
        type=str,
        default=None,
        choices=SUPPORTED_SEARCH_PROVIDERS,
        help="Search provider (default: INSIGHTS_SEARCH_PROVIDER env or valyu).",
    )
    parser.add_argument("--concurrency", type=int, default=5, help="Parallel player workers.")
    parser.add_argument("--delay-ms", type=int, default=200, help="Delay between player batches.")
    parser.add_argument("--start-index", type=int, default=0, help="Start index into ADP list.")
    parser.add_argument("--max-players", type=int, default=None, help="Optional player cap.")
    parser.add_argument("--force", action="store_true", help="Re-fetch even when cache exists.")
    return parser.parse_args()


def _fetch_player_searches(
    player,
    gateway: SearchGateway,
    cache_store: SearchCacheStore,
    query_builder: InsightQueryBuilder,
    force: bool,
    provider: str,
) -> tuple[str, str]:
    """Fetch all queries for one player.

    Returns
    -------
    tuple[str, str]
        Status label and sleeper id.
    """
    if cache_store.has_cached_provider(player.sleeper_id, provider) and not force:
        return ("skipped", player.sleeper_id)

    queries = query_builder.build_queries(player)
    for query in queries:
        execute_search_with_cache(gateway, cache_store, player, query, provider=provider)
    return ("fetched", player.sleeper_id)


def main() -> int:
    """Run the search fetch pipeline."""
    args = parse_args()
    try:
        provider = resolve_search_provider(args.search_provider)
        gateway = build_search_gateway(provider)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1

    config = load_runtime_config()
    selector = InsightPlayerSelector(config.paths.PLAYER_DATA_CSV, draft_year=args.year)
    players = selector.select(
        top_n=args.top_n,
        start_index=args.start_index,
        max_players=args.max_players,
    )

    cache_store = SearchCacheStore(insights_search_cache_dir(args.data_root))
    query_builder = InsightQueryBuilder()

    counts = {"fetched": 0, "skipped": 0, "failed": 0, "quota": 0}

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                _fetch_player_searches,
                player,
                gateway,
                cache_store,
                query_builder,
                args.force,
                provider,
            ): player
            for player in players
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching searches"):
            try:
                status, _ = future.result()
                counts[status] = counts.get(status, 0) + 1
            except QuotaExceededError:
                counts["quota"] += 1
                print("\nQuota exceeded. Re-run later or check provider billing.", file=sys.stderr)
                break
            except Exception as error:
                counts["failed"] += 1
                player = futures[future]
                print(f"\nFailed for {player.name} ({player.sleeper_id}): {error}", file=sys.stderr)
            sleep_between_batches(args.delay_ms)

    print(
        f"Done. provider={provider} fetched={counts['fetched']} skipped={counts['skipped']} "
        f"failed={counts['failed']} quota_errors={counts['quota']}"
    )
    return 0 if counts["failed"] == 0 and counts["quota"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
