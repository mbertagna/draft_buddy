"""Fetch and cache web search results for top ADP players and/or NFL teams."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, TypeVar

from tqdm import tqdm

from draft_buddy.config import load_runtime_config
from draft_buddy.data.cache_paths import (
    insights_league_search_cache_dir,
    insights_search_cache_dir,
    insights_team_search_cache_dir,
)
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
from draft_buddy.data.insights.team_context import InsightTeamContext
from draft_buddy.data.insights.team_query_builder import (
    LeagueInsightQueryBuilder,
    TeamInsightQueryBuilder,
)
from draft_buddy.data.insights.team_search_cache import (
    LEAGUE_CACHE_KEY,
    TeamSearchCacheStore,
    execute_team_search_with_cache,
)
from draft_buddy.data.insights.team_selector import InsightTeamSelector

SCOPE_CHOICES = ("players", "teams", "both")

ItemT = TypeVar("ItemT")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the search fetch script."""
    parser = argparse.ArgumentParser(
        description="Fetch web search results for top ADP players and/or NFL teams "
        "(Valyu or Google CSE)."
    )
    parser.add_argument("--year", type=int, default=2026, help="Draft year for queries.")
    parser.add_argument("--top-n", type=int, default=150, help="Number of top ADP players.")
    parser.add_argument("--data-root", type=str, default="./data", help="Data root directory.")
    parser.add_argument(
        "--scope",
        type=str,
        default="players",
        choices=SCOPE_CHOICES,
        help="What to fetch: player insights, team outlooks, or both (default: players).",
    )
    parser.add_argument(
        "--search-provider",
        type=str,
        default=None,
        choices=SUPPORTED_SEARCH_PROVIDERS,
        help="Search provider (default: INSIGHTS_SEARCH_PROVIDER env or valyu).",
    )
    parser.add_argument("--concurrency", type=int, default=5, help="Parallel worker count.")
    parser.add_argument("--delay-ms", type=int, default=200, help="Delay between fetch batches.")
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


def _fetch_team_searches(
    team,
    gateway: SearchGateway,
    cache_store: TeamSearchCacheStore,
    query_builder: TeamInsightQueryBuilder,
    force: bool,
    provider: str,
) -> tuple[str, str]:
    """Fetch all queries for one team.

    Returns
    -------
    tuple[str, str]
        Status label and team abbreviation.
    """
    if cache_store.has_cached_provider(team.team_abbr, provider) and not force:
        return ("skipped", team.team_abbr)

    queries = query_builder.build_queries(team)
    for query in queries:
        execute_team_search_with_cache(gateway, cache_store, team, query, provider=provider)
    return ("fetched", team.team_abbr)


def _fetch_league_searches(
    gateway: SearchGateway,
    cache_store: TeamSearchCacheStore,
    query_builder: LeagueInsightQueryBuilder,
    year: int,
    force: bool,
    provider: str,
) -> dict[str, int]:
    """Fetch the shared, league-wide (all-32-teams) queries once per run.

    These broad queries surface the same multi-team roundup articles that
    would otherwise be re-fetched redundantly for every individual team.

    Returns
    -------
    dict[str, int]
        Counts keyed by ``fetched``, ``skipped``, ``failed``, ``quota``.
    """
    counts = {"fetched": 0, "skipped": 0, "failed": 0, "quota": 0}
    league = InsightTeamContext(team_abbr=LEAGUE_CACHE_KEY, draft_year=year)
    if cache_store.has_cached_provider(league.team_abbr, provider) and not force:
        counts["skipped"] = 1
        return counts

    try:
        for query in query_builder.build_queries(year):
            execute_team_search_with_cache(gateway, cache_store, league, query, provider=provider)
        counts["fetched"] = 1
    except QuotaExceededError:
        counts["quota"] = 1
        print("\nQuota exceeded. Re-run later or check provider billing.", file=sys.stderr)
    except Exception as error:
        counts["failed"] = 1
        print(f"\nLeague-wide fetch failed: {error}", file=sys.stderr)

    return counts


def _run_fetch_batch(
    items: list[ItemT],
    fetch_one: Callable[[ItemT], tuple[str, str]],
    describe: Callable[[ItemT], str],
    concurrency: int,
    delay_ms: int,
    desc: str,
) -> dict[str, int]:
    """Fetch search results for a batch of items, tracking status counts.

    Parameters
    ----------
    items : list
        Items to fetch (players or teams).
    fetch_one : Callable
        Fetch function returning ``(status, identifier)`` for one item.
    describe : Callable
        Returns a human-readable label for an item, used in error messages.
    concurrency : int
        Parallel worker count.
    delay_ms : int
        Delay between fetch batches.
    desc : str
        Progress bar description.

    Returns
    -------
    dict[str, int]
        Counts keyed by ``fetched``, ``skipped``, ``failed``, ``quota``.
    """
    counts = {"fetched": 0, "skipped": 0, "failed": 0, "quota": 0}

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(fetch_one, item): item for item in items}
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            try:
                status, _ = future.result()
                counts[status] = counts.get(status, 0) + 1
            except QuotaExceededError:
                counts["quota"] += 1
                print("\nQuota exceeded. Re-run later or check provider billing.", file=sys.stderr)
                break
            except Exception as error:
                counts["failed"] += 1
                item = futures[future]
                print(f"\nFailed for {describe(item)}: {error}", file=sys.stderr)
            sleep_between_batches(delay_ms)

    return counts


def _run_player_fetch(
    args: argparse.Namespace,
    config,
    gateway: SearchGateway,
    provider: str,
) -> dict[str, int]:
    """Fetch and cache search results for top ADP players."""
    selector = InsightPlayerSelector(config.paths.PLAYER_DATA_CSV, draft_year=args.year)
    players = selector.select(
        top_n=args.top_n,
        start_index=args.start_index,
        max_players=args.max_players,
    )
    cache_store = SearchCacheStore(insights_search_cache_dir(args.data_root))
    query_builder = InsightQueryBuilder()

    return _run_fetch_batch(
        players,
        lambda player: _fetch_player_searches(
            player, gateway, cache_store, query_builder, args.force, provider
        ),
        describe=lambda player: f"{player.name} ({player.sleeper_id})",
        concurrency=args.concurrency,
        delay_ms=args.delay_ms,
        desc="Fetching player searches",
    )


def _run_league_fetch(
    args: argparse.Namespace,
    gateway: SearchGateway,
    provider: str,
) -> dict[str, int]:
    """Fetch and cache the shared league-wide roundup queries once."""
    cache_store = TeamSearchCacheStore(insights_league_search_cache_dir(args.data_root))
    query_builder = LeagueInsightQueryBuilder()
    return _fetch_league_searches(
        gateway, cache_store, query_builder, args.year, args.force, provider
    )


def _run_team_fetch(
    args: argparse.Namespace,
    config,
    gateway: SearchGateway,
    provider: str,
) -> dict[str, int]:
    """Fetch and cache search results for all NFL teams."""
    selector = InsightTeamSelector(config.paths.PLAYER_DATA_CSV, draft_year=args.year)
    teams = selector.select()
    cache_store = TeamSearchCacheStore(insights_team_search_cache_dir(args.data_root))
    query_builder = TeamInsightQueryBuilder()

    return _run_fetch_batch(
        teams,
        lambda team: _fetch_team_searches(
            team, gateway, cache_store, query_builder, args.force, provider
        ),
        describe=lambda team: team.team_abbr,
        concurrency=args.concurrency,
        delay_ms=args.delay_ms,
        desc="Fetching team searches",
    )


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
    succeeded = True

    if args.scope in ("players", "both"):
        counts = _run_player_fetch(args, config, gateway, provider)
        print(
            f"Done (players). provider={provider} fetched={counts['fetched']} "
            f"skipped={counts['skipped']} failed={counts['failed']} "
            f"quota_errors={counts['quota']}"
        )
        succeeded = succeeded and counts["failed"] == 0 and counts["quota"] == 0

    if args.scope in ("teams", "both"):
        league_counts = _run_league_fetch(args, gateway, provider)
        print(
            f"Done (league-wide). provider={provider} fetched={league_counts['fetched']} "
            f"skipped={league_counts['skipped']} failed={league_counts['failed']} "
            f"quota_errors={league_counts['quota']}"
        )
        succeeded = succeeded and league_counts["failed"] == 0 and league_counts["quota"] == 0

        counts = _run_team_fetch(args, config, gateway, provider)
        print(
            f"Done (teams). provider={provider} fetched={counts['fetched']} "
            f"skipped={counts['skipped']} failed={counts['failed']} "
            f"quota_errors={counts['quota']}"
        )
        succeeded = succeeded and counts["failed"] == 0 and counts["quota"] == 0

    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
