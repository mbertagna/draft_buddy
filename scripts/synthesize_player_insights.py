"""Synthesize structured player insights and/or team outlooks from cached search results."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, TypeVar

from tqdm import tqdm

from draft_buddy.config import load_runtime_config
from draft_buddy.data.cache_paths import (
    insights_league_search_cache_dir,
    insights_search_cache_dir,
    insights_synthesis_cache_dir,
    insights_team_search_cache_dir,
    insights_team_synthesis_cache_dir,
    player_insights_exports_dir,
    player_insights_output_path,
    team_insights_exports_dir,
    team_insights_output_path,
)
from draft_buddy.data.insights.cse_gateway import SearchCacheStore
from draft_buddy.data.insights.player_selector import InsightPlayerSelector
from draft_buddy.data.insights.schemas import PlayerInsight
from draft_buddy.data.insights.synthesis_factory import (
    build_synthesis_gateway,
    resolve_synthesis_model,
)
from draft_buddy.data.insights.synthesizer import (
    InsightSynthesizer,
    SynthesisCacheStore,
    merge_insights_file,
)
from draft_buddy.data.insights.team_schemas import TeamOutlook
from draft_buddy.data.insights.team_search_cache import TeamSearchCacheStore
from draft_buddy.data.insights.team_selector import InsightTeamSelector
from draft_buddy.data.insights.team_synthesizer import (
    TeamOutlookSynthesizer,
    TeamSynthesisCacheStore,
    merge_team_outlooks_file,
)
from draft_buddy.llm.model_registry import resolve_synthesis_provider

SCOPE_CHOICES = ("players", "teams", "both")

ItemT = TypeVar("ItemT")
ResultT = TypeVar("ResultT")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the synthesis script."""
    parser = argparse.ArgumentParser(
        description="Synthesize structured player insights and/or team outlooks "
        "from cached search results."
    )
    parser.add_argument("--year", type=int, default=2026, help="Draft year.")
    parser.add_argument("--top-n", type=int, default=150, help="Number of top ADP players.")
    parser.add_argument("--data-root", type=str, default="./data", help="Data root directory.")
    parser.add_argument(
        "--scope",
        type=str,
        default="players",
        choices=SCOPE_CHOICES,
        help="What to synthesize: player insights, team outlooks, or both (default: players).",
    )
    parser.add_argument("--concurrency", type=int, default=10, help="Parallel synthesis workers.")
    parser.add_argument("--start-index", type=int, default=0, help="Start index into ADP list.")
    parser.add_argument("--max-players", type=int, default=None, help="Optional player cap.")
    parser.add_argument(
        "--force", action="store_true", help="Re-synthesize even when cache exists."
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=os.environ.get("INSIGHTS_LLM_PROVIDER"),
        help="LLM provider: gemini or openrouter (optional; inferred from model when omitted).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model id (defaults to INSIGHTS_LLM_MODEL or gemini-2.5-flash).",
    )
    return parser.parse_args()


def _run_synthesis_batch(
    items: list[ItemT],
    synthesize_one: Callable[[ItemT], tuple[str, ResultT, str]],
    identify: Callable[[ItemT], str],
    desc: str,
    concurrency: int,
) -> tuple[dict[str, ResultT], dict[str, int]]:
    """Synthesize a batch of items in parallel, tracking status counts.

    Parameters
    ----------
    items : list
        Items to synthesize (players or teams).
    synthesize_one : Callable
        Synthesis function returning ``(identifier, result, status)``.
    identify : Callable
        Returns a human-readable label for an item, used in error messages.
    desc : str
        Progress bar description.
    concurrency : int
        Parallel worker count.

    Returns
    -------
    tuple[dict[str, ResultT], dict[str, int]]
        Results keyed by identifier, and status counts.
    """
    counts = {"synthesized": 0, "cached": 0, "failed": 0}
    results: dict[str, ResultT] = {}

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(synthesize_one, item): item for item in items}
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            item = futures[future]
            try:
                identifier, result, status = future.result()
                results[identifier] = result
                counts[status] += 1
            except Exception as error:
                counts["failed"] += 1
                print(f"\nFailed for {identify(item)}: {error}", file=sys.stderr)

    return results, counts


def _run_player_synthesis(args: argparse.Namespace, config, model: str, gateway) -> bool:
    """Synthesize player insights and write a merged export.

    Returns
    -------
    bool
        True when synthesis completed without failures.
    """
    selector = InsightPlayerSelector(config.paths.PLAYER_DATA_CSV, draft_year=args.year)
    players = selector.select(
        top_n=args.top_n,
        start_index=args.start_index,
        max_players=args.max_players,
    )

    search_cache = SearchCacheStore(insights_search_cache_dir(args.data_root))
    synthesis_cache = SynthesisCacheStore(insights_synthesis_cache_dir(args.data_root))
    synthesizer = InsightSynthesizer(gateway, search_cache, synthesis_cache)

    missing_search = [
        player.sleeper_id for player in players if not search_cache.has_manifest(player.sleeper_id)
    ]
    if missing_search:
        print(
            "Search cache missing for sleeper_ids: "
            + ", ".join(missing_search[:20])
            + (" ..." if len(missing_search) > 20 else ""),
            file=sys.stderr,
        )
        print(
            "Run fetch_player_insight_search.py --scope players first.",
            file=sys.stderr,
        )
        return False

    def _run(player) -> tuple[str, PlayerInsight, str]:
        if synthesis_cache.has_cache(player.sleeper_id) and not args.force:
            return player.sleeper_id, synthesis_cache.load(player.sleeper_id), "cached"
        insight = synthesizer.synthesize_player(player, force=args.force)
        return player.sleeper_id, insight, "synthesized"

    merged, counts = _run_synthesis_batch(
        players,
        _run,
        identify=lambda player: f"{player.name} ({player.sleeper_id})",
        desc="Synthesizing player insights",
        concurrency=args.concurrency,
    )

    insights_file = merge_insights_file(args.year, model, merged)
    exports_dir = player_insights_exports_dir(args.data_root)
    os.makedirs(exports_dir, exist_ok=True)
    output_path = player_insights_output_path(args.data_root, args.year, insights_file.generated_at)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(insights_file.model_dump(mode="json"), handle, indent=2)

    print(
        f"Done (players). synthesized={counts['synthesized']} cached={counts['cached']} "
        f"failed={counts['failed']} output={output_path}"
    )
    return counts["failed"] == 0


def _run_team_synthesis(args: argparse.Namespace, config, model: str, gateway) -> bool:
    """Synthesize team outlooks and write a merged export.

    Returns
    -------
    bool
        True when synthesis completed without failures.
    """
    selector = InsightTeamSelector(config.paths.PLAYER_DATA_CSV, draft_year=args.year)
    teams = selector.select()

    search_cache = TeamSearchCacheStore(insights_team_search_cache_dir(args.data_root))
    synthesis_cache = TeamSynthesisCacheStore(insights_team_synthesis_cache_dir(args.data_root))
    league_search_cache = TeamSearchCacheStore(insights_league_search_cache_dir(args.data_root))
    synthesizer = TeamOutlookSynthesizer(
        gateway, search_cache, synthesis_cache, league_search_cache
    )

    missing_search = [
        team.team_abbr for team in teams if not search_cache.has_manifest(team.team_abbr)
    ]
    if missing_search:
        print(
            "Search cache missing for teams: " + ", ".join(missing_search),
            file=sys.stderr,
        )
        print(
            "Run fetch_player_insight_search.py --scope teams first.",
            file=sys.stderr,
        )
        return False

    def _run(team) -> tuple[str, TeamOutlook, str]:
        if synthesis_cache.has_cache(team.team_abbr) and not args.force:
            return team.team_abbr, synthesis_cache.load(team.team_abbr), "cached"
        outlook = synthesizer.synthesize_team(team, force=args.force)
        return team.team_abbr, outlook, "synthesized"

    merged, counts = _run_synthesis_batch(
        teams,
        _run,
        identify=lambda team: team.team_abbr,
        desc="Synthesizing team outlooks",
        concurrency=args.concurrency,
    )

    outlooks_file = merge_team_outlooks_file(args.year, model, merged)
    exports_dir = team_insights_exports_dir(args.data_root)
    os.makedirs(exports_dir, exist_ok=True)
    output_path = team_insights_output_path(args.data_root, args.year, outlooks_file.generated_at)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(outlooks_file.model_dump(mode="json"), handle, indent=2)

    print(
        f"Done (teams). synthesized={counts['synthesized']} cached={counts['cached']} "
        f"failed={counts['failed']} output={output_path}"
    )
    return counts["failed"] == 0


def main() -> int:
    """Run the synthesis pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    provider = args.provider.strip() if args.provider else None
    try:
        model = resolve_synthesis_model(args.model)
        if provider:
            resolve_synthesis_provider(provider)
        gateway = build_synthesis_gateway(model=model, provider=provider)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1

    config = load_runtime_config()
    succeeded = True

    if args.scope in ("players", "both"):
        succeeded = _run_player_synthesis(args, config, model, gateway) and succeeded

    if args.scope in ("teams", "both"):
        succeeded = _run_team_synthesis(args, config, model, gateway) and succeeded

    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
