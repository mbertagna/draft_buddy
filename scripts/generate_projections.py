"""
Entry point: fetch stats, score players, merge ADP, and write generated player CSV.

Run from repo root with PYTHONPATH including ``src`` or through the Docker Compose
``data`` service.
"""

import argparse
import os

import pandas as pd

from draft_buddy.config import load_runtime_config
from draft_buddy.data import (
    FantasyDataProcessor,
    FantasyProsGateway,
    MarketDataUnavailable,
    MarketRankingRequest,
    SleeperCatalogBuilder,
    SleeperHttpGateway,
    add_static_player_value_metrics,
    adp_cache_dir,
    fantasypros_cache_dir,
    sleeper_cache_dir,
)
from draft_buddy.data.pipeline_diagnostics import build_pipeline_diagnostics
from draft_buddy.data.pipeline_report import export_pipeline_report
from draft_buddy.data.sleeper_catalog import format_search_rank_match_summary

DRAFTABLE_POSITIONS = ['QB', 'RB', 'WR', 'TE']
DATA_ROOT = './data'


def generated_output_dir(output_path: str) -> str:
    """
    Return the directory for this run's generated outputs, creating it if needed.

    Parameters
    ----------
    output_path : str
        League-scoped player CSV path from the active config.

    Returns
    -------
    str
        Parent directory of the generated player CSV.
    """
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_missing_nflverse_stats_report(missing_nflverse_stats_df: pd.DataFrame, output_dir: str) -> None:
    """
    Report Sleeper players with no nflverse stats match who are unlikely rookies.

    Parameters
    ----------
    missing_nflverse_stats_df : pd.DataFrame
        Players flagged by ``FantasyDataProcessor._find_likely_veterans_missing_stats``.
    output_dir : str
        Year-scoped directory to save the report into.
    """
    if missing_nflverse_stats_df.empty:
        print("No likely veterans are missing nflverse stats.")
        return

    missing_path = os.path.join(output_dir, 'sleeper_players_missing_nflverse_stats.csv')
    missing_nflverse_stats_df.to_csv(missing_path, index=False)
    print(
        f"⚠️  {len(missing_nflverse_stats_df)} likely veterans have no nflverse stats match. "
        f"Saved to '{missing_path}'"
    )


def check_sleeper_roster_coverage(cache_dir: str, sleeper_league_id: str, output_dir: str) -> None:
    """
    Verify the base-catalog filter didn't exclude a real league-rostered player.

    Parameters
    ----------
    cache_dir : str
        Directory used to cache the Sleeper player directory download.
    sleeper_league_id : str
        Sleeper league id to cross-check roster membership against.
    output_dir : str
        Year-scoped directory to save the report into.
    """
    print("\nChecking Sleeper roster coverage...")
    gateway = SleeperHttpGateway(cache_dir=cache_dir)
    catalog_builder = SleeperCatalogBuilder()

    all_players_df = gateway.fetch_all_players()
    catalog_df = catalog_builder.build_base_catalog(all_players_df, DRAFTABLE_POSITIONS)
    rostered_df = gateway.fetch_league_rosters(sleeper_league_id)

    excluded_df = catalog_builder.find_rostered_players_excluded_by_filter(
        all_players_df, rostered_df, catalog_df
    )
    if excluded_df.empty:
        print("No Sleeper-rostered players were excluded by the catalog filter.")
        return

    excluded_path = os.path.join(output_dir, 'sleeper_players_excluded_by_filter.csv')
    excluded_df.to_csv(excluded_path, index=False)
    print(
        f"⚠️  {len(excluded_df)} Sleeper-rostered players were excluded by the catalog filter. "
        f"Saved to '{excluded_path}'"
    )


def fantasypros_scoring_code(runtime_config) -> str:
    """Return FantasyPros scoring code for the active league rules.

    Parameters
    ----------
    runtime_config
        Loaded runtime configuration.

    Returns
    -------
    str
        FantasyPros scoring code: ``STD``, ``HALF``, or ``PPR``.
    """
    receptions = runtime_config.get_scoring_rules().get("receptions", 0)
    mapping = runtime_config.data.FANTASYPROS_SCORING_BY_RECEPTION_VALUE
    numeric_receptions = float(receptions or 0)
    normalized = str(int(numeric_receptions)) if numeric_receptions.is_integer() else str(numeric_receptions)
    return mapping.get(normalized, "STD")


def fantasypros_rankings_to_adp_dataframe(rankings_df: pd.DataFrame) -> pd.DataFrame:
    """Convert normalized FantasyPros API rankings into matcher input columns.

    Parameters
    ----------
    rankings_df : pandas.DataFrame
        Normalized rows from ``FantasyProsGateway``.

    Returns
    -------
    pandas.DataFrame
        DataFrame shaped like the existing FantasyPros HTML parser output.
    """
    if rankings_df.empty:
        return pd.DataFrame()
    adp_df = pd.DataFrame(
        {
            "Rank": rankings_df["adp"],
            "Player": rankings_df["player_name"],
            "Team": rankings_df["team"],
            "Bye": pd.NA,
            "POS": rankings_df["position"],
            "AVG": rankings_df["adp"],
            "rank_std": rankings_df.get("rank_std"),
            "rank_min": rankings_df.get("rank_min"),
            "rank_max": rankings_df.get("rank_max"),
            "market_source": rankings_df["source"],
            "market_fetched_at": rankings_df["fetched_at"],
        }
    )
    return adp_df[adp_df["AVG"].notna()].reset_index(drop=True)


def resolve_adp_input(runtime_config, draft_year: int, fallback_html_path: str):
    """Return API ADP data when available, otherwise the HTML fallback path.

    Parameters
    ----------
    runtime_config
        Loaded runtime configuration.
    draft_year : int
        Draft season year.
    fallback_html_path : str
        Existing manual FantasyPros HTML snapshot path.

    Returns
    -------
    str or pandas.DataFrame
        Matcher input for ADP merge.
    """
    if not runtime_config.data.FANTASYPROS_API_ENABLED:
        return fallback_html_path
    gateway = FantasyProsGateway(cache_dir=fantasypros_cache_dir(DATA_ROOT))
    request = MarketRankingRequest(
        season=draft_year,
        scoring=fantasypros_scoring_code(runtime_config),
        ranking_type="ADP",
    )
    try:
        rankings_df = gateway.fetch_consensus_rankings(request, allow_stale_cache=True)
    except MarketDataUnavailable as error:
        print(f"FantasyPros API ADP unavailable; using HTML fallback. Reason: {error}")
        return fallback_html_path
    api_adp_df = fantasypros_rankings_to_adp_dataframe(rankings_df)
    if api_adp_df.empty:
        print("FantasyPros API ADP returned no rows; using HTML fallback.")
        return fallback_html_path
    print(f"Loaded {len(api_adp_df)} FantasyPros API ADP rows.")
    return api_adp_df


def main(
    output_path,
    draft_year,
    runtime_config,
    sleeper_league_id=None,
    lookback_seasons=None,
):
    """
    Main function to run the data processing and merging pipeline.
    """
    pd.set_option('display.max_columns', None)

    resolved_lookback = (
        lookback_seasons
        if lookback_seasons is not None
        else runtime_config.data.LEGACY_STATS_LOOKBACK_SEASONS
    )
    stats_start_year = draft_year - resolved_lookback

    league_name = runtime_config.league.display_name or runtime_config.league.league_id
    print(f"--- Running Player Data Processor for {draft_year} Season ({league_name}) ---")
    print(
        f"Using {resolved_lookback}-season nflverse lookback "
        f"(start_year={stats_start_year}, veteran stats through {draft_year - 1})."
    )
    output_dir = generated_output_dir(output_path)

    processor = FantasyDataProcessor(
        scoring_rules=runtime_config.get_scoring_rules(),
        project_rookies=True,
        bye_weeks_override=runtime_config.season.bye_weeks,
        start_year=stats_start_year,
        positions=DRAFTABLE_POSITIONS,
        cache_dir=DATA_ROOT,
    )

    # ADP file path for the given season
    adp_file = os.path.join(
        adp_cache_dir(DATA_ROOT),
        f'fantasypros-{draft_year}-overall-adp-rankings.html',
    )

    process_result = processor.process_draft_data(
        draft_year=draft_year,
        adp_filepath=adp_file,
    )
    computed_players_df = process_result.draft_players_df
    missing_nflverse_stats_df = process_result.missing_nflverse_stats_df
    save_missing_nflverse_stats_report(missing_nflverse_stats_df, output_dir)

    if sleeper_league_id:
        check_sleeper_roster_coverage(
            cache_dir=sleeper_cache_dir(DATA_ROOT), sleeper_league_id=sleeper_league_id, output_dir=output_dir
        )

    adp_input = resolve_adp_input(runtime_config, draft_year, adp_file)

    merged_df, unmatched_df, borderline_df = processor.merge_adp_data(
        computed_df=computed_players_df,
        adp_filepath=adp_input,
        match_threshold=85
    )

    # Normalize column names for outputs
    for df in [merged_df, unmatched_df, borderline_df]:
        if not df.empty:
            if 'AVG' in df.columns and 'adp' not in df.columns:
                df.rename(columns={'AVG': 'adp'}, inplace=True)
            # Ensure single record per player_id in output CSVs where relevant
            if 'player_id' in df.columns:
                df.drop_duplicates(subset=['player_id'], keep='first', inplace=True)

    merged_df = merged_df.rename(columns={'player_display_name': 'name', 'total_pts': 'projected_points'})
    unmatched_df = unmatched_df.rename(columns={'player_display_name': 'name', 'total_pts': 'projected_points'})
    borderline_df = borderline_df.rename(columns={'player_display_name': 'name', 'total_pts': 'projected_points'})
    merged_df = add_static_player_value_metrics(merged_df)

    matched_adp_count = len(merged_df)
    total_adp_players = matched_adp_count + len(unmatched_df)

    search_rank_summary = (
        format_search_rank_match_summary(process_result.search_rank_report)
        if process_result.search_rank_report is not None
        else "Sleeper nflverse match report unavailable."
    )
    diagnostics = build_pipeline_diagnostics(
        league_id=runtime_config.league.league_id,
        league_name=league_name,
        draft_year=draft_year,
        lookback_seasons=resolved_lookback,
        stage_counts=process_result.stage_counts,
        search_rank_report=process_result.search_rank_report,
        search_rank_summary=search_rank_summary,
        missing_nflverse_stats_df=missing_nflverse_stats_df,
        unmatched_adp_df=unmatched_df,
        borderline_adp_df=borderline_df,
        total_adp_players=total_adp_players,
        matched_adp_count=matched_adp_count,
    )
    _json_path, html_path = export_pipeline_report(diagnostics, output_dir)
    print(f"Pipeline report: {html_path}")

    if not merged_df.empty:
        print(f"\n\n--- Successfully Merged Data for {draft_year} ---")
        display_cols = ['Rank', 'Player', 'Team', 'Pos', 'name', 'recent_team', 'projected_points', 'match_score', 'adp']
        print(merged_df[display_cols].head(10))

        merged_df.to_csv(output_path, index=False)
        print(f"\n✅ Saved final merged data to '{output_path}'")

    if not borderline_df.empty:
        borderline_path = os.path.join(output_dir, 'borderline_adp_matches.csv')
        borderline_df.to_csv(borderline_path, index=False)
        print(f"\n✅ Saved borderline cases to '{borderline_path}'")


def parse_cli_args(argv=None, default_year=2026):
    """Parse projection-generation CLI arguments.

    Parameters
    ----------
    argv : list of str, optional
        Argument vector excluding the program name.
    default_year : int, optional
        Draft year used when ``--year`` is omitted.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Process fantasy football data for a given year.")
    parser.add_argument(
        "--year",
        type=int,
        default=default_year,
        help="The draft year to process data for.",
    )
    parser.add_argument(
        "--sleeper_league_id",
        type=str,
        default=None,
        help=(
            "Optional Sleeper league id to verify the base-catalog filter did not "
            "exclude a rostered player. Defaults to the season overlay when set."
        ),
    )
    parser.add_argument(
        "--lookback-seasons",
        type=int,
        default=None,
        help=(
            "Number of completed seasons before --year to load from nflverse for "
            "veteran projections. Default: Config.data.LEGACY_STATS_LOOKBACK_SEASONS (2)."
        ),
    )
    return parser.parse_args(argv)


def resolve_sleeper_league_id(cli_league_id, runtime_config):
    """Return the Sleeper league id for optional roster-coverage checks.

    Parameters
    ----------
    cli_league_id : str or None
        Explicit CLI value.
    runtime_config
        Loaded runtime configuration.

    Returns
    -------
    str or None
        CLI id, overlay id when non-empty, or None.
    """
    if cli_league_id:
        return cli_league_id
    overlay_id = getattr(runtime_config.draft, "SLEEPER_LEAGUE_ID", "") or ""
    return overlay_id or None


if __name__ == "__main__":
    config = load_runtime_config()
    args = parse_cli_args(default_year=config.season.season)
    output_file_path = config.paths.PLAYER_DATA_CSV

    main(
        output_path=output_file_path,
        draft_year=args.year,
        runtime_config=config,
        sleeper_league_id=resolve_sleeper_league_id(args.sleeper_league_id, config),
        lookback_seasons=args.lookback_seasons,
    )
