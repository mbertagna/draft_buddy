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
    SleeperCatalogBuilder,
    SleeperHttpGateway,
    adp_cache_dir,
    sleeper_cache_dir,
)

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


def main(
    output_path,
    draft_year,
    rookie_projection_method,
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
        rookie_projection_method=rookie_projection_method,
        cache_dir=DATA_ROOT,
    )

    # ADP file path for the given season
    adp_file = os.path.join(
        adp_cache_dir(DATA_ROOT),
        f'fantasypros-{draft_year}-overall-adp-rankings.html',
    )

    computed_players_df, _, missing_nflverse_stats_df = processor.process_draft_data(
        draft_year=draft_year,
        adp_filepath=adp_file,
    )
    save_missing_nflverse_stats_report(missing_nflverse_stats_df, output_dir)

    if sleeper_league_id:
        check_sleeper_roster_coverage(
            cache_dir=sleeper_cache_dir(DATA_ROOT), sleeper_league_id=sleeper_league_id, output_dir=output_dir
        )

    merged_df, unmatched_df, borderline_df = processor.merge_adp_data(
        computed_df=computed_players_df,
        adp_filepath=adp_file,
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

    # Extra diagnostic: unmatched highest-ADP per position (ADP worst values)
    if not unmatched_df.empty:
        adp_col = 'adp' if 'adp' in unmatched_df.columns else ('Rank' if 'Rank' in unmatched_df.columns else None)
        if adp_col and 'Pos' in unmatched_df.columns:
            tmp = unmatched_df[['Pos', 'Player', 'Team', adp_col]].copy()
            tmp[adp_col] = pd.to_numeric(tmp[adp_col], errors='coerce')
            tmp = tmp[pd.notna(tmp[adp_col])]
            if not tmp.empty:
                # Normalize positions like WR1 -> WR
                tmp['PosBase'] = tmp['Pos'].astype(str).str.extract(r'([A-Za-z]+)')[0]
                print("\nUnmatched Highest-ADP per Position:")
                for position_value, g in tmp.groupby('PosBase'):
                    r = g.sort_values(by=adp_col, ascending=False).iloc[0]
                    print(f"- {position_value}: {r['Player']} (Team: {r.get('Team', 'N/A')}, ADP: {r[adp_col]})")
    # ID normalization and assignment now handled inside FantasyDataProcessor.
    # No additional ID manipulation required here.

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process fantasy football data for a given year.')
    parser.add_argument('--year', type=int, default=2025, help='The draft year to process data for.')
    parser.add_argument('--rookie_projection_method', type=str, default='draft', choices=['draft', 'adp', 'hybrid'],
                        help='Method to project rookie points: draft (slot scaling), adp (ADP interpolation), or hybrid (average).')
    parser.add_argument('--sleeper_league_id', type=str, default=None,
                        help='Optional Sleeper league id to verify the base-catalog filter did not exclude a rostered player.')
    parser.add_argument(
        '--lookback-seasons',
        type=int,
        default=None,
        help=(
            'Number of completed seasons before --year to load from nflverse for '
            'veteran projections. Default: Config.data.LEGACY_STATS_LOOKBACK_SEASONS (2).'
        ),
    )

    args = parser.parse_args()

    config = load_runtime_config()
    output_file_path = config.paths.PLAYER_DATA_CSV

    main(
        output_path=output_file_path,
        draft_year=args.year,
        rookie_projection_method=args.rookie_projection_method,
        runtime_config=config,
        sleeper_league_id=args.sleeper_league_id,
        lookback_seasons=args.lookback_seasons,
    )
