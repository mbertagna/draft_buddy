"""
Builds the Sleeper-anchored draft catalog.

Sleeper is treated as the source of truth for player identity, current
team/position, and roster status. This module filters Sleeper's full
player directory down to fantasy-relevant, rostered players and shapes
it into the base catalog that nflverse stats and FantasyPros ADP are
later attached onto.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import pandas as pd

INVALID_SLEEPER_SEARCH_RANK = 9_999_999
DEFAULT_TOP_SEARCH_RANK_REPORT_SIZE = 100
DEFAULT_SEARCH_RANK_SCAN_DEPTH = 400
MAX_MATCHED_NAMES_SHOWN_IN_RUN = 3

MatchCategory = Literal["matched", "gap", "rookie", "retired"]

CATALOG_COLUMN_RENAMES = {
    "full_name": "player_display_name",
    "team": "recent_team",
    "status": "sleeper_status",
    "injury_status": "sleeper_injury_status",
    "depth_chart_position": "sleeper_depth_chart_position",
}


class SleeperCatalogBuilder:
    """Builds and validates the Sleeper-sourced draft catalog."""

    def build_base_catalog(
        self, sleeper_players_df: pd.DataFrame, positions: Iterable[str]
    ) -> pd.DataFrame:
        """Filter and shape Sleeper's player directory into a base catalog.

        Parameters
        ----------
        sleeper_players_df : pd.DataFrame
            Full Sleeper player directory, as returned by
            ``SleeperGateway.fetch_all_players``.
        positions : Iterable[str]
            Fantasy-relevant positions to keep (e.g. ``["QB", "RB", "WR", "TE"]``).

        Returns
        -------
        pd.DataFrame
            One row per fantasy-relevant player with ``player_id``
            (int, derived from ``sleeper_id``), ``player_display_name``,
            ``position``, ``recent_team``, ``gsis_id``, ``sleeper_id``,
            ``sleeper_status``, ``sleeper_injury_status``,
            ``sleeper_depth_chart_position``, and ``years_exp``.
        """
        position_list = list(positions)
        is_fantasy_relevant = sleeper_players_df["position"].isin(position_list)
        is_on_team = sleeper_players_df["team"].notna()
        is_active_without_team = (
            sleeper_players_df["status"].eq("Active") & sleeper_players_df["team"].isna()
        )
        is_rosterable = is_fantasy_relevant & (is_on_team | is_active_without_team)
        catalog_df = sleeper_players_df[is_rosterable].rename(columns=CATALOG_COLUMN_RENAMES).copy()
        catalog_df["player_id"] = catalog_df["sleeper_id"].astype(int)
        return catalog_df.reset_index(drop=True)

    def find_rostered_players_excluded_by_filter(
        self,
        all_sleeper_players_df: pd.DataFrame,
        rostered_df: pd.DataFrame,
        catalog_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return league-rostered Sleeper players excluded by the base filter.

        A safety net confirming ``build_base_catalog``'s position/team filter
        didn't wrongly drop a player who is actually rostered in a real league.

        Parameters
        ----------
        all_sleeper_players_df : pd.DataFrame
            Full, unfiltered Sleeper player directory.
        rostered_df : pd.DataFrame
            Rostered players for one league, as returned by
            ``SleeperGateway.fetch_league_rosters``.
        catalog_df : pd.DataFrame
            The catalog produced by ``build_base_catalog``.

        Returns
        -------
        pd.DataFrame
            Sleeper player rows that are rostered in the league but absent
            from the catalog.
        """
        catalog_ids = set(catalog_df["sleeper_id"])
        rostered_ids = {value for value in rostered_df["sleeper_id"] if pd.notna(value)}
        excluded_ids = rostered_ids - catalog_ids
        return (
            all_sleeper_players_df[all_sleeper_players_df["sleeper_id"].isin(excluded_ids)]
            .reset_index(drop=True)
        )


@dataclass(frozen=True)
class SearchRankMatchRow:
    """One Sleeper-ranked player row in the nflverse match report.

    Parameters
    ----------
    search_rank : int
        Sleeper ``search_rank`` value.
    sleeper_id : str
        Sleeper player identifier.
    name : str
        Display name.
    position : str
        Position code.
    category : MatchCategory
        Match outcome or exclusion reason.
    """

    search_rank: int
    sleeper_id: str
    name: str
    position: str
    category: MatchCategory


@dataclass(frozen=True)
class SearchRankMatchReport:
    """Summary and scan rows for Sleeper-to-nflverse match coverage.

    Parameters
    ----------
    eligible_top_n : int
        Target number of eligible (non-rookie, non-retired) players evaluated.
    matched_count : int
        Eligible players with nflverse legacy stats.
    gap_count : int
        Eligible players missing nflverse stats.
    skipped_rookie_count : int
        Rookies skipped while filling the eligible pool.
    skipped_retired_count : int
        Retired/inactive players skipped while filling the eligible pool.
    pool_ranks_scanned : int
        Sleeper ranks walked while filling the eligible pool.
    scan_rows : tuple[SearchRankMatchRow, ...]
        Visual scan rows up to ``scan_depth``.
    """

    eligible_top_n: int
    matched_count: int
    gap_count: int
    skipped_rookie_count: int
    skipped_retired_count: int
    pool_ranks_scanned: int
    scan_rows: tuple[SearchRankMatchRow, ...]


def is_sleeper_rookie(player_row: pd.Series) -> bool:
    """Return whether Sleeper marks the player as an NFL rookie.

    Parameters
    ----------
    player_row : pd.Series
        Sleeper directory row with optional ``years_exp``.

    Returns
    -------
    bool
        ``True`` when ``years_exp == 0``.
    """
    years_exp = player_row.get("years_exp")
    if pd.isna(years_exp):
        return False
    return int(years_exp) == 0


def is_sleeper_retired_or_inactive(player_row: pd.Series, catalog_ids: set[str]) -> bool:
    """Return whether a ranked player should be excluded as retired/inactive.

    Players absent from the draft catalog, inactive without a team, and
    veteran free agents without a team are treated as retired/inactive ghosts.

    Parameters
    ----------
    player_row : pd.Series
        Sleeper directory row.
    catalog_ids : set[str]
        Sleeper ids present in the base draft catalog.

    Returns
    -------
    bool
        ``True`` when the player should be excluded from the eligible pool.
    """
    sleeper_id = str(player_row["sleeper_id"])
    if sleeper_id not in catalog_ids:
        return True

    team = player_row.get("team")
    has_team = pd.notna(team) and str(team).strip() != ""
    if has_team:
        return False

    status = player_row.get("status")
    years_exp = player_row.get("years_exp")
    is_veteran = pd.notna(years_exp) and int(years_exp) > 0
    if status == "Active" and is_veteran:
        return True
    return status != "Active"


def _ranked_skill_players(sleeper_players_df: pd.DataFrame, positions: Iterable[str]) -> pd.DataFrame:
    """Return skill-position Sleeper rows with valid search ranks, sorted."""
    position_list = list(positions)
    ranked_df = sleeper_players_df[
        sleeper_players_df["position"].isin(position_list)
        & sleeper_players_df["search_rank"].notna()
        & (sleeper_players_df["search_rank"] < INVALID_SLEEPER_SEARCH_RANK)
    ].copy()
    ranked_df["sleeper_id"] = ranked_df["sleeper_id"].astype(str)
    return ranked_df.sort_values(["search_rank", "sleeper_id"], kind="mergesort").reset_index(drop=True)


def _catalog_match_lookup(
    catalog_with_stats_df: pd.DataFrame,
) -> tuple[set[str], set[str]]:
    """Return matched and catalog sleeper ids from a stats-attached catalog."""
    id_column = "sleeper_id" if "sleeper_id" in catalog_with_stats_df.columns else "player_id"
    catalog_df = catalog_with_stats_df.copy()
    catalog_df[id_column] = catalog_df[id_column].astype(str)
    matched_ids = set(catalog_df.loc[~catalog_df["is_rookie_original"], id_column])
    catalog_ids = set(catalog_df[id_column])
    return matched_ids, catalog_ids


def _player_display_name(player_row: pd.Series) -> str:
    """Return the best available display name from a Sleeper row."""
    for column in ("full_name", "player_display_name"):
        value = player_row.get(column)
        if pd.notna(value) and str(value).strip():
            return str(value)
    return "Unknown"


def classify_search_rank_player(
    player_row: pd.Series,
    matched_ids: set[str],
    catalog_ids: set[str],
) -> MatchCategory:
    """Classify one Sleeper-ranked player for nflverse match reporting.

    Parameters
    ----------
    player_row : pd.Series
        Sleeper directory row.
    matched_ids : set[str]
        Sleeper ids with nflverse legacy stats attached.
    catalog_ids : set[str]
        Sleeper ids present in the base draft catalog.

    Returns
    -------
    MatchCategory
        ``matched``, ``gap``, ``rookie``, or ``retired``.
    """
    if is_sleeper_rookie(player_row):
        return "rookie"
    if is_sleeper_retired_or_inactive(player_row, catalog_ids):
        return "retired"

    sleeper_id = str(player_row["sleeper_id"])
    if sleeper_id in matched_ids:
        return "matched"
    return "gap"


def build_search_rank_nflverse_match_report(
    sleeper_players_df: pd.DataFrame,
    catalog_with_stats_df: pd.DataFrame,
    positions: Iterable[str],
    *,
    eligible_top_n: int = DEFAULT_TOP_SEARCH_RANK_REPORT_SIZE,
    scan_depth: int = DEFAULT_SEARCH_RANK_SCAN_DEPTH,
) -> SearchRankMatchReport:
    """Build a Sleeper rank-order nflverse match report.

    Walks Sleeper ``search_rank`` from the top down, skipping rookies and
    retired/inactive players until ``eligible_top_n`` eligible veterans are
    evaluated. Also collects up to ``scan_depth`` rows for console display.

    Parameters
    ----------
    sleeper_players_df : pd.DataFrame
        Full Sleeper player directory.
    catalog_with_stats_df : pd.DataFrame
        Catalog immediately after nflverse stats attach.
    positions : Iterable[str]
        Skill positions to include.
    eligible_top_n : int, optional
        Number of eligible players to evaluate for the headline match rate.
    scan_depth : int, optional
        Number of ranked rows to include in the visual scan section.

    Returns
    -------
    SearchRankMatchReport
        Structured report with counts and scan rows.
    """
    ranked_df = _ranked_skill_players(sleeper_players_df, positions)
    if ranked_df.empty:
        return SearchRankMatchReport(
            eligible_top_n=eligible_top_n,
            matched_count=0,
            gap_count=0,
            skipped_rookie_count=0,
            skipped_retired_count=0,
            pool_ranks_scanned=0,
            scan_rows=(),
        )

    matched_ids, catalog_ids = _catalog_match_lookup(catalog_with_stats_df)

    matched_count = 0
    gap_count = 0
    skipped_rookie_count = 0
    skipped_retired_count = 0
    pool_ranks_scanned = 0
    eligible_seen = 0
    scan_rows: list[SearchRankMatchRow] = []

    for _, player_row in ranked_df.iterrows():
        category = classify_search_rank_player(player_row, matched_ids, catalog_ids)
        match_row = SearchRankMatchRow(
            search_rank=int(player_row["search_rank"]),
            sleeper_id=str(player_row["sleeper_id"]),
            name=_player_display_name(player_row),
            position=str(player_row.get("position", "?")),
            category=category,
        )

        if len(scan_rows) < scan_depth:
            scan_rows.append(match_row)

        pool_full = eligible_seen >= eligible_top_n
        if not pool_full:
            pool_ranks_scanned += 1
            if category == "rookie":
                skipped_rookie_count += 1
            elif category == "retired":
                skipped_retired_count += 1
            else:
                eligible_seen += 1
                if category == "matched":
                    matched_count += 1
                else:
                    gap_count += 1

        if pool_full and len(scan_rows) >= scan_depth:
            break

    return SearchRankMatchReport(
        eligible_top_n=eligible_top_n,
        matched_count=matched_count,
        gap_count=gap_count,
        skipped_rookie_count=skipped_rookie_count,
        skipped_retired_count=skipped_retired_count,
        pool_ranks_scanned=pool_ranks_scanned,
        scan_rows=tuple(scan_rows),
    )


def format_search_rank_match_summary(report: SearchRankMatchReport) -> str:
    """Format the headline nflverse match summary for eligible top-ranked players.

    Parameters
    ----------
    report : SearchRankMatchReport
        Report from ``build_search_rank_nflverse_match_report``.

    Returns
    -------
    str
        One-line summary string.
    """
    evaluated = report.matched_count + report.gap_count
    return (
        f"Sleeper nflverse match: {report.matched_count}/{evaluated} eligible top-ranked "
        f"players matched (target {report.eligible_top_n}; scanned {report.pool_ranks_scanned} "
        f"Sleeper ranks, skipped {report.skipped_rookie_count} rookies and "
        f"{report.skipped_retired_count} retired/inactive)."
    )


def _print_collapsed_category_run(
    rows: list[SearchRankMatchRow],
    *,
    symbol: str,
    suffix: str,
    max_names_shown: int,
) -> None:
    """Print a collapsed run of scan rows sharing one category."""
    if not rows:
        return

    for row in rows[:max_names_shown]:
        print(f"  {symbol} {row.name} ({row.position})  rank {row.search_rank}{suffix}")

    remaining = len(rows) - max_names_shown
    if remaining > 0:
        label = rows[0].category.replace("_", " ")
        print(f"  {symbol} ({remaining} more {label} players)")


def print_search_rank_nflverse_match_report(
    report: SearchRankMatchReport,
    *,
    max_matched_names_shown: int = MAX_MATCHED_NAMES_SHOWN_IN_RUN,
) -> None:
    """Print a visual Sleeper rank scan focused on unmatched gaps.

    Matched and excluded-player runs are collapsed after a few sample names.

    Parameters
    ----------
    report : SearchRankMatchReport
        Report from ``build_search_rank_nflverse_match_report``.
    max_matched_names_shown : int, optional
        Number of consecutive matched names to print before collapsing a run.
    """
    print(format_search_rank_match_summary(report))
    if not report.scan_rows:
        return

    print("\nSleeper rank scan (gaps shown in full; matched and excluded runs collapsed):")
    category_suffix = {
        "matched": "",
        "gap": "  — no nflverse stats",
        "rookie": "  — rookie (excluded)",
        "retired": "  — retired/inactive (excluded)",
    }
    category_symbol = {
        "matched": "✓",
        "gap": "✗",
        "rookie": "-",
        "retired": "-",
    }
    collapse_categories = {"matched", "rookie", "retired"}

    index = 0
    scan_rows = list(report.scan_rows)
    while index < len(scan_rows):
        row = scan_rows[index]
        if row.category in collapse_categories:
            run_end = index + 1
            while run_end < len(scan_rows) and scan_rows[run_end].category == row.category:
                run_end += 1
            run_rows = scan_rows[index:run_end]
            max_names = max_matched_names_shown if row.category == "matched" else 1
            _print_collapsed_category_run(
                run_rows,
                symbol=category_symbol[row.category],
                suffix=category_suffix[row.category],
                max_names_shown=max_names,
            )
            index = run_end
            continue

        print(
            f"  {category_symbol[row.category]} {row.name} ({row.position})  "
            f"rank {row.search_rank}{category_suffix[row.category]}"
        )
        index += 1


def select_top_search_rank_players(
    sleeper_players_df: pd.DataFrame,
    positions: Iterable[str],
    top_n: int = DEFAULT_TOP_SEARCH_RANK_REPORT_SIZE,
) -> pd.DataFrame:
    """Return the top N Sleeper players by ``search_rank`` for given positions.

    Sleeper assigns lower ``search_rank`` values to more draft-relevant players.
    Rows with missing or sentinel ranks (``>= INVALID_SLEEPER_SEARCH_RANK``) are
    excluded before ranking.

    Parameters
    ----------
    sleeper_players_df : pd.DataFrame
        Full Sleeper player directory.
    positions : Iterable[str]
        Positions to include (for example ``["QB", "RB", "WR", "TE"]``).
    top_n : int, optional
        Number of top-ranked players to return.

    Returns
    -------
    pd.DataFrame
        Up to ``top_n`` rows sorted by ascending ``search_rank``, then
        ``sleeper_id`` for deterministic tie-breaking.
    """
    position_list = list(positions)
    ranked_df = sleeper_players_df[
        sleeper_players_df["position"].isin(position_list)
        & sleeper_players_df["search_rank"].notna()
        & (sleeper_players_df["search_rank"] < INVALID_SLEEPER_SEARCH_RANK)
    ].copy()
    ranked_df["sleeper_id"] = ranked_df["sleeper_id"].astype(str)
    ranked_df = ranked_df.sort_values(["search_rank", "sleeper_id"], kind="mergesort")
    return ranked_df.head(top_n).reset_index(drop=True)


def summarize_nflverse_match_for_top_search_rank(
    top_search_rank_df: pd.DataFrame,
    catalog_with_stats_df: pd.DataFrame,
) -> tuple[int, int, pd.DataFrame]:
    """Count nflverse legacy-stats matches among top Sleeper-ranked players.

    A player counts as matched when the catalog row has
    ``is_rookie_original == False`` after the nflverse stats attach step.

    Parameters
    ----------
    top_search_rank_df : pd.DataFrame
        Top-ranked Sleeper players from ``select_top_search_rank_players``.
    catalog_with_stats_df : pd.DataFrame
        Sleeper catalog after ``attach_legacy_stats_by_player_id``.

    Returns
    -------
    tuple[int, int, pd.DataFrame]
        ``(matched_count, total_count, unmatched_top_players_df)``.
    """
    if top_search_rank_df.empty:
        return 0, 0, pd.DataFrame()

    catalog_df = catalog_with_stats_df.copy()
    id_column = "sleeper_id" if "sleeper_id" in catalog_df.columns else "player_id"
    catalog_df[id_column] = catalog_df[id_column].astype(str)
    matched_ids = set(catalog_df.loc[~catalog_df["is_rookie_original"], id_column])
    catalog_ids = set(catalog_df[id_column])

    unmatched_rows = []
    matched_count = 0
    for _, row in top_search_rank_df.iterrows():
        sleeper_id = str(row["sleeper_id"])
        if sleeper_id in matched_ids:
            matched_count += 1
            continue
        failure_reason = "not in catalog" if sleeper_id not in catalog_ids else "no nflverse stats"
        unmatched_rows.append({**row.to_dict(), "match_failure_reason": failure_reason})

    unmatched_df = pd.DataFrame(unmatched_rows)
    return matched_count, len(top_search_rank_df), unmatched_df


def format_top_search_rank_match_summary(matched_count: int, total_count: int) -> str:
    """Format a one-line summary of top Sleeper player nflverse match coverage.

    Parameters
    ----------
    matched_count : int
        Number of top-ranked players with nflverse legacy stats.
    total_count : int
        Total top-ranked players evaluated.

    Returns
    -------
    str
        Human-readable summary string.

    Notes
    -----
    Prefer ``format_search_rank_match_summary`` for eligible-player reporting.
    """
    return (
        f"Sleeper top {total_count} players: "
        f"{matched_count}/{total_count} matched to nflverse legacy stats."
    )
