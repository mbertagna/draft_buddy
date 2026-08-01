"""Structured diagnostics for the player-data generation pipeline.

Collects stage counts, Sleeper-to-nflverse match coverage, draft-relevant
missing veterans, and ADP match summaries into JSON-serializable dataclasses
for HTML report rendering.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional
from urllib.parse import quote_plus

import pandas as pd

from draft_buddy.data.sleeper_catalog import (
    DEFAULT_SEARCH_RANK_SCAN_DEPTH,
    SearchRankMatchReport,
    SearchRankMatchRow,
)

SKILL_POSITIONS = frozenset({"QB", "RB", "WR", "TE"})
GOOGLE_SEARCH_SUFFIX = "nfl career timeline"


@dataclass(frozen=True)
class StageCounts:
    """Player counts at each pipeline stage.

    Parameters
    ----------
    sleeper_directory : int
        Rows in the full Sleeper player directory.
    catalog : int
        Fantasy-relevant skill players in the base catalog.
    gsis_resolved : int
        Catalog rows with a non-null ``nflverse_player_id``.
    nflverse_matched : int
        Catalog rows with legacy stats attached.
    rookie_projected : int
        Catalog rows flagged for rookie / no-stats projection.
    """

    sleeper_directory: int
    catalog: int
    gsis_resolved: int
    nflverse_matched: int
    rookie_projected: int


@dataclass(frozen=True)
class UnmatchedPlayerRow:
    """One unmatched or gap player for copyable review lists.

    Parameters
    ----------
    name : str
        Display name.
    position : str
        Position code.
    detail : str
        Rank, ADP, or other identifying context.
    reason : str
        Short failure reason.
    google_url : str
        Google search URL for career timeline lookup.
    copy_line : str
        One-line text suitable for bulk copy.
    """

    name: str
    position: str
    detail: str
    reason: str
    google_url: str
    copy_line: str


@dataclass(frozen=True)
class ScanPlayerRow:
    """One Sleeper rank-scan row for the interactive strip chart.

    Parameters
    ----------
    search_rank : int
        Sleeper search rank.
    name : str
        Display name.
    position : str
        Position code.
    category : str
        Match category label.
    """

    search_rank: int
    name: str
    position: str
    category: str


@dataclass(frozen=True)
class NflverseMatchSection:
    """Sleeper-to-nflverse match coverage for the report.

    Parameters
    ----------
    eligible_top_n : int
        Target eligible pool size.
    matched_count : int
        Eligible matched veterans.
    gap_count : int
        Eligible unmatched veterans.
    skipped_rookie_count : int
        Rookies skipped while filling the pool.
    skipped_retired_count : int
        Retired/inactive skipped while filling the pool.
    pool_ranks_scanned : int
        Ranks walked for the eligible pool.
    summary_line : str
        One-line console-style summary.
    scan_rows : tuple[ScanPlayerRow, ...]
        Rows for the rank-scan strip.
    gap_players : tuple[UnmatchedPlayerRow, ...]
        Copyable gap players from the scan.
    """

    eligible_top_n: int
    matched_count: int
    gap_count: int
    skipped_rookie_count: int
    skipped_retired_count: int
    pool_ranks_scanned: int
    summary_line: str
    scan_rows: tuple[ScanPlayerRow, ...] = ()
    gap_players: tuple[UnmatchedPlayerRow, ...] = ()


@dataclass(frozen=True)
class DraftRelevantMissingVeterans:
    """Likely veterans missing nflverse stats within draft-relevant ranks.

    Parameters
    ----------
    total_missing_veterans : int
        Full missing-veteran count before rank filter.
    max_search_rank : int
        Inclusive upper bound used for the draft-relevant filter.
    players : tuple[UnmatchedPlayerRow, ...]
        Filtered, rank-sorted copyable rows.
    """

    total_missing_veterans: int
    max_search_rank: int
    players: tuple[UnmatchedPlayerRow, ...] = ()


@dataclass(frozen=True)
class AdpMatchSection:
    """ADP merge coverage for the report.

    Parameters
    ----------
    total_adp_players : int
        Total ADP rows evaluated.
    matched_count : int
        Successfully matched ADP rows.
    borderline_count : int
        Borderline match count.
    unmatched_count : int
        Unmatched ADP row count.
    unmatched_dst_count : int
        Unmatched DST/DEF rows (excluded from skill copy list).
    skill_unmatched : tuple[UnmatchedPlayerRow, ...]
        Copyable unmatched skill-position ADP players.
    """

    total_adp_players: int
    matched_count: int
    borderline_count: int
    unmatched_count: int
    unmatched_dst_count: int
    skill_unmatched: tuple[UnmatchedPlayerRow, ...] = ()


@dataclass(frozen=True)
class PipelineDiagnostics:
    """Full diagnostics payload for one data-generation run.

    Parameters
    ----------
    league_id : str
        Active league identifier.
    league_name : str
        Display name for the league.
    draft_year : int
        Draft year processed.
    lookback_seasons : int
        nflverse lookback window.
    stage_counts : StageCounts
        Funnel counts.
    nflverse_match : NflverseMatchSection
        Sleeper-to-nflverse coverage.
    draft_relevant_missing : DraftRelevantMissingVeterans
        Filtered missing-veteran list.
    adp_match : AdpMatchSection
        ADP merge coverage.
    """

    league_id: str
    league_name: str
    draft_year: int
    lookback_seasons: int
    stage_counts: StageCounts
    nflverse_match: NflverseMatchSection
    draft_relevant_missing: DraftRelevantMissingVeterans
    adp_match: AdpMatchSection


@dataclass(frozen=True)
class ProcessDraftResult:
    """Return value from ``FantasyDataProcessor.process_draft_data``.

    Parameters
    ----------
    draft_players_df : pd.DataFrame
        Final draft player frame.
    weekly_projections : dict
        Weekly projection mapping.
    missing_nflverse_stats_df : pd.DataFrame
        Likely veterans missing nflverse stats.
    stage_counts : StageCounts
        Funnel counts captured during processing.
    search_rank_report : SearchRankMatchReport or None
        Rank-scan report when Sleeper path was used.
    """

    draft_players_df: pd.DataFrame
    weekly_projections: dict
    missing_nflverse_stats_df: pd.DataFrame
    stage_counts: StageCounts
    search_rank_report: Optional[SearchRankMatchReport] = None


def build_google_career_timeline_url(player_name: str) -> str:
    """Return a Google search URL for a player's NFL career timeline.

    Parameters
    ----------
    player_name : str
        Player display name.

    Returns
    -------
    str
        Absolute Google search URL.
    """
    query = f"{player_name} {GOOGLE_SEARCH_SUFFIX}"
    return f"https://www.google.com/search?q={quote_plus(query)}"


def format_gap_copy_line(name: str, position: str, search_rank: int) -> str:
    """Format one nflverse gap line for bulk copy.

    Parameters
    ----------
    name : str
        Player name.
    position : str
        Position code.
    search_rank : int
        Sleeper search rank.

    Returns
    -------
    str
        Copyable one-line summary.
    """
    return f"{name} ({position}) rank {search_rank} — no nflverse stats"


def format_adp_unmatched_copy_line(name: str, position: str, adp_value: Optional[float]) -> str:
    """Format one unmatched ADP line for bulk copy.

    Parameters
    ----------
    name : str
        Player name.
    position : str
        Position code.
    adp_value : float or None
        ADP or rank value when available.

    Returns
    -------
    str
        Copyable one-line summary.
    """
    if adp_value is None or pd.isna(adp_value):
        detail = "ADP n/a"
    else:
        detail = f"ADP {adp_value}"
    return f"{name} ({position}) {detail} — unmatched ADP"


def build_stage_counts(
    *,
    sleeper_directory: int,
    catalog: int,
    gsis_resolved: int,
    nflverse_matched: int,
    rookie_projected: int,
) -> StageCounts:
    """Build a ``StageCounts`` instance from raw integers.

    Parameters
    ----------
    sleeper_directory : int
        Sleeper directory size.
    catalog : int
        Base catalog size.
    gsis_resolved : int
        Rows with resolved nflverse ids.
    nflverse_matched : int
        Rows with legacy stats.
    rookie_projected : int
        Rows without legacy stats.

    Returns
    -------
    StageCounts
        Funnel counts.
    """
    return StageCounts(
        sleeper_directory=sleeper_directory,
        catalog=catalog,
        gsis_resolved=gsis_resolved,
        nflverse_matched=nflverse_matched,
        rookie_projected=rookie_projected,
    )


def empty_stage_counts() -> StageCounts:
    """Return zeroed stage counts for non-Sleeper processing paths.

    Returns
    -------
    StageCounts
        All-zero funnel counts.
    """
    return StageCounts(
        sleeper_directory=0,
        catalog=0,
        gsis_resolved=0,
        nflverse_matched=0,
        rookie_projected=0,
    )


def _unmatched_from_scan_gap(row: SearchRankMatchRow) -> UnmatchedPlayerRow:
    """Convert a scan gap row into a copyable unmatched player row."""
    copy_line = format_gap_copy_line(row.name, row.position, row.search_rank)
    return UnmatchedPlayerRow(
        name=row.name,
        position=row.position,
        detail=f"rank {row.search_rank}",
        reason="no nflverse stats",
        google_url=build_google_career_timeline_url(row.name),
        copy_line=copy_line,
    )


def build_nflverse_match_section(
    report: Optional[SearchRankMatchReport],
    summary_line: str,
) -> NflverseMatchSection:
    """Build the nflverse match section from a search-rank report.

    Parameters
    ----------
    report : SearchRankMatchReport or None
        Rank-scan report from the processor.
    summary_line : str
        One-line summary string.

    Returns
    -------
    NflverseMatchSection
        Structured match coverage for the report.
    """
    if report is None:
        return NflverseMatchSection(
            eligible_top_n=0,
            matched_count=0,
            gap_count=0,
            skipped_rookie_count=0,
            skipped_retired_count=0,
            pool_ranks_scanned=0,
            summary_line=summary_line,
        )

    scan_rows = tuple(
        ScanPlayerRow(
            search_rank=row.search_rank,
            name=row.name,
            position=row.position,
            category=row.category,
        )
        for row in report.scan_rows
    )
    gap_players = tuple(
        _unmatched_from_scan_gap(row) for row in report.scan_rows if row.category == "gap"
    )
    return NflverseMatchSection(
        eligible_top_n=report.eligible_top_n,
        matched_count=report.matched_count,
        gap_count=report.gap_count,
        skipped_rookie_count=report.skipped_rookie_count,
        skipped_retired_count=report.skipped_retired_count,
        pool_ranks_scanned=report.pool_ranks_scanned,
        summary_line=summary_line,
        scan_rows=scan_rows,
        gap_players=gap_players,
    )


def build_draft_relevant_missing_veterans(
    missing_nflverse_stats_df: pd.DataFrame,
    *,
    max_search_rank: int = DEFAULT_SEARCH_RANK_SCAN_DEPTH,
) -> DraftRelevantMissingVeterans:
    """Filter missing veterans to draft-relevant Sleeper ranks.

    Parameters
    ----------
    missing_nflverse_stats_df : pd.DataFrame
        Full missing-veteran frame from the processor.
    max_search_rank : int, optional
        Keep rows with ``search_rank`` below this threshold.

    Returns
    -------
    DraftRelevantMissingVeterans
        Filtered copyable list plus totals.
    """
    total = len(missing_nflverse_stats_df)
    if missing_nflverse_stats_df.empty or "search_rank" not in missing_nflverse_stats_df.columns:
        return DraftRelevantMissingVeterans(
            total_missing_veterans=total,
            max_search_rank=max_search_rank,
            players=(),
        )

    filtered = missing_nflverse_stats_df.copy()
    filtered["search_rank"] = pd.to_numeric(filtered["search_rank"], errors="coerce")
    filtered = filtered[filtered["search_rank"].notna() & (filtered["search_rank"] < max_search_rank)]
    filtered = filtered.sort_values(["search_rank", "player_display_name"], kind="mergesort")

    players: list[UnmatchedPlayerRow] = []
    for _, row in filtered.iterrows():
        name = str(row.get("player_display_name") or row.get("name") or "Unknown")
        position = str(row.get("position") or "?")
        search_rank = int(row["search_rank"])
        players.append(
            UnmatchedPlayerRow(
                name=name,
                position=position,
                detail=f"rank {search_rank}",
                reason="no nflverse stats",
                google_url=build_google_career_timeline_url(name),
                copy_line=format_gap_copy_line(name, position, search_rank),
            )
        )
    return DraftRelevantMissingVeterans(
        total_missing_veterans=total,
        max_search_rank=max_search_rank,
        players=tuple(players),
    )


def _adp_position_base(raw_position: Any) -> str:
    """Normalize ADP position labels like ``WR1`` to ``WR``."""
    text = str(raw_position or "")
    letters = "".join(character for character in text if character.isalpha()).upper()
    return letters or "?"


def build_adp_match_section(
    unmatched_df: pd.DataFrame,
    borderline_df: pd.DataFrame,
    *,
    total_adp_players: int,
    matched_count: int,
) -> AdpMatchSection:
    """Build the ADP match section from merge result frames.

    Parameters
    ----------
    unmatched_df : pd.DataFrame
        Unmatched ADP rows.
    borderline_df : pd.DataFrame
        Borderline ADP rows.
    total_adp_players : int
        Total ADP players evaluated.
    matched_count : int
        Successfully matched count.

    Returns
    -------
    AdpMatchSection
        Structured ADP coverage for the report.
    """
    unmatched_dst_count = 0
    skill_unmatched: list[UnmatchedPlayerRow] = []

    if not unmatched_df.empty:
        working = unmatched_df.copy()
        pos_col = "Pos" if "Pos" in working.columns else ("position" if "position" in working.columns else None)
        name_col = "Player" if "Player" in working.columns else ("name" if "name" in working.columns else None)
        adp_col = "adp" if "adp" in working.columns else ("AVG" if "AVG" in working.columns else None)
        if name_col is not None:
            for _, row in working.iterrows():
                position = _adp_position_base(row.get(pos_col)) if pos_col else "?"
                name = str(row.get(name_col) or "Unknown")
                if position in {"DST", "DEF", "TEAM"}:
                    unmatched_dst_count += 1
                    continue
                if position not in SKILL_POSITIONS:
                    continue
                adp_value = None
                if adp_col is not None:
                    adp_value = pd.to_numeric(row.get(adp_col), errors="coerce")
                skill_unmatched.append(
                    UnmatchedPlayerRow(
                        name=name,
                        position=position,
                        detail=(
                            f"ADP {adp_value}"
                            if adp_value is not None and pd.notna(adp_value)
                            else "ADP n/a"
                        ),
                        reason="unmatched ADP",
                        google_url=build_google_career_timeline_url(name),
                        copy_line=format_adp_unmatched_copy_line(name, position, adp_value),
                    )
                )

    return AdpMatchSection(
        total_adp_players=total_adp_players,
        matched_count=matched_count,
        borderline_count=len(borderline_df),
        unmatched_count=len(unmatched_df),
        unmatched_dst_count=unmatched_dst_count,
        skill_unmatched=tuple(skill_unmatched),
    )


def build_pipeline_diagnostics(
    *,
    league_id: str,
    league_name: str,
    draft_year: int,
    lookback_seasons: int,
    stage_counts: StageCounts,
    search_rank_report: Optional[SearchRankMatchReport],
    search_rank_summary: str,
    missing_nflverse_stats_df: pd.DataFrame,
    unmatched_adp_df: pd.DataFrame,
    borderline_adp_df: pd.DataFrame,
    total_adp_players: int,
    matched_adp_count: int,
) -> PipelineDiagnostics:
    """Assemble the full diagnostics payload for report rendering.

    Parameters
    ----------
    league_id : str
        League identifier.
    league_name : str
        League display name.
    draft_year : int
        Draft year.
    lookback_seasons : int
        nflverse lookback window.
    stage_counts : StageCounts
        Funnel counts from processing.
    search_rank_report : SearchRankMatchReport or None
        Rank-scan report.
    search_rank_summary : str
        One-line match summary.
    missing_nflverse_stats_df : pd.DataFrame
        Full missing-veteran frame.
    unmatched_adp_df : pd.DataFrame
        Unmatched ADP rows.
    borderline_adp_df : pd.DataFrame
        Borderline ADP rows.
    total_adp_players : int
        Total ADP rows.
    matched_adp_count : int
        Matched ADP rows.

    Returns
    -------
    PipelineDiagnostics
        Complete diagnostics payload.
    """
    return PipelineDiagnostics(
        league_id=league_id,
        league_name=league_name,
        draft_year=draft_year,
        lookback_seasons=lookback_seasons,
        stage_counts=stage_counts,
        nflverse_match=build_nflverse_match_section(search_rank_report, search_rank_summary),
        draft_relevant_missing=build_draft_relevant_missing_veterans(missing_nflverse_stats_df),
        adp_match=build_adp_match_section(
            unmatched_adp_df,
            borderline_adp_df,
            total_adp_players=total_adp_players,
            matched_count=matched_adp_count,
        ),
    )


def diagnostics_to_dict(diagnostics: PipelineDiagnostics) -> dict[str, Any]:
    """Convert diagnostics to a plain JSON-serializable dict.

    Parameters
    ----------
    diagnostics : PipelineDiagnostics
        Diagnostics payload.

    Returns
    -------
    dict[str, Any]
        Nested dictionary suitable for ``json.dump``.
    """
    return asdict(diagnostics)
