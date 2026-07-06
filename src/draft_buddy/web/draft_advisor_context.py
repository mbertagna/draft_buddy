"""Markdown context assembly for the live draft assistant."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from draft_buddy.core.entities import Player
from draft_buddy.data.insights.schemas import PlayerInsight
from draft_buddy.web.draft_advisor_filter import passes_gp_filter

POSITIONS: Tuple[str, ...] = ("QB", "RB", "WR", "TE")
POSITION_ORDER: Dict[str, int] = {position: index for index, position in enumerate(POSITIONS)}
DEFAULT_TOP_K = 5
HIGH_PRIORITY_TOP_K = 7


@dataclass(frozen=True)
class CandidateRow:
    """One player row in an assistant candidate table."""

    player: Player
    vorp: float


def position_top_k_map(rl_probs: Dict[str, float]) -> Dict[str, int]:
    """Map each position to top-K based on RL probability ranking.

    Parameters
    ----------
    rl_probs : Dict[str, float]
        Position probabilities from the RL model.

    Returns
    -------
    Dict[str, int]
        Position to K mapping. Top two positions get 7; others get 5.
    """
    ranked = sorted(
        POSITIONS,
        key=lambda position: (-rl_probs.get(position, 0.0), POSITION_ORDER[position]),
    )
    top_k = {position: DEFAULT_TOP_K for position in POSITIONS}
    for position in ranked[:2]:
        top_k[position] = HIGH_PRIORITY_TOP_K
    return top_k


def filter_available_players(
    players: Sequence[Player],
    gp_min: Optional[float],
) -> List[Player]:
    """Apply GP filter to available players.

    Parameters
    ----------
    players : Sequence[Player]
        Available player records.
    gp_min : Optional[float]
        Minimum games-played fraction.

    Returns
    -------
    List[Player]
        Filtered players.
    """
    return [player for player in players if passes_gp_filter(player.games_played_frac, gp_min)]


def build_candidate_rows(
    players: Sequence[Player],
    baselines: Dict[str, float],
) -> List[CandidateRow]:
    """Build candidate rows with VORP for each player.

    Parameters
    ----------
    players : Sequence[Player]
        Filtered available players.
    baselines : Dict[str, float]
        Positional replacement baselines.

    Returns
    -------
    List[CandidateRow]
        Candidate rows with computed VORP.
    """
    rows: List[CandidateRow] = []
    for player in players:
        baseline = baselines.get(player.position, 0.0)
        rows.append(CandidateRow(player=player, vorp=player.projected_points - baseline))
    return rows


def _top_by_vorp(rows: Sequence[CandidateRow], position: str, limit: int) -> List[CandidateRow]:
    """Return top candidates by VORP for one position."""
    position_rows = [row for row in rows if row.player.position == position]
    return sorted(position_rows, key=lambda row: (-row.vorp, row.player.player_id))[:limit]


def _top_by_adp(rows: Sequence[CandidateRow], position: str, limit: int) -> List[CandidateRow]:
    """Return top candidates by ADP for one position."""
    position_rows = [
        row
        for row in rows
        if row.player.position == position and np.isfinite(row.player.adp)
    ]
    return sorted(position_rows, key=lambda row: (row.player.adp, row.player.player_id))[:limit]


def collect_candidate_player_ids(
    rows: Sequence[CandidateRow],
    top_k_by_position: Dict[str, int],
) -> set[int]:
    """Collect the union of all shortlist player ids.

    Parameters
    ----------
    rows : Sequence[CandidateRow]
        All candidate rows after GP filtering.
    top_k_by_position : Dict[str, int]
        Per-position shortlist size.

    Returns
    -------
    set[int]
        Valid recommendation player ids.
    """
    valid_ids: set[int] = set()
    for position in POSITIONS:
        limit = top_k_by_position[position]
        for row in _top_by_vorp(rows, position, limit):
            valid_ids.add(row.player.player_id)
        for row in _top_by_adp(rows, position, limit):
            valid_ids.add(row.player.player_id)
    return valid_ids


def _format_adp(adp: float) -> str:
    """Format ADP for markdown tables."""
    if not np.isfinite(adp):
        return "N/A"
    return f"{adp:.1f}"


def _format_gp_frac(value: Any) -> str:
    """Format games-played fraction for markdown tables."""
    if value == "R":
        return "R"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(numeric):
        return "N/A"
    return f"{numeric:.2f}"


def _format_bye_week(bye_week: Optional[int]) -> str:
    """Format bye week for markdown tables."""
    if bye_week is None or (isinstance(bye_week, float) and np.isnan(bye_week)):
        return "N/A"
    return str(int(bye_week))


def _insight_cell(insight: Optional[PlayerInsight], field: str) -> str:
    """Return one insight field for a table cell."""
    if insight is None:
        return ""
    value = getattr(insight, field, "")
    if hasattr(value, "value"):
        return str(value.value)
    if isinstance(value, list):
        return ", ".join(str(item.value if hasattr(item, "value") else item) for item in value)
    return str(value)


def _render_candidate_table(title: str, rows: Sequence[CandidateRow], insights: Dict[int, PlayerInsight]) -> str:
    """Render one candidate markdown table."""
    if not rows:
        return f"### {title}\n\nNo candidates.\n"

    lines = [
        f"### {title}",
        "",
        "| player_id | name | vorp | adp | gp_frac | proj | bye | outlook | depth_role | playing_time | injury_risk | recovery | tags | confidence | fields_unknown |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        player = row.player
        insight = insights.get(player.player_id)
        lines.append(
            "| "
            + " | ".join(
                [
                    str(player.player_id),
                    player.name,
                    f"{row.vorp:.1f}",
                    _format_adp(player.adp),
                    _format_gp_frac(player.games_played_frac),
                    f"{player.projected_points:.1f}",
                    _format_bye_week(player.bye_week),
                    _insight_cell(insight, "outlook_phrase"),
                    _insight_cell(insight, "depth_role"),
                    _insight_cell(insight, "playing_time_tier"),
                    _insight_cell(insight, "injury_risk"),
                    _insight_cell(insight, "recovery_status"),
                    _insight_cell(insight, "tags"),
                    _insight_cell(insight, "overall_confidence"),
                    _insight_cell(insight, "fields_unknown"),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _compute_open_needs(
    roster_counts: Dict[str, int],
    roster_structure: Dict[str, int],
    total_bench_size: int,
) -> List[str]:
    """Return human-readable open roster needs."""
    needs: List[str] = []
    for position, required in roster_structure.items():
        if position == "FLEX":
            continue
        current = roster_counts.get(position, 0)
        open_slots = max(0, required - current)
        if open_slots:
            needs.append(f"{position} x{open_slots}")
    flex_required = roster_structure.get("FLEX", 0)
    flex_current = roster_counts.get("FLEX", 0)
    flex_open = max(0, flex_required - flex_current)
    if flex_open:
        needs.append(f"FLEX x{flex_open}")
    total_players = sum(roster_counts.values())
    bench_open = max(0, sum(roster_structure.values()) + total_bench_size - total_players)
    if bench_open:
        needs.append(f"bench x{bench_open}")
    return needs


def _format_rl_probs(rl_probs: Dict[str, float], top_k_by_position: Dict[str, int]) -> str:
    """Format RL probabilities and resulting K values."""
    lines = ["## RL position probabilities", ""]
    for position in POSITIONS:
        probability = rl_probs.get(position, 0.0)
        lines.append(
            f"- {position}: {probability:.0%} (candidate shortlist K={top_k_by_position[position]})"
        )
    lines.append("")
    return "\n".join(lines)


def build_advisor_context(
    *,
    ui_state: Dict[str, Any],
    advising_team_id: int,
    agent_team_id: int,
    roster_structure: Dict[str, int],
    total_bench_size: int,
    team_manager_mapping: Dict[int, str],
    candidate_rows: Sequence[CandidateRow],
    baselines: Dict[str, float],
    top_k_by_position: Dict[str, int],
    rl_probs: Dict[str, float],
    rl_degraded: bool,
    insights: Dict[int, PlayerInsight],
) -> str:
    """Build markdown context for the draft assistant LLM.

    Parameters
    ----------
    ui_state : Dict[str, Any]
        Draft UI state from the session.
    advising_team_id : int
        Team receiving advice.
    agent_team_id : int
        User agent team id.
    roster_structure : Dict[str, int]
        Starter slot requirements.
    total_bench_size : int
        Bench size per team.
    team_manager_mapping : Dict[int, str]
        Team id to manager name mapping.
    candidate_rows : Sequence[CandidateRow]
        GP-filtered candidate rows with VORP.
    baselines : Dict[str, float]
        Positional replacement baselines.
    top_k_by_position : Dict[str, int]
        Per-position shortlist sizes.
    rl_probs : Dict[str, float]
        RL position probabilities for the advising team.
    rl_degraded : bool
        Whether RL probabilities fell back to defaults.
    insights : Dict[int, PlayerInsight]
        Offline insights keyed by player id.

    Returns
    -------
    str
        Markdown context payload.
    """
    pick_number = ui_state.get("current_pick_number", 1)
    num_teams = ui_state.get("num_teams", 10)
    round_number = math.ceil(pick_number / num_teams) if num_teams else 1
    pick_in_round = ((pick_number - 1) % num_teams) + 1 if num_teams else pick_number
    manager_name = team_manager_mapping.get(advising_team_id, f"Team {advising_team_id}")
    snake_team = ui_state.get("snake_team_on_turn")
    override_active = ui_state.get("override_active", False)

    roster = ui_state.get("team_rosters", {}).get(advising_team_id, {})
    roster_counts = ui_state.get("roster_counts", {}).get(advising_team_id, {})
    bye_weeks = ui_state.get("team_bye_weeks", {}).get(advising_team_id, {})

    sections = [
        "## Field glossary",
        "- **vorp**: Value Over Replacement Player — projected_points minus positional baseline.",
        "- **adp**: Average draft position; lower = drafted earlier.",
        "- **gp_frac**: Fraction of games played last season; R = rookie (no NFL sample).",
        "- **outlook_phrase**: Short research summary (offline, may be missing).",
        "- **depth_role**: starter | co_starter | committee | backup | unknown",
        "- **playing_time_tier**: high | medium | low | unknown",
        "- **injury_risk / recovery_status**: From offline research when available.",
        "- **fields_unknown**: Insight fields with insufficient reporting — do not infer these.",
        "",
        "## Draft clock",
        f"- Pick {pick_number} (round {round_number}, pick {pick_in_round})",
        f"- Advising team: {manager_name} (team {advising_team_id})",
        f"- Agent team: {agent_team_id} | Snake turn team: {snake_team}",
        f"- Override active: {'yes' if override_active else 'no'}",
        "",
        "## Advising team roster",
    ]

    starter_lines = ["| slot | player | pos | proj | bye |", "| --- | --- | --- | --- | --- |"]
    for slot, players in (roster.get("starters") or {}).items():
        for player in players:
            starter_lines.append(
                f"| {slot} | {player['name']} | {player['position']} | "
                f"{player['projected_points']:.1f} | {_format_bye_week(player.get('bye_week'))} |"
            )
    for player in roster.get("bench") or []:
        starter_lines.append(
            f"| bench | {player['name']} | {player['position']} | "
            f"{player['projected_points']:.1f} | {_format_bye_week(player.get('bye_week'))} |"
        )
    if len(starter_lines) == 2:
        starter_lines.append("| — | empty | — | — | — |")
    sections.extend(starter_lines)
    sections.append("")

    open_needs = _compute_open_needs(roster_counts, roster_structure, total_bench_size)
    sections.extend(
        [
            "## Positional needs (computed)",
            "- Starters open: " + (", ".join(open_needs) if open_needs else "none"),
            "",
            "## Bye week pressure (weeks 4-14)",
        ]
    )
    heavy_weeks = []
    for week in range(4, 15):
        week_data = bye_weeks.get(week) or bye_weeks.get(str(week))
        if not week_data:
            continue
        total_on_bye = sum(week_data.values())
        if total_on_bye > 0:
            detail = ", ".join(f"{pos}:{count}" for pos, count in week_data.items() if count > 0)
            sections.append(f"- Week {week}: {total_on_bye} starters ({detail})")
            if total_on_bye >= 2:
                heavy_weeks.append(int(week))
    if len(sections) == sections.index("## Bye week pressure (weeks 4-14)") + 1:
        sections.append("- No bye conflicts yet.")
    sections.append("")

    sections.extend(["## Positional baselines", "", "| pos | baseline | available above baseline |", "| --- | --- | --- |"])
    for position in POSITIONS:
        baseline = baselines.get(position, 0.0)
        above = sum(1 for row in candidate_rows if row.player.position == position and row.vorp > 0)
        sections.append(f"| {position} | {baseline:.1f} | {above} |")
    sections.append("")

    if rl_degraded:
        sections.extend(
            [
                "## RL note",
                "RL model probabilities unavailable; using K=5 for all positions.",
                "",
            ]
        )
    sections.append(_format_rl_probs(rl_probs, top_k_by_position))
    sections.append("## Top candidates by position")
    sections.append("")

    for position in POSITIONS:
        limit = top_k_by_position[position]
        sections.append(_render_candidate_table(f"{position} — by VORP", _top_by_vorp(candidate_rows, position, limit), insights))
        sections.append(_render_candidate_table(f"{position} — by ADP", _top_by_adp(candidate_rows, position, limit), insights))

    sections.extend(
        [
            "## Instructions",
            "- Recommend exactly one player from the candidate tables above.",
            "- Prefer need-filling picks when VORP is close.",
            "- Cite insight outlook/summary when present; use stats only when insight is null.",
            "- If fields_unknown is non-empty, mention insufficient reporting — do not guess.",
            "- Do not recommend players not listed in the candidate tables.",
            "- Output JSON matching PickRecommendation schema only.",
        ]
    )
    return "\n".join(sections)
