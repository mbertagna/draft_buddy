"""Markdown context assembly for the live draft assistant."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from draft_buddy.core.entities import Player
from draft_buddy.core.stacking import calculate_stack_count
from draft_buddy.data.insights.schemas import PlayerInsight
from draft_buddy.web.draft_advisor_filter import passes_gp_filter

POSITIONS: Tuple[str, ...] = ("QB", "RB", "WR", "TE")
FLEX_ELIGIBLE_POSITIONS: Tuple[str, ...] = ("RB", "WR", "TE")
POSITION_ORDER: Dict[str, int] = {position: index for index, position in enumerate(POSITIONS)}
DEFAULT_TOP_K = 5
HIGH_PRIORITY_TOP_K = 7
OVERALL_TOP_K = 5
NEED_FILL_TOP_K = 3
RECENT_PICKS_LIMIT = 12
LEAGUE_TOP_PLAYERS_LIMIT = 2


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
    roster: Optional[Dict[str, Any]] = None,
    roster_structure: Optional[Dict[str, int]] = None,
) -> set[int]:
    """Collect the union of all shortlist player ids.

    Parameters
    ----------
    rows : Sequence[CandidateRow]
        All candidate rows after GP filtering.
    top_k_by_position : Dict[str, int]
        Per-position shortlist size.
    roster : dict, optional
        Advising team roster used for need-fill shortlists.
    roster_structure : dict, optional
        Starter slot requirements used for need-fill shortlists.

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

    for row in _top_overall_by_vorp(rows, OVERALL_TOP_K):
        valid_ids.add(row.player.player_id)

    if roster is not None and roster_structure is not None:
        for need_rows in _need_fill_candidate_groups(rows, roster, roster_structure).values():
            for row in need_rows:
                valid_ids.add(row.player.player_id)
    return valid_ids


def format_league_format_blurb(
    *,
    scoring_rules: Dict[str, Optional[float]],
    num_teams: int,
    roster_structure: Dict[str, int],
    total_bench_size: int,
) -> str:
    """Build a compact league format string for advisor context.

    Parameters
    ----------
    scoring_rules : dict
        Fantasy scoring rules keyed by stat name.
    num_teams : int
        Number of teams in the league.
    roster_structure : dict
        Starter slot requirements by position.
    total_bench_size : int
        Bench size per team.

    Returns
    -------
    str
        Human-readable league format blurb.
    """
    receptions = scoring_rules.get("receptions")
    if receptions == 1.0:
        ppr_label = "Full PPR"
    elif receptions == 0.5:
        ppr_label = "Half PPR"
    elif receptions is None:
        ppr_label = "Custom (receptions unset)"
    else:
        ppr_label = f"Custom (receptions={receptions})"

    starter_parts = []
    for position in (*POSITIONS, "FLEX"):
        count = roster_structure.get(position, 0)
        if count:
            starter_parts.append(f"{position}{count}")
    starters_text = " ".join(starter_parts) if starter_parts else "none"
    return (
        f"{ppr_label} · {num_teams} teams · starters {starters_text} · bench {total_bench_size}"
    )


def _top_overall_by_vorp(rows: Sequence[CandidateRow], limit: int) -> List[CandidateRow]:
    """Return top candidates by VORP across all positions."""
    return sorted(rows, key=lambda row: (-row.vorp, row.player.player_id))[:limit]


def _need_fill_candidate_groups(
    rows: Sequence[CandidateRow],
    roster: Dict[str, Any],
    roster_structure: Dict[str, int],
) -> Dict[str, List[CandidateRow]]:
    """Return top VORP candidates for each open starter or FLEX need."""
    starter_filled, flex_filled = _count_starter_slots_filled(roster)
    groups: Dict[str, List[CandidateRow]] = {}

    for position in POSITIONS:
        open_slots = max(0, roster_structure.get(position, 0) - starter_filled.get(position, 0))
        if open_slots <= 0:
            continue
        groups[position] = _top_by_vorp(rows, position, NEED_FILL_TOP_K)

    flex_open = max(0, roster_structure.get("FLEX", 0) - flex_filled)
    if flex_open > 0:
        flex_rows = [row for row in rows if row.player.position in FLEX_ELIGIBLE_POSITIONS]
        groups["FLEX"] = sorted(flex_rows, key=lambda row: (-row.vorp, row.player.player_id))[
            :NEED_FILL_TOP_K
        ]
    return groups


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


def _sanitize_table_cell(value: str) -> str:
    """Escape markdown table cell content."""
    return " ".join(value.replace("|", "/").split())


def _insight_cell(insight: Optional[PlayerInsight], field: str) -> str:
    """Return one insight field for a table cell."""
    if insight is None:
        return ""
    value = getattr(insight, field, "")
    if hasattr(value, "value"):
        return _sanitize_table_cell(str(value.value))
    if isinstance(value, list):
        joined = ", ".join(
            str(item.value if hasattr(item, "value") else item) for item in value
        )
        return _sanitize_table_cell(joined)
    return _sanitize_table_cell(str(value))


def _optional_player_field(value: Optional[str]) -> str:
    """Format an optional player metadata field for markdown tables."""
    if not value:
        return ""
    return _sanitize_table_cell(str(value))


def _render_candidate_table(title: str, rows: Sequence[CandidateRow], insights: Dict[int, PlayerInsight]) -> str:
    """Render one candidate markdown table."""
    if not rows:
        return f"### {title}\n\nNo candidates.\n"

    lines = [
        f"### {title}",
        "",
        "| player_id | name | nfl | status | injury | depth | vorp | adp | gp_frac | proj | bye | outlook | summary | depth_role | playing_time | injury_risk | recovery | tags | confidence | fields_unknown |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        player = row.player
        insight = insights.get(player.player_id)
        lines.append(
            "| "
            + " | ".join(
                [
                    str(player.player_id),
                    _sanitize_table_cell(player.name),
                    _optional_player_field(player.team),
                    _optional_player_field(player.sleeper_status),
                    _optional_player_field(player.sleeper_injury_status),
                    _optional_player_field(player.sleeper_depth_chart_position),
                    f"{row.vorp:.1f}",
                    _format_adp(player.adp),
                    _format_gp_frac(player.games_played_frac),
                    f"{player.projected_points:.1f}",
                    _format_bye_week(player.bye_week),
                    _insight_cell(insight, "outlook_phrase"),
                    _insight_cell(insight, "summary"),
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


def _roster_players_flat(roster: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return all players on a roster as flat player dicts."""
    players_flat = roster.get("players_flat")
    if isinstance(players_flat, list):
        return players_flat

    players: List[Dict[str, Any]] = []
    for slot_players in (roster.get("starters") or {}).values():
        if isinstance(slot_players, list):
            players.extend(slot_players)
    bench = roster.get("bench") or []
    if isinstance(bench, list):
        players.extend(bench)
    return players


def _count_players_by_position(roster: Dict[str, Any]) -> Dict[str, int]:
    """Count rostered players by skill position."""
    counts = {position: 0 for position in POSITIONS}
    for player in _roster_players_flat(roster):
        position = player.get("position")
        if position in counts:
            counts[position] += 1
    return counts


def _count_starter_slots_filled(roster: Dict[str, Any]) -> Tuple[Dict[str, int], int]:
    """Return dedicated starter slot fill counts and FLEX fill count."""
    starters = roster.get("starters") or {}
    filled = {
        position: len(starters.get(position) or [])
        for position in POSITIONS
    }
    flex_filled = len(starters.get("FLEX") or [])
    return filled, flex_filled


def _format_position_targets_table(
    roster: Dict[str, Any],
    roster_structure: Dict[str, int],
    bench_maxes: Dict[str, int],
    total_bench_size: int,
    total_roster_size: int,
) -> str:
    """Render current-vs-target counts for each roster slot type."""
    on_roster = _count_players_by_position(roster)
    starter_filled, flex_filled = _count_starter_slots_filled(roster)
    roster_size = len(_roster_players_flat(roster))
    bench_count = len(roster.get("bench") or [])

    lines = [
        "## Roster targets",
        "",
        "How many players the advising team has vs starter requirements and per-position caps.",
        "",
        "| pos | on_roster | starters_req | starters_open | pos_cap | room_at_pos |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for position in POSITIONS:
        starters_required = roster_structure.get(position, 0)
        starters_open = max(0, starters_required - starter_filled.get(position, 0))
        position_cap = starters_required + bench_maxes.get(position, 0)
        room_at_position = max(0, position_cap - on_roster.get(position, 0))
        lines.append(
            f"| {position} | {on_roster.get(position, 0)} | {starters_required} | "
            f"{starters_open} | {position_cap} | {room_at_position} |"
        )

    flex_required = roster_structure.get("FLEX", 0)
    flex_open = max(0, flex_required - flex_filled)
    lines.extend(
        [
            f"| FLEX | {flex_filled} filled | {flex_required} | {flex_open} | — | — |",
            f"| bench | {bench_count} | {total_bench_size} | {max(0, total_bench_size - bench_count)} | — | — |",
            f"| total | {roster_size} | {total_roster_size} | {max(0, total_roster_size - roster_size)} | — | — |",
            "",
            "- **starters_req**: dedicated starter slots still to fill at this position.",
            "- **pos_cap**: max players draftable at this position (starters + bench max).",
            "- **room_at_pos**: pos_cap minus current players at that position.",
            "- RB/WR/TE can also fill FLEX before counting toward bench.",
            "",
        ]
    )
    return "\n".join(lines)


def _format_recent_picks(recent_picks: Sequence[Dict[str, Any]]) -> str:
    """Render recent draft picks for league-flow context."""
    lines = ["## Recent picks", ""]
    if not recent_picks:
        lines.append("- No picks yet.")
        lines.append("")
        return "\n".join(lines)

    lines.extend(
        [
            "| pick | team | player | pos |",
            "| --- | --- | --- | --- |",
        ]
    )
    for pick in recent_picks:
        lines.append(
            f"| {pick.get('pick_number', '?')} | {pick.get('team_id', '?')} | "
            f"{pick.get('player_name', '?')} | {pick.get('position', '?')} |"
        )
    lines.append("")
    return "\n".join(lines)


def _format_league_snapshot(
    *,
    advising_team_id: int,
    team_manager_mapping: Dict[int, str],
    team_rosters: Dict[int, Dict[str, Any]],
    roster_counts: Dict[int, Dict[str, int]],
    team_bye_weeks: Dict[int, Dict[Any, Dict[str, int]]],
    roster_structure: Dict[str, int],
    bench_maxes: Dict[str, int],
    total_bench_size: int,
) -> str:
    """Render compact needs summaries for other teams in the league."""
    lines = ["## League snapshot (other teams)", ""]
    other_team_lines: List[str] = []

    for team_id in sorted(team_rosters):
        if team_id == advising_team_id:
            continue

        display_name = team_manager_mapping.get(team_id, f"Team {team_id}")
        roster = team_rosters.get(team_id, {})
        counts = roster_counts.get(team_id, {})
        open_needs = _compute_open_needs(counts, roster_structure, total_bench_size)
        needs_text = ", ".join(open_needs) if open_needs else "roster complete"

        top_players = sorted(
            _roster_players_flat(roster),
            key=lambda player: float(player.get("projected_points") or 0.0),
            reverse=True,
        )[:LEAGUE_TOP_PLAYERS_LIMIT]
        top_text = ", ".join(
            f"{player.get('name', '?')} ({player.get('position', '?')})" for player in top_players
        )
        if not top_text:
            top_text = "none yet"

        heavy_bye_weeks: List[int] = []
        team_byes = team_bye_weeks.get(team_id, {})
        for week in range(4, 15):
            week_data = team_byes.get(week) or team_byes.get(str(week))
            if not week_data:
                continue
            if sum(week_data.values()) >= 2:
                heavy_bye_weeks.append(week)
        bye_text = f"W{', W'.join(str(week) for week in heavy_bye_weeks)}" if heavy_bye_weeks else "none"

        on_roster = _count_players_by_position(roster)
        position_summary = ", ".join(f"{position}:{on_roster[position]}" for position in POSITIONS)
        other_team_lines.append(
            f"- Team {team_id} ({display_name}): {position_summary}; "
            f"open needs: {needs_text}; top: {top_text}; bye pressure: {bye_text}"
        )

    if not other_team_lines:
        lines.append("- No other teams in league.")
    else:
        lines.extend(other_team_lines)
    lines.append("")
    return "\n".join(lines)


def _format_stack_summary(roster: Dict[str, Any]) -> str:
    """Render QB stack opportunities for the advising team roster."""
    players = [
        Player(
            player_id=int(player.get("player_id") or 0),
            name=str(player.get("name") or ""),
            position=str(player.get("position") or ""),
            projected_points=float(player.get("projected_points") or 0.0),
            team=player.get("team"),
        )
        for player in _roster_players_flat(roster)
    ]
    stack_count = calculate_stack_count(players)
    lines = ["## Stack summary", ""]

    team_positions: Dict[str, Dict[str, int]] = {}
    for player in players:
        if not player.team or player.position not in {"QB", "WR", "TE"}:
            continue
        if player.team not in team_positions:
            team_positions[player.team] = {"QB": 0, "WR": 0, "TE": 0}
        team_positions[player.team][player.position] += 1

    stack_details = []
    for team, positions in sorted(team_positions.items()):
        if positions["QB"] <= 0:
            continue
        pass_catchers = positions["WR"] + positions["TE"]
        if pass_catchers <= 0:
            stack_details.append(f"- {team}: QB without same-team WR/TE yet")
            continue
        pairs = positions["QB"] * pass_catchers
        stack_details.append(
            f"- {team}: {positions['QB']} QB, {positions['WR']} WR, {positions['TE']} TE "
            f"({pairs} stack pair{'s' if pairs != 1 else ''})"
        )

    if stack_count > 0:
        lines.append(f"- Total QB-WR/TE stack pairs: {stack_count}")

    if not stack_details:
        lines.append("- No NFL-team stack context yet.")
    else:
        lines.extend(stack_details)
    lines.append("")
    return "\n".join(lines)


def _format_nfl_team(player: Dict[str, Any]) -> str:
    """Format NFL team abbreviation for roster tables."""
    team = player.get("team")
    if team:
        return str(team)
    return "N/A"


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
    bench_maxes: Dict[str, int],
    total_bench_size: int,
    team_manager_mapping: Dict[int, str],
    candidate_rows: Sequence[CandidateRow],
    baselines: Dict[str, float],
    top_k_by_position: Dict[str, int],
    rl_probs: Dict[str, float],
    rl_degraded: bool,
    insights: Dict[int, PlayerInsight],
    recent_picks: Sequence[Dict[str, Any]] | None = None,
    league_format_blurb: str = "",
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
    bench_maxes : Dict[str, int]
        Maximum bench players allowed per position.
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
    recent_picks : Sequence[Dict[str, Any]], optional
        Recent draft picks for league-flow context.
    league_format_blurb : str, optional
        Compact scoring and roster format summary.

    Returns
    -------
    str
        Markdown context payload.
    """
    pick_number = ui_state.get("current_pick_number", 1)
    num_teams = ui_state.get("num_teams", 10)
    round_number = math.ceil(pick_number / num_teams) if num_teams else 1
    pick_in_round = ((pick_number - 1) % num_teams) + 1 if num_teams else pick_number
    display_name = team_manager_mapping.get(advising_team_id, f"Team {advising_team_id}")
    snake_team = ui_state.get("snake_team_on_turn")
    override_active = ui_state.get("override_active", False)

    roster = ui_state.get("team_rosters", {}).get(advising_team_id, {})
    roster_counts = ui_state.get("roster_counts", {}).get(advising_team_id, {})
    bye_weeks = ui_state.get("team_bye_weeks", {}).get(advising_team_id, {})
    total_roster_size = int(ui_state.get("total_roster_size_per_team") or 0)
    team_rosters = ui_state.get("team_rosters", {})
    all_roster_counts = ui_state.get("roster_counts", {})
    all_team_bye_weeks = ui_state.get("team_bye_weeks", {})
    pick_history = list(recent_picks or [])
    format_blurb = league_format_blurb or format_league_format_blurb(
        scoring_rules={},
        num_teams=int(num_teams or 0),
        roster_structure=roster_structure,
        total_bench_size=total_bench_size,
    )

    sections = [
        "## Field glossary",
        "- **vorp**: Value Over Replacement Player — projected_points minus positional baseline.",
        "- **adp**: Average draft position; lower = drafted earlier.",
        "- **gp_frac**: Fraction of games played last season; R = rookie (no NFL sample).",
        "- **nfl / status / injury / depth**: NFL team and Sleeper roster/injury/depth metadata when known.",
        "- **outlook_phrase**: Short research summary (offline, may be missing).",
        "- **summary**: Longer offline research blurb (up to two sentences; may be missing).",
        "- **depth_role**: starter | co_starter | committee | backup | unknown",
        "- **playing_time_tier**: high | medium | low | unknown",
        "- **injury_risk / recovery_status**: From offline research when available.",
        "- **fields_unknown**: Insight fields with insufficient reporting — do not infer these.",
        "",
        "## League format",
        f"- {format_blurb}",
        "",
        "## Draft clock",
        f"- Pick {pick_number} (round {round_number}, pick {pick_in_round})",
        f"- Advising team: {display_name} (team {advising_team_id})",
        f"- Agent team: {agent_team_id} | Snake turn team: {snake_team}",
        f"- Override active: {'yes' if override_active else 'no'}",
        "",
        "## Advising team roster",
    ]

    starter_lines = ["| slot | player | nfl | pos | proj | bye |", "| --- | --- | --- | --- | --- | --- |"]
    for slot, players in (roster.get("starters") or {}).items():
        for player in players:
            starter_lines.append(
                f"| {slot} | {player['name']} | {_format_nfl_team(player)} | {player['position']} | "
                f"{player['projected_points']:.1f} | {_format_bye_week(player.get('bye_week'))} |"
            )
    for player in roster.get("bench") or []:
        starter_lines.append(
            f"| bench | {player['name']} | {_format_nfl_team(player)} | {player['position']} | "
            f"{player['projected_points']:.1f} | {_format_bye_week(player.get('bye_week'))} |"
        )
    if len(starter_lines) == 2:
        starter_lines.append("| — | empty | — | — | — | — |")
    sections.extend(starter_lines)
    sections.append("")

    sections.append(
        _format_position_targets_table(
            roster=roster,
            roster_structure=roster_structure,
            bench_maxes=bench_maxes,
            total_bench_size=total_bench_size,
            total_roster_size=total_roster_size,
        )
    )
    sections.append(_format_stack_summary(roster))

    open_needs = _compute_open_needs(roster_counts, roster_structure, total_bench_size)
    sections.extend(
        [
            "## Positional needs (urgent)",
            "- Open slots: " + (", ".join(open_needs) if open_needs else "none"),
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

    sections.append(_format_recent_picks(pick_history))
    sections.append(
        _format_league_snapshot(
            advising_team_id=advising_team_id,
            team_manager_mapping=team_manager_mapping,
            team_rosters=team_rosters,
            roster_counts=all_roster_counts,
            team_bye_weeks=all_team_bye_weeks,
            roster_structure=roster_structure,
            bench_maxes=bench_maxes,
            total_bench_size=total_bench_size,
        )
    )

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

    sections.append("## Best available overall")
    sections.append("")
    sections.append(
        _render_candidate_table(
            f"Top {OVERALL_TOP_K} by VORP (any position)",
            _top_overall_by_vorp(candidate_rows, OVERALL_TOP_K),
            insights,
        )
    )

    need_groups = _need_fill_candidate_groups(candidate_rows, roster, roster_structure)
    sections.append("## Best for open starter needs")
    sections.append("")
    if not need_groups:
        sections.append("No open dedicated starter or FLEX slots.")
        sections.append("")
    else:
        for need_label, need_rows in need_groups.items():
            sections.append(
                _render_candidate_table(
                    f"{need_label} — top {NEED_FILL_TOP_K} by VORP",
                    need_rows,
                    insights,
                )
            )

    sections.append("## Top candidates by position")
    sections.append("")

    for position in POSITIONS:
        limit = top_k_by_position[position]
        sections.append(_render_candidate_table(f"{position} — by VORP", _top_by_vorp(candidate_rows, position, limit), insights))
        sections.append(_render_candidate_table(f"{position} — by ADP", _top_by_adp(candidate_rows, position, limit), insights))

    sections.extend(
        [
            "## Instructions",
            "- The reader knows little about fantasy football or the NFL; use plain English and briefly define jargon.",
            "- Recommend exactly one player from the candidate or decision-board tables above.",
            "- Compare best overall vs best need-fill vs ADP value; pick one and say what you are giving up.",
            "- Prefer need-filling picks when VORP is close.",
            "- Use roster targets and league snapshot when weighing positional runs and scarcity.",
            "- Consider bye-week pressure and stack opportunities from the advising team roster.",
            "- Cite insight summary/outlook when present; use stats only when insight is null or empty.",
            "- If fields_unknown is non-empty, mention insufficient reporting — do not guess or invent backstory.",
            "- Do not recommend players not listed in the candidate or decision-board tables.",
            "- Populate plain_english_recap (2-3 sentences), rationale_bullets (2-5), risks (up to 3), and alternates.",
            "- Output JSON matching PickRecommendation schema only.",
        ]
    )
    return "\n".join(sections)
