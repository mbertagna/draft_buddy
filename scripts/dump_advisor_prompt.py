"""Dump the live draft assistant system prompt and user context to a file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from draft_buddy.config import load_runtime_config
from draft_buddy.data.insights.loader import load_latest_player_insights
from draft_buddy.web.draft_advisor_context import (
    DEFAULT_TOP_K,
    POSITIONS,
    build_advisor_context,
    build_candidate_rows,
    filter_available_players,
    format_league_format_blurb,
    position_top_k_map,
)
from draft_buddy.web.draft_advisor_filter import exclude_ignored_players
from draft_buddy.web.draft_advisor_service import RECENT_PICKS_LIMIT, SYSTEM_PROMPT
from draft_buddy.web.session import DraftSession

from run_webapp import RlInferenceProvider


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for advisor prompt dumping."""
    parser = argparse.ArgumentParser(
        description=(
            "Write the draft assistant system prompt and user context without calling an LLM. "
            "By default loads the current draft from data/draft_state.json."
        )
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Start from a fresh draft at pick 1 instead of loading draft_state.json.",
    )
    parser.add_argument(
        "--state-file",
        type=str,
        default="",
        help="Override draft state path (default: configured DRAFT_STATE_FILE).",
    )
    parser.add_argument(
        "--team-id",
        type=int,
        default=None,
        help="Advising team id (default: team currently on the clock).",
    )
    parser.add_argument(
        "--gp-min",
        type=float,
        default=None,
        help="Minimum games-played fraction filter (matches UI Min GP Frac).",
    )
    parser.add_argument(
        "--ignore-player-id",
        type=int,
        action="append",
        default=[],
        dest="ignore_player_ids",
        help="Exclude a player id from candidates (repeatable).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="advisor_prompt.txt",
        help="Output file path (default: advisor_prompt.txt).",
    )
    return parser.parse_args()


def _build_recent_pick_summaries(session: DraftSession) -> list[dict[str, object]]:
    """Return recent draft picks with player names for advisor context."""
    summaries: list[dict[str, object]] = []
    for pick in session.draft_history[-RECENT_PICKS_LIMIT:]:
        player = session.player_catalog.get(pick.player_id)
        summaries.append(
            {
                "pick_number": pick.pick_number,
                "team_id": pick.team_id,
                "player_name": player.name if player is not None else f"Player {pick.player_id}",
                "position": player.position if player is not None else "?",
            }
        )
    return summaries


def build_prompts(
    session: DraftSession,
    *,
    team_id: int | None,
    gp_min: float | None,
    ignore_player_ids: list[int],
    insights_root: str,
) -> tuple[str, str, dict[str, object]]:
    """Build system and user prompts for the advising team.

    Parameters
    ----------
    session : DraftSession
        Loaded draft session.
    team_id : int | None
        Advising team, or ``None`` to use the team on the clock.
    gp_min : float | None
        Optional minimum games-played fraction filter.
    ignore_player_ids : list[int]
        Player ids to exclude from candidate pools.
    insights_root : str
        Data root used to resolve the latest insights export.

    Returns
    -------
    tuple[str, str, dict[str, object]]
        System prompt, user context, and summary metadata.
    """
    ui_state = session.get_ui_state()
    advising_team_id = team_id or ui_state.get("current_team_picking")
    if advising_team_id is None:
        raise ValueError("No team is currently on the clock (draft may be complete).")
    if not (1 <= advising_team_id <= session.num_teams):
        raise ValueError(f"Invalid team id {advising_team_id}.")

    available_players = [
        session.player_catalog.require(player_id)
        for player_id in session.available_player_ids
    ]
    available_players = exclude_ignored_players(available_players, ignore_player_ids)
    filtered_players = filter_available_players(available_players, gp_min)
    if not filtered_players:
        raise ValueError("No players remain after GP and ignore filters.")

    baselines = session.get_positional_baselines()
    candidate_rows = build_candidate_rows(filtered_players, baselines)
    suggestion = session.get_ai_suggestion_for_team(advising_team_id, ignore_player_ids)
    rl_degraded = bool(suggestion.get("error"))
    if rl_degraded:
        rl_probs = {position: 0.25 for position in POSITIONS}
        top_k_by_position = {position: DEFAULT_TOP_K for position in POSITIONS}
    else:
        rl_probs = {
            position: float(suggestion.get(position, 0.0)) for position in POSITIONS
        }
        top_k_by_position = position_top_k_map(rl_probs)

    advising_roster = ui_state.get("team_rosters", {}).get(advising_team_id, {})
    league_format_blurb = format_league_format_blurb(
        scoring_rules=session.scoring_rules,
        num_teams=session.num_teams,
        roster_structure=session.roster_structure,
        total_bench_size=session.total_bench_size,
    )
    context = build_advisor_context(
        ui_state=ui_state,
        advising_team_id=advising_team_id,
        agent_team_id=session.agent_team_id,
        roster_structure=session.roster_structure,
        bench_maxes=session.bench_maxes,
        total_bench_size=session.total_bench_size,
        team_manager_mapping=session.team_manager_mapping,
        candidate_rows=candidate_rows,
        baselines=baselines,
        top_k_by_position=top_k_by_position,
        rl_probs=rl_probs,
        rl_degraded=rl_degraded,
        insights=load_latest_player_insights(insights_root).players,
        recent_picks=_build_recent_pick_summaries(session),
        league_format_blurb=league_format_blurb,
        platform_bench_maxes=session.platform_bench_maxes,
    )
    meta = {
        "pick_number": ui_state.get("current_pick_number"),
        "team_id": advising_team_id,
        "history_len": len(session.draft_history),
        "rl_degraded": rl_degraded,
        "candidate_count": len(filtered_players),
        "roster_keys": list(advising_roster.keys()),
    }
    return SYSTEM_PROMPT, context, meta


def main() -> None:
    """Load draft state and write advisor prompts to disk."""
    args = parse_args()
    config = load_runtime_config()
    session = DraftSession(config, inference_provider=RlInferenceProvider(config))

    if args.reset:
        session.reset()
    else:
        state_path = args.state_file or config.paths.DRAFT_STATE_FILE
        session.load_state(state_path)

    system_prompt, user_context, meta = build_prompts(
        session,
        team_id=args.team_id,
        gp_min=args.gp_min,
        ignore_player_ids=args.ignore_player_ids,
        insights_root=config.paths.DATA_DIR,
    )

    output_path = Path(args.output)
    output_path.write_text(
        "=== SYSTEM PROMPT ===\n"
        + system_prompt
        + "\n\n=== USER CONTEXT ===\n"
        + user_context,
        encoding="utf-8",
    )
    print(
        f"Wrote {output_path.resolve()} "
        f"(pick={meta['pick_number']}, team={meta['team_id']}, "
        f"history={meta['history_len']}, rl_degraded={meta['rl_degraded']})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
