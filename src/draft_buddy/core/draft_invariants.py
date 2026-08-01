"""Pure invariant checks for draft state consistency."""

from __future__ import annotations

from typing import List

from draft_buddy.core.draft_state import DraftState


def collect_invariant_errors(state: DraftState) -> List[str]:
    """Return human-readable invariant violations for a draft state.

    Parameters
    ----------
    state : DraftState
        Mutable draft state to inspect.

    Returns
    -------
    list of str
        Empty when the state is consistent.
    """
    errors: List[str] = []
    rostered_ids: set[int] = set()
    for team_id, roster in state.team_rosters.items():
        if roster.size > state.total_roster_size_per_team:
            errors.append(
                f"Team {team_id} roster size {roster.size} exceeds "
                f"cap {state.total_roster_size_per_team}."
            )
        for player_id in roster.player_ids:
            if player_id in rostered_ids:
                errors.append(f"Player {player_id} appears on multiple rosters.")
            rostered_ids.add(player_id)
            if player_id in state.available_player_ids:
                errors.append(f"Rostered player {player_id} is still marked available.")

    board_counts: dict[int, int] = {}
    for team_id, rounds in state.visual_board.items():
        roster = state.team_rosters.get(team_id)
        roster_ids = set(roster.player_ids) if roster is not None else set()
        for round_index, player_id in rounds.items():
            if player_id is None:
                continue
            board_counts[player_id] = board_counts.get(player_id, 0) + 1
            if player_id not in roster_ids:
                errors.append(
                    f"Visual board cell ({team_id}, {round_index}) has player "
                    f"{player_id} not on that team's roster."
                )

    for player_id, count in board_counts.items():
        if count != 1:
            errors.append(f"Player {player_id} appears {count} times on the visual board.")

    for player_id in rostered_ids:
        if player_id not in board_counts:
            errors.append(f"Rostered player {player_id} is missing from the visual board.")

    for action in state.action_history:
        if action.action_type == "pick":
            history_len = len(state.draft_history)
        elif action.action_type == "transfer":
            history_len = len(state.transfer_history)
        elif action.action_type == "swap":
            history_len = len(state.swap_history)
        else:
            errors.append(f"Unknown action type in action_history: {action.action_type}.")
            continue
        if action.history_index < 0 or action.history_index >= history_len:
            errors.append(
                f"Action {action.action_type} history_index {action.history_index} "
                f"is out of range for length {history_len}."
            )

    return errors


def assert_invariants(state: DraftState) -> None:
    """Raise ``ValueError`` when draft state invariants are violated.

    Parameters
    ----------
    state : DraftState
        Mutable draft state to validate.

    Raises
    ------
    ValueError
        When one or more invariants fail.
    """
    errors = collect_invariant_errors(state)
    if errors:
        raise ValueError("; ".join(errors))
