"""Snake-draft pick numbering for position guide exports."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PickPlacement:
    """Overall and round placement for one user pick in a snake draft.

    Parameters
    ----------
    user_pick_index : int
        One-based index of the user's pick in the draft (1st pick, 2nd pick, ...).
    round : int
        Draft round (1-based).
    overall_pick_number : int
        Overall pick number in the full draft order.
    """

    user_pick_index: int
    round: int
    overall_pick_number: int


def overall_pick_number(num_teams: int, draft_slot: int, round_number: int) -> int:
    """Return the overall pick number for a slot in a snake round.

    Parameters
    ----------
    num_teams : int
        Number of teams in the league.
    draft_slot : int
        User's draft slot (1-based).
    round_number : int
        Draft round (1-based).

    Returns
    -------
    int
        Overall pick number in the draft.
    """
    if round_number % 2 == 1:
        return (round_number - 1) * num_teams + draft_slot
    return round_number * num_teams - draft_slot + 1


def pick_placement(
    num_teams: int, draft_slot: int, user_pick_index: int
) -> PickPlacement:
    """Map a user pick index to round and overall pick number.

    Parameters
    ----------
    num_teams : int
        Number of teams in the league.
    draft_slot : int
        User's draft slot (1-based).
    user_pick_index : int
        One-based index of the user's pick.

    Returns
    -------
    PickPlacement
        Round and overall pick metadata.
    """
    round_number = user_pick_index
    overall = overall_pick_number(num_teams, draft_slot, round_number)
    return PickPlacement(
        user_pick_index=user_pick_index,
        round=round_number,
        overall_pick_number=overall,
    )
