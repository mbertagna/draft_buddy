"""Tests for draft state invariant checks."""

from __future__ import annotations

import pytest

from draft_buddy.core.draft_invariants import assert_invariants, collect_invariant_errors
from draft_buddy.core.entities import DraftAction


def test_collect_invariant_errors_detects_rostered_available_player(draft_state, player_catalog) -> None:
    """Verify rostered players must not remain available."""
    draft_state.add_player_to_roster(1, player_catalog.require(1))
    draft_state.place_player_visual(1, 0, 1)
    draft_state.available_player_ids.add(1)

    errors = collect_invariant_errors(draft_state)

    assert any("still marked available" in error for error in errors)


def test_collect_invariant_errors_detects_missing_visual_placement(draft_state, player_catalog) -> None:
    """Verify rostered players must appear on the visual board."""
    draft_state.add_player_to_roster(1, player_catalog.require(1))

    errors = collect_invariant_errors(draft_state)

    assert any("missing from the visual board" in error for error in errors)


def test_collect_invariant_errors_detects_board_player_not_on_roster(draft_state) -> None:
    """Verify visual board players must belong to that team's roster."""
    draft_state.place_player_visual(1, 0, 99)

    errors = collect_invariant_errors(draft_state)

    assert any("not on that team's roster" in error for error in errors)


def test_collect_invariant_errors_detects_out_of_range_action_index(draft_state) -> None:
    """Verify action history indexes must reference existing history entries."""
    draft_state.append_action(DraftAction(action_type="pick", history_index=5))

    errors = collect_invariant_errors(draft_state)

    assert any("out of range" in error for error in errors)


def test_assert_invariants_raises_on_violation(draft_state, player_catalog) -> None:
    """Verify assert_invariants raises a ValueError for broken state."""
    draft_state.add_player_to_roster(1, player_catalog.require(1))
    draft_state.place_player_visual(1, 0, 1)
    draft_state.available_player_ids.add(1)

    with pytest.raises(ValueError, match="still marked available"):
        assert_invariants(draft_state)


def test_assert_invariants_accepts_consistent_pick(draft_state, player_catalog) -> None:
    """Verify a normal pick placement satisfies all invariants."""
    draft_state.add_player_to_roster(1, player_catalog.require(1))
    draft_state.available_player_ids.discard(1)
    draft_state.place_player_visual(1, 0, 1)

    assert_invariants(draft_state)


def test_collect_invariant_errors_allows_display_only_board_player(draft_state) -> None:
    """Verify K/DST display picks may sit on the board without a roster slot."""
    draft_state.display_only_player_ids.add(99)
    draft_state.place_player_visual(1, 0, 99)

    errors = collect_invariant_errors(draft_state)

    assert errors == []


def test_collect_invariant_errors_detects_shelved_available_overlap(draft_state) -> None:
    """Verify shelved players must not remain available."""
    draft_state.shelved_player_ids.add(2)
    draft_state.available_player_ids.add(2)

    errors = collect_invariant_errors(draft_state)

    assert any("Shelved player 2 is still marked available" in error for error in errors)


def test_collect_invariant_errors_detects_rostered_shelved_player(
    draft_state, player_catalog
) -> None:
    """Verify rostered players must not remain shelved."""
    draft_state.add_player_to_roster(1, player_catalog.require(1))
    draft_state.place_player_visual(1, 0, 1)
    draft_state.shelved_player_ids.add(1)

    errors = collect_invariant_errors(draft_state)

    assert any("still marked shelved" in error for error in errors)