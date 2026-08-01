"""Tests for fantasy rules-engine branch behavior."""

from __future__ import annotations


def test_rules_engine_manual_allows_bench_pick_when_position_bench_is_full(
    draft_state, player_catalog, rules_engine
) -> None:
    """Verify manual drafting can use open generic bench space."""
    draft_state.add_player_to_roster(1, player_catalog.require(1))

    assert rules_engine.can_draft_manual(draft_state, 1, "QB", player_catalog) is True


def test_rules_engine_simulated_rejects_position_when_position_bench_is_full(
    draft_state, player_catalog, rules_engine
) -> None:
    """Verify simulated drafting enforces per-position bench maxima."""
    draft_state.add_player_to_roster(1, player_catalog.require(1))

    assert rules_engine.can_draft_simulated(draft_state, 1, "QB", player_catalog) is False


def test_rules_engine_simulated_allows_flex_eligible_position_with_open_flex(
    draft_state, player_catalog, rules_engine
) -> None:
    """Verify simulated drafting allows RB/WR/TE picks into an open flex slot."""
    draft_state.add_player_to_roster(1, player_catalog.require(2))

    assert rules_engine.can_draft_simulated(draft_state, 1, "RB", player_catalog) is True


def test_rules_engine_rejects_position_without_available_players(
    draft_state, player_catalog, rules_engine
) -> None:
    """Verify unavailable positions are rejected before roster checks."""
    for player_id in [4, 8, 12, 16]:
        draft_state.available_player_ids.remove(player_id)

    assert rules_engine.can_draft_manual(draft_state, 1, "TE", player_catalog) is False


def test_rules_engine_rejects_pick_when_roster_is_full(draft_state, player_catalog, rules_engine) -> None:
    """Verify both draft modes reject picks once the roster reaches capacity."""
    for player_id in [1, 2, 3, 4, 6, 7, 8]:
        draft_state.add_player_to_roster(1, player_catalog.require(player_id))

    assert rules_engine.can_draft_simulated(draft_state, 1, "RB", player_catalog) is False
