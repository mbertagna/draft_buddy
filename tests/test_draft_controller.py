"""Tests for shared draft orchestration."""

from __future__ import annotations

import json

import numpy as np
import pytest

from draft_buddy.core import BotGM, DraftController


class StubBot(BotGM):
    """Minimal deterministic bot used for controller tests."""

    def execute_pick(
        self,
        team_id: int,
        available_player_ids: set,
        player_catalog,
        team_roster,
        roster_structure: dict,
        bench_maxes: dict,
        can_draft_position_fn,
        try_select_player_fn,
        **kwargs,
    ):
        """Return the best available RB when possible."""
        _ = (team_id, team_roster, roster_structure, bench_maxes, kwargs)
        is_valid, player = try_select_player_fn(team_id, "RB", available_player_ids)
        return player if is_valid else None


class NullBot(BotGM):
    """Bot that declines to make a selection."""

    def execute_pick(
        self,
        team_id: int,
        available_player_ids: set,
        player_catalog,
        team_roster,
        roster_structure: dict,
        bench_maxes: dict,
        can_draft_position_fn,
        try_select_player_fn,
        **kwargs,
    ):
        """Always return no pick."""
        _ = (
            team_id,
            available_player_ids,
            player_catalog,
            team_roster,
            roster_structure,
            bench_maxes,
            can_draft_position_fn,
            try_select_player_fn,
            kwargs,
        )
        return None


def test_draft_controller_drafts_and_undos_one_pick(config, draft_state, player_catalog, rules_engine) -> None:
    """Verify the controller applies and undoes typed picks."""
    controller = DraftController(
        state=draft_state,
        player_catalog=player_catalog,
        rules_engine=rules_engine,
        action_to_position={0: "QB", 1: "RB", 2: "WR", 3: "TE"},
    )
    controller.draft_player(1)
    controller.undo_last_pick()

    assert draft_state.current_pick_index == 0 and draft_state.roster_for_team(1).player_ids == [] and 1 in draft_state.available_player_ids


def test_draft_controller_simulates_bot_pick(config, draft_state, player_catalog, rules_engine) -> None:
    """Verify bot-driven simulation delegates pick application to shared workflow."""
    controller = DraftController(
        state=draft_state,
        player_catalog=player_catalog,
        rules_engine=rules_engine,
        action_to_position={0: "QB", 1: "RB", 2: "WR", 3: "TE"},
        bot_factory=lambda _team_id: StubBot(),
    )
    drafted_player = controller.simulate_single_pick(manual_draft_teams=set())

    assert drafted_player.position == "RB" and draft_state.draft_history[0].player_id == drafted_player.player_id


def test_draft_controller_action_mask_is_boolean_vector(draft_controller) -> None:
    """Verify action masks are typed and aligned with the action space."""
    mask = draft_controller.get_action_mask_for_team(1)

    assert isinstance(mask, np.ndarray) and mask.dtype == bool and mask.shape == (4,)


def test_draft_controller_rejects_unknown_player_and_clears_override(draft_controller) -> None:
    """Verify invalid player ids raise and clear any override team."""
    draft_controller.set_override_team(3)

    with pytest.raises(ValueError, match="is not available"):
        draft_controller.draft_player(9999)

    assert draft_controller.state.override_team_id is None


def test_draft_controller_rejects_already_drafted_player_and_clears_override(draft_controller) -> None:
    """Verify drafting an unavailable player restores normal turn order."""
    draft_controller.draft_player(1)
    draft_controller.set_override_team(3)

    with pytest.raises(ValueError, match="is not available"):
        draft_controller.draft_player(1)

    assert draft_controller.state.override_team_id is None


def test_draft_controller_rejects_illegal_position_and_clears_override(
    draft_controller, draft_state, player_catalog
) -> None:
    """Verify illegal position picks raise and clear overrides."""
    for player_id in [1, 2, 3, 4, 6, 7, 8]:
        draft_state.add_player_to_roster(1, player_catalog.require(player_id))
    draft_controller.set_override_team(1)

    with pytest.raises(ValueError, match="cannot draft a QB"):
        draft_controller.draft_player(5)

    assert draft_controller.state.override_team_id is None


def test_draft_controller_rejects_pick_after_draft_concludes(draft_controller) -> None:
    """Verify manual picks stop once the draft cursor passes the order."""
    draft_controller.state.current_pick_index = len(draft_controller.draft_order)

    with pytest.raises(ValueError, match="draft has already concluded"):
        draft_controller.draft_player(1)


def test_draft_controller_undo_requires_existing_history(draft_controller) -> None:
    """Verify undo fails when no prior pick exists."""
    with pytest.raises(ValueError, match="No picks to undo"):
        draft_controller.undo_last_pick()


def test_draft_controller_save_and_load_round_trips_state(
    tmp_path, config, draft_state, player_catalog, rules_engine
) -> None:
    """Verify persisted state restores picks, rosters, and counts."""
    controller = DraftController(
        state=draft_state,
        player_catalog=player_catalog,
        rules_engine=rules_engine,
        action_to_position={0: "QB", 1: "RB", 2: "WR", 3: "TE"},
    )
    controller.draft_player(1)
    controller.draft_player(6)
    file_path = tmp_path / "draft_state.json"
    controller.save_state(str(file_path))
    restored_state = draft_state.__class__(
        all_player_ids=set(player_catalog.player_ids),
        draft_order=[1, 2, 3, 4],
        roster_structure=config.draft.ROSTER_STRUCTURE,
        bench_maxes=config.draft.BENCH_MAXES,
        total_roster_size_per_team=sum(config.draft.ROSTER_STRUCTURE.values()) + config.draft.TOTAL_BENCH_SIZE,
        agent_team_id=config.draft.AGENT_START_POSITION,
    )
    restored_controller = DraftController(
        state=restored_state,
        player_catalog=player_catalog,
        rules_engine=rules_engine,
        action_to_position={0: "QB", 1: "RB", 2: "WR", 3: "TE"},
    )

    restored_controller.load_state(str(file_path))

    assert json.dumps(restored_controller.state.to_dict(), sort_keys=True) == json.dumps(
        controller.state.to_dict(), sort_keys=True
    )


def test_draft_controller_load_state_ignores_missing_file(draft_controller, tmp_path) -> None:
    """Verify loading a missing file leaves the draft untouched."""
    missing_path = tmp_path / "missing.json"
    original_state = draft_controller.state.to_dict()

    draft_controller.load_state(str(missing_path))

    assert draft_controller.state.to_dict() == original_state


def test_draft_controller_reset_clears_cached_bots(draft_state, player_catalog, rules_engine) -> None:
    """Verify reset drops cached bots and refreshes the player pool."""
    call_log: list[int] = []
    controller = DraftController(
        state=draft_state,
        player_catalog=player_catalog,
        rules_engine=rules_engine,
        action_to_position={0: "QB", 1: "RB", 2: "WR", 3: "TE"},
        bot_factory=lambda team_id: call_log.append(team_id) or StubBot(),
    )
    controller._get_bot(2)

    controller.reset([4, 3, 2, 1], agent_team_id=4)
    controller._get_bot(2)

    assert call_log == [2, 2]


def test_draft_controller_simulate_single_pick_rejects_manual_team(draft_controller) -> None:
    """Verify simulation stops when the active team is manual."""
    with pytest.raises(ValueError, match="manual team's turn"):
        draft_controller.simulate_single_pick(manual_draft_teams={1})


def test_draft_controller_simulate_single_pick_rejects_completed_draft(draft_controller) -> None:
    """Verify simulation rejects picks after the draft ends."""
    draft_controller.state.current_pick_index = len(draft_controller.draft_order)

    with pytest.raises(ValueError, match="draft has already concluded"):
        draft_controller.simulate_single_pick(manual_draft_teams=set())


def test_draft_controller_simulate_remaining_advances_until_completion(
    draft_state, player_catalog, rules_engine
) -> None:
    """Verify scheduled simulation keeps drafting until the order is exhausted."""
    controller = DraftController(
        state=draft_state,
        player_catalog=player_catalog,
        rules_engine=rules_engine,
        action_to_position={0: "QB", 1: "RB", 2: "WR", 3: "TE"},
        bot_factory=lambda _team_id: StubBot(),
    )

    controller.simulate_remaining(manual_draft_teams=set())

    assert controller.current_pick_index == len(controller.draft_order)


def test_draft_controller_try_select_player_uses_provided_candidate_ids(draft_controller) -> None:
    """Verify player selection honors an explicit candidate pool."""
    is_valid, player = draft_controller.try_select_player_for_team(1, "QB", {5, 13})

    assert is_valid and player.player_id == 5


def test_draft_controller_try_select_player_rejects_disallowed_position(
    draft_controller, draft_state, player_catalog
) -> None:
    """Verify selection fails when the team cannot legally take the position."""
    for player_id in [1, 5, 9, 13, 2, 3, 4]:
        draft_state.add_player_to_roster(1, player_catalog.require(player_id))
    is_valid, player = draft_controller.try_select_player_for_team(1, "QB")

    assert is_valid is False and player is None


def test_draft_controller_uses_override_team_for_manual_pick(draft_controller) -> None:
    """Verify override team picks are applied to the selected team."""
    draft_controller.set_override_team(3)

    draft_controller.draft_player(1)

    assert draft_controller.state.draft_history[-1].team_id == 3


def test_draft_controller_computes_zero_baseline_when_position_is_empty(
    draft_controller, draft_state, player_catalog
) -> None:
    """Verify empty positional pools produce zero-valued baselines."""
    for player_id in [1, 5, 9, 13]:
        draft_state.available_player_ids.remove(player_id)
    baselines = draft_controller.get_positional_baselines()

    assert baselines["QB"] == 0.0


def test_draft_controller_smooths_positional_baseline_from_nearby_players(draft_controller) -> None:
    """Verify baselines use the neighboring replacement-level players."""
    baselines = draft_controller.get_positional_baselines()

    assert baselines["RB"] == pytest.approx((205.0 + 190.0 + 190.0) / 3.0)


def test_draft_controller_falls_back_to_random_pick_when_bot_returns_none(
    monkeypatch, draft_state, player_catalog, rules_engine
) -> None:
    """Verify controller uses the random fallback when a bot declines to pick."""
    controller = DraftController(
        state=draft_state,
        player_catalog=player_catalog,
        rules_engine=rules_engine,
        action_to_position={0: "QB", 1: "RB", 2: "WR", 3: "TE"},
        bot_factory=lambda _team_id: NullBot(),
    )
    monkeypatch.setattr("random.choice", lambda players: players[0])

    drafted_player = controller.simulate_single_pick(manual_draft_teams=set())

    assert drafted_player.player_id == 1
