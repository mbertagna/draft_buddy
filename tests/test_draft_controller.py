"""Tests for shared draft orchestration."""

from __future__ import annotations

import numpy as np

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
