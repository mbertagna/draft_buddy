"""Tests for web-layer draft sessions."""

from __future__ import annotations

from typing import Any, Dict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from draft_buddy.core import InferenceProvider
from draft_buddy.web.session import DraftSession, DraftSessionManager


class _PolicySimStubBot:
    """Policy stub that picks QB when state callbacks are provided."""

    def __init__(self, action_to_position: Dict[str, Any]) -> None:
        self._action_to_position = action_to_position

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
        build_state_fn=None,
        get_action_mask_fn=None,
        **kwargs,
    ):
        """Return the best available QB when policy callbacks are wired."""
        _ = (team_roster, roster_structure, bench_maxes, kwargs)
        if build_state_fn is None or get_action_mask_fn is None:
            return None
        _ = build_state_fn(team_id)
        _ = get_action_mask_fn(team_id)
        _valid, player = try_select_player_fn(team_id, "QB", available_player_ids)
        return player


class StubInferenceProvider(InferenceProvider):
    """Simple inference provider used by session tests."""

    def create_bot(self, team_id: int, strategy_config: Dict[str, Any], action_to_position: Dict[int, str]):
        """Return no model-backed bot for tests."""
        _ = (team_id, strategy_config, action_to_position)
        return None

    def create_policy_sim_bot(self, action_to_position: Dict[int, str]):
        """Return a deterministic policy bot for simulated picks."""
        return _PolicySimStubBot(action_to_position)

    def build_state_vector(self, team_id: int, draft_state, player_catalog) -> np.ndarray:
        """Return a deterministic state vector for tests."""
        _ = (team_id, draft_state, player_catalog)
        return np.array([1.0, 2.0], dtype=np.float32)

    def predict_action_probabilities(
        self,
        team_id: int,
        draft_state,
        player_catalog,
        action_to_position: Dict[int, str],
        get_action_mask_fn,
    ) -> Dict[str, float]:
        """Return a deterministic position distribution for tests."""
        _ = (team_id, draft_state, player_catalog, get_action_mask_fn)
        return {action_to_position[0]: 0.7, action_to_position[1]: 0.2, action_to_position[2]: 0.1, action_to_position[3]: 0.0}


def test_draft_session_builds_ui_state_from_controller_and_catalog(config, player_catalog) -> None:
    """Verify UI state resolves player ids through the catalog."""
    session = DraftSession(config)
    session.draft_player(1)
    ui_state = session.get_ui_state()

    assert ui_state["team_rosters"][1]["players_flat"][0]["player_id"] == 1 and ui_state["current_pick_number"] == 2


def test_draft_session_ai_suggestion_uses_inference_provider(config, player_catalog) -> None:
    """Verify team suggestions flow through the injected inference abstraction."""
    session = DraftSession(config, inference_provider=StubInferenceProvider())

    assert session.get_ai_suggestion_for_team(1)["QB"] == 0.7


def test_draft_session_get_ai_suggestion_returns_draft_over_error(config) -> None:
    """Verify AI suggestion fails cleanly after the draft concludes."""
    session = DraftSession(config)
    session._state.current_pick_index = len(session.draft_order)

    assert session.get_ai_suggestion() == {"error": "Draft is over."}


def test_draft_session_rejects_invalid_team_for_ai_suggestion(config) -> None:
    """Verify invalid team ids return a descriptive error."""
    session = DraftSession(config, inference_provider=StubInferenceProvider())

    assert session.get_ai_suggestion_for_team(99) == {"error": "Invalid team id 99."}


def test_draft_session_returns_model_not_loaded_error_without_provider(config) -> None:
    """Verify team suggestions require an inference provider."""
    session = DraftSession(config)

    assert session.get_ai_suggestion_for_team(1) == {"error": "AI model not loaded."}


def test_draft_session_restores_available_players_after_ignored_ids(config) -> None:
    """Verify ignored players are removed temporarily and restored afterward."""
    session = DraftSession(config, inference_provider=StubInferenceProvider())
    original_available = set(session.available_player_ids)

    result = session.get_ai_suggestion_for_team(1, ignore_player_ids=[1, 9999])

    assert result["QB"] == 0.7 and session.available_player_ids == original_available


def test_draft_session_wraps_inference_errors(config) -> None:
    """Verify inference failures are surfaced as error payloads."""

    class FailingProvider(StubInferenceProvider):
        def predict_action_probabilities(self, *args, **kwargs):
            raise RuntimeError("prediction failed")

    session = DraftSession(config, inference_provider=FailingProvider())

    assert session.get_ai_suggestion_for_team(1) == {"error": "prediction failed"}


def test_draft_session_get_ai_suggestions_all_returns_model_error_without_provider(config) -> None:
    """Verify all-team suggestions require an inference provider."""
    session = DraftSession(config)

    assert session.get_ai_suggestions_all() == {"error": "AI model not loaded."}


def test_draft_session_set_current_team_picking_validates_range(config) -> None:
    """Verify invalid override team ids raise ValueError."""
    session = DraftSession(config)

    with pytest.raises(ValueError, match="Invalid team ID"):
        session.set_current_team_picking(9)


def test_draft_session_reset_restores_pick_cursor(config) -> None:
    """Verify reset returns the session to a fresh draft state."""
    session = DraftSession(config)
    first_player_id = session.player_catalog.player_ids[0]
    session.draft_player(first_player_id)

    session.reset()

    assert session.current_pick_number == 1 and len(session.draft_history) == 0


def test_draft_session_transfers_player_and_unified_undo_restores_it(config, player_catalog) -> None:
    """Verify session transfer behavior delegates to chronological undo."""
    session = DraftSession(config)
    session.draft_player(1)

    session.transfer_player(player_id=1, to_team_id=2)
    session.undo_last_pick()

    assert session.team_rosters[1].player_ids == [1] and session.team_rosters[2].player_ids == []


def test_get_ui_state_includes_visual_board(config, player_catalog) -> None:
    """Verify UI state exposes visual board placements for the frontend."""
    session = DraftSession(config)
    session.draft_player(1)
    ui_state = session.get_ui_state()

    assert ui_state["visual_board"][1][0] == 1


def test_draft_session_transfer_to_round_and_swap(config, player_catalog) -> None:
    """Verify session transfer-to-round and swap update visual board state."""
    session = DraftSession(config)
    session.draft_player(1)
    session.draft_player(2)

    session.transfer_player(player_id=1, to_team_id=1, to_round=2)
    session.swap_players(1, 2)
    ui_state = session.get_ui_state()

    assert ui_state["visual_board"][1][2] == 2
    assert ui_state["visual_board"][2][0] == 1


def test_draft_session_create_bot_strategy_falls_back_when_provider_returns_none(config) -> None:
    """Verify bot creation falls back to configured core strategies."""
    session = DraftSession(config, inference_provider=StubInferenceProvider())
    config.opponent.OPPONENT_TEAM_STRATEGIES[2] = {"logic": "AGENT_MODEL"}

    bot = session._create_bot_strategy(2)

    assert bot is not None


def test_draft_session_aggregate_bye_weeks_counts_positions(config, player_dataframe) -> None:
    """Verify bye-week aggregation groups counts by week and position."""
    player_dataframe.to_csv(config.paths.PLAYER_DATA_CSV, index=False)
    session = DraftSession(config)
    qb_player = next(
        player for player in session.player_catalog if player.position == "QB" and player.bye_week is not None
    )
    rb_player = next(
        player for player in session.player_catalog if player.position == "RB" and player.bye_week is not None
    )
    session.draft_player(qb_player.player_id)
    session._controller.apply_pick(team_id=2, player_id=rb_player.player_id, is_manual_pick=False)

    bye_weeks = session._aggregate_bye_weeks()

    assert bye_weeks[1][int(qb_player.bye_week)]["QB"] == 1 and bye_weeks[2][int(rb_player.bye_week)]["RB"] == 1


def test_draft_session_manager_get_or_create_loads_and_resets_empty_state(config, monkeypatch) -> None:
    """Verify session manager resets and saves when persisted state has no draft order."""
    loaded = {"called": False}
    saved = {"called": False}

    def fake_load(self, file_path: str) -> None:
        _ = file_path
        loaded["called"] = True
        self._state.draft_order = []

    def fake_save(self, file_path: str) -> None:
        _ = file_path
        saved["called"] = True

    monkeypatch.setattr(DraftSession, "load_state", fake_load)
    monkeypatch.setattr(DraftSession, "save_state", fake_save)
    manager = DraftSessionManager(config)

    session = manager.get_or_create("abc")

    assert loaded["called"] is True and saved["called"] is True and session.draft_order


def test_draft_session_manager_create_new_replaces_existing_session(config) -> None:
    """Verify create_new stores a fresh session under the session id."""
    manager = DraftSessionManager(config)

    first = manager.create_new("abc")
    second = manager.create_new("abc")

    assert first is not second and manager.get_or_create("abc") is second


def test_get_ui_state_exposes_snake_team_and_override_flag(config, player_catalog) -> None:
    """Verify UI state exposes snake turn and override metadata."""
    session = DraftSession(config)
    ui_state = session.get_ui_state()

    assert ui_state["snake_team_on_turn"] == 1
    assert ui_state["override_active"] is False
    assert ui_state["current_team_picking"] == 1

    session.set_current_team_picking(2)
    ui_state = session.get_ui_state()

    assert ui_state["snake_team_on_turn"] == 1
    assert ui_state["override_active"] is True
    assert ui_state["current_team_picking"] == 2


class TeamAwareStubInferenceProvider(StubInferenceProvider):
    """Return different probabilities per team for override tests."""

    def predict_action_probabilities(
        self,
        team_id: int,
        draft_state,
        player_catalog,
        action_to_position: Dict[str, Any],
        get_action_mask_fn,
    ) -> Dict[str, float]:
        """Return team-specific position probabilities."""
        _ = (draft_state, player_catalog, get_action_mask_fn)
        if team_id == 2:
            return {
                action_to_position[0]: 0.1,
                action_to_position[1]: 0.1,
                action_to_position[2]: 0.7,
                action_to_position[3]: 0.1,
            }
        return {
            action_to_position[0]: 0.7,
            action_to_position[1]: 0.1,
            action_to_position[2]: 0.1,
            action_to_position[3]: 0.1,
        }


def test_ai_suggestion_follows_override_team(config, player_catalog) -> None:
    """Verify RL suggestions use the overridden on-clock team."""
    session = DraftSession(config, inference_provider=TeamAwareStubInferenceProvider())

    assert session.get_ai_suggestion()["QB"] == 0.7

    session.set_current_team_picking(2)

    assert session.get_ai_suggestion()["WR"] == 0.7


def test_simulate_single_pick_uses_policy_when_requested(config, player_catalog) -> None:
    """Verify policy simulation mode uses the inference provider bot."""
    session = DraftSession(config, inference_provider=StubInferenceProvider())

    session.simulate_single_pick(use_policy=True)

    drafted = session.draft_history[0]
    player = session.player_catalog.require(drafted.player_id)
    assert player.position == "QB"


def test_simulate_single_pick_policy_mode_requires_model(config, player_catalog) -> None:
    """Verify policy simulation fails cleanly without an inference provider."""
    session = DraftSession(config)

    with pytest.raises(ValueError, match="Policy model not loaded"):
        session.simulate_single_pick(use_policy=True)

