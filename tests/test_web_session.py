"""Tests for web-layer draft sessions."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from draft_buddy.core import InferenceProvider
from draft_buddy.web.session import DraftSession


class StubInferenceProvider(InferenceProvider):
    """Simple inference provider used by session tests."""

    def create_bot(self, team_id: int, strategy_config: Dict[str, Any], action_to_position: Dict[int, str]):
        """Return no model-backed bot for tests."""
        _ = (team_id, strategy_config, action_to_position)
        return None

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
