"""Tests for the RL inference provider used by the web application."""

from __future__ import annotations

import torch

from run_webapp import RlInferenceProvider


class _TemperatureCapturingModel:
    """Stub policy model that records the temperature used for inference."""

    def __init__(self) -> None:
        self.received_temperature = None

    def get_action_probabilities(self, _state_tensor, action_mask=None, temperature=1.0):
        """Return a fixed distribution and capture the temperature argument."""
        self.received_temperature = temperature
        _ = action_mask
        return torch.tensor([[0.7, 0.2, 0.1, 0.0]])


def test_rl_inference_provider_forwards_suggestion_temperature(
    config, player_catalog, draft_state, draft_controller
) -> None:
    """Verify UI suggestions use the configured policy softmax temperature."""
    config.training.POLICY_SUGGESTION_TEMPERATURE = 2.5
    provider = RlInferenceProvider(config)
    model = _TemperatureCapturingModel()
    provider._suggestion_model = model
    action_to_position = {0: "QB", 1: "RB", 2: "WR", 3: "TE"}

    provider.predict_action_probabilities(
        team_id=1,
        draft_state=draft_state,
        player_catalog=player_catalog,
        action_to_position=action_to_position,
        get_action_mask_fn=draft_controller.get_action_mask_for_team,
    )

    assert model.received_temperature == 2.5


def test_config_defaults_policy_suggestion_temperature_to_one_point_five() -> None:
    """Verify the default UI suggestion temperature matches the position guide."""
    from draft_buddy.config import Config

    assert Config().training.POLICY_SUGGESTION_TEMPERATURE == 1.5
