"""Tests for policy network and model-driven bot behavior."""

import numpy as np
import torch

from draft_buddy.domain.entities import Player
from draft_buddy.rl.agent_bot import AgentModelBotGM
from draft_buddy.rl.policy_network import PolicyNetwork


def test_policy_network_forward_outputs_expected_logit_shape():
    """Verify forward pass returns (1, 4) logits for one state."""
    model = PolicyNetwork(input_dim=8, output_dim=4, hidden_dim=16)
    state = torch.zeros((1, 8), dtype=torch.float32)
    logits = model.forward(state)

    assert tuple(logits.shape) == (1, 4)


def test_agent_model_bot_execute_pick_translates_best_action_to_wr_selection():
    """Verify model argmax action index maps to WR pick."""
    class _MockModel:
        def get_action_probabilities(self, state_tensor, action_mask=None):
            del state_tensor, action_mask
            return torch.tensor([[0.1, 0.2, 0.9, 0.0]], dtype=torch.float32)

    wr_player = Player(2, "WR Target", "WR", 200.0, adp=12.0)
    player_map = {
        1: Player(1, "QB Target", "QB", 250.0, adp=8.0),
        2: wr_player,
    }
    bot = AgentModelBotGM(_MockModel(), {0: "QB", 1: "RB", 2: "WR", 3: "TE"})

    def can_draft(team_id, position, is_manual=False):
        del team_id, is_manual
        return position in {"QB", "WR"}

    def try_select(team_id, position, available_ids):
        del team_id, available_ids
        if position == "WR":
            return True, wr_player
        return False, None

    picked = bot.execute_pick(
        team_id=1,
        available_player_ids={1, 2},
        player_map=player_map,
        roster_counts={"QB": 0, "RB": 0, "WR": 0, "TE": 0, "FLEX": 0},
        roster_structure={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
        bench_maxes={"QB": 1, "RB": 1, "WR": 1, "TE": 1},
        can_draft_position_fn=can_draft,
        try_select_player_fn=try_select,
        build_state_fn=lambda team_id: np.zeros(8, dtype=np.float32),
        get_action_mask_fn=lambda team_id: np.array([True, True, True, True]),
    )

    assert picked.position == "WR"
