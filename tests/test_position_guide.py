"""Tests for offline position guide generation."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
import torch

from draft_buddy.data.cache_paths import position_guide_output_path
from draft_buddy.rl.draft_gym_env import DraftGymEnv
from draft_buddy.rl.position_guide.pick_numbers import overall_pick_number
from draft_buddy.rl.position_guide.schemas import (
    PositionGuideFile,
    PositionGuidePick,
    PositionProbabilities,
)
from draft_buddy.rl.position_guide.simulator import average_position_probabilities
from draft_buddy.rl.run_utils import find_latest_checkpoint_in_dir


def test_snake_pick_number_12teams_slot5_round1() -> None:
    """Round 1 overall pick for slot 5 in a 12-team league is pick 5."""
    assert overall_pick_number(12, 5, 1) == 5


def test_snake_pick_number_12teams_slot5_round2() -> None:
    """Round 2 overall pick for slot 5 in a 12-team league is pick 20."""
    assert overall_pick_number(12, 5, 2) == 20


def test_snake_pick_number_10teams_slot5_round1() -> None:
    """Round 1 overall pick for slot 5 in a 10-team league is pick 5."""
    assert overall_pick_number(10, 5, 1) == 5


def test_position_guide_output_path_includes_num_teams(tmp_path) -> None:
    """Export paths include league size in the filename."""
    generated_at = datetime(2026, 7, 5, 22, 45, tzinfo=timezone.utc)
    path_12 = position_guide_output_path(str(tmp_path), 12, 5, 2026, generated_at, ext="json")
    path_10 = position_guide_output_path(str(tmp_path), 10, 5, 2026, generated_at, ext="html")
    assert "12teams" in path_12
    assert "10teams" in path_10


def test_average_position_probabilities() -> None:
    """Two probability samples average element-wise."""
    result = average_position_probabilities(
        [{"QB": 0.2, "RB": 0.8, "WR": 0.0, "TE": 0.0}, {"QB": 0.4, "RB": 0.6, "WR": 0.0, "TE": 0.0}]
    )
    assert result["QB"] == pytest.approx(0.3)


def test_position_guide_schema_roundtrip() -> None:
    """PositionGuideFile serializes and validates through Pydantic."""
    guide = PositionGuideFile(
        generated_at=datetime(2026, 7, 5, tzinfo=timezone.utc),
        draft_year=2026,
        draft_slot=5,
        num_teams=12,
        simulations=100,
        checkpoint_path="models/12_teams_random_start/v3/checkpoint_episode_1.pth",
        checkpoint_episode=1,
        player_data_csv="data/generated_player_data.csv",
        enabled_state_features=["current_pick_number"],
        roster_structure={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2},
        total_user_picks=1,
        picks=[
            PositionGuidePick(
                user_pick_index=1,
                overall_pick_number=5,
                round=1,
                positions=PositionProbabilities(QB=0.1, RB=0.7, WR=0.15, TE=0.05),
                top_position="RB",
                sample_count=100,
            )
        ],
    )
    restored = PositionGuideFile.model_validate(guide.model_dump(mode="json"))
    assert restored.picks[0].top_position == "RB"


def test_simulator_smoke_two_rollouts(config, player_catalog) -> None:
    """Two rollouts with a mock policy produce aggregated pick rows."""
    env = DraftGymEnv(config, training=True, player_catalog=player_catalog)

    class _FixedPolicy:
        """Return a fixed RB-favored distribution for every state."""

        def get_action_probabilities(self, state_tensor, action_mask=None):
            probs = torch.tensor([0.05, 0.7, 0.15, 0.1], dtype=torch.float32)
            if action_mask is not None:
                mask = torch.tensor(action_mask, dtype=torch.bool)
                masked = probs.clone()
                masked[~mask] = 0.0
                total = masked.sum()
                if total > 0:
                    masked = masked / total
                probs = masked
            return probs.unsqueeze(0)

    env.agent_model = _FixedPolicy()

    aggregates: dict[int, list[dict[str, float]]] = {}
    for _ in range(2):
        env.reset()
        user_pick_index = 0
        while env.team_rosters[env.agent_team_id].size < env.total_roster_size_per_team:
            if env._controller.team_on_clock != env.agent_team_id:
                break
            suggestion = env.get_ai_suggestion_for_team(env.agent_team_id)
            user_pick_index += 1
            aggregates.setdefault(user_pick_index, []).append(suggestion)
            action = int(np.argmax([suggestion[pos] for pos in ["QB", "RB", "WR", "TE"]]))
            _, _, done, _, _ = env.step(action)
            if done:
                break

    averaged = average_position_probabilities(aggregates[1])
    assert averaged["RB"] > averaged["WR"]


def test_find_latest_checkpoint_in_dir_returns_highest_episode(tmp_path) -> None:
    """Checkpoint resolver picks the file with the largest episode number."""
    (tmp_path / "checkpoint_episode_10.pth").write_text("a", encoding="utf-8")
    (tmp_path / "checkpoint_episode_200.pth").write_text("b", encoding="utf-8")
    resolved = find_latest_checkpoint_in_dir(str(tmp_path))
    assert resolved is not None
    assert resolved.endswith("checkpoint_episode_200.pth")
