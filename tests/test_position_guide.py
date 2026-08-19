"""Tests for offline position guide generation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
import torch

from draft_buddy.data.cache_paths import model_adp_output_path, position_guide_output_path
from draft_buddy.rl.draft_gym_env import DraftGymEnv
from draft_buddy.rl.position_guide.exporter import export_model_adp, export_position_guide
from draft_buddy.rl.position_guide.pick_numbers import overall_pick_number
from draft_buddy.rl.position_guide.schemas import (
    ModelAdpEntry,
    ModelAdpFile,
    PositionGuideFile,
    PositionGuidePick,
    PositionProbabilities,
    TopPlayerEntry,
)
from draft_buddy.rl.position_guide.simulator import (
    PositionGuideSimulator,
    average_position_probabilities,
    sample_position,
)


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


def test_model_adp_output_path_includes_num_teams(tmp_path) -> None:
    """Model-ADP export paths include league size in the filename."""
    generated_at = datetime(2026, 7, 5, 22, 45, tzinfo=timezone.utc)
    path = model_adp_output_path(str(tmp_path), 12, 2026, generated_at, ext="json")

    assert "12teams" in path and "model_adp" in path


def test_average_position_probabilities() -> None:
    """Two probability samples average element-wise."""
    result = average_position_probabilities(
        [{"QB": 0.2, "RB": 0.8, "WR": 0.0, "TE": 0.0}, {"QB": 0.4, "RB": 0.6, "WR": 0.0, "TE": 0.0}]
    )
    assert result["QB"] == pytest.approx(0.3)


def test_sample_position_favors_higher_probability_position() -> None:
    """Sampling with a seeded RNG over many draws favors the higher-probability position."""
    import random

    rng = random.Random(1234)
    suggestion = {"QB": 0.0, "RB": 0.9, "WR": 0.1, "TE": 0.0}
    counts = {"QB": 0, "RB": 0, "WR": 0, "TE": 0}
    for _ in range(200):
        counts[sample_position(suggestion, rng)] += 1

    assert counts["RB"] > counts["WR"]


def test_sample_position_falls_back_to_uniform_when_all_zero() -> None:
    """Sampling with an all-zero distribution still returns a valid position."""
    import random

    rng = random.Random(1)
    result = sample_position({"QB": 0.0, "RB": 0.0, "WR": 0.0, "TE": 0.0}, rng)

    assert result in {"QB", "RB", "WR", "TE"}


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
        temperature=1.5,
        picks=[
            PositionGuidePick(
                user_pick_index=1,
                overall_pick_number=5,
                round=1,
                positions=PositionProbabilities(QB=0.1, RB=0.7, WR=0.15, TE=0.05),
                top_position="RB",
                sample_count=100,
                top_players={
                    "RB": [TopPlayerEntry(player_id=2, name="RB One", times_drafted=70, share=1.0)]
                },
            )
        ],
    )
    restored = PositionGuideFile.model_validate(guide.model_dump(mode="json"))

    assert restored.picks[0].top_position == "RB"
    assert restored.schema_version == 2
    assert restored.generation_mode == "self_play"
    assert restored.picks[0].top_players["RB"][0].name == "RB One"


def test_model_adp_schema_roundtrip() -> None:
    """ModelAdpFile serializes and validates through Pydantic."""
    model_adp = ModelAdpFile(
        generated_at=datetime(2026, 7, 5, tzinfo=timezone.utc),
        draft_year=2026,
        num_teams=12,
        simulations=100,
        checkpoint_path="models/12_teams_random_start/v3/checkpoint_episode_1.pth",
        checkpoint_episode=1,
        player_data_csv="data/generated_player_data.csv",
        temperature=1.5,
        players=[
            ModelAdpEntry(
                player_id=2,
                name="RB One",
                position="RB",
                model_adp=4.2,
                std_dev=1.1,
                times_drafted=95,
                draft_rate=0.95,
                market_adp=2.0,
            )
        ],
    )
    restored = ModelAdpFile.model_validate(model_adp.model_dump(mode="json"))

    assert restored.players[0].model_adp == pytest.approx(4.2)


class _FixedRbPolicy:
    """Deterministic policy that favors RB, for simulator smoke tests."""

    def get_action_probabilities(self, _state_tensor, action_mask=None, temperature=1.0):
        """Return a fixed RB-favored, masked, temperature-scaled distribution."""
        probs = torch.tensor([0.05, 0.7, 0.15, 0.1], dtype=torch.float32)
        if action_mask is not None:
            mask = torch.tensor(action_mask, dtype=torch.bool)
            probs = probs.clone()
            probs[~mask] = 0.0
            total = probs.sum()
            if total > 0:
                probs = probs / total
        return probs.unsqueeze(0)


def test_self_play_simulator_produces_guides_for_every_slot(
    config, player_catalog, monkeypatch
) -> None:
    """Two self-play rollouts populate every slot's guide and top players."""
    monkeypatch.setattr(
        DraftGymEnv, "_load_agent_model", lambda self: setattr(self, "agent_model", _FixedRbPolicy())
    )

    simulator = PositionGuideSimulator(
        config=config,
        num_teams=config.draft.NUM_TEAMS,
        checkpoint_path="unused.pth",
        draft_year=2026,
        simulations=2,
        seed=1,
        temperature=1.0,
        player_data_csv=config.paths.PLAYER_DATA_CSV,
        show_progress=False,
    )

    guides, model_adp = simulator.run()

    assert set(guides.keys()) == set(range(1, config.draft.NUM_TEAMS + 1))
    for guide in guides.values():
        assert guide.generation_mode == "self_play"
        assert guide.schema_version == 2
        assert guide.temperature == pytest.approx(1.0)
    first_slot_guide = guides[1]
    assert first_slot_guide.picks, "expected at least one recorded pick for slot 1"
    assert any(pick.top_players for pick in first_slot_guide.picks)
    assert model_adp.players, "expected at least one model-ADP entry"
    assert all(0.0 <= entry.draft_rate <= 1.0 for entry in model_adp.players)


def test_self_play_simulator_applies_prune_and_limit_adp_pool(
    config, player_catalog, monkeypatch
) -> None:
    """Verify guide pool flags restrict available ids and appear in export metadata."""
    monkeypatch.setattr(
        DraftGymEnv, "_load_agent_model", lambda self: setattr(self, "agent_model", _FixedRbPolicy())
    )
    monkeypatch.setattr(
        "draft_buddy.rl.position_guide.simulator.load_player_catalog",
        lambda *_args, **_kwargs: player_catalog,
    )
    inactive = player_catalog.require(2)
    object.__setattr__(inactive, "sleeper_status", "Inactive")
    limit_n = max(8, config.draft.NUM_TEAMS * 2)

    captured_pools: list[set[int]] = []
    original_reset = DraftGymEnv.reset

    def capturing_reset(self, seed=None, options=None):
        result = original_reset(self, seed=seed, options=options)
        captured_pools.append(set(self.available_player_ids))
        return result

    monkeypatch.setattr(DraftGymEnv, "reset", capturing_reset)

    simulator = PositionGuideSimulator(
        config=config,
        num_teams=config.draft.NUM_TEAMS,
        checkpoint_path="unused.pth",
        draft_year=2026,
        simulations=1,
        seed=1,
        temperature=1.0,
        player_data_csv=config.paths.PLAYER_DATA_CSV,
        show_progress=False,
        prune_inactive=True,
        limit_adp=limit_n,
    )

    guides, model_adp = simulator.run()

    assert simulator._draft_pool_ids is not None
    assert 2 not in simulator._draft_pool_ids
    assert 2 in player_catalog.player_ids
    assert len(simulator._draft_pool_ids) >= min(limit_n, len(player_catalog) - 1)
    assert captured_pools and captured_pools[0] == simulator._draft_pool_ids
    guide = guides[1]
    assert guide.prune_inactive is True
    assert guide.limit_adp == limit_n
    assert guide.draft_pool_size == len(simulator._draft_pool_ids)
    assert model_adp.draft_pool_size == guide.draft_pool_size


def test_export_position_guide_writes_json_and_html(tmp_path) -> None:
    """Verify position guide export writes both JSON and HTML files."""
    guide = PositionGuideFile(
        generated_at=datetime(2026, 7, 5, tzinfo=timezone.utc),
        draft_year=2026,
        draft_slot=1,
        num_teams=4,
        simulations=10,
        checkpoint_path="models/checkpoint_episode_1.pth",
        checkpoint_episode=1,
        player_data_csv="data/generated_player_data.csv",
        enabled_state_features=[],
        roster_structure={"QB": 1},
        total_user_picks=1,
        temperature=1.5,
        picks=[
            PositionGuidePick(
                user_pick_index=1,
                overall_pick_number=1,
                round=1,
                positions=PositionProbabilities(QB=0.1, RB=0.7, WR=0.15, TE=0.05),
                top_position="RB",
                sample_count=10,
                top_players={
                    "RB": [TopPlayerEntry(player_id=2, name="RB One", times_drafted=7, share=0.7)]
                },
            )
        ],
    )

    json_path, html_path = export_position_guide(guide, str(tmp_path))

    assert "RB One" in open(html_path, encoding="utf-8").read()
    assert json_path.endswith(".json")


def test_export_model_adp_writes_json_and_html(tmp_path) -> None:
    """Verify model-ADP export writes both JSON and HTML files."""
    model_adp = ModelAdpFile(
        generated_at=datetime(2026, 7, 5, tzinfo=timezone.utc),
        draft_year=2026,
        num_teams=4,
        simulations=10,
        checkpoint_path="models/checkpoint_episode_1.pth",
        checkpoint_episode=1,
        player_data_csv="data/generated_player_data.csv",
        temperature=1.5,
        players=[
            ModelAdpEntry(
                player_id=2,
                name="RB One",
                position="RB",
                model_adp=4.2,
                std_dev=1.1,
                times_drafted=9,
                draft_rate=0.9,
                market_adp=2.0,
            )
        ],
    )

    json_path, html_path = export_model_adp(model_adp, str(tmp_path))

    assert "RB One" in open(html_path, encoding="utf-8").read()
    assert json_path.endswith(".json")
