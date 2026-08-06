"""Tests for RL feature engineering on canonical draft state."""

from __future__ import annotations

from draft_buddy.rl.feature_extractor import FeatureExtractor
from draft_buddy.rl.state_normalizer import StateNormalizer


def test_feature_extractor_returns_vector_aligned_to_enabled_features(config, draft_state, player_catalog) -> None:
    """Verify feature extraction consumes DraftState and PlayerCatalog directly."""
    draft_state.add_player_to_roster(1, player_catalog.require(1))
    draft_state.add_player_to_roster(1, player_catalog.require(2))
    extractor = FeatureExtractor(config, StateNormalizer(config))
    vector = extractor.extract(draft_state, player_catalog, 1)

    assert len(vector) == len(config.training.ENABLED_STATE_FEATURES)


def test_available_slot_features_clamp_to_remaining_roster_room(
    config, draft_state, player_catalog
) -> None:
    """Verify available position slots never exceed remaining total roster room."""
    config.draft.BENCH_MAXES = {"QB": 3, "RB": 8, "WR": 8, "TE": 4}
    config.training.ENABLED_STATE_FEATURES = [
        "available_roster_slots_qb",
        "available_roster_slots_rb",
        "available_roster_slots_wr",
        "available_roster_slots_te",
    ]
    for player_id in [1, 2, 3, 4, 6, 7]:
        draft_state.add_player_to_roster(1, player_catalog.require(player_id))

    extractor = FeatureExtractor(config, StateNormalizer(config))
    feature_map = extractor.build_state_map_for_team(
        draft_state=draft_state,
        player_catalog=player_catalog,
        team_id=1,
        global_features={},
    )
    remaining = draft_state.total_roster_size_per_team - draft_state.roster_for_team(1).size

    assert remaining == 1
    assert feature_map["available_roster_slots_qb"] == 1.0
    assert feature_map["available_roster_slots_rb"] == 1.0
    assert feature_map["available_roster_slots_wr"] == 1.0
    assert feature_map["available_roster_slots_te"] == 1.0
