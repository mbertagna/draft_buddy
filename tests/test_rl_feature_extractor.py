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
