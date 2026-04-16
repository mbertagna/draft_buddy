"""Tests for RL feature extraction behavior."""

from types import SimpleNamespace

from draft_buddy.config import Config
from draft_buddy.core.draft_state import DraftState
from draft_buddy.domain.entities import Player
from draft_buddy.draft_env.state_normalizer import StateNormalizer
from draft_buddy.rl.feature_extractor import FeatureExtractor


def _build_feature_test_objects():
    """Build deterministic config, state, and player map for extractor tests."""
    config = Config()
    config.draft.NUM_TEAMS = 4
    config.draft.ROSTER_STRUCTURE = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}
    config.draft.BENCH_MAXES = {"QB": 1, "RB": 1, "WR": 1, "TE": 1}
    config.training.ENABLED_STATE_FEATURES = [
        "best_available_qb_vorp",
        "rb_scarcity",
        "rb_imminent_threat",
    ]
    players = [
        Player(1, "QB1", "QB", 300.0),
        Player(2, "QB2", "QB", 280.0),
        Player(11, "QB3", "QB", 260.0),
        Player(12, "QB4", "QB", 240.0),
        Player(3, "RB1", "RB", 250.0),
        Player(4, "RB2", "RB", 240.0),
        Player(5, "RB3", "RB", 230.0),
        Player(6, "RB4", "RB", 220.0),
        Player(7, "WR1", "WR", 210.0),
        Player(8, "WR2", "WR", 205.0),
    ]
    player_map = {player.player_id: player for player in players}
    state = DraftState(
        all_player_ids=set(player_map.keys()),
        draft_order=[1, 2, 3, 4, 3, 2, 1],
        roster_structure=config.draft.ROSTER_STRUCTURE,
        bench_maxes=config.draft.BENCH_MAXES,
        total_roster_size_per_team=sum(config.draft.ROSTER_STRUCTURE.values())
        + sum(config.draft.BENCH_MAXES.values()),
        agent_team_id=1,
    )
    extractor = FeatureExtractor(config, StateNormalizer(config))
    return SimpleNamespace(config=config, state=state, player_map=player_map, extractor=extractor)


def test_extract_returns_array_length_matching_enabled_state_features():
    """Verify normalized feature vector length matches configured feature count."""
    ctx = _build_feature_test_objects()
    features = ctx.extractor.extract(ctx.state, ctx.player_map, 1)

    assert len(features) == len(ctx.config.training.ENABLED_STATE_FEATURES)


def test_compute_global_state_features_golden_master_best_available_qb_vorp():
    """Golden-master lock for best_available_qb_vorp output."""
    ctx = _build_feature_test_objects()
    global_features = ctx.extractor.compute_global_state_features(ctx.state, ctx.player_map)

    assert global_features["best_available_qb_vorp"] == 53.33333333333334


def test_compute_global_state_features_golden_master_rb_scarcity():
    """Golden-master lock for rb_scarcity output."""
    ctx = _build_feature_test_objects()
    global_features = ctx.extractor.compute_global_state_features(ctx.state, ctx.player_map)

    assert global_features["rb_scarcity"] == 30.0


def test_calculate_imminent_threat_counts_three_rb_needy_teams_between_picks():
    """Verify imminent threat count for RB starter needs."""
    ctx = _build_feature_test_objects()
    ctx.state.set_current_pick_idx(0)
    threat = ctx.extractor.calculate_imminent_threat(ctx.state, ctx.state.get_rosters(), 1, "RB")

    assert threat == 3


def test_compute_global_state_features_handles_empty_available_pool_without_crashing():
    """Verify empty available pool yields safe zero scarcity output."""
    ctx = _build_feature_test_objects()
    ctx.state.replace_available_player_ids(set())
    global_features = ctx.extractor.compute_global_state_features(ctx.state, ctx.player_map)

    assert global_features["rb_scarcity"] == 0.0
