"""Behavioral tests for the scoring engine."""

import pandas as pd

from draft_buddy.logic.scoring.engine import ScoringEngine


def test_normalize_rule_weights_prefers_touchdown_and_xp_rule_names():
    """Verify preferred rule names suppress their legacy aliases."""
    normalized = ScoringEngine._normalize_rule_weights(
        {
            "passing_touchdowns": 6,
            "passing_tds": 4,
            "xp_made": 1,
            "pat_made": 2,
        }
    )

    assert "passing_tds" not in normalized and "pat_made" not in normalized


def test_normalize_rule_weights_drops_fg_buckets_when_yardage_rule_is_present():
    """Verify yardage-based FG scoring disables bucket scoring."""
    normalized = ScoringEngine._normalize_rule_weights(
        {"fg_made_yards": 0.1, "fg_made_0_39": 3, "fg_made_40_49": 4}
    )

    assert "fg_made_0_39" not in normalized and "fg_made_40_49" not in normalized


def test_normalize_rule_weights_drops_total_fumbles_when_specific_components_exist():
    """Verify specific fumble rules suppress the aggregate total rule."""
    normalized = ScoringEngine._normalize_rule_weights(
        {"total_fumbles_lost": -2, "rushing_fumbles_lost": -3}
    )

    assert "total_fumbles_lost" not in normalized


def test_derive_fg_buckets_from_lists_counts_each_distance_range():
    """Verify made-field-goal lists populate the expected bucket counts."""
    df = pd.DataFrame([{"fg_made_list": "18;25;37;45;54;61"}])

    ScoringEngine._derive_fg_buckets_from_lists(df)

    assert [df.iloc[0][column] for column in ["fg_made_0_19", "fg_made_20_29", "fg_made_30_39", "fg_made_40_49", "fg_made_50_59", "fg_made_60_"]] == [1, 1, 1, 1, 1, 1]


def test_compute_total_made_fg_yards_sums_list_distances():
    """Verify FG yardage total is derived from made-kick distance lists."""
    yards = ScoringEngine._compute_total_made_fg_yards(pd.DataFrame([{"fg_made_list": "53;44;28"}]))

    assert int(yards.iloc[0]) == 125


def test_apply_scoring_uses_yardage_rule_without_double_counting_fg_buckets():
    """Verify bucket rules are ignored when yardage-based FG scoring is enabled."""
    df = pd.DataFrame([{"fg_made_list": "40", "xp_made": 0, "fg_missed": 0}])

    scored = ScoringEngine.apply_scoring(
        df,
        {"fg_made_yards": 0.1, "fg_made_40_49": 4},
    )

    assert float(scored["total_pts"].iloc[0]) == 4.0


def test_apply_team_def_scoring_uses_explicit_return_touchdown_rules_instead_of_aggregate():
    """Verify explicit KR/PR touchdown rules suppress aggregate special-teams TD scoring."""
    df = pd.DataFrame(
        [{"kick_return_touchdowns": 1, "punt_return_touchdowns": 1, "st_def_td": 2}]
    )

    scored = ScoringEngine.apply_team_def_scoring(
        df,
        {"kick_return_touchdowns": 6, "punt_return_touchdowns": 6, "st_def_td": 3},
    )

    assert float(scored["def_total_pts"].iloc[0]) == 12.0


def test_apply_team_def_scoring_applies_points_allowed_tiers():
    """Verify points-allowed tiers map to their configured score values."""
    df = pd.DataFrame([{"points_allowed": 0}, {"points_allowed": 30}, {"points_allowed": 40}])

    scored = ScoringEngine.apply_team_def_scoring(
        df,
        {
            "def_points_allowed_0": 10,
            "def_points_allowed_28_34": -1,
            "def_points_allowed_35_plus": -4,
        },
    )

    assert list(scored["def_total_pts"]) == [10.0, -1.0, -4.0]


def test_coerce_numeric_turns_non_numeric_values_into_zero():
    """Verify numeric coercion safely zero-fills invalid values."""
    df = pd.DataFrame([{"passing_yards": "abc"}])

    ScoringEngine._coerce_numeric(df, ["passing_yards"])

    assert float(df["passing_yards"].iloc[0]) == 0.0
