"""Tests for ADP fuzzy matching service."""

from unittest.mock import mock_open, patch

import pandas as pd

from draft_buddy.data_pipeline.adp_matcher import AdpMatcher, _standardize_name


def test_standardize_name_removes_suffix_from_junior_name():
    """Verify Jr. suffix is stripped from names."""
    assert _standardize_name("Odell Beckham Jr.") == "odell beckham"


def test_standardize_name_removes_suffix_from_roman_numeral_name():
    """Verify Roman numeral suffix is stripped from names."""
    assert _standardize_name("Patrick Mahomes II") == "patrick mahomes"


def test_standardize_name_removes_diacritics_and_lowercases():
    """Verify diacritics are removed and text is lowercased."""
    assert _standardize_name("Jósé Núñez") == "jose nunez"


def test_standardize_name_returns_none_for_nan_values():
    """Verify NaN names produce None."""
    assert _standardize_name(pd.NA) is None


@patch("builtins.open", new_callable=mock_open)
def test_merge_adp_data_accepts_boosted_team_pos_fuzzy_match(mock_file):
    """Verify team/position bonus can push fuzzy score above threshold."""
    adp_csv = "Player,Team,POS,AVG\nPat Mahomes,KC,QB1,12\n"
    mock_file.return_value.read.return_value = adp_csv
    computed_df = pd.DataFrame(
        [
            {
                "player_id": 15,
                "player_display_name": "Patrick Mahomes",
                "position": "QB",
                "recent_team": "KC",
                "total_pts": 350.0,
            }
        ]
    )
    matcher = AdpMatcher()
    merged_df, _, _ = matcher.merge_adp_data(computed_df, "fake_adp.csv", match_threshold=95)

    assert len(merged_df) == 1


@patch("builtins.open", new_callable=mock_open)
def test_merge_adp_data_routes_low_score_player_to_unmatched(mock_file):
    """Verify low-score fuzzy matches are rejected from merged output."""
    adp_csv = "Player,Team,POS,AVG\nCompletely Different,ATL,QB1,12\n"
    mock_file.return_value.read.return_value = adp_csv
    computed_df = pd.DataFrame(
        [
            {
                "player_id": 99,
                "player_display_name": "Patrick Mahomes",
                "position": "QB",
                "recent_team": "KC",
                "total_pts": 350.0,
            }
        ]
    )
    matcher = AdpMatcher()
    _, unmatched_df, _ = matcher.merge_adp_data(computed_df, "fake_adp.csv", match_threshold=99)

    assert len(unmatched_df) == 1


@patch("builtins.open", side_effect=OSError("boom"))
def test_merge_adp_data_returns_empty_triplet_when_file_load_fails(mock_file):
    """Verify ADP load exceptions return empty DataFrames."""
    del mock_file
    matcher = AdpMatcher()
    merged_df, unmatched_df, borderline_df = matcher.merge_adp_data(pd.DataFrame(), "bad.csv")

    assert merged_df.empty and unmatched_df.empty and borderline_df.empty


@patch("builtins.open", new_callable=mock_open)
def test_merge_adp_data_produces_borderline_rows_near_threshold(mock_file):
    """Verify near-threshold misses are captured in borderline output."""
    adp_csv = "Player,Team,POS,AVG\nPat Mahomes,KC,QB1,12\n"
    mock_file.return_value.read.return_value = adp_csv
    computed_df = pd.DataFrame(
        [
            {
                "player_id": 15,
                "player_display_name": "Patrick Mahones",
                "position": "QB",
                "recent_team": "LV",
                "total_pts": 350.0,
            }
        ]
    )
    matcher = AdpMatcher()
    _, _, borderline_df = matcher.merge_adp_data(computed_df, "fake_adp.csv", match_threshold=90)

    assert len(borderline_df) >= 1


@patch("builtins.open", new_callable=mock_open)
def test_merge_adp_data_maps_team_aliases_to_standard_forms(mock_file):
    """Verify ADP team aliases (LA) map to LAR."""
    adp_csv = "Player,Team,POS,AVG\nCooper Kupp,LA,WR1,10\n"
    mock_file.return_value.read.return_value = adp_csv
    computed_df = pd.DataFrame(
        [
            {
                "player_id": 5,
                "player_display_name": "Cooper Kupp",
                "position": "WR",
                "recent_team": "LAR",
                "total_pts": 250.0,
            }
        ]
    )
    matcher = AdpMatcher()
    merged_df, _, _ = matcher.merge_adp_data(computed_df, "fake_adp.csv", match_threshold=85)

    assert str(merged_df.iloc[0]["Team"]) == "LAR"


@patch("builtins.open", new_callable=mock_open)
def test_merge_adp_data_accepts_rank_column_when_avg_missing(mock_file):
    """Verify rank column can drive sorting when AVG is absent."""
    adp_csv = "Player,Team,POS,Rank\nPatrick Mahomes,KC,QB1,2\n"
    mock_file.return_value.read.return_value = adp_csv
    computed_df = pd.DataFrame(
        [
            {
                "player_id": 1,
                "player_display_name": "Patrick Mahomes",
                "position": "QB",
                "recent_team": "KC",
                "total_pts": 300.0,
            }
        ]
    )
    matcher = AdpMatcher()
    merged_df, _, _ = matcher.merge_adp_data(computed_df, "fake_adp.csv", match_threshold=85)

    assert len(merged_df) == 1


@patch("builtins.open", new_callable=mock_open)
def test_merge_adp_data_handles_missing_team_and_pos_columns(mock_file):
    """Verify merge works when ADP file has no team/position columns."""
    adp_csv = "Player,AVG\nPatrick Mahomes,2\n"
    mock_file.return_value.read.return_value = adp_csv
    computed_df = pd.DataFrame(
        [{"player_id": 1, "player_display_name": "Patrick Mahomes", "position": "QB", "recent_team": "KC"}]
    )
    matcher = AdpMatcher()
    merged_df, _, _ = matcher.merge_adp_data(
        computed_df,
        "fake_adp.csv",
        match_threshold=85,
        adp_col_map={"Player": "Player", "Team": "NoTeam", "POS": "NoPos"},
    )

    assert len(merged_df) == 1


@patch("builtins.open", new_callable=mock_open)
def test_merge_adp_data_handles_computed_df_without_player_id_column(mock_file):
    """Verify matcher can run even without roster player_id column."""
    adp_csv = "Player,Team,POS,AVG\nPatrick Mahomes,KC,QB1,2\n"
    mock_file.return_value.read.return_value = adp_csv
    computed_df = pd.DataFrame(
        [{"player_display_name": "Patrick Mahomes", "position": "QB", "recent_team": "KC"}]
    )
    matcher = AdpMatcher()
    merged_df, _, _ = matcher.merge_adp_data(computed_df, "fake_adp.csv", match_threshold=85)

    assert len(merged_df) == 1


@patch("builtins.open", new_callable=mock_open)
def test_merge_adp_data_skips_rows_with_missing_standardized_name(mock_file):
    """Verify ADP rows with missing names are skipped."""
    adp_csv = "Player,Team,POS,AVG\n,KC,QB1,2\n"
    mock_file.return_value.read.return_value = adp_csv
    matcher = AdpMatcher()
    _, unmatched_df, _ = matcher.merge_adp_data(
        pd.DataFrame([{"player_id": 1, "player_display_name": "Patrick Mahomes", "position": "QB"}]),
        "fake_adp.csv",
        match_threshold=85,
    )

    assert len(unmatched_df) == 1


@patch("builtins.open", new_callable=mock_open)
def test_merge_adp_data_resolves_duplicate_exact_names_by_position(mock_file):
    """Verify exact duplicate names can be disambiguated by position."""
    adp_csv = "Player,Team,POS,AVG\nChris Smith,KC,RB1,20\n"
    mock_file.return_value.read.return_value = adp_csv
    computed_df = pd.DataFrame(
        [
            {"player_id": 1, "player_display_name": "Chris Smith", "position": "WR", "recent_team": "KC"},
            {"player_id": 2, "player_display_name": "Chris Smith", "position": "RB", "recent_team": "KC"},
        ]
    )
    matcher = AdpMatcher()
    merged_df, _, _ = matcher.merge_adp_data(computed_df, "fake_adp.csv", match_threshold=85)

    assert int(merged_df.iloc[0]["matched_player_id"]) == 2


@patch("builtins.open", new_callable=mock_open)
def test_merge_adp_data_handles_files_without_rank_or_avg_columns(mock_file):
    """Verify merge succeeds when neither AVG nor Rank exists."""
    adp_csv = "Player,Team,POS\nPatrick Mahomes,KC,QB1\n"
    mock_file.return_value.read.return_value = adp_csv
    computed_df = pd.DataFrame(
        [{"player_id": 1, "player_display_name": "Patrick Mahomes", "position": "QB", "recent_team": "KC"}]
    )
    matcher = AdpMatcher()
    merged_df, _, _ = matcher.merge_adp_data(computed_df, "fake_adp.csv", match_threshold=85)

    assert len(merged_df) == 1
