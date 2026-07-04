"""Tests for ADP matching behavior."""

from __future__ import annotations

import pandas as pd

from draft_buddy.data.adp_matcher import AdpMatcher, _standardize_name


def test_standardize_name_removes_suffix_and_diacritics() -> None:
    """Verify name normalization strips suffixes and accents."""
    assert _standardize_name("Ámon-Ra St. Brown Jr.") == "amon-ra st. brown"


def test_standardize_name_returns_none_for_nan() -> None:
    """Verify NaN names normalize to None."""
    assert _standardize_name(float("nan")) is None


def test_clean_adp_content_rewrites_known_bad_tokens() -> None:
    """Verify ADP content cleaning applies regex repairs."""
    matcher = AdpMatcher()

    assert '","NO","12","' in matcher._clean_adp_content('","N","12 O","')


def test_merge_adp_data_returns_empty_frames_when_file_load_fails(tmp_path) -> None:
    """Verify load failures return empty outputs instead of raising."""
    matcher = AdpMatcher()
    computed_df = pd.DataFrame([{"player_id": 1, "player_display_name": "Josh Allen", "position": "QB", "recent_team": "BUF"}])
    merged_df, unmatched_df, borderline_df = matcher.merge_adp_data(computed_df, str(tmp_path / "missing.csv"))

    assert merged_df.empty and unmatched_df.empty and borderline_df.empty


def test_merge_adp_data_matches_exact_single_candidate(tmp_path) -> None:
    """Verify exact standardized-name matches attach the computed player id."""
    adp_path = tmp_path / "adp.csv"
    adp_path.write_text("Player,Team,POS,AVG\nJosh Allen,BUF,QB,1\n", encoding="utf-8")
    computed_df = pd.DataFrame(
        [{"player_id": 1, "player_display_name": "Josh Allen", "position": "QB", "recent_team": "BUF"}]
    )
    merged_df, unmatched_df, borderline_df = AdpMatcher().merge_adp_data(computed_df, str(adp_path))

    assert list(merged_df["matched_player_id"]) == [1] and unmatched_df.empty and borderline_df.empty


def test_split_combined_player_column_extracts_name_team_and_bye() -> None:
    """Verify a combined 'Player (Team / Bye)' cell splits into separate columns."""
    adp_df = pd.DataFrame({"Player (Team / Bye)": ["Jahmyr Gibbs   DET (6)"], "POS": ["RB1"], "AVG": [1.0]})

    result_df = AdpMatcher()._split_combined_player_column(adp_df)

    assert result_df.loc[0, ["Player", "Team", "Bye"]].tolist() == ["Jahmyr Gibbs", "DET", "6"]


def test_split_combined_player_column_falls_back_to_raw_text_without_team() -> None:
    """Verify a combined cell with no team/bye suffix keeps the full name and leaves Team empty."""
    adp_df = pd.DataFrame({"Player (Team / Bye)": ["Stefon Diggs"], "POS": ["WR56"], "AVG": [135.0]})

    result_df = AdpMatcher()._split_combined_player_column(adp_df)

    assert result_df.loc[0, "Player"] == "Stefon Diggs" and pd.isna(result_df.loc[0, "Team"])


def test_split_combined_player_column_returns_unchanged_when_no_combined_column() -> None:
    """Verify the standard export format (separate Player/Team columns) is left untouched."""
    adp_df = pd.DataFrame({"Player": ["Josh Allen"], "Team": ["BUF"]})

    result_df = AdpMatcher()._split_combined_player_column(adp_df)

    pd.testing.assert_frame_equal(result_df, adp_df)


def test_merge_adp_data_matches_players_from_combined_player_column(tmp_path) -> None:
    """Verify end-to-end matching works when the ADP file uses the combined column format."""
    adp_path = tmp_path / "adp.csv"
    adp_path.write_text("Rank,Player (Team / Bye),POS,AVG\n1,Josh Allen   BUF (7),QB1,1.0\n", encoding="utf-8")
    computed_df = pd.DataFrame(
        [{"player_id": 1, "player_display_name": "Josh Allen", "position": "QB", "recent_team": "BUF"}]
    )
    merged_df, unmatched_df, borderline_df = AdpMatcher().merge_adp_data(computed_df, str(adp_path))

    assert list(merged_df["matched_player_id"]) == [1] and unmatched_df.empty and borderline_df.empty


def test_merge_adp_data_populates_borderline_bucket_for_near_match(tmp_path) -> None:
    """Verify near-threshold misses are captured in the borderline output."""
    adp_path = tmp_path / "adp.csv"
    adp_path.write_text("Player,Team,POS,AVG\nJahs Allen,MIA,QB,1\n", encoding="utf-8")
    computed_df = pd.DataFrame(
        [{"player_id": 1, "player_display_name": "Josh Allen", "position": "QB", "recent_team": "BUF"}]
    )
    _merged_df, unmatched_df, borderline_df = AdpMatcher().merge_adp_data(
        computed_df,
        str(adp_path),
        match_threshold=95,
    )

    assert not unmatched_df.empty and not borderline_df.empty
