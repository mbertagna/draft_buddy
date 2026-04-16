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
