"""Tests for team name variant lookups."""

from draft_buddy.data.insights.team_names import TEAM_FULL_NAMES, team_name_variants


def test_team_full_names_covers_all_32_teams() -> None:
    """Verify the lookup has an entry for every NFL team."""
    assert len(TEAM_FULL_NAMES) == 32


def test_team_name_variants_includes_full_name_nickname_and_abbr() -> None:
    """Verify variants include the full name, nickname, and abbreviation."""
    variants = team_name_variants("SF")

    assert variants == ["San Francisco 49ers", "49ers", "SF"]


def test_team_name_variants_handles_multi_word_city() -> None:
    """Verify multi-word cities still resolve to a single-word nickname."""
    variants = team_name_variants("KC")

    assert variants == ["Kansas City Chiefs", "Chiefs", "KC"]


def test_team_name_variants_falls_back_to_abbr_for_unknown_team() -> None:
    """Verify an unrecognized abbreviation falls back to just itself."""
    variants = team_name_variants("ZZZ")

    assert variants == ["ZZZ"]
