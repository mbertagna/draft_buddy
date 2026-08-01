"""Tests for ADP matching behavior."""

from __future__ import annotations

import pandas as pd

from draft_buddy.data.adp_matcher import AdpMatcher
from draft_buddy.data.name_matching import standardize_name as _standardize_name


_MINIMAL_ADP_HTML = """
<table class="mcu-table mcu-table__table reports__table-inner">
  <caption class="hidden-aria">Average Draft Position (ADP) - PPR Leagues 2026</caption>
  <thead>
    <tr>
      <th>Rank</th><th>Player (Bye)</th><th>POS</th><th>AVG</th>
    </tr>
  </thead>
  <tbody class="mcu-table__data">
    <tr class="mcu-table__row mcu-table__data-row">
      <td class="mcu-table__cell mcu-table__cell--rank"><div>1</div></td>
      <td class="mcu-table__cell mcu-table__cell--player">
        <a class="fp-player-link" fp-player-name="Josh Allen">Josh Allen</a>
        <span class="reports__player-team">BUF (7)</span>
      </td>
      <td class="mcu-table__cell mcu-table__cell--pos"><div>QB1</div></td>
      <td class="mcu-table__cell mcu-table__cell--avg"><div>1.0</div></td>
    </tr>
  </tbody>
</table>
"""

_SUFFIX_PLAYER_ADP_HTML = """
<table class="mcu-table reports__table-inner">
  <caption>Average Draft Position (ADP) - PPR Leagues 2026</caption>
  <tbody>
    <tr class="mcu-table__row mcu-table__data-row">
      <td class="mcu-table__cell mcu-table__cell--rank"><div>12</div></td>
      <td class="mcu-table__cell mcu-table__cell--player">
        <a fp-player-name="James Cook III">James Cook III</a>
        <span class="reports__player-team">BUF (7)</span>
      </td>
      <td class="mcu-table__cell mcu-table__cell--pos"><div>RB6</div></td>
      <td class="mcu-table__cell mcu-table__cell--avg"><div>12.5</div></td>
    </tr>
  </tbody>
</table>
"""

_NEAR_MATCH_ADP_HTML = """
<table class="mcu-table reports__table-inner">
  <caption>Average Draft Position (ADP)</caption>
  <tbody>
    <tr class="mcu-table__row mcu-table__data-row">
      <td class="mcu-table__cell mcu-table__cell--rank"><div>1</div></td>
      <td class="mcu-table__cell mcu-table__cell--player">
        <a fp-player-name="Jahs Allen">Jahs Allen</a>
        <span class="reports__player-team">MIA (6)</span>
      </td>
      <td class="mcu-table__cell mcu-table__cell--pos"><div>QB1</div></td>
      <td class="mcu-table__cell mcu-table__cell--avg"><div>1.0</div></td>
    </tr>
  </tbody>
</table>
"""


def test_standardize_name_removes_suffix_and_diacritics() -> None:
    """Verify name normalization strips suffixes and accents."""
    assert _standardize_name("Ámon-Ra St. Brown Jr.") == "amon-ra st. brown"


def test_standardize_name_returns_none_for_nan() -> None:
    """Verify NaN names normalize to None."""
    assert _standardize_name(float("nan")) is None


def test_parse_team_bye_extracts_team_and_week() -> None:
    """Verify team/bye span text splits into separate fields."""
    team, bye = AdpMatcher()._parse_team_bye("DET (6)")

    assert team == "DET" and bye == "6"


def test_parse_team_bye_returns_none_for_invalid_text() -> None:
    """Verify unparseable team/bye text returns None values."""
    team, bye = AdpMatcher()._parse_team_bye("Free Agent")

    assert team is None and bye is None


def test_load_adp_dataframe_parses_html_rows(tmp_path) -> None:
    """Verify HTML ADP snapshots load into Rank/Player/Team/Bye/POS/AVG columns."""
    adp_path = tmp_path / "fantasypros-2026-overall-adp-rankings.html"
    adp_path.write_text(_MINIMAL_ADP_HTML, encoding="utf-8")

    adp_df = AdpMatcher()._load_adp_dataframe(str(adp_path))

    assert list(adp_df.columns) == ["Rank", "Player", "Team", "Bye", "POS", "AVG"]
    assert adp_df.loc[0].to_dict() == {
        "Rank": "1",
        "Player": "Josh Allen",
        "Team": "BUF",
        "Bye": "7",
        "POS": "QB1",
        "AVG": "1.0",
    }


def test_load_adp_dataframe_keeps_name_suffixes(tmp_path) -> None:
    """Verify suffix names like James Cook III stay intact from HTML attributes."""
    adp_path = tmp_path / "adp.html"
    adp_path.write_text(_SUFFIX_PLAYER_ADP_HTML, encoding="utf-8")

    adp_df = AdpMatcher()._load_adp_dataframe(str(adp_path))

    assert adp_df.loc[0, "Player"] == "James Cook III"
    assert adp_df.loc[0, "Team"] == "BUF"
    assert adp_df.loc[0, "Bye"] == "7"


def test_merge_adp_data_returns_empty_frames_when_file_load_fails(tmp_path) -> None:
    """Verify load failures return empty outputs instead of raising."""
    matcher = AdpMatcher()
    computed_df = pd.DataFrame(
        [{"player_id": 1, "player_display_name": "Josh Allen", "position": "QB", "recent_team": "BUF"}]
    )
    merged_df, unmatched_df, borderline_df = matcher.merge_adp_data(
        computed_df, str(tmp_path / "missing.html")
    )

    assert merged_df.empty and unmatched_df.empty and borderline_df.empty


def test_merge_adp_data_matches_exact_single_candidate(tmp_path) -> None:
    """Verify exact standardized-name matches attach the computed player id."""
    adp_path = tmp_path / "adp.html"
    adp_path.write_text(_MINIMAL_ADP_HTML, encoding="utf-8")
    computed_df = pd.DataFrame(
        [{"player_id": 1, "player_display_name": "Josh Allen", "position": "QB", "recent_team": "BUF"}]
    )
    merged_df, unmatched_df, borderline_df = AdpMatcher().merge_adp_data(computed_df, str(adp_path))

    assert list(merged_df["matched_player_id"]) == [1] and unmatched_df.empty and borderline_df.empty


def test_merge_adp_data_matches_suffix_player_from_html(tmp_path) -> None:
    """Verify end-to-end matching works for HTML rows with name suffixes."""
    adp_path = tmp_path / "adp.html"
    adp_path.write_text(_SUFFIX_PLAYER_ADP_HTML, encoding="utf-8")
    computed_df = pd.DataFrame(
        [
            {
                "player_id": 12,
                "player_display_name": "James Cook",
                "position": "RB",
                "recent_team": "BUF",
            }
        ]
    )
    merged_df, unmatched_df, borderline_df = AdpMatcher().merge_adp_data(computed_df, str(adp_path))

    assert list(merged_df["matched_player_id"]) == [12] and unmatched_df.empty and borderline_df.empty


def test_merge_adp_data_populates_borderline_bucket_for_near_match(tmp_path) -> None:
    """Verify near-threshold misses are captured in the borderline output."""
    adp_path = tmp_path / "adp.html"
    adp_path.write_text(_NEAR_MATCH_ADP_HTML, encoding="utf-8")
    computed_df = pd.DataFrame(
        [{"player_id": 1, "player_display_name": "Josh Allen", "position": "QB", "recent_team": "BUF"}]
    )
    _merged_df, unmatched_df, borderline_df = AdpMatcher().merge_adp_data(
        computed_df,
        str(adp_path),
        match_threshold=95,
    )

    assert not unmatched_df.empty and not borderline_df.empty
