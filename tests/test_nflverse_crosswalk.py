"""Tests for nflverse sleeper_id crosswalk building."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from draft_buddy.data.cache_paths import nflverse_cache_dir
from draft_buddy.data.nflverse_crosswalk import NflverseCrosswalkBuilder


def test_build_keeps_latest_season_row_per_sleeper_id(tmp_path: Path) -> None:
    """Verify merged roster files dedupe to the latest season per sleeper_id."""
    cache_dir = nflverse_cache_dir(str(tmp_path))
    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True)
    older_roster = pd.DataFrame(
        [
            {
                "season": 2024,
                "sleeper_id": 9226,
                "gsis_id": "00-0039040",
                "draft_number": 99,
            }
        ]
    )
    newer_roster = pd.DataFrame(
        [
            {
                "season": 2026,
                "sleeper_id": 9226,
                "gsis_id": "00-0039040",
                "draft_number": 84,
            }
        ]
    )
    older_roster.to_csv(cache_dir_path / "roster_2024.csv", index=False)
    newer_roster.to_csv(cache_dir_path / "roster_2026.csv", index=False)

    crosswalk_df = NflverseCrosswalkBuilder().build(str(tmp_path), draft_year=2026)

    assert int(crosswalk_df.iloc[0]["draft_number"]) == 84


def test_build_returns_empty_frame_when_no_roster_files_exist(tmp_path: Path) -> None:
    """Verify an empty crosswalk is returned when no roster cache exists."""
    crosswalk_df = NflverseCrosswalkBuilder().build(str(tmp_path), draft_year=2026)

    assert crosswalk_df.empty
