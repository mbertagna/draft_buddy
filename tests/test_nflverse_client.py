"""Tests for nflverse data loading behavior."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from draft_buddy.data.nflverse_client import NflverseCsvDownloader


def test_normalize_player_id_strips_non_digits(tmp_path: Path) -> None:
    """Verify player_id normalization removes non-numeric characters."""
    downloader = NflverseCsvDownloader(str(tmp_path))
    dataframe = pd.DataFrame({"player_id": ["abc123", "456-x", ""]})
    downloader._normalize_player_id(dataframe)

    assert list(dataframe["player_id"].astype("object")) == [123, 456, pd.NA]


def test_download_file_uses_cached_csv_without_fetch(monkeypatch, tmp_path: Path) -> None:
    """Verify cached files are read directly without downloading."""
    cached_file = tmp_path / "cached.csv"
    cached_file.write_text("value\n1\n", encoding="utf-8")
    downloader = NflverseCsvDownloader(str(tmp_path))
    called = {"downloaded": False}
    monkeypatch.setattr(downloader, "_download_from_url", lambda *_args, **_kwargs: called.__setitem__("downloaded", True))
    dataframe = downloader.download_file("cached.csv", "https://example.com/cached.csv")

    assert int(dataframe.iloc[0]["value"]) == 1 and called["downloaded"] is False


def test_fetch_player_pool_filters_season_and_position(monkeypatch, tmp_path: Path) -> None:
    """Verify fetch_player_pool splits legacy and draft-year rows after filtering."""
    downloader = NflverseCsvDownloader(str(tmp_path))
    stats_df = pd.DataFrame(
        [
            {"player_id": "1", "season": 2024, "position": "QB", "player_display_name": "A", "recent_team": "BUF"},
            {"player_id": "1", "season": 2025, "position": "QB", "player_display_name": "A", "recent_team": "BUF"},
            {"player_id": "2", "season": 2025, "position": "K", "player_display_name": "B", "recent_team": "BUF"},
        ]
    )
    roster_df = pd.DataFrame(
        [
            {"full_name": "A", "position": "QB", "team": "BUF", "player_id": "1"},
            {"full_name": "B", "position": "K", "team": "BUF", "player_id": "2"},
        ]
    )
    monkeypatch.setattr(
        downloader,
        "download_file",
        lambda file_name, _url: stats_df.copy() if "player_stats" in file_name else roster_df.copy(),
    )
    draft_pool_df, legacy_stats_df, draft_year_stats_df = downloader.fetch_player_pool(
        draft_year=2025,
        positions=["QB"],
        start_year=2024,
        end_year=2025,
    )

    assert len(draft_pool_df) == 1 and len(legacy_stats_df) == 1 and len(draft_year_stats_df) == 1
