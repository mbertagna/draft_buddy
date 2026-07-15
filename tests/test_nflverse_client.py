"""Tests for nflverse data loading behavior."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests

from draft_buddy.data.nflverse_client import NflverseCsvDownloader


def test_normalize_player_id_column_strips_non_digits(tmp_path: Path) -> None:
    """Verify player_id normalization removes non-numeric characters."""
    downloader = NflverseCsvDownloader(str(tmp_path))
    dataframe = pd.DataFrame({"player_id": ["abc123", "456-x", ""]})
    downloader._normalize_player_id_column(dataframe)

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


def test_download_file_fetches_when_cache_is_missing(monkeypatch, tmp_path: Path) -> None:
    """Verify missing cache files are downloaded before reading."""
    downloader = NflverseCsvDownloader(str(tmp_path))

    def fake_download(_url: str, file_path: str) -> None:
        Path(file_path).write_text("value\n2\n", encoding="utf-8")

    monkeypatch.setattr(downloader, "_download_from_url", fake_download)

    dataframe = downloader.download_file("missing.csv", "https://example.com/missing.csv")

    assert int(dataframe.iloc[0]["value"]) == 2


def test_download_file_redownloads_when_cache_is_stale(monkeypatch, tmp_path: Path) -> None:
    """Verify a stale cached file is refreshed via a new download."""
    cached_file = tmp_path / "cached.csv"
    cached_file.write_text("value\n1\n", encoding="utf-8")
    old_mtime = time.time() - 1000
    os.utime(cached_file, (old_mtime, old_mtime))
    downloader = NflverseCsvDownloader(str(tmp_path), cache_max_age_seconds=500)

    def fake_download(_url: str, file_path: str) -> None:
        Path(file_path).write_text("value\n2\n", encoding="utf-8")

    monkeypatch.setattr(downloader, "_download_from_url", fake_download)

    dataframe = downloader.download_file("cached.csv", "https://example.com/cached.csv")

    assert int(dataframe.iloc[0]["value"]) == 2


def test_download_file_reuses_cache_within_max_age(monkeypatch, tmp_path: Path) -> None:
    """Verify a cache file younger than the max age is not re-downloaded."""
    cached_file = tmp_path / "cached.csv"
    cached_file.write_text("value\n1\n", encoding="utf-8")
    downloader = NflverseCsvDownloader(str(tmp_path), cache_max_age_seconds=500)
    called = {"downloaded": False}
    monkeypatch.setattr(downloader, "_download_from_url", lambda *_args, **_kwargs: called.__setitem__("downloaded", True))

    dataframe = downloader.download_file("cached.csv", "https://example.com/cached.csv")

    assert int(dataframe.iloc[0]["value"]) == 1 and called["downloaded"] is False


def test_fetch_legacy_stats_filters_by_player_id_allowlist(monkeypatch, tmp_path: Path) -> None:
    """Verify fetch_legacy_stats keeps allowlisted ids regardless of weekly position."""
    downloader = NflverseCsvDownloader(str(tmp_path))
    stats_df = pd.DataFrame(
        [
            {
                "player_id": "00-0000001",
                "season": 2024,
                "position": "QB",
                "player_display_name": "A",
                "recent_team": "BUF",
            },
            {
                "player_id": "00-0000001",
                "season": 2025,
                "position": "QB",
                "player_display_name": "A",
                "recent_team": "BUF",
            },
            {
                "player_id": "00-0040718",
                "season": 2025,
                "position": "CB",
                "player_display_name": "Travis Hunter",
                "recent_team": "JAX",
            },
            {
                "player_id": "00-0000002",
                "season": 2025,
                "position": "WR",
                "player_display_name": "Excluded",
                "recent_team": "KC",
            },
        ]
    )
    monkeypatch.setattr(downloader, "_fetch_stats_player_week_range", lambda _start, _end: stats_df.copy())

    legacy_stats_df, draft_year_stats_df = downloader.fetch_legacy_stats(
        draft_year=2025,
        start_year=2024,
        end_year=2025,
        nflverse_player_ids={1, 40718},
    )

    assert len(legacy_stats_df) == 1
    assert set(draft_year_stats_df["player_id"].astype(int)) == {1, 40718}
    assert "Travis Hunter" in set(draft_year_stats_df["player_display_name"])
    assert "Excluded" not in set(draft_year_stats_df["player_display_name"])


def test_fetch_legacy_stats_returns_empty_when_allowlist_empty(monkeypatch, tmp_path: Path) -> None:
    """Verify an empty allowlist skips loading weekly stats frames."""
    downloader = NflverseCsvDownloader(str(tmp_path))
    called = {"fetched": False}

    def fake_fetch(_start: int, _end: int) -> pd.DataFrame:
        called["fetched"] = True
        return pd.DataFrame(
            [{"player_id": "1", "season": 2025, "position": "QB", "player_display_name": "A"}]
        )

    monkeypatch.setattr(downloader, "_fetch_stats_player_week_range", fake_fetch)

    legacy_stats_df, draft_year_stats_df = downloader.fetch_legacy_stats(
        draft_year=2025,
        start_year=2024,
        end_year=2025,
        nflverse_player_ids=[],
    )

    assert legacy_stats_df.empty and draft_year_stats_df.empty and called["fetched"] is False


def test_fetch_stats_player_week_range_concatenates_seasons(monkeypatch, tmp_path: Path) -> None:
    """Verify each season in the range is fetched and concatenated."""
    downloader = NflverseCsvDownloader(str(tmp_path))
    season_frames = {
        2024: pd.DataFrame(
            [{"player_id": "1", "season": 2024, "position": "RB", "player_display_name": "Vet", "recent_team": "BUF"}]
        ),
        2025: pd.DataFrame(
            [
                {
                    "player_id": "00-0040666",
                    "season": 2025,
                    "position": "RB",
                    "player_display_name": "Omarion Hampton",
                    "team": "LAC",
                }
            ]
        ),
    }
    monkeypatch.setattr(
        downloader,
        "_fetch_season_player_stats",
        lambda season: season_frames.get(season, pd.DataFrame()).copy(),
    )

    merged_df = downloader._fetch_stats_player_week_range(2024, 2025)

    assert set(merged_df["season"].astype(int)) == {2024, 2025}
    assert "Omarion Hampton" in set(merged_df["player_display_name"])


def test_fetch_season_player_stats_returns_empty_frame_on_404(monkeypatch, tmp_path: Path) -> None:
    """Verify unpublished seasons return an empty frame instead of raising."""
    downloader = NflverseCsvDownloader(str(tmp_path))

    def raise_not_found(_url: str, _file_path: str) -> None:
        response = requests.Response()
        response.status_code = 404
        raise requests.HTTPError(response=response)

    monkeypatch.setattr(downloader, "_download_from_url", raise_not_found)

    season_df = downloader._fetch_season_player_stats(2026)

    assert season_df.empty


def test_fetch_stats_player_week_range_skips_unpublished_future_season(monkeypatch, tmp_path: Path) -> None:
    """Verify a 404 on a future season still returns data for published seasons."""
    downloader = NflverseCsvDownloader(str(tmp_path))
    season_frames = {
        2024: pd.DataFrame(
            [{"player_id": "1", "season": 2024, "position": "QB", "player_display_name": "A", "recent_team": "BUF"}]
        ),
        2025: pd.DataFrame(
            [{"player_id": "1", "season": 2025, "position": "QB", "player_display_name": "A", "recent_team": "BUF"}]
        ),
        2026: pd.DataFrame(),
    }
    monkeypatch.setattr(
        downloader,
        "_fetch_season_player_stats",
        lambda season: season_frames.get(season, pd.DataFrame()).copy(),
    )

    merged_df = downloader._fetch_stats_player_week_range(2024, 2026)

    assert set(merged_df["season"].astype(int)) == {2024, 2025}


def test_fetch_draft_year_roster_creates_fallback_player_ids_when_roster_ids_missing(
    monkeypatch, tmp_path: Path
) -> None:
    """Verify roster rows receive generated player ids when the source file lacks them."""
    downloader = NflverseCsvDownloader(str(tmp_path))
    roster_df = pd.DataFrame([{"full_name": "A", "position": "QB", "team": "BUF"}])
    monkeypatch.setattr(downloader, "ensure_roster_cached", lambda _draft_year: None)
    monkeypatch.setattr(pd, "read_csv", lambda _path: roster_df.copy())

    draft_pool_df = downloader.fetch_draft_year_roster(
        draft_year=2025,
        positions=["QB"],
    )

    assert int(draft_pool_df.iloc[0]["player_id"]) == 0
