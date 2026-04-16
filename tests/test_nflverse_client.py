"""Tests for nflverse downloader client."""

from unittest.mock import mock_open, patch

import pandas as pd

from draft_buddy.data_pipeline.nflverse_client import NflverseCsvDownloader


class _MockResponse:
    """Simple context-managed response mock."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.headers = {"content-length": "4"}

    def __enter__(self):
        """Enter context."""
        return self

    def __exit__(self, exc_type, exc, tb):
        """Exit context."""
        return None

    def raise_for_status(self):
        """No-op status check."""
        return None

    def iter_content(self, chunk_size=8192):
        """Yield configured byte chunks."""
        del chunk_size
        return iter(self._chunks)


def test_normalize_player_id_strips_non_numeric_characters():
    """Verify player IDs keep only digits."""
    downloader = NflverseCsvDownloader(cache_dir="./data")
    dataframe = pd.DataFrame({"player_id": ["00-0012345"]})
    downloader._normalize_player_id(dataframe)

    assert int(dataframe.loc[0, "player_id"]) == 12345


def test_normalize_player_id_converts_empty_to_pd_na():
    """Verify empty player ID values become missing Int64 values."""
    downloader = NflverseCsvDownloader(cache_dir="./data")
    dataframe = pd.DataFrame({"player_id": [""]})
    downloader._normalize_player_id(dataframe)

    assert pd.isna(dataframe.loc[0, "player_id"])


def test_normalize_player_id_skips_when_column_is_missing():
    """Verify missing id column is safely ignored."""
    downloader = NflverseCsvDownloader(cache_dir="./data")
    dataframe = pd.DataFrame({"other": [1]})
    downloader._normalize_player_id(dataframe)

    assert "other" in dataframe.columns


@patch("draft_buddy.data_pipeline.nflverse_client.pd.read_csv")
@patch("builtins.open", new_callable=mock_open)
@patch("draft_buddy.data_pipeline.nflverse_client.requests.get")
@patch("draft_buddy.data_pipeline.nflverse_client.os.path.exists", return_value=False)
def test_download_file_requests_network_when_cache_missing(
    mock_exists, mock_get, mock_file, mock_read_csv
):
    """Verify missing cache triggers network download."""
    del mock_exists, mock_file
    mock_get.return_value = _MockResponse([b"a", b"b"])
    mock_read_csv.return_value = pd.DataFrame({"x": [1]})
    downloader = NflverseCsvDownloader(cache_dir="/tmp/cache")
    downloader.download_file("player_stats.csv", "https://example.com/player_stats.csv")

    assert mock_get.called


@patch("draft_buddy.data_pipeline.nflverse_client.pd.read_csv")
@patch("builtins.open", new_callable=mock_open)
@patch("draft_buddy.data_pipeline.nflverse_client.requests.get")
@patch("draft_buddy.data_pipeline.nflverse_client.os.path.exists", return_value=False)
def test_download_file_writes_stream_chunks_to_cache_path(
    mock_exists, mock_get, mock_file, mock_read_csv
):
    """Verify streamed chunks are written to local cache file."""
    del mock_exists
    mock_get.return_value = _MockResponse([b"ab", b"cd"])
    mock_read_csv.return_value = pd.DataFrame({"x": [1]})
    downloader = NflverseCsvDownloader(cache_dir="/tmp/cache")
    downloader.download_file("player_stats.csv", "https://example.com/player_stats.csv")
    handle = mock_file()

    assert handle.write.call_count == 2


@patch("draft_buddy.data_pipeline.nflverse_client.pd.read_csv")
@patch("draft_buddy.data_pipeline.nflverse_client.os.path.getsize", return_value=1)
@patch("draft_buddy.data_pipeline.nflverse_client.os.path.exists", return_value=True)
@patch.object(NflverseCsvDownloader, "_download_from_url")
def test_download_file_skips_network_when_cached_file_exists(
    mock_download, mock_exists, mock_getsize, mock_read_csv
):
    """Verify cached non-empty files do not trigger download."""
    del mock_exists, mock_getsize
    mock_read_csv.return_value = pd.DataFrame({"x": [1]})
    downloader = NflverseCsvDownloader(cache_dir="/tmp/cache")
    downloader.download_file("player_stats.csv", "https://example.com/player_stats.csv")

    assert mock_download.called is False


@patch.object(NflverseCsvDownloader, "download_file")
def test_fetch_player_pool_splits_legacy_stats_before_draft_year(mock_download_file):
    """Verify legacy stats include only seasons before draft year."""
    ps_df = pd.DataFrame(
        [
            {"player_id": "00-0001", "season": 2023, "position": "QB", "pass_yards": 1},
            {"player_id": "00-0002", "season": 2024, "position": "QB", "pass_yards": 2},
        ]
    )
    psk_df = pd.DataFrame(
        [
            {"player_id": "00-0001", "season": 2023, "position": "QB"},
            {"player_id": "00-0002", "season": 2024, "position": "QB"},
        ]
    )
    roster_df = pd.DataFrame(
        [{"player_id": "00-0001", "full_name": "Player One", "team": "KC", "position": "QB"}]
    )
    mock_download_file.side_effect = [ps_df, psk_df, roster_df]
    downloader = NflverseCsvDownloader(cache_dir="./data")
    _, legacy_stats_df, _ = downloader.fetch_player_pool(2024, ["QB"], 2023, 2024)

    assert set(legacy_stats_df["season"].unique()) == {2023}


@patch.object(NflverseCsvDownloader, "download_file")
def test_fetch_player_pool_splits_draft_year_stats_for_selected_year(mock_download_file):
    """Verify draft-year stats include only season equal to draft year."""
    ps_df = pd.DataFrame(
        [
            {"player_id": "00-0001", "season": 2023, "position": "QB", "pass_yards": 1},
            {"player_id": "00-0002", "season": 2024, "position": "QB", "pass_yards": 2},
        ]
    )
    psk_df = pd.DataFrame(
        [
            {"player_id": "00-0001", "season": 2023, "position": "QB"},
            {"player_id": "00-0002", "season": 2024, "position": "QB"},
        ]
    )
    roster_df = pd.DataFrame(
        [{"player_id": "00-0002", "full_name": "Player Two", "team": "BUF", "position": "QB"}]
    )
    mock_download_file.side_effect = [ps_df, psk_df, roster_df]
    downloader = NflverseCsvDownloader(cache_dir="./data")
    _, _, draft_year_stats_df = downloader.fetch_player_pool(2024, ["QB"], 2023, 2024)

    assert set(draft_year_stats_df["season"].unique()) == {2024}


@patch.object(NflverseCsvDownloader, "download_file")
def test_fetch_player_pool_creates_player_id_from_index_when_roster_id_missing(mock_download_file):
    """Verify roster rows get fallback player_id when source column is absent."""
    ps_df = pd.DataFrame([{"player_id": "00-0001", "season": 2024, "position": "QB"}])
    psk_df = pd.DataFrame([{"player_id": "00-0001", "season": 2024, "position": "QB"}])
    roster_df = pd.DataFrame([{"full_name": "Player No Id", "team": "KC", "position": "QB"}])
    mock_download_file.side_effect = [ps_df, psk_df, roster_df]
    downloader = NflverseCsvDownloader(cache_dir="./data")
    draft_pool_df, _, _ = downloader.fetch_player_pool(2024, ["QB"], 2024, 2024)

    assert int(draft_pool_df.iloc[0]["player_id"]) == 0
