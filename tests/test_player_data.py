"""Tests for player loader and player-data utilities."""

from unittest.mock import mock_open, patch

import pandas as pd
import pytest

from draft_buddy.data.player_loader import _create_dummy_csv, _generate_mock_adp, load_player_data
from draft_buddy.data_pipeline.player_data_utils import _convert_to_simulation_format, get_simulation_dfs
from draft_buddy.domain.entities import Player


@patch("draft_buddy.data.player_loader.pd.read_csv")
def test_load_player_data_parses_rows_into_player_entities(mock_read_csv):
    """Verify load_player_data returns Player entity instances."""
    mock_read_csv.return_value = pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "A",
                "position": "qb",
                "projected_points": 100.0,
                "adp": 10.0,
                "games_played_frac": 1.0,
                "bye_week": 7,
                "team": "KC",
            }
        ]
    )
    players = load_player_data("fake.csv", {"enabled": True, "weights": {}, "sort_order_ascending": True})

    assert isinstance(players[0], Player)


@patch("draft_buddy.data.player_loader.pd.read_csv")
def test_load_player_data_parses_bye_week_as_int(mock_read_csv):
    """Verify bye_week values are converted to int."""
    mock_read_csv.return_value = pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "A",
                "position": "QB",
                "projected_points": 100.0,
                "adp": 10.0,
                "games_played_frac": 1.0,
                "bye_week": 7,
                "team": "KC",
            }
        ]
    )
    players = load_player_data("fake.csv", {"enabled": True, "weights": {}, "sort_order_ascending": True})

    assert players[0].bye_week == 7


@patch("draft_buddy.data.player_loader._create_dummy_csv")
@patch("draft_buddy.data.player_loader.pd.read_csv")
def test_load_player_data_calls_dummy_csv_creation_on_missing_file(mock_read_csv, mock_create_dummy):
    """Verify missing source CSV triggers dummy CSV generation."""
    mock_read_csv.side_effect = [
        FileNotFoundError("missing"),
        pd.DataFrame(
            [{"player_id": 1, "name": "A", "position": "QB", "projected_points": 10.0, "adp": 1.0}]
        ),
    ]
    load_player_data("missing.csv", {"enabled": True, "weights": {}, "sort_order_ascending": True})

    assert mock_create_dummy.called


@patch("draft_buddy.data.player_loader._create_dummy_csv")
@patch("draft_buddy.data.player_loader.pd.read_csv")
def test_load_player_data_recovers_after_dummy_creation(mock_read_csv, mock_create_dummy):
    """Verify loader returns players after fallback CSV creation."""
    del mock_create_dummy
    mock_read_csv.side_effect = [
        FileNotFoundError("missing"),
        pd.DataFrame(
            [{"player_id": 1, "name": "A", "position": "QB", "projected_points": 10.0, "adp": 1.0}]
        ),
    ]
    players = load_player_data("missing.csv", {"enabled": True, "weights": {}, "sort_order_ascending": True})

    assert len(players) == 1


def test_generate_mock_adp_assigns_sequential_ranks_by_descending_projected_points():
    """Verify mock ADP assigns rank 1 to highest weighted score."""
    players = [
        Player(1, "P1", "QB", 300.0),
        Player(2, "P2", "RB", 200.0),
        Player(3, "P3", "WR", 100.0),
    ]
    config = {
        "enabled": True,
        "weights": {"projected_points": 1.0},
        "sort_order_ascending": False,
    }
    ranked = _generate_mock_adp(players, config)
    adp_map = {player.player_id: player.adp for player in ranked}

    assert adp_map == {1: 1, 2: 2, 3: 3}


def test_convert_to_simulation_format_reshapes_weekly_projection_dict_to_pts_list():
    """Verify simulation format conversion builds positional points list."""
    converted = _convert_to_simulation_format({1: {"pos": "QB", 1: 15.0, 2: 15.0}})

    assert converted == {1: {"pos": "QB", "pts": [15.0, 15.0]}}


def test_load_player_data_raises_value_error_when_required_column_is_missing():
    """Verify loader fails fast when required schema columns are absent."""
    with patch("draft_buddy.data.player_loader.pd.read_csv", return_value=pd.DataFrame([{"name": "A"}])):
        with pytest.raises(ValueError):
            load_player_data("fake.csv", {"enabled": True, "weights": {}, "sort_order_ascending": True})


@patch("draft_buddy.data.player_loader.pd.read_csv")
def test_load_player_data_preserves_rookie_games_played_flag(mock_read_csv):
    """Verify games_played_frac='R' remains string flag."""
    mock_read_csv.return_value = pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "A",
                "position": "QB",
                "projected_points": 100.0,
                "adp": 10.0,
                "games_played_frac": "R",
            }
        ]
    )
    players = load_player_data("fake.csv", {"enabled": True, "weights": {}, "sort_order_ascending": True})

    assert players[0].games_played_frac == "R"


@patch("draft_buddy.data.player_loader.pd.read_csv")
def test_load_player_data_uses_recent_team_when_available(mock_read_csv):
    """Verify loader prefers recent_team as team source column."""
    mock_read_csv.return_value = pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "A",
                "position": "QB",
                "projected_points": 100.0,
                "adp": 10.0,
                "games_played_frac": 1.0,
                "recent_team": "KC",
            }
        ]
    )
    players = load_player_data("fake.csv", {"enabled": True, "weights": {}, "sort_order_ascending": True})

    assert players[0].team == "KC"


@patch("draft_buddy.data.player_loader._generate_mock_adp")
@patch("draft_buddy.data.player_loader.pd.read_csv")
def test_load_player_data_generates_mock_adp_when_adp_column_is_missing(mock_read_csv, mock_generate):
    """Verify ADP generation runs when CSV has no ADP column."""
    mock_read_csv.return_value = pd.DataFrame(
        [{"player_id": 1, "name": "A", "position": "QB", "projected_points": 100.0}]
    )
    mock_generate.side_effect = lambda players, config: players
    load_player_data("fake.csv", {"enabled": True, "weights": {}, "sort_order_ascending": True})

    assert mock_generate.called


def test_generate_mock_adp_raises_when_disabled():
    """Verify mock ADP generation fails when disabled by config."""
    with pytest.raises(ValueError):
        _generate_mock_adp([Player(1, "A", "QB", 1.0)], {"enabled": False, "weights": {}, "sort_order_ascending": True})


def test_generate_mock_adp_ignores_unknown_weight_attributes():
    """Verify unknown weight keys do not break ADP generation."""
    players = [Player(1, "A", "QB", 300.0)]
    ranked = _generate_mock_adp(
        players,
        {"enabled": True, "weights": {"unknown_field": 10.0}, "sort_order_ascending": True},
    )

    assert ranked[0].adp == 1


def test_convert_to_simulation_format_uses_position_fallback_key():
    """Verify conversion reads 'position' when 'pos' is missing."""
    converted = _convert_to_simulation_format({1: {"position": "RB", 1: 10.0}})

    assert converted[1]["pos"] == "RB"


def test_convert_to_simulation_format_passes_through_pts_list_when_present():
    """Verify existing pts array is preserved."""
    converted = _convert_to_simulation_format({1: {"pos": "WR", "pts": [1.0, 2.0]}})

    assert converted[1]["pts"] == [1.0, 2.0]


def test_convert_to_simulation_format_supports_numeric_string_week_keys():
    """Verify numeric string week keys are included in order."""
    converted = _convert_to_simulation_format({1: {"pos": "TE", "1": 7.0, "2": 8.0}})

    assert converted[1]["pts"] == [7.0, 8.0]


def test_convert_to_simulation_format_returns_empty_pts_when_no_week_data():
    """Verify missing week keys produce empty points list."""
    converted = _convert_to_simulation_format({1: {"pos": "QB"}})

    assert converted[1]["pts"] == []


def test_get_simulation_dfs_converts_processor_output_to_simulator_format(monkeypatch):
    """Verify get_simulation_dfs returns converted weekly projection format."""
    class _FakeProcessor:
        def __init__(self, **kwargs):
            del kwargs

        def process_draft_data(self, **kwargs):
            del kwargs
            return pd.DataFrame([{"player_id": 1}]), {1: {"position": "QB", 1: 10.0, 2: 11.0}}

    monkeypatch.setattr("draft_buddy.data_pipeline.player_data_utils.FantasyDataProcessor", _FakeProcessor)
    _, weekly = get_simulation_dfs(season=2025, ps_start_year=2020)

    assert weekly == {1: {"pos": "QB", "pts": [10.0, 11.0]}}


@patch("draft_buddy.data.player_loader.os.makedirs")
@patch("builtins.open", new_callable=mock_open)
def test_create_dummy_csv_writes_fallback_rows(mock_file, mock_makedirs):
    """Verify fallback CSV writer emits dummy player rows."""
    del mock_makedirs
    _create_dummy_csv("/tmp/fallback.csv")

    assert mock_file().write.called
