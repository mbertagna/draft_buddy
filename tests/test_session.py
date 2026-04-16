"""Tests for web session orchestration and persistence."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from draft_buddy.domain.entities import Player
from draft_buddy.web.session import DraftSession, DraftSessionManager


def _mock_players():
    """Return deterministic player list for session tests."""
    return [
        Player(101, "QB One", "QB", 120.0, adp=1.0, bye_week=7, team="KC"),
        Player(102, "RB One", "RB", 110.0, adp=2.0, bye_week=8, team="BUF"),
        Player(103, "WR One", "WR", 100.0, adp=3.0, bye_week=9, team="MIA"),
        Player(104, "TE One", "TE", 90.0, adp=4.0, bye_week=10, team="NE"),
    ]


@patch("draft_buddy.web.session.load_player_data", return_value=_mock_players())
def test_draft_session_manager_get_or_create_returns_same_instance_for_same_id(
    mock_load_player_data, mock_config
):
    """Verify repeated get_or_create returns identical object reference."""
    del mock_load_player_data
    manager = DraftSessionManager(mock_config, inference_provider=None)
    session_a = manager.get_or_create("test_id")
    session_b = manager.get_or_create("test_id")

    assert session_a is session_b


@patch("draft_buddy.web.session.load_player_data", return_value=_mock_players())
def test_draft_session_save_and_load_reconstructs_rostered_players(
    mock_load_player_data, mock_config, tmp_path
):
    """Verify save/load round-trip restores rostered players."""
    del mock_load_player_data
    state_file = Path(tmp_path) / "draft_state.json"
    session_one = DraftSession(mock_config, inference_provider=None)
    session_one.draft_player(101)
    session_one.save_state(str(state_file))

    session_two = DraftSession(mock_config, inference_provider=None)
    session_two.load_state(str(state_file))
    rostered_ids = [player.player_id for player in session_two.teams_rosters[1]["PLAYERS"]]

    assert rostered_ids == [101]


@patch("draft_buddy.web.session.load_player_data", return_value=_mock_players())
def test_draft_session_save_and_load_reconstructs_draft_history(
    mock_load_player_data, mock_config, tmp_path
):
    """Verify save/load round-trip restores draft history entries."""
    del mock_load_player_data
    state_file = Path(tmp_path) / "draft_state.json"
    session_one = DraftSession(mock_config, inference_provider=None)
    session_one.draft_player(101)
    session_one.save_state(str(state_file))

    session_two = DraftSession(mock_config, inference_provider=None)
    session_two.load_state(str(state_file))

    assert session_two._draft_history == session_one._draft_history


@patch("draft_buddy.web.session.load_player_data", return_value=_mock_players())
def test_get_ai_suggestion_returns_error_when_inference_provider_missing(
    mock_load_player_data, mock_config
):
    """Verify missing inference provider returns expected error payload."""
    del mock_load_player_data
    session = DraftSession(mock_config, inference_provider=None)

    assert session.get_ai_suggestion() == {"error": "AI model not loaded."}


@patch("draft_buddy.web.session.load_player_data", return_value=_mock_players())
def test_get_ai_suggestion_for_team_restores_ignored_players_when_prediction_raises(
    mock_load_player_data, mock_config
):
    """Verify ignored IDs are restored even when prediction call raises."""
    del mock_load_player_data
    provider = MagicMock()
    provider.predict_action_probabilities.side_effect = RuntimeError("prediction failed")
    session = DraftSession(mock_config, inference_provider=provider)
    before_ids = set(session.available_players_ids)
    ignored_id = 101
    session.get_ai_suggestion_for_team(team_id=1, ignore_player_ids=[ignored_id])

    assert session.available_players_ids == before_ids


@patch("draft_buddy.web.session.load_player_data", return_value=_mock_players())
def test_session_property_team_manager_mapping_exposes_config_mapping(mock_load_player_data, mock_config):
    """Verify team manager mapping property delegates to config."""
    del mock_load_player_data
    session = DraftSession(mock_config, inference_provider=None)

    assert session.team_manager_mapping == mock_config.draft.TEAM_MANAGER_MAPPING


@patch("draft_buddy.web.session.load_player_data", return_value=_mock_players())
def test_session_property_roster_structure_exposes_config_structure(mock_load_player_data, mock_config):
    """Verify roster structure property delegates to config."""
    del mock_load_player_data
    session = DraftSession(mock_config, inference_provider=None)

    assert session.roster_structure == mock_config.draft.ROSTER_STRUCTURE


@patch("draft_buddy.web.session.load_player_data", return_value=_mock_players())
def test_session_property_bench_maxes_exposes_config_bench_maxes(mock_load_player_data, mock_config):
    """Verify bench maxes property delegates to config."""
    del mock_load_player_data
    session = DraftSession(mock_config, inference_provider=None)

    assert session.bench_maxes == mock_config.draft.BENCH_MAXES


@patch("draft_buddy.web.session.load_player_data", return_value=_mock_players())
def test_session_property_agent_team_id_matches_state(mock_load_player_data, mock_config):
    """Verify agent team id property reads from draft state."""
    del mock_load_player_data
    session = DraftSession(mock_config, inference_provider=None)

    assert session.agent_team_id == mock_config.draft.AGENT_START_POSITION


@patch("draft_buddy.web.session.load_player_data", return_value=_mock_players())
def test_session_get_ui_state_returns_dictionary(mock_load_player_data, mock_config):
    """Verify UI state adapter returns dict payload."""
    del mock_load_player_data
    session = DraftSession(mock_config, inference_provider=None)

    assert isinstance(session.get_ui_state(), dict)


@patch("draft_buddy.web.session.load_player_data", return_value=_mock_players())
def test_session_load_state_returns_early_when_file_missing(mock_load_player_data, mock_config, tmp_path):
    """Verify load_state is a no-op for missing files."""
    del mock_load_player_data
    session = DraftSession(mock_config, inference_provider=None)
    missing = Path(tmp_path) / "missing_state.json"
    session.load_state(str(missing))

    assert session.current_pick_number == 1


@patch("draft_buddy.web.session.load_player_data", return_value=_mock_players())
def test_draft_player_raises_when_player_unavailable(mock_load_player_data, mock_config):
    """Verify draft_player rejects unavailable player ids."""
    del mock_load_player_data
    session = DraftSession(mock_config, inference_provider=None)
    raised = False
    try:
        session.draft_player(99999)
    except ValueError:
        raised = True

    assert raised is True


@patch("draft_buddy.web.session.load_player_data", return_value=_mock_players())
def test_draft_player_raises_when_manual_rules_reject_position(mock_load_player_data, mock_config):
    """Verify draft_player raises when manual rules deny the pick."""
    del mock_load_player_data
    session = DraftSession(mock_config, inference_provider=None)
    with patch.object(session._rules, "can_draft_manual", return_value=False):
        raised = False
        try:
            session.draft_player(101)
        except ValueError:
            raised = True

    assert raised is True


@patch("draft_buddy.web.session.load_player_data", return_value=_mock_players())
def test_get_ai_suggestion_returns_draft_over_error_when_pick_index_exhausted(
    mock_load_player_data, mock_config
):
    """Verify get_ai_suggestion reports draft-over condition."""
    del mock_load_player_data
    session = DraftSession(mock_config, inference_provider=None)
    session._state.set_current_pick_idx(len(session.draft_order))

    assert session.get_ai_suggestion() == {"error": "Draft is over."}


@patch("draft_buddy.web.session.load_player_data", return_value=_mock_players())
def test_get_ai_suggestion_for_team_rejects_invalid_team_id(mock_load_player_data, mock_config):
    """Verify invalid team ids return error payload."""
    del mock_load_player_data
    session = DraftSession(mock_config, inference_provider=MagicMock())

    assert "Invalid team id" in session.get_ai_suggestion_for_team(team_id=999)["error"]


def test_draft_session_manager_get_or_create_resets_and_saves_when_loaded_session_has_empty_order(
    mock_config,
):
    """Verify manager reinitializes empty-order sessions."""
    fake_session = MagicMock()
    fake_session.draft_order = []
    with patch("draft_buddy.web.session.DraftSession", return_value=fake_session):
        manager = DraftSessionManager(mock_config, inference_provider=None)
        manager.get_or_create("session-x")

    assert fake_session.reset.called


@patch("draft_buddy.web.session.load_player_data", return_value=_mock_players())
def test_draft_session_save_state_uses_atomic_temp_file_replace(mock_load_player_data, mock_config, tmp_path):
    """Verify session persistence writes through a temporary file before replacing."""
    del mock_load_player_data
    session = DraftSession(mock_config, inference_provider=None)
    target = Path(tmp_path) / "draft_state.json"

    with patch("draft_buddy.web.session.os.replace") as mock_replace:
        session.save_state(str(target))

    assert mock_replace.call_args.args[0].endswith(".tmp")


@patch("draft_buddy.web.session.load_player_data", return_value=_mock_players())
def test_get_positional_baselines_returns_zero_when_position_has_no_available_players(
    mock_load_player_data, mock_config
):
    """Verify missing available players yield a zero baseline for that position."""
    del mock_load_player_data
    session = DraftSession(mock_config, inference_provider=None)
    session._state.replace_available_player_ids({102, 103, 104})

    assert session.get_positional_baselines()["QB"] == 0.0


@patch("draft_buddy.web.session.load_player_data", return_value=_mock_players())
def test_undo_last_pick_raises_when_history_is_empty(mock_load_player_data, mock_config):
    """Verify undo rejects empty history."""
    del mock_load_player_data
    session = DraftSession(mock_config, inference_provider=None)

    with pytest.raises(ValueError, match="No picks to undo"):
        session.undo_last_pick()


@patch("draft_buddy.web.session.load_player_data", return_value=_mock_players())
def test_undo_last_pick_restores_previous_pick_number(mock_load_player_data, mock_config):
    """Verify undo rewinds the global pick counter."""
    del mock_load_player_data
    session = DraftSession(mock_config, inference_provider=None)
    original_pick = session.current_pick_number
    session.draft_player(101)

    session.undo_last_pick()

    assert session.current_pick_number == original_pick


@patch("draft_buddy.web.session.load_player_data", return_value=_mock_players())
def test_simulate_single_pick_rejects_manual_teams(mock_load_player_data, mock_config):
    """Verify manual teams cannot be auto-simulated."""
    del mock_load_player_data
    mock_config.draft.MANUAL_DRAFT_TEAMS = [1]
    mock_config.draft.AGENT_START_POSITION = 2
    session = DraftSession(mock_config, inference_provider=None)
    session.set_current_team_picking(1)

    with pytest.raises(ValueError, match="manual team's turn"):
        session.simulate_single_pick()


@patch("draft_buddy.web.session.load_player_data", return_value=_mock_players())
def test_try_select_player_for_team_returns_false_when_position_pool_is_empty(
    mock_load_player_data, mock_config
):
    """Verify position selection fails when no players of that position remain."""
    del mock_load_player_data
    session = DraftSession(mock_config, inference_provider=None)

    picked, player = session._try_select_player_for_team(team_id=1, position_choice="K", available_ids={101, 102})

    assert picked is False and player is None


@patch("draft_buddy.web.session.load_player_data", return_value=_mock_players())
def test_try_select_player_for_team_returns_false_when_rules_reject_position(
    mock_load_player_data, mock_config
):
    """Verify position selection fails when roster rules reject the pick."""
    del mock_load_player_data
    session = DraftSession(mock_config, inference_provider=None)

    with patch.object(session, "_can_draft_position", return_value=False):
        picked, player = session._try_select_player_for_team(team_id=1, position_choice="QB", available_ids={101})

    assert picked is False and player is None


@patch("draft_buddy.web.session.load_player_data", return_value=_mock_players())
def test_create_bot_strategy_falls_back_when_agent_model_provider_returns_none(
    mock_load_player_data, mock_config
):
    """Verify AGENT_MODEL configs fall back to the regular bot factory when provider declines."""
    del mock_load_player_data
    provider = MagicMock()
    provider.create_bot.return_value = None
    session = DraftSession(mock_config, inference_provider=provider)
    strategy_config = {"logic": "AGENT_MODEL"}

    with patch("draft_buddy.web.session.create_bot_gm", return_value="fallback-bot") as mock_create_bot:
        strategy = session._create_bot_strategy(team_id=2, strategy_config=strategy_config)

    assert strategy == "fallback-bot" and mock_create_bot.called


@patch("draft_buddy.web.session.load_player_data", return_value=_mock_players())
def test_get_ai_suggestions_all_collects_results_for_each_team(mock_load_player_data, mock_config):
    """Verify bulk AI suggestions aggregate every team response."""
    del mock_load_player_data
    session = DraftSession(mock_config, inference_provider=MagicMock())

    with patch.object(session, "get_ai_suggestion_for_team", side_effect=lambda team_id: {"team": team_id}):
        suggestions = session.get_ai_suggestions_all()

    assert suggestions[mock_config.draft.NUM_TEAMS] == {"team": mock_config.draft.NUM_TEAMS}


@patch("draft_buddy.web.session.load_player_data", return_value=_mock_players())
def test_generate_snake_draft_order_truncates_rounds_when_player_pool_is_small(
    mock_load_player_data, mock_config
):
    """Verify snake order truncates to the available number of full rounds."""
    del mock_load_player_data
    mock_config.draft.NUM_TEAMS = 3
    session = DraftSession(mock_config, inference_provider=None)
    session.all_players_data = session.all_players_data[:4]

    assert session._generate_snake_draft_order() == [1, 2, 3]
