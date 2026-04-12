import io
import os

import csv
import numpy as np
import pandas as pd
from flask import Flask, jsonify, make_response, request, send_from_directory, session

from draft_buddy.config import Config
from draft_buddy.utils.data_utils import load_player_data
from draft_buddy.logic.draft_service import DraftService
from draft_buddy.logic.season_simulation_service import SeasonSimulationService
from draft_buddy.logic.draft_presentation_service import DraftPresentationService
from draft_buddy.logic.draft_rest_simulation import simulate_scheduled_picks_remaining
from draft_buddy.draft_env.fantasy_draft_env import FantasyFootballDraftEnv
from draft_buddy.utils.season_simulation_fast import simulate_season_fast

app = Flask(__name__, static_folder="../frontend")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")

config = Config()
draft_service = DraftService(config, env_factory=FantasyFootballDraftEnv)
season_simulation_service = SeasonSimulationService(config)

all_players = load_player_data(config.paths.PLAYER_DATA_CSV, config.draft.MOCK_ADP_CONFIG)
player_map = {p.player_id: p for p in all_players}


def _get_session_id() -> str:
    """
    Return the session ID for the current request.

    Uses the Flask session to persist the draft session ID across requests.
    Creates a new UUID if none exists.

    Returns
    -------
    str
        The draft session ID for the current request.
    """
    if not session.get("draft_session_id"):
        import uuid
        session["draft_session_id"] = str(uuid.uuid4())
    return session["draft_session_id"]


def _get_draft_env():
    """
    Return the draft environment for the current session.

    Fetches or creates a FantasyFootballDraftEnv instance keyed by the
    session ID.

    Returns
    -------
    FantasyFootballDraftEnv
        The draft environment for the current session.
    """
    return draft_service.get_or_create_draft(_get_session_id())


def _get_draft_state():
    """
    Build the draft state dictionary for the current session.

    Aggregates roster data, pick order, team points, and configuration
    into a single dictionary suitable for API responses.

    Returns
    -------
    dict
        Draft state.
    """
    draft_env = _get_draft_env()
    return DraftPresentationService.get_ui_state(draft_env)

# API route for backend status
@app.route('/api/hello')
def hello_world():
    return {'message': 'Hello from DRAFT BUDDY backend!'}

@app.route("/api/draft/new", methods=["POST"])
def create_new_draft():
    """Archives the current draft and creates a fresh one for this session."""
    draft_service.create_new_draft(_get_session_id())
    return jsonify(_get_draft_state())

@app.route("/api/draft/state")
def draft_state():
    """Returns the complete current state of the draft."""
    return jsonify(_get_draft_state())

@app.route("/api/draft/pick", methods=["POST"])
def draft_pick():
    draft_env = _get_draft_env()
    data = request.get_json()
    player_id = data.get("player_id")
    if not player_id:
        return jsonify({"error": "Player ID is required"}), 400

    try:
        draft_env.draft_player(player_id)
        draft_env.save_state(config.paths.DRAFT_STATE_FILE)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(_get_draft_state())

@app.route("/api/draft/undo", methods=["POST"])
def undo_pick():
    draft_env = _get_draft_env()
    try:
        draft_env.undo_last_pick()
        draft_env.save_state(config.paths.DRAFT_STATE_FILE)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(_get_draft_state())

@app.route("/api/draft/override_team", methods=["POST"])
def override_team():
    draft_env = _get_draft_env()
    data = request.get_json()
    team_id = data.get("team_id")
    if team_id is None:
        return jsonify({"error": "Team ID is required"}), 400
    try:
        draft_env.set_current_team_picking(team_id)
        draft_env.save_state(config.paths.DRAFT_STATE_FILE)
    except ValueError as e:
        return jsonify({"warning": str(e), "info": "Override allowed for manual post-draft picks"}), 200

    return jsonify(_get_draft_state())

@app.route("/api/draft/simulate_pick", methods=["POST"])
def simulate_pick():
    draft_env = _get_draft_env()
    try:
        draft_env.simulate_single_pick()
        draft_env.save_state(config.paths.DRAFT_STATE_FILE)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(_get_draft_state())


@app.route("/api/draft/simulate_rest", methods=["POST"])
def simulate_rest():
    """Simulates all remaining scheduled picks in one request; saves state once."""
    draft_env = _get_draft_env()
    try:
        simulate_scheduled_picks_remaining(draft_env)
        draft_env.save_state(config.paths.DRAFT_STATE_FILE)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(_get_draft_state())

@app.route("/api/draft/ai_suggestion")
def ai_suggestion():
    draft_env = _get_draft_env()
    suggestion = draft_env.get_ai_suggestion()
    return jsonify(suggestion)

@app.route("/api/draft/ai_suggestions_all")
def ai_suggestions_all():
    draft_env = _get_draft_env()
    suggestions = draft_env.get_ai_suggestions_all()
    return jsonify(suggestions)

@app.route("/api/draft/ai_suggestion_for_team")
def ai_suggestion_for_team():
    draft_env = _get_draft_env()
    try:
        team_id_str = request.args.get('team_id')
        if team_id_str is None:
            return jsonify({'error': 'team_id is required'}), 400
        team_id = int(team_id_str)
    except ValueError:
        return jsonify({'error': 'team_id must be an integer'}), 400

    # Optional: ignore list for blind prediction
    ignore_raw = request.args.get('ignore')  # comma-separated ids
    ignore_ids = None
    if ignore_raw:
        try:
            ignore_ids = [int(x) for x in ignore_raw.split(',') if x.strip()]
        except ValueError:
            return jsonify({'error': 'ignore must be a comma-separated list of integers'}), 400

    suggestion = draft_env.get_ai_suggestion_for_team(team_id, ignore_player_ids=ignore_ids)
    return jsonify(suggestion)

@app.route("/api/draft/summary")
def draft_summary():
    draft_env = _get_draft_env()
    summary = draft_env.get_draft_summary()
    return jsonify(summary)

@app.route("/api/draft/export_csv")
def export_csv():
    draft_env = _get_draft_env()

    si = io.StringIO()
    cw = csv.writer(si)

    # Write header
    header = [
        'Pick Number', 'Team ID', 'Player ID', 'Name', 'Position', 'Team', 'Bye Week',
        'Projected Points', 'ADP', 'Games Played %'
    ]
    cw.writerow(header)

    # Write draft history
    for pick in draft_env._draft_history:
        player = draft_env.player_map.get(pick['player_id'])
        if player:
            cw.writerow([
                pick['previous_pick_number'],
                pick['team_id'],
                player.player_id,
                player.name,
                player.position,
                getattr(player, 'team', 'N/A'),
                player.bye_week if player.bye_week and not np.isnan(player.bye_week) else 'N/A',
                player.projected_points,
                player.adp if np.isfinite(player.adp) else 'N/A',
                f"{player.games_played_frac:.2f}" if isinstance(player.games_played_frac, (int, float)) else player.games_played_frac
            ])

    # Prepare response
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=draft_results.csv"
    output.headers["Content-type"] = "text/csv"
    return output

@app.route("/api/simulate_season", methods=["POST"])
def simulate_season():
    """Runs a full season simulation using current in-memory draft state."""
    draft_env = _get_draft_env()
    try:
        results = season_simulation_service.simulate_season(draft_env)
        return jsonify(results)
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Season simulation failed: {e}'}), 500

@app.route("/api/players")
def get_players():
    position_filter = request.args.get('position')
    search_query = request.args.get('search')
    sort_by = request.args.get('sort_by', 'vorp')  # one of: player_id, name, position, projected_points, vorp, games_played_frac, adp, bye_week
    sort_dir = request.args.get('sort_dir', 'desc').lower()  # 'asc' or 'desc'
    reverse = (sort_dir == 'desc')

    draft_env = _get_draft_env()
    if draft_env is not None:
        source_players = [player_map[p_id] for p_id in draft_env.available_players_ids]
    else:
        source_players = all_players

    filtered_players = source_players

    # Apply position filtering
    if position_filter:
        positions_to_match = [pos.strip().upper() for pos in position_filter.split(',')]
        filtered_players = [p for p in filtered_players if p.position in positions_to_match]

    # Apply search query filtering
    if search_query:
        search_query_lower = search_query.lower()
        filtered_players = [p for p in filtered_players if search_query_lower in p.name.lower()]

    # Get positional baselines for VORP calculation
    baselines = draft_env.get_positional_baselines()

    # Precompute VORP map to sort efficiently
    player_vorp_map = {p.player_id: (p.projected_points - baselines.get(p.position, 0)) for p in filtered_players}

    def sort_key(p):
        if sort_by == 'vorp':
            val = player_vorp_map.get(p.player_id, 0.0)
        elif sort_by == 'adp':
            # Lower ADP is better; treat inf/missing as very large so they sink on asc
            val = p.adp if np.isfinite(p.adp) else float('inf')
        elif sort_by == 'projected_points':
            val = p.projected_points
        elif sort_by == 'games_played_frac':
            val = p.games_played_frac
            if val == 'R':
                return -1.0  # Sort rookies consistently
            return val if np.isfinite(val) else -1.0
        elif sort_by == 'position':
            val = p.position
        elif sort_by == 'name':
            val = p.name.lower()
        elif sort_by == 'bye_week':
            val = p.bye_week if (p.bye_week is not None and not np.isnan(p.bye_week)) else 99
        elif sort_by == 'team':
            val = p.team.lower() if p.team else 'N/A'
        elif sort_by == 'player_id':
            val = p.player_id
        else:
            # Default to VORP if unknown key
            val = player_vorp_map.get(p.player_id, 0.0)
        # Secondary key for determinism
        return (val, p.player_id)

    # Sort in place
    filtered_players.sort(key=sort_key, reverse=reverse)

    # Convert Player objects to a list of dictionaries for JSON serialization
    players_data = []
    for p in filtered_players:
        player_vorp = player_vorp_map.get(p.player_id, 0.0)
        player_dict = {
            'player_id': p.player_id,
            'name': p.name,
            'position': p.position,
            'projected_points': p.projected_points,
            'vorp': player_vorp,
            'games_played_frac': p.games_played_frac,
            'adp': None if np.isinf(p.adp) else p.adp,
            'bye_week': p.bye_week if p.bye_week and not np.isnan(p.bye_week) else 'N/A',
            'team': getattr(p, 'team', None)
        }
        players_data.append(player_dict)

    return jsonify(players_data)

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host="0.0.0.0", port=port)
