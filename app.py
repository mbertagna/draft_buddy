import os
import io
import csv
from flask import Flask, send_from_directory, jsonify, request, make_response
from data_utils import load_player_data
from config import Config
import numpy as np
from fantasy_draft_env import FantasyFootballDraftEnv

# Use 'frontend' as the static folder, and serve files from it
app = Flask(__name__, static_folder='frontend')

# Load configuration
config = Config()

# Load player data once when the app starts
all_players = load_player_data(config.PLAYER_DATA_CSV, config.MOCK_ADP_CONFIG)
player_map = {p.player_id: p for p in all_players}

# Global variable to hold the draft environment
draft_env = None

def get_draft_state():
    """Helper function to create the draft state dictionary."""
    if not draft_env:
        return None
    
    # Determine the current team on the clock, considering overrides
    team_on_clock = draft_env._overridden_team_id if draft_env._overridden_team_id is not None else \
                    (draft_env.draft_order[draft_env.current_pick_idx] if draft_env.current_pick_idx < len(draft_env.draft_order) else None)

    return {
        'draft_order': draft_env.draft_order,
        'current_pick_number': draft_env.current_pick_number,
        'current_team_picking': team_on_clock,
        'team_rosters': {team_id: [p.to_dict() for p in roster_data['PLAYERS']] for team_id, roster_data in draft_env.teams_rosters.items()},
        'roster_counts': {team_id: {pos: roster_data[pos] for pos in ['QB', 'RB', 'WR', 'TE', 'FLEX']} for team_id, roster_data in draft_env.teams_rosters.items()},
        'team_projected_points': {team_id: sum(p.projected_points for p in roster_data['PLAYERS']) for team_id, roster_data in draft_env.teams_rosters.items()},
        'manual_draft_teams': list(draft_env.manual_draft_teams)
    }

# API route for backend status
@app.route('/api/hello')
def hello_world():
    return {'message': 'Hello from DRAFT BUDDY backend!'}

# API route for player data
@app.route('/api/players')
def get_players():
    position_filter = request.args.get('position')
    search_query = request.args.get('search')

    # If a draft is in progress, use the available players from the environment
    if draft_env:
        source_players = [player_map[p_id] for p_id in draft_env.available_players_ids]
    else:
        source_players = all_players

    filtered_players = source_players

    # Apply position filtering
    if position_filter:
        positions_to_match = [pos.strip().upper() for pos in position_filter.split(',')]
        filtered_players = [
            p for p in filtered_players if p.position in positions_to_match
        ]

    # Apply search query filtering
    if search_query:
        search_query_lower = search_query.lower()
        filtered_players = [
            p for p in filtered_players if search_query_lower in p.name.lower()
        ]

    # Convert Player objects to a list of dictionaries for JSON serialization
    players_data = []
    for p in filtered_players:
        player_dict = {
            'player_id': p.player_id,
            'name': p.name,
            'position': p.position,
            'projected_points': p.projected_points,
            'adp': None if np.isinf(p.adp) else p.adp
        }
        players_data.append(player_dict)

    # Sort players by ADP (ascending), with None values at the end
    players_data.sort(key=lambda x: (x['adp'] is None, x['adp']))

    return jsonify(players_data)

# API route for creating a new draft
@app.route('/api/draft/new', methods=['POST'])
def create_new_draft():
    global draft_env
    # Initialize a new environment
    draft_env = FantasyFootballDraftEnv(config)
    # Reset the environment to get the initial state
    draft_env.reset()

    return jsonify(get_draft_state())

@app.route('/api/draft/pick', methods=['POST'])
def draft_pick():
    global draft_env
    if not draft_env:
        return jsonify({'error': 'Draft has not been started'}), 400

    data = request.get_json()
    player_id = data.get('player_id')
    if not player_id:
        return jsonify({'error': 'Player ID is required'}), 400

    try:
        draft_env.draft_player(player_id)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    return jsonify(get_draft_state())

@app.route('/api/draft/undo', methods=['POST'])
def undo_pick():
    global draft_env
    if not draft_env:
        return jsonify({'error': 'Draft has not been started'}), 400
    try:
        draft_env.undo_last_pick()
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    
    return jsonify(get_draft_state())

@app.route('/api/draft/override_team', methods=['POST'])
def override_team():
    global draft_env
    if not draft_env:
        return jsonify({'error': 'Draft has not been started'}), 400
    data = request.get_json()
    team_id = data.get('team_id')
    if team_id is None:
        return jsonify({'error': 'Team ID is required'}), 400
    try:
        draft_env.set_current_team_picking(team_id)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    
    return jsonify(get_draft_state())

@app.route('/api/draft/simulate_pick', methods=['POST'])
def simulate_pick():
    global draft_env
    if not draft_env:
        return jsonify({'error': 'Draft has not been started'}), 400
    try:
        draft_env.simulate_single_pick()
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    
    return jsonify(get_draft_state())

@app.route('/api/draft/ai_suggestion')
def ai_suggestion():
    global draft_env
    if not draft_env:
        return jsonify({'error': 'Draft has not been started'}), 400

    suggestion = draft_env.get_ai_suggestion()
    return jsonify({'suggestion': suggestion})

@app.route('/api/draft/summary')
def draft_summary():
    global draft_env
    if not draft_env:
        return jsonify({'error': 'Draft has not been started'}), 400

    summary = draft_env.get_draft_summary()
    return jsonify(summary)

@app.route('/api/draft/export_csv')
def export_csv():
    global draft_env
    if not draft_env:
        return jsonify({'error': 'Draft has not been started'}), 400

    # Create a string buffer to hold CSV data
    si = io.StringIO()
    cw = csv.writer(si)

    # Write header
    cw.writerow(['Pick Number', 'Team ID', 'Player ID', 'Player Name', 'Position', 'Projected Points', 'ADP'])

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
                player.projected_points,
                player.adp if np.isfinite(player.adp) else 'N/A'
            ])

    # Prepare response
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=draft_results.csv"
    output.headers["Content-type"] = "text/csv"
    return output

# Serve frontend application
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
