import os
import io
import csv
import datetime
from flask import Flask, send_from_directory, jsonify, request, make_response
from data_utils import load_player_data
from config import Config
import numpy as np
from fantasy_draft_env import FantasyFootballDraftEnv
from utils.season_simulation_fast import simulate_season_fast
import pandas as pd

# Use 'frontend' as the static folder, and serve files from it
app = Flask(__name__, static_folder='frontend')

# Load configuration
config = Config()

# Load player data once when the app starts
all_players = load_player_data(config.PLAYER_DATA_CSV, config.MOCK_ADP_CONFIG)
player_map = {p.player_id: p for p in all_players}

# Initialize the draft environment when the application starts
draft_env = FantasyFootballDraftEnv(config)
# Try to load an existing state
draft_env.load_state(config.DRAFT_STATE_FILE)

# If the draft is empty (e.g., new start or corrupted file), create and save a new one
if not draft_env.draft_order:
    print("No valid draft state found or draft is empty. Creating a new draft.")
    draft_env.reset()
    draft_env.save_state(config.DRAFT_STATE_FILE)

def get_draft_state():
    """Helper function to create the draft state dictionary."""
    if not draft_env:
        return None

    # Determine the current team on the clock, considering overrides
    team_on_clock = draft_env._overridden_team_id if draft_env._overridden_team_id is not None else \
                    (draft_env.draft_order[draft_env.current_pick_idx] if draft_env.current_pick_idx < len(draft_env.draft_order) else None)

    # Get structured rosters and points for all teams
    structured_rosters = {}
    team_points_summary = {}
    team_is_full = {}

    for team_id, roster_data in draft_env.teams_rosters.items():
        starters, bench, _ = draft_env._categorize_roster_by_slots(
            roster_data['PLAYERS'],
            draft_env.config.ROSTER_STRUCTURE,
            draft_env.config.BENCH_MAXES
        )

        # Calculate starter points
        starter_points = sum(p.projected_points for pos_list in starters.values() for p in pos_list)
        
        # Calculate bench points
        bench_points = sum(p.projected_points for p in bench)

        team_points_summary[team_id] = {
            'starters_total': starter_points,
            'bench_total': bench_points
        }

        structured_rosters[team_id] = {
            'starters': {pos: [p.to_dict() for p in players] for pos, players in starters.items()},
            'bench': [p.to_dict() for p in bench]
        }

        # Determine if team is full (starters + bench)
        team_is_full[team_id] = (len(roster_data['PLAYERS']) >= draft_env.total_roster_size_per_team)

    return {
        'draft_order': draft_env.draft_order,
        'current_pick_number': draft_env.current_pick_number,
        'current_team_picking': team_on_clock,
        'team_rosters': structured_rosters,
        'roster_counts': {team_id: {pos: roster_data[pos] for pos in ['QB', 'RB', 'WR', 'TE', 'FLEX']} for team_id, roster_data in draft_env.teams_rosters.items()},
        'team_projected_points': {team_id: sum(p.projected_points for p in roster_data['PLAYERS']) for team_id, roster_data in draft_env.teams_rosters.items()},
        'manual_draft_teams': list(draft_env.manual_draft_teams),
        'roster_structure': draft_env.config.ROSTER_STRUCTURE,
        'team_is_full': team_is_full,
        'team_points_summary': team_points_summary,
        'num_teams': draft_env.config.NUM_TEAMS,
        'team_bye_weeks': { 
            team_id: {
                week: {pos: int(count) for pos, count in zip(*np.unique([p.position for p in roster_data['PLAYERS'] if p.bye_week == week], return_counts=True))}
                for week in set(p.bye_week for p in roster_data['PLAYERS'] if p.bye_week and not np.isnan(p.bye_week))
            }
            for team_id, roster_data in draft_env.teams_rosters.items()
        }
    }

# API route for backend status
@app.route('/api/hello')
def hello_world():
    return {'message': 'Hello from DRAFT BUDDY backend!'}

# API route for creating a new draft
@app.route('/api/draft/new', methods=['POST'])
def create_new_draft():
    """Saves the current draft state and resets the environment."""
    global draft_env

    # Archive the current state if the draft has started
    if draft_env and draft_env.current_pick_number > 1:
        saved_states_dir = 'saved_states'
        if not os.path.exists(saved_states_dir):
            os.makedirs(saved_states_dir)
        
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        archive_filename = os.path.join(saved_states_dir, f'draft_state_{timestamp}.json')
        draft_env.save_state(archive_filename)
        print(f"Saved current draft state to {archive_filename}")

    draft_env.reset()
    draft_env.save_state(config.DRAFT_STATE_FILE)
    return jsonify(get_draft_state())

# API route for getting the current draft state
@app.route('/api/draft/state')
def draft_state():
    """Returns the complete current state of the draft."""
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
        draft_env.save_state(config.DRAFT_STATE_FILE)
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
        draft_env.save_state(config.DRAFT_STATE_FILE)
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
        draft_env.save_state(config.DRAFT_STATE_FILE)
    except ValueError as e:
        # Relax override restriction for UI: treat as success if draft is over
        # so user can proceed to manually fill rosters by setting team then picking.
        # Still return error message for visibility but 200 to allow UI flow.
        return jsonify({'warning': str(e), 'info': 'Override allowed for manual post-draft picks'}), 200
    
    return jsonify(get_draft_state())

@app.route('/api/draft/simulate_pick', methods=['POST'])
def simulate_pick():
    global draft_env
    if not draft_env:
        return jsonify({'error': 'Draft has not been started'}), 400
    try:
        draft_env.simulate_single_pick()
        draft_env.save_state(config.DRAFT_STATE_FILE)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    
    return jsonify(get_draft_state())

@app.route('/api/draft/ai_suggestion')
def ai_suggestion():
    global draft_env
    if not draft_env:
        return jsonify({'error': 'Draft has not been started'}), 400

    suggestion = draft_env.get_ai_suggestion()
    return jsonify(suggestion)

@app.route('/api/draft/ai_suggestions_all')
def ai_suggestions_all():
    global draft_env
    if not draft_env:
        return jsonify({'error': 'Draft has not been started'}), 400
    suggestions = draft_env.get_ai_suggestions_all()
    return jsonify(suggestions)

@app.route('/api/draft/ai_suggestion_for_team')
def ai_suggestion_for_team():
    global draft_env
    if not draft_env:
        return jsonify({'error': 'Draft has not been started'}), 400
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

@app.route('/api/simulate_season', methods=['POST'])
def simulate_season():
    """Runs a full season simulation using current in-memory draft state and returns results.

    This does not modify any saved state on disk.
    """
    global draft_env
    if not draft_env:
        return jsonify({'error': 'Draft has not been started'}), 400

    # Build rosters: manager name -> list of player_ids
    rosters = {}
    for team_id, roster_data in draft_env.teams_rosters.items():
        manager_name = config.TEAM_MANAGER_MAPPING.get(team_id)
        if not manager_name:
            continue
        rosters[manager_name] = [p.player_id for p in roster_data['PLAYERS']]

    # Load matchups: prefer size-specific file when available
    default_matchups_filename = 'red_league_matchups_2025.csv'
    size_specific_filename = f"red_league_matchups_2025_{config.NUM_TEAMS}_team.csv"
    candidates = [
        os.path.join(config.DATA_DIR, size_specific_filename),
        os.path.join(config.DATA_DIR, default_matchups_filename),
    ]
    matchups_path = None
    for p in candidates:
        if os.path.exists(p):
            matchups_path = p
            break
    if matchups_path is None:
        matchups_path = os.path.join(config.DATA_DIR, default_matchups_filename)
    try:
        matchups_df = pd.read_csv(matchups_path)
    except FileNotFoundError:
        return jsonify({'error': f'Matchups file not found at {matchups_path}'}), 400

    # Week-to-week points dict. Prefer env's prepared dict if available
    wtw_dict = getattr(draft_env, 'wtw_dict', None)
    if wtw_dict is None:
        wtw_dict = {p.player_id: {'pts': [p.projected_points] * 18, 'pos': p.position} for p in draft_env.all_players_data}

    try:
        num_playoff_teams = int(getattr(config, 'REGULAR_SEASON_REWARD', {}).get('NUM_PLAYOFF_TEAMS', 6))
        regular_results_df, regular_records, playoff_results_df, playoffs_tree, winner = simulate_season_fast(
            wtw_dict, matchups_df, rosters, 2025, '', False, num_playoff_teams
        )
    except Exception as e:
        return jsonify({'error': f'Season simulation failed: {e}'}), 500

    # Convert playoff results to structured JSON for nicer UI
    playoff_results = []
    try:
        for _, row in playoff_results_df.iterrows():
            playoff_results.append({
                'week': int(row['Week']),
                'matchup': int(row['Matchup']),
                'away_manager': None if pd.isna(row['Away Manager(s)']) else str(row['Away Manager(s)']),
                'away_score': None if pd.isna(row['Away Score']) else float(row['Away Score']),
                'home_manager': None if pd.isna(row['Home Manager(s)']) else str(row['Home Manager(s)']),
                'home_score': None if pd.isna(row['Home Score']) else float(row['Home Score'])
            })
    except Exception:
        playoff_results = []

    return jsonify({
        'regular_season_records': regular_records,
        'playoff_tree': playoffs_tree,
        'playoff_results': playoff_results,
        'winner': winner
    })

@app.route('/api/players')
def get_players():
    position_filter = request.args.get('position')
    search_query = request.args.get('search')
    sort_by = request.args.get('sort_by', 'vorp')  # one of: player_id, name, position, projected_points, vorp, games_played_frac, adp, bye_week
    sort_dir = request.args.get('sort_dir', 'desc').lower()  # 'asc' or 'desc'
    reverse = (sort_dir == 'desc')

    # If a draft is in progress, use the available players from the environment
    if draft_env:
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
