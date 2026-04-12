import logging
from collections import defaultdict
from typing import List, Dict, Tuple, Any

# Setup logger for the module
logger = logging.getLogger(__name__)

def categorize_roster_by_slots(
    team_roster: List[Any], 
    roster_structure: Dict[str, int], 
    bench_maxes: Dict[str, int]
) -> Tuple[Dict[str, List[Any]], List[Any], List[Any]]:
    """
    Categorizes players into starters, bench, and flex players based on roster structure
    and projected points, ensuring no player is double-counted.

    Parameters
    ----------
    team_roster : list
        List of player objects with 'position', 'projected_points', and 'player_id' attributes.
    roster_structure : dict
        Mapping of position to number of required starters (e.g., {'QB': 1, 'FLEX': 1}).
    bench_maxes : dict
        Mapping of position to maximum allowed bench players.

    Returns
    -------
    tuple
        A tuple containing:
        - starters: dict mapping position to list of player objects.
        - bench: list of player objects assigned to the bench.
        - flex_players: list of player objects assigned to the FLEX spot.
    """
    starters = defaultdict(list)
    bench = []
    flex_players = []

    # Create a copy of the roster to avoid modifying the original list
    sorted_roster = sorted(team_roster, key=lambda p: p.projected_points, reverse=True)
    
    # A set to keep track of players who have already been assigned a spot
    assigned_player_ids = set()

    # --- 1. Fill mandatory starter positions ---
    for pos in roster_structure.keys():
        if pos == 'FLEX': continue # Handle FLEX separately
        
        # Find the best players for this position
        pos_players = [p for p in sorted_roster if p.position == pos and getattr(p, 'player_id', id(p)) not in assigned_player_ids]
        
        # Assign the required number of starters
        num_starters_for_pos = roster_structure.get(pos, 0)
        for i in range(min(len(pos_players), num_starters_for_pos)):
            player = pos_players[i]
            starters[pos].append(player)
            assigned_player_ids.add(getattr(player, 'player_id', id(player)))

    # --- 2. Identify candidates for FLEX and Bench ---
    remaining_players = [p for p in sorted_roster if getattr(p, 'player_id', id(p)) not in assigned_player_ids]
    flex_candidates = [p for p in remaining_players if p.position in ['RB', 'WR', 'TE']]

    # --- 3. Fill FLEX spots from the best candidates ---
    num_flex_spots = roster_structure.get('FLEX', 0)
    flex_players = sorted(flex_candidates, key=lambda p: p.projected_points, reverse=True)[:num_flex_spots]
    for player in flex_players:
        starters['FLEX'].append(player)
        assigned_player_ids.add(getattr(player, 'player_id', id(player)))

    # --- 4. Assign all remaining players to the bench ---
    bench = [p for p in sorted_roster if getattr(p, 'player_id', id(p)) not in assigned_player_ids]

    return starters, bench, flex_players

def calculate_roster_scores(
    team_roster: List[Any], 
    roster_structure: Dict[str, int], 
    bench_maxes: Dict[str, int]
) -> Dict[str, float]:
    """
    Calculates projected points for starters, bench, and combined based on a team's drafted players.

    Parameters
    ----------
    team_roster : list
        List of player objects.
    roster_structure : dict
        Mapping of position to number of required starters.
    bench_maxes : dict
        Mapping of position to maximum allowed bench players.

    Returns
    -------
    dict
        Dictionary containing total and average points for starters, bench, and flex, plus roster size.
    """
    starters_dict, bench_list, flex_list = categorize_roster_by_slots(team_roster, roster_structure, bench_maxes)
    
    starter_points = sum(p.projected_points for pos_list in starters_dict.values() for p in pos_list if p not in flex_list)
    flex_points = sum(p.projected_points for p in flex_list)
    bench_points = sum(p.projected_points for p in bench_list)
    
    # Check if there are unoptimally assigned players (overflow beyond position capacity).
    # For RB/WR/TE, capacity includes FLEX slots since those positions can fill FLEX.
    temp_pos_counts = defaultdict(int)
    for pos, pos_list in starters_dict.items():
        if pos == 'FLEX':
            continue
        for p in pos_list:
            temp_pos_counts[p.position] += 1

    for pos, pos_list in starters_dict.items():
        if pos != 'FLEX':
            continue
        for p in pos_list:
            temp_pos_counts[p.position] += 1

    for p in bench_list:
        pos_limit = roster_structure.get(p.position, 0) + bench_maxes.get(p.position, 0)
        if p.position in ('RB', 'WR', 'TE'):
            pos_limit += roster_structure.get('FLEX', 0)
        if temp_pos_counts[p.position] < pos_limit:
            temp_pos_counts[p.position] += 1
        else:
            logger.warning(
                f"Player {p.name if hasattr(p, 'name') else p} ({p.position}) could not be optimally "
                "assigned to a starter/flex spot. Placed on bench."
            )

    combined_total_points = starter_points + bench_points + flex_points
    
    # Calculate average points
    num_starters = sum(len(lst) for pos, lst in starters_dict.items() if pos != 'FLEX') + len(flex_list)
    num_bench = len(bench_list)

    avg_starter_points = (starter_points + flex_points) / num_starters if num_starters > 0 else 0
    avg_bench_points = bench_points / num_bench if num_bench > 0 else 0
    avg_combined_points = combined_total_points / (num_starters + num_bench) if (num_starters + num_bench) > 0 else 0

    return {
        'starters_total_points': starter_points + flex_points,  # simulate.py treated flex as part of starters in some contexts, but tracked separate
        'bench_total_points': bench_points,
        'flex_total_points': flex_points,
        'combined_total_points': combined_total_points,
        'starters_avg_points': avg_starter_points,
        'bench_avg_points': avg_bench_points,
        'combined_avg_points': avg_combined_points,
        'roster_size': len(team_roster)
    }
