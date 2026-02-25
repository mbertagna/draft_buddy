import os
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union

# Define Player data structure
@dataclass
class Player:
    player_id: int
    name: str
    position: str # QB, RB, WR, TE
    projected_points: float
    games_played_frac: Union[float, str] = 1.0
    adp: float = field(default=np.inf) # Default to infinity if no ADP
    bye_week: Optional[int] = None
    team: Optional[str] = None

    def to_dict(self):
        return {
            'player_id': self.player_id,
            'name': self.name,
            'position': self.position,
            'projected_points': self.projected_points,
            'games_played_frac': self.games_played_frac,
            'adp': self.adp if np.isfinite(self.adp) else None,
            'bye_week': self.bye_week,
            'team': self.team
        }

def _generate_mock_adp(players: List[Player], adp_config: Dict) -> List[Player]:
    """
    Generates mock ADP for players based on configurable weighted attributes.
    Modifies players in place.
    """
    if not adp_config['enabled']:
        raise ValueError("Mock ADP generation is disabled but 'adp' column is missing.")

    # Create a list of tuples (weighted_score, player_id)
    weighted_scores = []
    for player in players:
        score = 0.0
        for attr, weight in adp_config['weights'].items():
            if hasattr(player, attr):
                score += getattr(player, attr) * weight
            else:
                print(f"Warning: Attribute '{attr}' not found in Player object for mock ADP generation.")
        weighted_scores.append((score, player.player_id))

    # Sort based on weighted scores
    # If sort_order_ascending is True, lower score gets lower ADP (e.g., age)
    # If sort_order_ascending is False, higher score gets lower ADP (e.g., projected points)
    weighted_scores.sort(key=lambda x: x[0], reverse=not adp_config['sort_order_ascending'])

    # Assign ADPs based on rank
    player_id_to_adp = {score_tuple[1]: i + 1 for i, score_tuple in enumerate(weighted_scores)}

    # Update player objects with new ADP
    for player in players:
        player.adp = player_id_to_adp.get(player.player_id, np.inf) # Assign generated ADP or keep inf

    print(f"Generated mock ADP for {len(players)} players based on weights: {adp_config['weights']}")
    return players


def load_player_data(filepath: str, adp_config: Dict) -> List[Player]:
    """
    Loads player data from a CSV file, handles missing ADP by generating mock ADP.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: Player data CSV not found at {filepath}. Please ensure it exists.")
        print("Creating a dummy CSV for demonstration purposes...")
        _create_dummy_csv(filepath)
        df = pd.read_csv(filepath) # Try loading again

    required_columns = ['player_id', 'name', 'position', 'projected_points']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}' in player data CSV.")

    players = []
    has_adp_column = 'adp' in df.columns
    has_games_played_frac_column = 'games_played_frac' in df.columns
    has_bye_week_column = 'bye_week' in df.columns
    # Team column can be named 'recent_team' from our merge, or 'team'
    team_col = 'recent_team' if 'recent_team' in df.columns else ('team' if 'team' in df.columns else None)

    for _, row in df.iterrows():
        player_id = int(row['player_id'])
        name = str(row['name'])
        position = str(row['position']).upper() # Ensure consistent casing
        projected_points = float(row['projected_points'])
        adp = float(row['adp']) if has_adp_column and pd.notna(row['adp']) else np.inf
        
        games_played_frac_val = row['games_played_frac'] if has_games_played_frac_column and pd.notna(row['games_played_frac']) else 1.0
        if games_played_frac_val == 'R':
            games_played_frac = 'R'
        else:
            games_played_frac = float(games_played_frac_val)

        bye_week = int(row['bye_week']) if has_bye_week_column and pd.notna(row['bye_week']) else None

        team = str(row[team_col]) if team_col and pd.notna(row[team_col]) else None
        players.append(Player(player_id, name, position, projected_points, games_played_frac, adp, bye_week, team))

    # If ADP column was missing or all values were NaN, generate mock ADP
    if not has_adp_column or all(p.adp == np.inf for p in players):
        print("ADP column missing or all values NaN. Generating mock ADP...")
        players = _generate_mock_adp(players, adp_config)
    else:
        print(f"Loaded {len(players)} with existing ADP.")

    # Sort players by ADP for easier processing later (lower ADP means drafted earlier)
    players.sort(key=lambda p: p.adp)

    return players

def _create_dummy_csv(filepath: str):
    """
    Creates a small dummy CSV file for initial testing if the actual file is not found.
    """
    dummy_data = """player_id,name,position,projected_points,adp,games_played_frac
1001,Patrick Mahomes,QB,378.5,15.2,1.0
1002,Josh Allen,QB,350.1,25.5,1.0
1003,Christian McCaffrey,RB,320.0,1.1,0.9
1004,Austin Ekeler,RB,290.5,5.8,1.0
1005,Justin Jefferson,WR,310.0,3.2,1.0
1006,Ja'Marr Chase,WR,295.5,7.5,1.0
1007,Travis Kelce,TE,280.0,10.0,1.0
1008,Mark Andrews,TE,250.0,20.1,0.85
1009,Saquon Barkley,RB,280.0,9.0,0.8
1010,Cooper Kupp,WR,270.0,12.0,0.75
1011,Jalen Hurts,QB,340.0,30.0,1.0
1012,Jonathan Taylor,RB,270.0,11.0,0.9
1013,Tyreek Hill,WR,280.0,14.0,1.0
1014,T.J. Hockenson,TE,200.0,35.0,0.8
1015,Stefon Diggs,WR,260.0,16.0,1.0
1016,Nick Chubb,RB,250.0,13.0,0.95
1017,CeeDee Lamb,WR,255.0,18.0,1.0
1018,Amari Cooper,WR,240.0,22.0,1.0
1019,George Kittle,TE,220.0,28.0,0.85
1020,Dak Prescott,QB,300.0,40.0,1.0
"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(dummy_data)
    print(f"Dummy CSV created at: {filepath}")


def calculate_stack_count(roster_players: List[Player]) -> int:
    """
    Calculate the total number of valid QB-WR/TE stacks on a given roster.
    
    A valid stack is defined as a Quarterback and a Wide Receiver or Tight End 
    sharing the same NFL team identifier.
    
    Parameters
    ----------
    roster_players : List[Player]
        List of Player objects representing the current roster
        
    Returns
    -------
    int
        Total number of valid stacks found on the roster
        
    Examples
    --------
    >>> qb = Player(1, "Josh Allen", "QB", 350.0, team="BUF")
    >>> wr = Player(2, "Stefon Diggs", "WR", 280.0, team="BUF") 
    >>> te = Player(3, "Dawson Knox", "TE", 150.0, team="BUF")
    >>> rb = Player(4, "James Cook", "RB", 200.0, team="BUF")
    >>> calculate_stack_count([qb, wr, te, rb])
    2
    """
    if not roster_players:
        return 0
    
    # Group players by team
    team_positions = {}
    for player in roster_players:
        if player.team is None:
            continue
        if player.team not in team_positions:
            team_positions[player.team] = {'QB': 0, 'WR': 0, 'TE': 0}
        if player.position in ['QB', 'WR', 'TE']:
            team_positions[player.team][player.position] += 1
    
    # Count stacks for each team
    total_stacks = 0
    for team, positions in team_positions.items():
        qb_count = positions['QB']
        stack_targets = positions['WR'] + positions['TE']
        # Each QB can stack with each WR/TE on the same team
        total_stacks += qb_count * stack_targets
    
    return total_stacks