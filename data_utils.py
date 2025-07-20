import os
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# Define Player data structure
@dataclass
class Player:
    player_id: int
    name: str
    position: str # QB, RB, WR, TE
    projected_points: float
    adp: float = field(default=np.inf) # Default to infinity if no ADP

    def to_dict(self):
        return {
            'player_id': self.player_id,
            'name': self.name,
            'position': self.position,
            'projected_points': self.projected_points,
            'adp': self.adp if np.isfinite(self.adp) else None
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

    for _, row in df.iterrows():
        player_id = int(row['player_id'])
        name = str(row['name'])
        position = str(row['position']).upper() # Ensure consistent casing
        projected_points = float(row['projected_points'])
        adp = float(row['adp']) if has_adp_column and pd.notna(row['adp']) else np.inf

        players.append(Player(player_id, name, position, projected_points, adp))

    # If ADP column was missing or all values were NaN, generate mock ADP
    if not has_adp_column or all(p.adp == np.inf for p in players):
        print("ADP column missing or all values NaN. Generating mock ADP...")
        players = _generate_mock_adp(players, adp_config)
    else:
        print(f"Loaded {len(players)} players with existing ADP.")

    # Sort players by ADP for easier processing later (lower ADP means drafted earlier)
    players.sort(key=lambda p: p.adp)

    return players

def _create_dummy_csv(filepath: str):
    """
    Creates a small dummy CSV file for initial testing if the actual file is not found.
    """
    dummy_data = """player_id,name,position,projected_points,adp
1001,Patrick Mahomes,QB,378.5,15.2
1002,Josh Allen,QB,350.1,25.5
1003,Christian McCaffrey,RB,320.0,1.1
1004,Austin Ekeler,RB,290.5,5.8
1005,Justin Jefferson,WR,310.0,3.2
1006,Ja'Marr Chase,WR,295.5,7.5
1007,Travis Kelce,TE,280.0,10.0
1008,Mark Andrews,TE,250.0,20.1
1009,Saquon Barkley,RB,280.0,9.0
1010,Cooper Kupp,WR,270.0,12.0
1011,Jalen Hurts,QB,340.0,30.0
1012,Jonathan Taylor,RB,270.0,11.0
1013,Tyreek Hill,WR,280.0,14.0
1014,T.J. Hockenson,TE,200.0,35.0
1015,Stefon Diggs,WR,260.0,16.0
1016,Nick Chubb,RB,250.0,13.0
1017,CeeDee Lamb,WR,255.0,18.0
1018,Amari Cooper,WR,240.0,22.0
1019,George Kittle,TE,220.0,28.0
1020,Dak Prescott,QB,300.0,40.0
"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(dummy_data)
    print(f"Dummy CSV created at: {filepath}")