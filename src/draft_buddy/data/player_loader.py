"""Player loading and ADP enrichment for draft workflows."""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd

from draft_buddy.domain.entities import Player


def load_player_data(filepath: str, adp_config: Dict) -> List[Player]:
    """Load players from CSV and ensure ADP values exist.

    Parameters
    ----------
    filepath : str
        Source CSV path.
    adp_config : Dict
        Mock ADP generation config if ADP values are missing.

    Returns
    -------
    List[Player]
        Loaded and ADP-sorted players.
    """
    try:
        dataframe = pd.read_csv(filepath)
    except FileNotFoundError:
        _create_dummy_csv(filepath)
        dataframe = pd.read_csv(filepath)

    required_columns = ["player_id", "name", "position", "projected_points"]
    for column_name in required_columns:
        if column_name not in dataframe.columns:
            raise ValueError(f"Missing required column '{column_name}' in player data CSV.")

    players: List[Player] = []
    has_adp_column = "adp" in dataframe.columns
    has_games_played_fraction = "games_played_frac" in dataframe.columns
    has_bye_week_column = "bye_week" in dataframe.columns
    team_column = (
        "recent_team"
        if "recent_team" in dataframe.columns
        else ("team" if "team" in dataframe.columns else None)
    )

    for _, row in dataframe.iterrows():
        player_id = int(row["player_id"])
        name = str(row["name"])
        position = str(row["position"]).upper()
        projected_points = float(row["projected_points"])
        adp = float(row["adp"]) if has_adp_column and pd.notna(row["adp"]) else np.inf
        games_played_fraction = (
            row["games_played_frac"]
            if has_games_played_fraction and pd.notna(row["games_played_frac"])
            else 1.0
        )
        if games_played_fraction != "R":
            games_played_fraction = float(games_played_fraction)
        bye_week = int(row["bye_week"]) if has_bye_week_column and pd.notna(row["bye_week"]) else None
        team = str(row[team_column]) if team_column and pd.notna(row[team_column]) else None
        players.append(
            Player(
                player_id,
                name,
                position,
                projected_points,
                games_played_fraction,
                adp,
                bye_week,
                team,
            )
        )

    if not has_adp_column or all(player.adp == np.inf for player in players):
        players = _generate_mock_adp(players, adp_config)

    players.sort(key=lambda player: player.adp)
    return players


def _generate_mock_adp(players: List[Player], adp_config: Dict) -> List[Player]:
    """Generate ADP rankings from weighted player attributes.

    Parameters
    ----------
    players : List[Player]
        Players to rank.
    adp_config : Dict
        Configuration containing weights and sort direction.

    Returns
    -------
    List[Player]
        Players with generated ADP values.
    """
    if not adp_config["enabled"]:
        raise ValueError("Mock ADP generation is disabled but ADP data is missing.")
    weighted_scores = []
    for player in players:
        weighted_score = 0.0
        for attribute, weight in adp_config["weights"].items():
            if hasattr(player, attribute):
                weighted_score += getattr(player, attribute) * weight
        weighted_scores.append((weighted_score, player.player_id))
    weighted_scores.sort(
        key=lambda score_tuple: score_tuple[0],
        reverse=not adp_config["sort_order_ascending"],
    )
    player_id_to_adp = {
        score_tuple[1]: index + 1 for index, score_tuple in enumerate(weighted_scores)
    }
    for player in players:
        player.adp = player_id_to_adp.get(player.player_id, np.inf)
    return players


def _create_dummy_csv(filepath: str) -> None:
    """Create a fallback dummy player CSV when source data is missing.

    Parameters
    ----------
    filepath : str
        Destination path for generated CSV.
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
"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as file_obj:
        file_obj.write(dummy_data)
