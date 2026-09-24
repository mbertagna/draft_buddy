"""Player loading and ADP enrichment for draft workflows."""

from __future__ import annotations

from dataclasses import replace
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from draft_buddy.core import Player, PlayerCatalog


def load_player_catalog(filepath: str, adp_config: Dict) -> PlayerCatalog:
    """Load a player catalog from CSV and ensure ADP values exist.

    Parameters
    ----------
    filepath : str
        Source CSV path.
    adp_config : Dict
        Mock ADP generation config if ADP values are missing.

    Returns
    -------
    PlayerCatalog
        Loaded catalog sorted by ADP.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"Generated player data not found: {filepath}. "
            "Run `docker compose run --rm data` for the active league."
        )
    dataframe = pd.read_csv(filepath)

    required_columns = ["player_id", "name", "position", "projected_points"]
    for column_name in required_columns:
        if column_name not in dataframe.columns:
            raise ValueError(f"Missing required column '{column_name}' in player data CSV.")

    players: list[Player] = []
    has_adp_column = "adp" in dataframe.columns
    has_games_played_fraction = "games_played_frac" in dataframe.columns
    has_bye_week_column = "bye_week" in dataframe.columns
    team_column = (
        "recent_team"
        if "recent_team" in dataframe.columns
        else ("team" if "team" in dataframe.columns else None)
    )
    sleeper_columns = {
        "sleeper_id": "sleeper_id" in dataframe.columns,
        "sleeper_status": "sleeper_status" in dataframe.columns,
        "sleeper_injury_status": "sleeper_injury_status" in dataframe.columns,
        "sleeper_depth_chart_position": "sleeper_depth_chart_position" in dataframe.columns,
    }

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
                player_id=player_id,
                name=name,
                position=position,
                projected_points=projected_points,
                games_played_frac=games_played_fraction,
                adp=adp,
                bye_week=bye_week,
                team=team,
                **_read_sleeper_fields(row, sleeper_columns),
                **_read_valuation_fields(row),
                **_read_form_fields(row),
            )
        )

    if not has_adp_column or all(player.adp == np.inf for player in players):
        players = _generate_mock_adp(players, adp_config)

    players.sort(key=lambda player: player.adp)
    return PlayerCatalog(players)


def _read_sleeper_fields(row: pd.Series, sleeper_columns: Dict[str, bool]) -> Dict[str, Optional[str]]:
    """Read optional Sleeper-derived columns from one CSV row.

    Parameters
    ----------
    row : pd.Series
        Source CSV row.
    sleeper_columns : Dict[str, bool]
        Mapping of Sleeper field name to whether the column exists in the CSV.

    Returns
    -------
    Dict[str, Optional[str]]
        Sleeper field values, or None for missing columns/values.
    """
    return {
        field_name: (str(row[field_name]) if has_column and pd.notna(row[field_name]) else None)
        for field_name, has_column in sleeper_columns.items()
    }


def _read_valuation_fields(row: pd.Series) -> Dict[str, Optional[float] | Optional[str]]:
    """Read optional static valuation fields from a CSV row.

    Parameters
    ----------
    row : pandas.Series
        Source CSV row.

    Returns
    -------
    Dict[str, Optional[float] | Optional[str]]
        Optional valuation values keyed by ``Player`` constructor fields.
    """
    float_fields = [
        "historical_weekly_median",
        "projected_points_p20",
        "projected_points_p50",
        "projected_points_p80",
        "effective_sample_size",
        "data_confidence",
        "market_implied_points",
        "projection_edge_points",
        "market_premium_points",
        "projection_edge_z",
        "agreement_score",
        "stability_score",
    ]
    text_fields = ["projection_basis", "risk_tier", "player_archetype"]
    values: Dict[str, Optional[float] | Optional[str]] = {}
    for field_name in float_fields:
        values[field_name] = _optional_float(row.get(field_name))
    for field_name in text_fields:
        values[field_name] = _optional_string(row.get(field_name))
    return values


def _read_form_fields(row: pd.Series) -> Dict[str, Optional[float] | Optional[int]]:
    """Read weekly distribution and current-season scoring columns.

    Parameters
    ----------
    row : pandas.Series
        Source CSV row.

    Returns
    -------
    dict
        Optional form fields for ``Player``.
    """
    games = _optional_float(row.get("current_season_games"))
    order = _optional_float(row.get("depth_chart_order"))
    if order is None:
        order = _optional_float(row.get("sleeper_depth_chart_order"))
    return {
        "weekly_points_mean": _optional_float(row.get("weekly_points_mean")),
        "weekly_points_median": _optional_float(row.get("weekly_points_median")),
        "weekly_points_floor": _optional_float(row.get("weekly_points_floor")),
        "weekly_points_ceiling": _optional_float(row.get("weekly_points_ceiling")),
        "current_season_ppg": _optional_float(row.get("current_season_ppg")),
        "current_season_games": int(games or 0),
        "sleeper_depth_chart_order": None if order is None else int(order),
    }


def _optional_float(value) -> Optional[float]:
    """Return a finite float value or ``None``."""
    if value is None or pd.isna(value):
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _optional_string(value) -> Optional[str]:
    """Return a non-empty string or ``None``."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _generate_mock_adp(players: list[Player], adp_config: Dict) -> list[Player]:
    """Generate ADP rankings from weighted player attributes.

    Parameters
    ----------
    players : List[Player]
        Players to rank.
    adp_config : Dict
        Configuration containing weights and sort direction.

    Returns
    -------
    list[Player]
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
    return [
        replace(player, adp=player_id_to_adp.get(player.player_id, np.inf))
        for player in players
    ]
