"""Stateless season simulation interfaces."""

from draft_buddy.simulator.evaluator import generate_round_robin_schedule, simulate_season_fast
from draft_buddy.simulator.service import SeasonSimulationService
from draft_buddy.simulator.team_identity import parse_team_id, team_label

__all__ = [
    "SeasonSimulationService",
    "generate_round_robin_schedule",
    "parse_team_id",
    "simulate_season_fast",
    "team_label",
]
