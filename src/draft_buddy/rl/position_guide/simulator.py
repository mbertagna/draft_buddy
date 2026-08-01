"""Monte Carlo simulation and aggregation for position guides."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List

from tqdm import tqdm

from draft_buddy.config import Config
from draft_buddy.data import load_player_catalog
from draft_buddy.rl.draft_gym_env import DraftGymEnv
from draft_buddy.rl.position_guide.pick_numbers import pick_placement
from draft_buddy.rl.position_guide.schemas import (
    POSITION_CODES,
    PositionGuideFile,
    PositionGuidePick,
    PositionProbabilities,
)


def average_position_probabilities(
    samples: List[Dict[str, float]],
) -> Dict[str, float]:
    """Compute element-wise mean of position probability samples.

    Parameters
    ----------
    samples : List[Dict[str, float]]
        Per-simulation probability dicts keyed by position code.

    Returns
    -------
    Dict[str, float]
        Averaged probabilities for each position.
    """
    if not samples:
        return {position: 0.0 for position in POSITION_CODES}
    totals = {position: 0.0 for position in POSITION_CODES}
    for sample in samples:
        for position in POSITION_CODES:
            totals[position] += float(sample.get(position, 0.0))
    count = float(len(samples))
    return {position: totals[position] / count for position in POSITION_CODES}


def top_position_from_probabilities(probabilities: Dict[str, float]) -> str:
    """Return the position with the highest probability.

    Parameters
    ----------
    probabilities : Dict[str, float]
        Position probability map.

    Returns
    -------
    str
        Position code with maximum probability.
    """
    return max(POSITION_CODES, key=lambda position: probabilities[position])


def extract_checkpoint_episode(checkpoint_path: str) -> int:
    """Parse episode number from a checkpoint filename.

    Parameters
    ----------
    checkpoint_path : str
        Path to a checkpoint file.

    Returns
    -------
    int
        Episode number, or ``0`` when not parseable.
    """
    match = re.search(r"checkpoint_episode_(\d+)\.pth$", checkpoint_path)
    if match is None:
        return 0
    return int(match.group(1))


class PositionGuideSimulator:
    """Run Monte Carlo draft rollouts and aggregate position probabilities."""

    def __init__(
        self,
        config: Config,
        draft_slot: int,
        num_teams: int,
        checkpoint_path: str,
        draft_year: int,
        simulations: int,
        seed: int,
        player_data_csv: str | None = None,
        show_progress: bool = True,
    ) -> None:
        """Initialize the simulator with draft and model settings.

        Parameters
        ----------
        config : Config
            Base configuration to copy and override.
        draft_slot : int
            User draft slot (1-based).
        num_teams : int
            Number of teams in the league.
        checkpoint_path : str
            Policy checkpoint used for suggestions and rollouts.
        draft_year : int
            Draft year for export metadata.
        simulations : int
            Number of Monte Carlo rollouts.
        seed : int
            Base random seed for reproducibility.
        player_data_csv : str | None, optional
            Player CSV path override.
        show_progress : bool, optional
            When ``True``, display a tqdm progress bar during rollouts.
        """
        self._base_config = config
        self._draft_slot = draft_slot
        self._num_teams = num_teams
        self._checkpoint_path = checkpoint_path
        self._draft_year = draft_year
        self._simulations = simulations
        self._seed = seed
        self._player_data_csv = player_data_csv or config.paths.PLAYER_DATA_CSV
        self._show_progress = show_progress

    def run(self) -> PositionGuideFile:
        """Execute simulations and build a position guide export.

        Returns
        -------
        PositionGuideFile
            Aggregated position guide ready for export.

        Raises
        ------
        RuntimeError
            When the policy model fails to load or inference returns an error.
        """
        runtime_config = self._build_runtime_config()
        player_catalog = load_player_catalog(
            self._player_data_csv, runtime_config.draft.MOCK_ADP_CONFIG
        )
        env = DraftGymEnv(
            runtime_config, training=True, player_catalog=player_catalog
        )
        if env.agent_model is None:
            raise RuntimeError(
                f"Failed to load policy model from checkpoint: {self._checkpoint_path}"
            )

        aggregates: Dict[int, List[Dict[str, float]]] = defaultdict(list)
        total_roster_size = env.total_roster_size_per_team

        simulation_range = range(self._simulations)
        if self._show_progress:
            simulation_range = tqdm(
                simulation_range,
                desc="Simulations",
                unit="sim",
                total=self._simulations,
            )

        for simulation_index in simulation_range:
            env.reset(seed=self._seed + simulation_index)
            user_pick_index = 0
            while env.team_rosters[env.agent_team_id].size < total_roster_size:
                team_on_clock = env._controller.team_on_clock
                if team_on_clock != env.agent_team_id:
                    break
                suggestion = env.get_ai_suggestion_for_team(env.agent_team_id)
                if "error" in suggestion:
                    raise RuntimeError(suggestion["error"])
                user_pick_index += 1
                aggregates[user_pick_index].append(suggestion)
                action = max(
                    env.action_to_position,
                    key=lambda action_index: float(
                        suggestion.get(env.action_to_position[action_index], 0.0)
                    ),
                )
                _observation, _reward, done, _truncated, _info = env.step(action)
                if done:
                    break

        picks = self._build_pick_rows(aggregates)
        return PositionGuideFile(
            generated_at=datetime.now(timezone.utc),
            draft_year=self._draft_year,
            draft_slot=self._draft_slot,
            num_teams=self._num_teams,
            simulations=self._simulations,
            checkpoint_path=self._checkpoint_path,
            checkpoint_episode=extract_checkpoint_episode(self._checkpoint_path),
            player_data_csv=self._player_data_csv,
            enabled_state_features=list(runtime_config.training.ENABLED_STATE_FEATURES),
            roster_structure=dict(runtime_config.draft.ROSTER_STRUCTURE),
            total_user_picks=len(picks),
            picks=picks,
        )

    def _build_runtime_config(self) -> Config:
        """Return a config copy with guide-specific overrides applied."""
        runtime_config = Config.from_dict(self._base_config.to_dict())
        runtime_config.draft.NUM_TEAMS = self._num_teams
        runtime_config.draft.AGENT_START_POSITION = self._draft_slot
        runtime_config.draft.RANDOMIZE_AGENT_START_POSITION = False
        runtime_config.training.MODEL_PATH_TO_LOAD = self._checkpoint_path
        return runtime_config

    def _build_pick_rows(
        self, aggregates: Dict[int, List[Dict[str, float]]]
    ) -> List[PositionGuidePick]:
        """Convert raw aggregates into ordered pick rows.

        Parameters
        ----------
        aggregates : Dict[int, List[Dict[str, float]]]
            Raw probability samples keyed by user pick index.

        Returns
        -------
        List[PositionGuidePick]
            Sorted pick rows for export.
        """
        picks: List[PositionGuidePick] = []
        for user_pick_index in sorted(aggregates):
            averaged = average_position_probabilities(aggregates[user_pick_index])
            placement = pick_placement(self._num_teams, self._draft_slot, user_pick_index)
            top_position = top_position_from_probabilities(averaged)
            picks.append(
                PositionGuidePick(
                    user_pick_index=user_pick_index,
                    overall_pick_number=placement.overall_pick_number,
                    round=placement.round,
                    positions=PositionProbabilities(**averaged),
                    top_position=top_position,
                    sample_count=len(aggregates[user_pick_index]),
                )
            )
        return picks
