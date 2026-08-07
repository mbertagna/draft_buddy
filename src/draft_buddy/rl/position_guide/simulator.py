"""All-slots self-play Monte Carlo simulation for position guides."""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from tqdm import tqdm

from draft_buddy.config import Config
from draft_buddy.core.entities import Player, PlayerCatalog
from draft_buddy.data import load_player_catalog
from draft_buddy.rl.draft_gym_env import DraftGymEnv
from draft_buddy.rl.position_guide.pick_numbers import pick_placement
from draft_buddy.rl.position_guide.schemas import (
    POSITION_CODES,
    ModelAdpEntry,
    ModelAdpFile,
    PositionGuideFile,
    PositionGuidePick,
    PositionProbabilities,
    TopPlayerEntry,
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


def sample_position(suggestion: Dict[str, float], rng: random.Random) -> str:
    """Sample one position from a (temperature-scaled) probability map.

    Parameters
    ----------
    suggestion : Dict[str, float]
        Position probabilities, already action-masked and temperature
        scaled by the caller.
    rng : random.Random
        Seeded random generator, for reproducible simulations.

    Returns
    -------
    str
        Sampled position code. Falls back to a uniform choice across all
        position codes in the degenerate case where every probability is
        zero (no legal action was reflected in the suggestion).
    """
    positions = list(POSITION_CODES)
    weights = [max(suggestion.get(position, 0.0), 0.0) for position in positions]
    if sum(weights) <= 0.0:
        return rng.choice(positions)
    return rng.choices(positions, weights=weights, k=1)[0]


@dataclass
class _ModelAdpAccumulator:
    """Running mean/variance accumulator for one player's overall pick number."""

    count: int = 0
    _sum: float = field(default=0.0, repr=False)
    _sum_sq: float = field(default=0.0, repr=False)

    def add(self, overall_pick_number: int) -> None:
        """Record one observed overall pick number."""
        self.count += 1
        self._sum += overall_pick_number
        self._sum_sq += overall_pick_number**2

    @property
    def mean(self) -> float:
        """Return the mean overall pick number, or ``0.0`` when empty."""
        return self._sum / self.count if self.count else 0.0

    @property
    def std_dev(self) -> float:
        """Return the population standard deviation of overall pick number."""
        if self.count < 2:
            return 0.0
        variance = (self._sum_sq / self.count) - (self.mean**2)
        return max(variance, 0.0) ** 0.5


class PositionGuideSimulator:
    """Run all-slots self-play Monte Carlo draft rollouts.

    Every team samples its own temperature-scaled policy suggestion to make
    its actual pick each turn ("self-play"), so the actual draft history
    feeding a team's later suggestions is always generated consistently
    with what the exported guide reports for that team. One batch of
    simulations therefore yields self-consistent position guides for every
    draft slot simultaneously, plus a model-derived ADP ranking pooled
    across all players and picks.
    """

    def __init__(
        self,
        config: Config,
        num_teams: int,
        checkpoint_path: str,
        draft_year: int,
        simulations: int,
        seed: int,
        temperature: float = 1.5,
        player_data_csv: str | None = None,
        show_progress: bool = True,
    ) -> None:
        """Initialize the simulator with draft and model settings.

        Parameters
        ----------
        config : Config
            Base configuration to copy and override.
        num_teams : int
            Number of teams in the league.
        checkpoint_path : str
            Policy checkpoint used for suggestions and self-play picks.
        draft_year : int
            Draft year for export metadata.
        simulations : int
            Number of Monte Carlo rollouts.
        seed : int
            Base random seed for reproducibility.
        temperature : float, optional
            Softmax temperature applied to every suggestion, both for what
            is reported in the guide and for sampling the actual self-play
            pick. Values above ``1.0`` soften overconfident distributions.
        player_data_csv : str | None, optional
            Player CSV path override.
        show_progress : bool, optional
            When ``True``, display a tqdm progress bar during rollouts.
        """
        self._base_config = config
        self._num_teams = num_teams
        self._checkpoint_path = checkpoint_path
        self._draft_year = draft_year
        self._simulations = simulations
        self._seed = seed
        self._temperature = temperature
        self._player_data_csv = player_data_csv or config.paths.PLAYER_DATA_CSV
        self._show_progress = show_progress

    def run(self) -> tuple[Dict[int, PositionGuideFile], ModelAdpFile]:
        """Execute self-play simulations and build guide and ADP exports.

        Returns
        -------
        tuple[Dict[int, PositionGuideFile], ModelAdpFile]
            Position guide keyed by draft slot (1-based), and the
            model-derived ADP export pooled across all slots.

        Raises
        ------
        RuntimeError
            When the policy model fails to load or a suggestion errors.
        """
        runtime_config = self._build_runtime_config()
        player_catalog = load_player_catalog(
            self._player_data_csv, runtime_config.draft.MOCK_ADP_CONFIG
        )
        env = DraftGymEnv(runtime_config, training=False, player_catalog=player_catalog)
        if env.agent_model is None:
            raise RuntimeError(
                f"Failed to load policy model from checkpoint: {self._checkpoint_path}"
            )

        suggestion_aggregates: Dict[int, Dict[int, List[Dict[str, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        top_player_tally: Dict[int, Dict[int, Dict[str, Counter]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(Counter))
        )
        model_adp_tally: Dict[int, _ModelAdpAccumulator] = defaultdict(_ModelAdpAccumulator)

        simulation_range = range(self._simulations)
        if self._show_progress:
            simulation_range = tqdm(
                simulation_range, desc="Simulations", unit="sim", total=self._simulations
            )

        for simulation_index in simulation_range:
            self._run_one_draft(
                env,
                seed=self._seed + simulation_index,
                suggestion_aggregates=suggestion_aggregates,
                top_player_tally=top_player_tally,
                model_adp_tally=model_adp_tally,
            )

        guides = self._build_guides(suggestion_aggregates, top_player_tally, player_catalog)
        model_adp_file = self._build_model_adp_file(model_adp_tally, player_catalog)
        return guides, model_adp_file

    def _run_one_draft(
        self,
        env: DraftGymEnv,
        seed: int,
        suggestion_aggregates: Dict[int, Dict[int, List[Dict[str, float]]]],
        top_player_tally: Dict[int, Dict[int, Dict[str, Counter]]],
        model_adp_tally: Dict[int, _ModelAdpAccumulator],
    ) -> None:
        """Simulate one full self-play draft and record its data.

        Parameters
        ----------
        env : DraftGymEnv
            Environment reused across simulations.
        seed : int
            Seed for this simulation's reset and sampling RNG.
        suggestion_aggregates : Dict[int, Dict[int, List[Dict[str, float]]]]
            Accumulator: team id -> pick ordinal -> suggestion samples.
        top_player_tally : Dict[int, Dict[int, Dict[str, Counter]]]
            Accumulator: team id -> pick ordinal -> position -> player
            frequency counter.
        model_adp_tally : Dict[int, _ModelAdpAccumulator]
            Accumulator: player id -> overall pick number statistics.
        """
        env.reset(seed=seed)
        rng = random.Random(seed)
        pick_ordinal_by_team: Dict[int, int] = defaultdict(int)

        while True:
            team_id = env._controller.team_on_clock
            if team_id is None:
                break
            suggestion = env.get_ai_suggestion_for_team(team_id, temperature=self._temperature)
            if "error" in suggestion:
                raise RuntimeError(suggestion["error"])
            pick_ordinal_by_team[team_id] += 1
            pick_ordinal = pick_ordinal_by_team[team_id]
            suggestion_aggregates[team_id][pick_ordinal].append(suggestion)

            position = sample_position(suggestion, rng)
            drafted_player = self._resolve_pick(env, team_id, position)
            if drafted_player is None:
                break

            overall_pick_number = env._controller.current_pick_number
            env._controller.apply_pick(
                team_id=team_id, player_id=drafted_player.player_id, is_manual_pick=False
            )
            env._invalidate_sorted_available_cache()
            top_player_tally[team_id][pick_ordinal][position][drafted_player.player_id] += 1
            model_adp_tally[drafted_player.player_id].add(overall_pick_number)

    def _resolve_pick(
        self, env: DraftGymEnv, team_id: int, position: str
    ) -> Optional[Player]:
        """Return the player drafted for a sampled position, with fallback.

        Parameters
        ----------
        env : DraftGymEnv
            Environment driving the shared draft state.
        team_id : int
            Team currently on the clock.
        position : str
            Position sampled from the team's suggestion.

        Returns
        -------
        Player or None
            Selected player, or ``None`` when no legal pick exists at all
            (the draft cannot continue for this team).
        """
        is_valid, drafted_player = env._controller.try_select_player_for_team(
            team_id, position
        )
        if is_valid and drafted_player is not None:
            return drafted_player
        eligible = [
            env.player_catalog.require(player_id)
            for player_id in env._controller.available_player_ids
            if env.player_catalog.get(player_id)
            and env._controller.can_draft_position(
                team_id, env.player_catalog.require(player_id).position, is_manual=False
            )
        ]
        return min(eligible, key=lambda player: player.adp) if eligible else None

    def _build_runtime_config(self) -> Config:
        """Return a config copy with guide-specific overrides applied."""
        runtime_config = Config.from_dict(self._base_config.to_dict())
        runtime_config.draft.NUM_TEAMS = self._num_teams
        runtime_config.draft.RANDOMIZE_AGENT_START_POSITION = False
        runtime_config.training.MODEL_PATH_TO_LOAD = self._checkpoint_path
        return runtime_config

    def _build_guides(
        self,
        suggestion_aggregates: Dict[int, Dict[int, List[Dict[str, float]]]],
        top_player_tally: Dict[int, Dict[int, Dict[str, Counter]]],
        player_catalog: PlayerCatalog,
    ) -> Dict[int, PositionGuideFile]:
        """Build one ``PositionGuideFile`` per draft slot from aggregates."""
        generated_at = datetime.now(timezone.utc)
        runtime_config = self._build_runtime_config()
        guides: Dict[int, PositionGuideFile] = {}
        for slot in range(1, self._num_teams + 1):
            picks = self._build_pick_rows(
                slot,
                suggestion_aggregates.get(slot, {}),
                top_player_tally.get(slot, {}),
                player_catalog,
            )
            guides[slot] = PositionGuideFile(
                generated_at=generated_at,
                draft_year=self._draft_year,
                draft_slot=slot,
                num_teams=self._num_teams,
                simulations=self._simulations,
                checkpoint_path=self._checkpoint_path,
                checkpoint_episode=extract_checkpoint_episode(self._checkpoint_path),
                player_data_csv=self._player_data_csv,
                enabled_state_features=list(runtime_config.training.ENABLED_STATE_FEATURES),
                roster_structure=dict(runtime_config.draft.ROSTER_STRUCTURE),
                total_user_picks=len(picks),
                temperature=self._temperature,
                picks=picks,
            )
        return guides

    def _build_pick_rows(
        self,
        slot: int,
        aggregates: Dict[int, List[Dict[str, float]]],
        top_players_by_pick: Dict[int, Dict[str, Counter]],
        player_catalog: PlayerCatalog,
    ) -> List[PositionGuidePick]:
        """Convert one slot's raw aggregates into ordered pick rows.

        Parameters
        ----------
        slot : int
            Draft slot (1-based) these rows belong to.
        aggregates : Dict[int, List[Dict[str, float]]]
            Raw probability samples keyed by pick ordinal.
        top_players_by_pick : Dict[int, Dict[str, Counter]]
            Drafted-player frequency counters keyed by pick ordinal and
            position.
        player_catalog : PlayerCatalog
            Catalog used to resolve player names.

        Returns
        -------
        List[PositionGuidePick]
            Sorted pick rows for export.
        """
        picks: List[PositionGuidePick] = []
        for pick_ordinal in sorted(aggregates):
            averaged = average_position_probabilities(aggregates[pick_ordinal])
            placement = pick_placement(self._num_teams, slot, pick_ordinal)
            top_position = top_position_from_probabilities(averaged)
            picks.append(
                PositionGuidePick(
                    user_pick_index=pick_ordinal,
                    overall_pick_number=placement.overall_pick_number,
                    round=placement.round,
                    positions=PositionProbabilities(**averaged),
                    top_position=top_position,
                    sample_count=len(aggregates[pick_ordinal]),
                    top_players=self._build_top_players(
                        top_players_by_pick.get(pick_ordinal, {}), player_catalog
                    ),
                )
            )
        return picks

    def _build_top_players(
        self, tally_by_position: Dict[str, Counter], player_catalog: PlayerCatalog
    ) -> Dict[str, List[TopPlayerEntry]]:
        """Return up to five most-frequently-drafted players per position.

        Parameters
        ----------
        tally_by_position : Dict[str, Counter]
            Player-id frequency counters keyed by position.
        player_catalog : PlayerCatalog
            Catalog used to resolve player names.

        Returns
        -------
        Dict[str, List[TopPlayerEntry]]
            Top-5 entries per position with at least one recorded pick.
        """
        result: Dict[str, List[TopPlayerEntry]] = {}
        for position, counts in tally_by_position.items():
            total = sum(counts.values())
            if total <= 0:
                continue
            entries = []
            for player_id, times_drafted in counts.most_common(5):
                player = player_catalog.get(player_id)
                name = player.name if player is not None else f"Player {player_id}"
                entries.append(
                    TopPlayerEntry(
                        player_id=player_id,
                        name=name,
                        times_drafted=times_drafted,
                        share=times_drafted / total,
                    )
                )
            result[position] = entries
        return result

    def _build_model_adp_file(
        self,
        model_adp_tally: Dict[int, _ModelAdpAccumulator],
        player_catalog: PlayerCatalog,
    ) -> ModelAdpFile:
        """Build the league-wide model-derived ADP export.

        Parameters
        ----------
        model_adp_tally : Dict[int, _ModelAdpAccumulator]
            Overall pick number statistics keyed by player id.
        player_catalog : PlayerCatalog
            Catalog used to resolve player names, positions, and market ADP.

        Returns
        -------
        ModelAdpFile
            Players ranked by mean overall pick number across all
            simulations and slots.
        """
        entries: List[ModelAdpEntry] = []
        for player_id, accumulator in model_adp_tally.items():
            player = player_catalog.get(player_id)
            if player is None:
                continue
            entries.append(
                ModelAdpEntry(
                    player_id=player_id,
                    name=player.name,
                    position=player.position,
                    model_adp=accumulator.mean,
                    std_dev=accumulator.std_dev,
                    times_drafted=accumulator.count,
                    draft_rate=accumulator.count / self._simulations,
                    market_adp=player.adp if math.isfinite(player.adp) else None,
                )
            )
        entries.sort(key=lambda entry: entry.model_adp)
        return ModelAdpFile(
            generated_at=datetime.now(timezone.utc),
            draft_year=self._draft_year,
            num_teams=self._num_teams,
            simulations=self._simulations,
            checkpoint_path=self._checkpoint_path,
            checkpoint_episode=extract_checkpoint_episode(self._checkpoint_path),
            player_data_csv=self._player_data_csv,
            temperature=self._temperature,
            players=entries,
        )
