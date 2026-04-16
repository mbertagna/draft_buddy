"""Reward calculation for RL drafting loops."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from draft_buddy.core.roster_utils import calculate_roster_scores
from draft_buddy.core.stacking import calculate_stack_count
from draft_buddy.simulator.evaluator import simulate_season_fast


class RewardCalculator:
    """Calculate incremental and terminal draft rewards."""

    @staticmethod
    def calculate_step_reward(
        config, env, drafted_player, prev_starter_points: float
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate one-step reward and diagnostics.

        Parameters
        ----------
        config : Config
            Runtime reward settings.
        env : DraftGymEnv
            Active environment instance.
        drafted_player : Player
            Player selected this step.
        prev_starter_points : float
            Starter points before the current pick.

        Returns
        -------
        Tuple[float, Dict[str, Any]]
            Step reward and metadata.
        """
        reward = 0.0
        info: Dict[str, Any] = {}
        current_roster = env.resolve_roster_players(env.agent_team_id)
        if config.reward.ENABLE_INTERMEDIATE_REWARD:
            if config.reward.INTERMEDIATE_REWARD_MODE == "STATIC":
                reward += config.reward.INTERMEDIATE_REWARD_VALUE
            elif config.reward.INTERMEDIATE_REWARD_MODE == "PROPORTIONAL":
                scaling = getattr(config.reward, "PROPORTIONAL_REWARD_SCALING_FACTOR", 1.0)
                reward += drafted_player.projected_points * scaling

        if (
            config.reward.ENABLE_PICK_SHAPING_REWARD
            and config.reward.ENABLE_ROSTER_SLOT_WEIGHTED_REWARD
        ):
            current_scores = calculate_roster_scores(
                current_roster, config.draft.ROSTER_STRUCTURE, config.draft.BENCH_MAXES
            )
            current_starter_points = current_scores["starters_total_points"]
            delta = current_starter_points - prev_starter_points
            if delta > 0:
                reward += delta * config.reward.PICK_SHAPING_STARTER_DELTA_WEIGHT
                info["pick_shaping_delta"] = delta

        if config.reward.ENABLE_VORP_PICK_SHAPING:
            vorp = env._calculate_vorp(drafted_player.position)
            reward += vorp * config.reward.VORP_PICK_SHAPING_WEIGHT
            info["vorp_shaping"] = vorp

        if config.reward.ENABLE_STACKING_REWARD:
            post_stack_count = calculate_stack_count(current_roster)
            pre_stack_count = calculate_stack_count(current_roster[:-1])
            stack_increase = post_stack_count - pre_stack_count
            if stack_increase > 0:
                stacking_reward = stack_increase * config.reward.STACKING_REWARD_WEIGHT
                reward += stacking_reward
                info["stacking_reward"] = stacking_reward
        return reward, info

    @staticmethod
    def calculate_final_reward(config, env, matchups_df: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
        """Calculate terminal reward when an episode ends."""
        reward = 0.0
        info: Dict[str, Any] = {}
        agent_team_id = env.agent_team_id
        agent_roster = env.resolve_roster_players(agent_team_id)
        agent_scores = calculate_roster_scores(
            agent_roster, config.draft.ROSTER_STRUCTURE, config.draft.BENCH_MAXES
        )
        if config.reward.ENABLE_ROSTER_SLOT_WEIGHTED_REWARD:
            agent_final_weighted_score = (
                agent_scores["starters_total_points"] * config.reward.STARTER_POINTS_WEIGHT
            ) + (agent_scores["bench_total_points"] * config.reward.BENCH_POINTS_WEIGHT)
            info["raw_starter_points"] = agent_scores["starters_total_points"]
            info["raw_bench_points"] = agent_scores["bench_total_points"]
        else:
            agent_final_weighted_score = agent_scores["combined_total_points"]
        info["final_score_agent"] = agent_final_weighted_score

        if len(agent_roster) >= env.total_roster_size_per_team and config.reward.BONUS_FOR_FULL_ROSTER > 0:
            reward += config.reward.BONUS_FOR_FULL_ROSTER

        if getattr(config.reward, "ENABLE_FINAL_BASE_REWARD", True):
            reward += agent_final_weighted_score
            info["base_reward_component"] = agent_final_weighted_score

        if config.reward.ENABLE_COMPETITIVE_REWARD:
            opponent_scores = []
            for team_id in env.team_rosters:
                if team_id == agent_team_id:
                    continue
                opponent_roster = env.resolve_roster_players(team_id)
                opponent = calculate_roster_scores(
                    opponent_roster,
                    config.draft.ROSTER_STRUCTURE,
                    config.draft.BENCH_MAXES,
                )
                if config.reward.ENABLE_ROSTER_SLOT_WEIGHTED_REWARD:
                    opponent_score = (
                        opponent["starters_total_points"] * config.reward.STARTER_POINTS_WEIGHT
                    ) + (opponent["bench_total_points"] * config.reward.BENCH_POINTS_WEIGHT)
                else:
                    opponent_score = opponent["combined_total_points"]
                opponent_scores.append(opponent_score)

            if opponent_scores:
                opponent_scores_np = np.array(opponent_scores)
                comp_component = 0.0
                if config.reward.COMPETITIVE_REWARD_MODE == "MAX_OPPONENT_DIFFERENCE":
                    max_opp = np.max(opponent_scores_np)
                    comp_component = agent_final_weighted_score - max_opp
                    info["competitive_mode"] = "Max Opponent Difference"
                    info["target_opponent_score"] = max_opp
                elif config.reward.COMPETITIVE_REWARD_MODE == "AVG_OPPONENT_DIFFERENCE":
                    avg_opp = np.mean(opponent_scores_np)
                    comp_component = agent_final_weighted_score - avg_opp
                    info["competitive_mode"] = "Average Opponent Difference"
                    info["target_opponent_score"] = avg_opp
                if config.reward.COMPETITIVE_REWARD_MODE != "SEASON_SIM":
                    reward += comp_component
                    info["competitive_reward_component"] = comp_component
                if config.reward.ENABLE_OPPONENT_STD_DEV_PENALTY and len(opponent_scores_np) > 1:
                    std_penalty = np.std(opponent_scores_np) * config.reward.OPPONENT_STD_DEV_PENALTY_WEIGHT
                    reward -= std_penalty
                    info["opponent_std_dev_penalty"] = std_penalty

        if (
            config.reward.COMPETITIVE_REWARD_MODE == "SEASON_SIM"
            or config.reward.ENABLE_SEASON_SIM_REWARD
        ) and matchups_df is not None and not matchups_df.empty:
            sim_rosters = {
                config.draft.TEAM_MANAGER_MAPPING.get(team_id): list(team_roster.player_ids)
                for team_id, team_roster in env.team_rosters.items()
                if config.draft.TEAM_MANAGER_MAPPING.get(team_id)
            }
            try:
                num_playoff_teams = int(config.reward.REGULAR_SEASON_REWARD.get("NUM_PLAYOFF_TEAMS", 6))
                _, regular_records, playoff_results_df, _, winner = simulate_season_fast(
                    env.weekly_projections,
                    matchups_df,
                    sim_rosters,
                    2025,
                    "",
                    False,
                    num_playoff_teams,
                )
                agent_manager_name = config.draft.TEAM_MANAGER_MAPPING.get(agent_team_id)
                sim_reward = 0.0
                seed_reward, made_playoffs, seed_value = RewardCalculator.compute_regular_season_reward(
                    config, regular_records, agent_manager_name
                )
                sim_reward += seed_reward
                info["made_playoffs"] = bool(made_playoffs)
                if made_playoffs:
                    info["playoff_seed"] = int(seed_value)
                    info["regular_season_seed_reward"] = float(seed_reward)
                placement_reward, placement_label = RewardCalculator.compute_playoff_placement_reward(
                    config, regular_records, playoff_results_df, winner, agent_manager_name
                )
                sim_reward += placement_reward
                info["playoff_placement"] = placement_label
                info["playoff_placement_reward"] = float(placement_reward)
                reward += sim_reward
                info["season_sim_reward"] = sim_reward
            except Exception as error:
                print(f"Error during season simulation in RewardCalculator: {error}")
        return reward, info

    @staticmethod
    def compute_regular_season_reward(
        config, regular_records: List[Tuple], agent_manager_name: str
    ) -> Tuple[float, bool, Optional[int]]:
        """Compute regular season playoff-seeding reward."""
        reward_config = config.reward.REGULAR_SEASON_REWARD
        if not reward_config:
            return 0.0, False, None
        num_playoff_teams = int(reward_config.get("NUM_PLAYOFF_TEAMS", 6))
        seeding_list = [record[0] for record in regular_records[:num_playoff_teams]]
        if agent_manager_name not in seeding_list:
            return 0.0, False, None
        seed_index = seeding_list.index(agent_manager_name)
        seed = seed_index + 1
        total_reward = float(reward_config.get("MAKE_PLAYOFFS_BONUS", 0.0))
        mode = reward_config.get("SEED_REWARD_MODE", "LINEAR").upper()
        if mode == "MAPPING":
            mapping = reward_config.get("SEED_REWARD_MAPPING", {})
            seed_component = float(mapping.get(seed, 0.0))
        else:
            seed_max = float(reward_config.get("SEED_REWARD_MAX", 0.0))
            seed_min = float(reward_config.get("SEED_REWARD_MIN", 0.0))
            if num_playoff_teams <= 1:
                seed_component = seed_max
            else:
                interpolation = (seed - 1) / (num_playoff_teams - 1)
                seed_component = seed_max + (seed_min - seed_max) * interpolation
        total_reward += seed_component
        return total_reward, True, seed

    @staticmethod
    def compute_playoff_placement_reward(
        config,
        regular_records: List[Tuple],
        playoff_results_df: pd.DataFrame,
        winner: str,
        agent_manager_name: str,
    ) -> Tuple[float, str]:
        """Compute playoff placement reward and label."""
        reward_config = config.reward.PLAYOFF_PLACEMENT_REWARDS
        if not reward_config:
            return 0.0, "NON_PLAYOFF"
        num_playoff_teams = int(config.reward.REGULAR_SEASON_REWARD.get("NUM_PLAYOFF_TEAMS", 6))
        playoff_teams = [record[0] for record in regular_records[:num_playoff_teams]]
        if agent_manager_name not in playoff_teams:
            return float(reward_config.get("NON_PLAYOFF", 0.0)), "NON_PLAYOFF"
        if winner == agent_manager_name:
            return float(reward_config.get("CHAMPION", 0.0)), "CHAMPION"
        try:
            last_row = playoff_results_df.iloc[-1]
            finalist = (
                last_row["Home Manager(s)"]
                if last_row["Home Manager(s)"] != winner
                else last_row["Away Manager(s)"]
            )
        except Exception:
            finalist = None
        if finalist == agent_manager_name:
            return float(reward_config.get("RUNNER_UP", 0.0)), "RUNNER_UP"
        try:
            appears_mask = (
                playoff_results_df["Home Manager(s)"] == agent_manager_name
            ) | (playoff_results_df["Away Manager(s)"] == agent_manager_name)
            appearances = playoff_results_df[appears_mask]
            if appearances.empty:
                return float(reward_config.get("NON_PLAYOFF", 0.0)), "NON_PLAYOFF"
            last_week = appearances["Week"].max()
            max_week = playoff_results_df["Week"].max()
            if last_week == max_week:
                return float(reward_config.get("RUNNER_UP", 0.0)), "RUNNER_UP"
            if last_week == max_week - 1:
                return float(reward_config.get("SEMIFINALIST", 0.0)), "SEMIFINALIST"
            return float(reward_config.get("QUARTERFINALIST", 0.0)), "QUARTERFINALIST"
        except Exception:
            return float(reward_config.get("QUARTERFINALIST", 0.0)), "QUARTERFINALIST"
