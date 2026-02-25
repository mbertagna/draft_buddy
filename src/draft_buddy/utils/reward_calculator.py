import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from draft_buddy.utils.season_simulation_fast import simulate_season_fast
from draft_buddy.utils.data_utils import calculate_stack_count

class RewardCalculator:
    """
    Calculates rewards for the fantasy football draft environment,
    including intermediate shaping rewards and end-of-episode simulation rewards.
    """

    @staticmethod
    def calculate_step_reward(config, env, drafted_player, prev_starter_points: float) -> Tuple[float, Dict[str, Any]]:
        """
        Calculates reward for a single action/step.
        """
        reward = 0.0
        info = {}

        # 1. Intermediate Reward (Static or Proportional)
        if config.reward.ENABLE_INTERMEDIATE_REWARD:
            if config.reward.INTERMEDIATE_REWARD_MODE == 'STATIC':
                reward += config.reward.INTERMEDIATE_REWARD_VALUE
            elif config.reward.INTERMEDIATE_REWARD_MODE == 'PROPORTIONAL':
                # Note: PROPORTIONAL_REWARD_SCALING_FACTOR was suggested for removal if unused, 
                # but we'll keep it here if it's in the dataclass.
                scaling = getattr(config.reward, 'PROPORTIONAL_REWARD_SCALING_FACTOR', 1.0)
                reward += drafted_player.projected_points * scaling

        # 2. Pick Shaping (Starter Delta)
        if config.reward.ENABLE_PICK_SHAPING_REWARD and config.reward.ENABLE_ROSTER_SLOT_WEIGHTED_REWARD:
            from draft_buddy.utils.roster_utils import calculate_roster_scores
            curr_roster = env.teams_rosters[env.agent_team_id]['PLAYERS']
            curr_scores = calculate_roster_scores(
                curr_roster, config.draft.ROSTER_STRUCTURE, config.draft.BENCH_MAXES
            )
            curr_starter_points = curr_scores['starters_total_points']
            delta = curr_starter_points - prev_starter_points
            if delta > 0:
                reward += delta * config.reward.PICK_SHAPING_STARTER_DELTA_WEIGHT
                info['pick_shaping_delta'] = delta

        # 3. VORP Shaping
        if config.reward.ENABLE_VORP_PICK_SHAPING:
            vorp = env._calculate_vorp(drafted_player.position)
            reward += vorp * config.reward.VORP_PICK_SHAPING_WEIGHT
            info['vorp_shaping'] = vorp

        # 4. Stacking Reward
        if config.reward.ENABLE_STACKING_REWARD:
            roster = env.teams_rosters[env.agent_team_id]['PLAYERS']
            post_stack_count = calculate_stack_count(roster)
            # Calculate count BEFORE this pick to get delta
            pre_stack_count = calculate_stack_count(roster[:-1])
            stack_increase = post_stack_count - pre_stack_count
            if stack_increase > 0:
                stacking_reward = stack_increase * config.reward.STACKING_REWARD_WEIGHT
                reward += stacking_reward
                info['stacking_reward'] = stacking_reward
        
        return reward, info

    @staticmethod
    def calculate_final_reward(config, env, matchups_df: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
        """
        Calculates the terminal reward at the end of a draft episode.
        """
        reward = 0.0
        info = {}
        agent_team_id = env.agent_team_id
        from draft_buddy.utils.roster_utils import calculate_roster_scores

        # --- 1. Agent Weighted Score ---
        agent_roster = env.teams_rosters[agent_team_id]['PLAYERS']
        agent_scores = calculate_roster_scores(
            agent_roster, config.draft.ROSTER_STRUCTURE, config.draft.BENCH_MAXES
        )
        
        if config.reward.ENABLE_ROSTER_SLOT_WEIGHTED_REWARD:
            agent_final_weighted_score = (agent_scores['starters_total_points'] * config.reward.STARTER_POINTS_WEIGHT) + \
                                         (agent_scores['bench_total_points'] * config.reward.BENCH_POINTS_WEIGHT)
            info['raw_starter_points'] = agent_scores['starters_total_points']
            info['raw_bench_points'] = agent_scores['bench_total_points']
        else:
            agent_final_weighted_score = agent_scores['combined_total_points']
        
        info['final_score_agent'] = agent_final_weighted_score

        # --- 2. Bonus for Full Roster ---
        if len(agent_roster) >= env.total_roster_size_per_team and config.reward.BONUS_FOR_FULL_ROSTER > 0:
            reward += config.reward.BONUS_FOR_FULL_ROSTER

        # --- 3. Base Reward Component ---
        if getattr(config.reward, 'ENABLE_FINAL_BASE_REWARD', True):
            reward += agent_final_weighted_score
            info['base_reward_component'] = agent_final_weighted_score

        # --- 4. Competitive Reward (Opponent Difference) ---
        if config.reward.ENABLE_COMPETITIVE_REWARD:
            opponent_scores = []
            for team_id, roster_data in env.teams_rosters.items():
                if team_id == agent_team_id:
                    continue
                
                opp_scores = calculate_roster_scores(
                    roster_data['PLAYERS'], config.draft.ROSTER_STRUCTURE, config.draft.BENCH_MAXES
                )
                if config.reward.ENABLE_ROSTER_SLOT_WEIGHTED_REWARD:
                    opp_score = (opp_scores['starters_total_points'] * config.reward.STARTER_POINTS_WEIGHT) + \
                                (opp_scores['bench_total_points'] * config.reward.BENCH_POINTS_WEIGHT)
                else:
                    opp_score = opp_scores['combined_total_points']
                opponent_scores.append(opp_score)

            if opponent_scores:
                opponent_scores_np = np.array(opponent_scores)
                comp_component = 0.0

                if config.reward.COMPETITIVE_REWARD_MODE == 'MAX_OPPONENT_DIFFERENCE':
                    max_opp = np.max(opponent_scores_np)
                    comp_component = agent_final_weighted_score - max_opp
                    info['competitive_mode'] = 'Max Opponent Difference'
                    info['target_opponent_score'] = max_opp
                elif config.reward.COMPETITIVE_REWARD_MODE == 'AVG_OPPONENT_DIFFERENCE':
                    avg_opp = np.mean(opponent_scores_np)
                    comp_component = agent_final_weighted_score - avg_opp
                    info['competitive_mode'] = 'Average Opponent Difference'
                    info['target_opponent_score'] = avg_opp
                
                if config.reward.COMPETITIVE_REWARD_MODE != 'SEASON_SIM':
                    reward += comp_component
                    info['competitive_reward_component'] = comp_component

                if config.reward.ENABLE_OPPONENT_STD_DEV_PENALTY and len(opponent_scores_np) > 1:
                    std_penalty = np.std(opponent_scores_np) * config.reward.OPPONENT_STD_DEV_PENALTY_WEIGHT
                    reward -= std_penalty
                    info['opponent_std_dev_penalty'] = std_penalty

        # --- 5. Season Simulation Reward ---
        if (config.reward.COMPETITIVE_REWARD_MODE == 'SEASON_SIM' or config.reward.ENABLE_SEASON_SIM_REWARD) and \
           matchups_df is not None and not matchups_df.empty:
            
            sim_rosters = {
                config.draft.TEAM_MANAGER_MAPPING.get(tid): [p.player_id for p in r['PLAYERS']] 
                for tid, r in env.teams_rosters.items() if config.draft.TEAM_MANAGER_MAPPING.get(tid)
            }
            
            try:
                num_playoff_teams = int(config.reward.REGULAR_SEASON_REWARD.get('NUM_PLAYOFF_TEAMS', 6))
                _, regular_records, playoff_results_df, _, winner = simulate_season_fast(
                    env.weekly_projections, matchups_df, sim_rosters, 2025, "", False, num_playoff_teams
                )
                
                agent_manager_name = config.draft.TEAM_MANAGER_MAPPING.get(agent_team_id)
                sim_reward = 0.0

                # Regular Season Component
                seed_r, made_p, seed_v = RewardCalculator.compute_regular_season_reward(config, regular_records, agent_manager_name)
                sim_reward += seed_r
                info['made_playoffs'] = bool(made_p)
                if made_p:
                    info['playoff_seed'] = int(seed_v)
                    info['regular_season_seed_reward'] = float(seed_r)

                # Playoff Placement Component
                place_r, place_l = RewardCalculator.compute_playoff_placement_reward(
                    config, regular_records, playoff_results_df, winner, agent_manager_name
                )
                sim_reward += place_r
                info['playoff_placement'] = place_l
                info['playoff_placement_reward'] = float(place_r)
                
                reward += sim_reward
                info['season_sim_reward'] = sim_reward
            except Exception as e:
                print(f"Error during season simulation in RewardCalculator: {e}")

        return reward, info

    @staticmethod
    def compute_regular_season_reward(config, regular_records: List[Tuple], agent_manager_name: str) -> Tuple[float, bool, int]:
        """
        Returns (seed_reward, made_playoffs_flag, seed) based on config.REGULAR_SEASON_REWARD.
        """
        cfg = config.reward.REGULAR_SEASON_REWARD
        if not cfg:
            return 0.0, False, None

        num_playoff_teams = int(cfg.get('NUM_PLAYOFF_TEAMS', 6))
        seeding_list = [record[0] for record in regular_records[:num_playoff_teams]]

        if agent_manager_name not in seeding_list:
            return 0.0, False, None

        seed_index = seeding_list.index(agent_manager_name)
        seed = seed_index + 1
        total_reward = float(cfg.get('MAKE_PLAYOFFS_BONUS', 0.0))

        mode = cfg.get('SEED_REWARD_MODE', 'LINEAR').upper()
        if mode == 'MAPPING':
            mapping = cfg.get('SEED_REWARD_MAPPING', {})
            seed_component = float(mapping.get(seed, 0.0))
        else:
            seed_max = float(cfg.get('SEED_REWARD_MAX', 0.0))
            seed_min = float(cfg.get('SEED_REWARD_MIN', 0.0))
            if num_playoff_teams <= 1:
                seed_component = seed_max
            else:
                t = (seed - 1) / (num_playoff_teams - 1)
                seed_component = seed_max + (seed_min - seed_max) * t

        total_reward += seed_component
        return total_reward, True, seed

    @staticmethod
    def compute_playoff_placement_reward(config, regular_records: List[Tuple], playoff_results_df: pd.DataFrame, winner: str, agent_manager_name: str) -> Tuple[float, str]:
        """
        Returns (placement_reward, placement_label) for playoff placement.
        """
        cfg = config.reward.PLAYOFF_PLACEMENT_REWARDS
        if not cfg:
            return 0.0, 'NON_PLAYOFF'

        num_playoff_teams = int(config.reward.REGULAR_SEASON_REWARD.get('NUM_PLAYOFF_TEAMS', 6))
        playoff_teams = [record[0] for record in regular_records[:num_playoff_teams]]
        if agent_manager_name not in playoff_teams:
            return float(cfg.get('NON_PLAYOFF', 0.0)), 'NON_PLAYOFF'

        if winner == agent_manager_name:
            return float(cfg.get('CHAMPION', 0.0)), 'CHAMPION'

        try:
            last_row = playoff_results_df.iloc[-1]
            finalist = last_row['Home Manager(s)'] if last_row['Home Manager(s)'] != winner else last_row['Away Manager(s)']
        except Exception:
            finalist = None

        if finalist == agent_manager_name:
            return float(cfg.get('RUNNER_UP', 0.0)), 'RUNNER_UP'

        try:
            appears_mask = (playoff_results_df['Home Manager(s)'] == agent_manager_name) | (playoff_results_df['Away Manager(s)'] == agent_manager_name)
            appearances = playoff_results_df[appears_mask]
            if appearances.empty:
                return float(cfg.get('NON_PLAYOFF', 0.0)), 'NON_PLAYOFF'

            last_week = appearances['Week'].max()
            max_week = playoff_results_df['Week'].max()

            if last_week == max_week:
                return float(cfg.get('RUNNER_UP', 0.0)), 'RUNNER_UP'
            elif last_week == max_week - 1:
                return float(cfg.get('SEMIFINALIST', 0.0)), 'SEMIFINALIST'
            else:
                return float(cfg.get('QUARTERFINALIST', 0.0)), 'QUARTERFINALIST'
        except Exception:
            return float(cfg.get('QUARTERFINALIST', 0.0)), 'QUARTERFINALIST'
