"""Reinforcement-learning package boundaries."""

from __future__ import annotations

__all__ = [
    "AgentModelBotGM",
    "CheckpointManager",
    "DraftGymEnv",
    "FeatureExtractor",
    "MetricsLogger",
    "PolicyNetwork",
    "ReinforceAgent",
    "RewardCalculator",
    "find_latest_checkpoint",
    "get_run_name",
    "save_run_metadata",
    "setup_run_directories",
]


def __getattr__(name: str):
    """Lazily expose RL symbols to avoid import cycles."""
    if name == "DraftGymEnv":
        from draft_buddy.rl.draft_gym_env import DraftGymEnv as draft_gym_env

        return draft_gym_env
    if name == "PolicyNetwork":
        from draft_buddy.rl.policy_network import PolicyNetwork as policy_network

        return policy_network
    if name == "ReinforceAgent":
        from draft_buddy.rl.reinforce_agent import ReinforceAgent as reinforce_agent

        return reinforce_agent
    if name == "CheckpointManager":
        from draft_buddy.rl.checkpoint_manager import CheckpointManager as checkpoint_manager

        return checkpoint_manager
    if name == "MetricsLogger":
        from draft_buddy.rl.metrics_logger import MetricsLogger as metrics_logger

        return metrics_logger
    if name == "find_latest_checkpoint":
        from draft_buddy.rl.run_utils import find_latest_checkpoint as find_latest_checkpoint_func

        return find_latest_checkpoint_func
    if name == "get_run_name":
        from draft_buddy.rl.run_utils import get_run_name as get_run_name_func

        return get_run_name_func
    if name == "save_run_metadata":
        from draft_buddy.rl.run_utils import save_run_metadata as save_run_metadata_func

        return save_run_metadata_func
    if name == "setup_run_directories":
        from draft_buddy.rl.run_utils import setup_run_directories as setup_run_directories_func

        return setup_run_directories_func
    if name == "FeatureExtractor":
        from draft_buddy.rl.feature_extractor import FeatureExtractor as feature_extractor

        return feature_extractor
    if name == "RewardCalculator":
        from draft_buddy.rl.reward_calculator import RewardCalculator as reward_calculator

        return reward_calculator
    if name == "AgentModelBotGM":
        from draft_buddy.rl.agent_bot import AgentModelBotGM as agent_model_bot

        return agent_model_bot
    raise AttributeError(f"module 'draft_buddy.rl' has no attribute {name!r}")
