"""Run the Draft Buddy FastAPI web application."""

import os
from typing import Any, Dict, Optional

import numpy as np
import torch
import uvicorn

from draft_buddy.config import Config, load_runtime_config
from draft_buddy.core import BotGM, DraftState, InferenceProvider, PlayerCatalog
from draft_buddy.data.insights.loader import load_latest_player_insights
from draft_buddy.rl.agent_bot import AgentModelBotGM
from draft_buddy.rl.checkpoint_manager import CheckpointManager
from draft_buddy.rl.feature_extractor import FeatureExtractor
from draft_buddy.rl.policy_network import PolicyNetwork
from draft_buddy.rl.state_normalizer import StateNormalizer
from draft_buddy.web.advisor_factory import AdvisorGatewayRegistry, build_advisor_registry
from draft_buddy.web.app import create_app
from draft_buddy.web.draft_advisor_service import DraftAdvisorService
from draft_buddy.web.session import DraftSessionManager


class RlInferenceProvider(InferenceProvider):
    """RL-backed inference provider composed at application startup."""

    def __init__(self, config: Config, action_space_size: int = 4) -> None:
        self._config = config
        self._action_space_size = action_space_size
        self._suggestion_temperature = config.training.POLICY_SUGGESTION_TEMPERATURE
        self._feature_extractor = FeatureExtractor(config, StateNormalizer(config))
        self._suggestion_model = self._load_policy_model(config.training.MODEL_PATH_TO_LOAD)
        self._opponent_models = self._load_opponent_models()

    def create_bot(
        self, team_id: int, strategy_config: Dict[str, Any], action_to_position: Dict[int, str]
    ) -> BotGM | None:
        """Create an AGENT_MODEL bot when a model is available.

        Parameters
        ----------
        team_id : int
            Team receiving the strategy.
        strategy_config : Dict[str, Any]
            Strategy configuration for the team.
        action_to_position : Dict[int, str]
            Action-index to position mapping.

        Returns
        -------
        BotGM | None
            Model-backed bot instance or ``None``.
        """
        if strategy_config.get("logic") != "AGENT_MODEL":
            return None
        model = self._opponent_models.get(team_id)
        if model is None:
            return None
        return AgentModelBotGM(model, action_to_position)

    def create_policy_sim_bot(self, action_to_position: Dict[int, str]) -> BotGM | None:
        """Return a bot backed by the loaded suggestion policy checkpoint.

        Parameters
        ----------
        action_to_position : Dict[int, str]
            Action-index to position mapping.

        Returns
        -------
        BotGM | None
            Policy-backed bot, or ``None`` when the suggestion model is unavailable.
        """
        if self._suggestion_model is None:
            return None
        return AgentModelBotGM(self._suggestion_model, action_to_position)

    def build_state_vector(
        self, team_id: int, draft_state: DraftState, player_catalog: PlayerCatalog
    ) -> np.ndarray:
        """Build normalized feature vector for team perspective.

        Parameters
        ----------
        team_id : int
            Team perspective for feature extraction.
        draft_state : DraftState
            Current draft state.
        player_catalog : PlayerCatalog
            Shared player catalog.

        Returns
        -------
        np.ndarray
            Normalized feature vector.
        """
        return self._feature_extractor.extract(draft_state, player_catalog, team_id)

    def predict_action_probabilities(
        self,
        team_id: int,
        draft_state: DraftState,
        player_catalog: PlayerCatalog,
        action_to_position: Dict[int, str],
        get_action_mask_fn,
    ) -> Dict[str, float]:
        """Return position probability distribution from the suggestion model.

        Parameters
        ----------
        team_id : int
            Team perspective for prediction.
        draft_state : DraftState
            Current draft state.
        player_catalog : PlayerCatalog
            Shared player catalog.
        action_to_position : Dict[int, str]
            Action-index to position mapping.
        get_action_mask_fn : callable
            Callback returning a valid-action mask for a team id.

        Returns
        -------
        Dict[str, float]
            Position probability map.
        """
        if self._suggestion_model is None:
            raise ValueError("AI model not loaded.")
        state = self.build_state_vector(team_id, draft_state, player_catalog)
        action_mask = get_action_mask_fn(team_id)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_probs_tensor = self._suggestion_model.get_action_probabilities(
                state_tensor,
                action_mask=action_mask,
                temperature=self._suggestion_temperature,
            )
            action_probs = action_probs_tensor.squeeze().tolist()
        return {action_to_position[index]: float(prob) for index, prob in enumerate(action_probs)}

    def _load_opponent_models(self) -> Dict[int, PolicyNetwork]:
        """Load configured opponent models keyed by team id.

        Returns
        -------
        Dict[int, PolicyNetwork]
            Team-indexed loaded policy networks.
        """
        models: Dict[int, PolicyNetwork] = {}
        for team_id in range(1, self._config.draft.NUM_TEAMS + 1):
            if team_id == self._config.draft.AGENT_START_POSITION:
                continue
            strategy = self._config.opponent.OPPONENT_TEAM_STRATEGIES.get(
                team_id, self._config.opponent.DEFAULT_OPPONENT_STRATEGY
            )
            if strategy.get("logic") != "AGENT_MODEL":
                continue
            model_path_key = strategy.get("model_path_key")
            model_path = self._config.opponent.OPPONENT_MODEL_PATHS.get(model_path_key, "")
            if not model_path:
                continue
            model = self._load_policy_model(model_path)
            if model is not None:
                models[team_id] = model
        return models

    def _load_policy_model(self, model_path: str) -> Optional[PolicyNetwork]:
        """Load one policy model checkpoint with compatibility checks.

        Parameters
        ----------
        model_path : str
            Filesystem path to a checkpoint file.

        Returns
        -------
        Optional[PolicyNetwork]
            Loaded model in eval mode, or ``None`` on failure.
        """
        if not model_path or not os.path.exists(model_path):
            return None
        input_dim = len(self._config.training.ENABLED_STATE_FEATURES)
        model = PolicyNetwork(input_dim, self._action_space_size, self._config.training.HIDDEN_DIM)
        checkpoint_manager = CheckpointManager(model, value_network=None, optimizer=None)
        try:
            checkpoint_manager.load_checkpoint(model_path, self._config, is_training=False)
            model.eval()
            return model
        except Exception:
            return None


def main() -> None:
    """Start the web application server."""
    config = load_runtime_config()
    inference_provider = RlInferenceProvider(config)
    session_manager = DraftSessionManager(config, inference_provider=inference_provider)
    loaded_insights = load_latest_player_insights(config.paths.DATA_DIR)
    advisor_registry = build_advisor_registry()
    advisor_service = DraftAdvisorService(advisor_registry) if advisor_registry else None
    if advisor_registry is None:
        print(
            "Draft assistant disabled: set GEMINI_API_KEY and/or OPENROUTER_API_KEY in .env."
        )
    app = create_app(
        config=config,
        session_manager=session_manager,
        loaded_insights=loaded_insights,
        advisor_service=advisor_service,
        advisor_registry=advisor_registry,
    )
    port = int(os.environ.get("PORT", 5001))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
