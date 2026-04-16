"""Inference provider abstraction for composition-root model integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

import numpy as np

from draft_buddy.core.bot_gm import BotGM
from draft_buddy.core.draft_state import DraftState
from draft_buddy.core.entities import PlayerCatalog


class InferenceProvider(ABC):
    """Abstract inference integration for model-backed draft behavior.

    Implementations live in outer layers and may use framework-specific tools
    such as PyTorch. Core and web layers depend only on this abstraction.
    """

    @abstractmethod
    def create_bot(
        self, team_id: int, strategy_config: Dict[str, Any], action_to_position: Dict[int, str]
    ) -> BotGM | None:
        """Return a model-backed bot for a team when available.

        Parameters
        ----------
        team_id : int
            Team receiving the strategy.
        strategy_config : Dict[str, Any]
            Team strategy configuration.
        action_to_position : Dict[int, str]
            Action-index to position mapping.

        Returns
        -------
        BotGM | None
            Bot implementation or ``None`` when unavailable.
        """

    @abstractmethod
    def build_state_vector(
        self, team_id: int, draft_state: DraftState, player_catalog: PlayerCatalog
    ) -> np.ndarray:
        """Build normalized model features for a team perspective.

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
            Normalized feature vector for inference.
        """

    @abstractmethod
    def predict_action_probabilities(
        self,
        team_id: int,
        draft_state: DraftState,
        player_catalog: PlayerCatalog,
        action_to_position: Dict[int, str],
        get_action_mask_fn: Callable[[int], np.ndarray],
    ) -> Dict[str, float]:
        """Predict action probabilities for a team.

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
        get_action_mask_fn : Callable[[int], np.ndarray]
            Callback returning valid-action mask for a team.

        Returns
        -------
        Dict[str, float]
            Position-keyed probability values.
        """
