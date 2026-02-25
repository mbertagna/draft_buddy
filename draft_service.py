"""
Draft session management for the web layer.

Handles draft instances per session, enabling concurrent draft sessions
without global mutable state.
"""

import threading
from typing import Dict, Optional

from config import Config
from fantasy_draft_env import FantasyFootballDraftEnv


class DraftService:
    """
    Manages draft environment instances per session.

    Routes handlers retrieve draft state based on session or user ID,
    allowing concurrent draft sessions without race conditions.
    """

    def __init__(self, config: Config):
        """
        Initialize the draft service.

        Parameters
        ----------
        config : Config
            Application configuration.
        """
        self._config = config
        self._drafts: Dict[str, FantasyFootballDraftEnv] = {}
        self._lock = threading.Lock()

    def get_or_create_draft(self, session_id: str) -> FantasyFootballDraftEnv:
        """
        Returns the draft environment for the session, creating one if needed.

        Parameters
        ----------
        session_id : str
            Unique identifier for the draft session.

        Returns
        -------
        FantasyFootballDraftEnv
            The draft environment for this session.
        """
        with self._lock:
            if session_id not in self._drafts:
                env = FantasyFootballDraftEnv(self._config, training=False)
                env.load_state(self._config.DRAFT_STATE_FILE)
                if not env.draft_order:
                    env.reset()
                    env.save_state(self._config.DRAFT_STATE_FILE)
                self._drafts[session_id] = env
            return self._drafts[session_id]

    def get_draft(self, session_id: str) -> Optional[FantasyFootballDraftEnv]:
        """
        Returns the draft for the session if it exists.

        Parameters
        ----------
        session_id : str
            Session identifier.

        Returns
        -------
        FantasyFootballDraftEnv or None
            The draft environment, or None if not found.
        """
        with self._lock:
            return self._drafts.get(session_id)

    def create_new_draft(self, session_id: str) -> FantasyFootballDraftEnv:
        """
        Archives the current draft (if any) and creates a fresh one.

        Parameters
        ----------
        session_id : str
            Session identifier.

        Returns
        -------
        FantasyFootballDraftEnv
            The new draft environment.
        """
        import datetime
        import os

        with self._lock:
            existing = self._drafts.get(session_id)
            if existing and existing.current_pick_number > 1:
                saved_states_dir = "saved_states"
                os.makedirs(saved_states_dir, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                archive_path = os.path.join(saved_states_dir, f"draft_state_{timestamp}.json")
                existing.save_state(archive_path)
                print(f"Saved current draft state to {archive_path}")

            env = FantasyFootballDraftEnv(self._config, training=False)
            env.reset()
            env.save_state(self._config.DRAFT_STATE_FILE)
            self._drafts[session_id] = env
            return env
