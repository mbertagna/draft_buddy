"""Read-only draft session that mirrors a live Sleeper draft."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from draft_buddy.config import Config
from draft_buddy.core import InferenceProvider
from draft_buddy.data.cache_paths import sleeper_cache_dir
from draft_buddy.data.sleeper_client import SleeperGateway, SleeperHttpGateway
from draft_buddy.data.sleeper_draft_sync import SleeperDraftSyncService
from draft_buddy.data.sleeper_player_resolver import PlayerResolver
from draft_buddy.web.session import DraftSession, SyncReadOnlyError

_READ_ONLY_MESSAGE = "Draft is in Sleeper sync mode"


class SleeperSyncedSession(DraftSession):
    """Draft Buddy companion session driven exclusively by Sleeper picks."""

    def __init__(
        self,
        config: Config,
        inference_provider: Optional[InferenceProvider] = None,
        sleeper_gateway: Optional[SleeperGateway] = None,
    ) -> None:
        validate_sleeper_sync_config(config)
        super().__init__(config, inference_provider=inference_provider)
        self._gateway = sleeper_gateway or SleeperHttpGateway(
            sleeper_cache_dir(config.paths.DATA_DIR)
        )
        self._sync_service = SleeperDraftSyncService(self._gateway)
        directory_df = self._safe_fetch_directory()
        self._resolver = PlayerResolver(self.player_catalog, directory_df)
        self._last_sleeper_fetch_at = 0.0
        self._last_synced_pick_no = 0
        self._sleeper_draft_status: Optional[str] = None
        self._sync_error: Optional[str] = None
        self._slot_to_roster_id: Dict[str, Any] = {}
        self._bootstrap_from_sleeper()

    def get_ui_state(self) -> Dict[str, Any]:
        """Return dashboard state plus Sleeper sync metadata."""
        payload = super().get_ui_state()
        payload["sleeper_sync"] = True
        payload["sleeper_draft_id"] = self._config.draft.SLEEPER_DRAFT_ID
        payload["sleeper_draft_status"] = self._sleeper_draft_status
        payload["last_synced_pick_no"] = self._last_synced_pick_no
        payload["visual_round_count"] = self._state.visual_round_count
        payload["sync_error"] = self._sync_error
        payload["sleeper_sync_poll_seconds"] = self._config.draft.SLEEPER_SYNC_POLL_SECONDS
        payload["display_picks_by_player_id"] = {
            str(player_id): self.player_catalog.require(player_id).to_dict()
            for player_id in self._state.display_only_player_ids
            if player_id in self.player_catalog
        }
        return payload

    def sync_from_sleeper(self, force: bool = False) -> None:
        """Fetch new Sleeper picks when the min interval has elapsed.

        Parameters
        ----------
        force : bool, optional
            When True, ignore the coalesce interval.
        """
        interval = max(int(self._config.draft.SLEEPER_SYNC_POLL_SECONDS), 0)
        now = time.monotonic()
        if not force and interval > 0 and (now - self._last_sleeper_fetch_at) < interval:
            return
        try:
            high_water, errors = self._sync_service.sync_new_picks(
                draft_id=self._config.draft.SLEEPER_DRAFT_ID,
                controller=self._controller,
                resolver=self._resolver,
                roster_id_to_team_id=self._config.draft.SLEEPER_ROSTER_ID_TO_TEAM_ID,
                last_synced_pick_no=self._last_synced_pick_no,
                slot_to_roster_id=self._slot_to_roster_id,
            )
            self.player_catalog = self._controller.player_catalog
            self._resolver.replace_catalog(self.player_catalog)
            self.weekly_projections = self.player_catalog.to_weekly_projections()
            self._last_synced_pick_no = high_water
            self._sync_error = "; ".join(errors) if errors else None
            if high_water >= len(self.draft_order):
                self._sleeper_draft_status = "complete"
            elif high_water > 0:
                self._sleeper_draft_status = "drafting"
            self._last_sleeper_fetch_at = now
        except Exception as error:
            self._sync_error = str(error)

    def reset(self) -> None:
        """Rebuild local state by replaying the Sleeper draft from pick 0."""
        self._bootstrap_from_sleeper()

    def draft_player(self, player_id: int) -> None:
        """Reject local picks while Sleeper is the source of truth."""
        _ = player_id
        raise SyncReadOnlyError(_READ_ONLY_MESSAGE)

    def undo_last_pick(self) -> None:
        """Reject undo while Sleeper is the source of truth."""
        raise SyncReadOnlyError(_READ_ONLY_MESSAGE)

    def transfer_player(
        self, player_id: int, to_team_id: int, to_round: Optional[int] = None
    ) -> None:
        """Reject transfers while Sleeper is the source of truth."""
        _ = (player_id, to_team_id, to_round)
        raise SyncReadOnlyError(_READ_ONLY_MESSAGE)

    def swap_players(self, player_id_1: int, player_id_2: int) -> None:
        """Reject swaps while Sleeper is the source of truth."""
        _ = (player_id_1, player_id_2)
        raise SyncReadOnlyError(_READ_ONLY_MESSAGE)

    def set_current_team_picking(self, team_id: int) -> None:
        """Reject clock overrides while Sleeper is the source of truth."""
        _ = team_id
        raise SyncReadOnlyError(_READ_ONLY_MESSAGE)

    def simulate_single_pick(self, use_policy: bool = False) -> None:
        """Reject simulated picks while Sleeper is the source of truth."""
        _ = use_policy
        raise SyncReadOnlyError(_READ_ONLY_MESSAGE)

    def simulate_scheduled_picks_remaining(self, use_policy: bool = False) -> None:
        """Reject auto-draft while Sleeper is the source of truth."""
        _ = use_policy
        raise SyncReadOnlyError(_READ_ONLY_MESSAGE)

    def _bootstrap_from_sleeper(self) -> None:
        """Reset local state and replay every Sleeper pick."""
        draft_id = self._config.draft.SLEEPER_DRAFT_ID
        draft_meta = self._sync_service.fetch_draft(draft_id)
        rounds = _draft_round_count(draft_meta, self.total_roster_size_per_team)
        draft_order = self._sync_service.build_draft_order(
            draft_meta,
            self._config.draft.SLEEPER_ROSTER_ID_TO_TEAM_ID,
            self.num_teams,
            rounds,
        )
        self._controller.reset(
            draft_order=draft_order,
            agent_team_id=self._config.draft.AGENT_START_POSITION,
        )
        self._state.ensure_visual_board(num_teams=self.num_teams, rounds=rounds)
        self._last_synced_pick_no = 0
        self._last_sleeper_fetch_at = 0.0
        self._sleeper_draft_status = str(draft_meta.get("status") or "drafting")
        self._sync_error = None
        self._slot_to_roster_id = dict(draft_meta.get("slot_to_roster_id") or {})
        self._resolver.replace_catalog(self.player_catalog)
        self.sync_from_sleeper(force=True)

    def _safe_fetch_directory(self):
        """Return the Sleeper player directory, or None when unavailable."""
        try:
            return self._gateway.fetch_all_players()
        except Exception:
            return None


def validate_sleeper_sync_config(config: Config) -> None:
    """Raise when required Sleeper sync settings are missing.

    Parameters
    ----------
    config : Config
        Runtime configuration.

    Raises
    ------
    ValueError
        If draft id or roster mapping is incomplete.
    """
    if not str(config.draft.SLEEPER_DRAFT_ID).strip():
        raise ValueError("SLEEPER_DRAFT_ID is required when Sleeper sync is enabled.")
    mapping = config.draft.SLEEPER_ROSTER_ID_TO_TEAM_ID
    if not mapping:
        raise ValueError("SLEEPER_ROSTER_ID_TO_TEAM_ID is required when Sleeper sync is enabled.")
    team_ids = sorted(int(team_id) for team_id in mapping.values())
    expected = list(range(1, config.draft.NUM_TEAMS + 1))
    if team_ids != expected:
        raise ValueError(
            "SLEEPER_ROSTER_ID_TO_TEAM_ID must map onto team ids "
            f"1..{config.draft.NUM_TEAMS} exactly once."
        )


def _draft_round_count(draft_meta: Dict[str, Any], fallback: int) -> int:
    """Return Sleeper settings.rounds or the local roster-size fallback."""
    settings = draft_meta.get("settings") or {}
    rounds = settings.get("rounds")
    try:
        parsed = int(rounds)
    except (TypeError, ValueError):
        return fallback
    return max(parsed, fallback)
