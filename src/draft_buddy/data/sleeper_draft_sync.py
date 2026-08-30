"""Mirror Sleeper draft picks into local DraftState."""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

import pandas as pd

from draft_buddy.core.draft_controller import DraftController
from draft_buddy.data.sleeper_client import SleeperGateway
from draft_buddy.data.sleeper_player_resolver import PlayerResolver, is_skill_position

SNAKE_DRAFT_TYPE = "snake"


class SleeperDraftSyncService:
    """Translate Sleeper draft payloads into local controller mutations."""

    def __init__(self, gateway: SleeperGateway) -> None:
        """
        Parameters
        ----------
        gateway : SleeperGateway
            Source of draft metadata and picks.
        """
        self._gateway = gateway

    def fetch_draft(self, draft_id: str) -> dict:
        """Return raw Sleeper draft metadata.

        Parameters
        ----------
        draft_id : str
            Sleeper draft identifier.

        Returns
        -------
        dict
            Draft metadata payload.
        """
        return self._gateway.fetch_draft(draft_id)

    def build_draft_order(
        self,
        draft_meta: Mapping[str, Any],
        roster_id_to_team_id: Mapping[int, int],
        num_teams: int,
        rounds: int,
    ) -> list[int]:
        """Build a snake draft-order list of internal team ids.

        Parameters
        ----------
        draft_meta : Mapping
            Sleeper draft metadata including ``slot_to_roster_id``.
        roster_id_to_team_id : Mapping[int, int]
            Sleeper roster id to internal team id.
        num_teams : int
            League size.
        rounds : int
            Number of draft rounds.

        Returns
        -------
        list[int]
            Team ids in pick order.

        Raises
        ------
        ValueError
            If the draft is not snake or a slot/roster mapping is missing.
        """
        draft_type = str(draft_meta.get("type") or "").lower()
        if draft_type != SNAKE_DRAFT_TYPE:
            raise ValueError(f"Unsupported Sleeper draft type: {draft_meta.get('type')!r}.")
        slot_to_roster = draft_meta.get("slot_to_roster_id") or {}
        first_round: list[int] = []
        for slot in range(1, num_teams + 1):
            roster_id = _lookup_int_keyed(slot_to_roster, slot)
            if roster_id is None:
                raise ValueError(f"Missing slot_to_roster_id for draft slot {slot}.")
            team_id = roster_id_to_team_id.get(int(roster_id))
            if team_id is None:
                raise ValueError(f"No team_id mapped for Sleeper roster_id {roster_id}.")
            first_round.append(int(team_id))
        draft_order: list[int] = []
        round_count = max(int(rounds), 1)
        for round_number in range(round_count):
            team_ids = first_round if round_number % 2 == 0 else list(reversed(first_round))
            draft_order.extend(team_ids)
        return draft_order

    def sync_new_picks(
        self,
        draft_id: str,
        controller: DraftController,
        resolver: PlayerResolver,
        roster_id_to_team_id: Mapping[int, int],
        last_synced_pick_no: int,
        slot_to_roster_id: Optional[Mapping[Any, Any]] = None,
    ) -> tuple[int, list[str]]:
        """Apply Sleeper picks newer than the high-water mark.

        Parameters
        ----------
        draft_id : str
            Sleeper draft identifier.
        controller : DraftController
            Local draft workflow.
        resolver : PlayerResolver
            Catalog/directory/metadata resolver.
        roster_id_to_team_id : Mapping[int, int]
            Sleeper roster id to internal team id.
        last_synced_pick_no : int
            Highest pick_no already applied.
        slot_to_roster_id : Mapping, optional
            Sleeper ``slot_to_roster_id``. Used when a pick's ``roster_id``
            is null, which is common for mock drafts.

        Returns
        -------
        tuple[int, list[str]]
            New high-water pick_no and non-fatal error messages.
        """
        picks_df = self._gateway.fetch_draft_picks(draft_id)
        errors: list[str] = []
        high_water = int(last_synced_pick_no)
        if picks_df is None or picks_df.empty:
            return high_water, errors

        ordered = picks_df.sort_values("pick_no", kind="mergesort")
        for row in ordered.to_dict(orient="records"):
            pick_no = _as_int(row.get("pick_no"))
            if pick_no is None or pick_no <= last_synced_pick_no:
                continue
            try:
                self._apply_one_pick(
                    row,
                    controller,
                    resolver,
                    roster_id_to_team_id,
                    slot_to_roster_id,
                )
            except Exception as error:
                errors.append(f"pick_no {pick_no}: {error}")
            high_water = max(high_water, pick_no)
        return high_water, errors

    def _apply_one_pick(
        self,
        row: Mapping[str, Any],
        controller: DraftController,
        resolver: PlayerResolver,
        roster_id_to_team_id: Mapping[int, int],
        slot_to_roster_id: Optional[Mapping[Any, Any]] = None,
    ) -> None:
        """Resolve and apply a single Sleeper pick row."""
        sleeper_id = row.get("player_id")
        if sleeper_id is None or (isinstance(sleeper_id, float) and math.isnan(sleeper_id)):
            raise ValueError("Pick is missing player_id.")
        roster_id = _resolve_roster_id(row, slot_to_roster_id)
        if roster_id is None:
            raise ValueError("Pick is missing roster_id and draft_slot.")
        team_id = roster_id_to_team_id.get(roster_id)
        if team_id is None:
            raise ValueError(f"No team_id mapped for Sleeper roster_id {roster_id}.")

        player = resolver.resolve(str(sleeper_id), row)
        if player.player_id not in controller.player_catalog:
            updated_catalog = controller.player_catalog.with_added_player(player)
            controller.player_catalog = updated_catalog
            resolver.replace_catalog(updated_catalog)
        if player.player_id in controller.state.shelved_player_ids:
            controller.unshelve_players([player.player_id])

        if is_skill_position(player.position):
            try:
                controller.apply_pick(team_id=team_id, player_id=player.player_id, is_manual_pick=False)
                return
            except ValueError:
                controller.apply_display_pick(team_id, player.player_id)
                return
        controller.apply_display_pick(team_id, player.player_id)


def _resolve_roster_id(
    row: Mapping[str, Any],
    slot_to_roster_id: Optional[Mapping[Any, Any]],
) -> Optional[int]:
    """Return a Sleeper roster id from the pick, falling back to draft slot."""
    roster_id = _as_int(row.get("roster_id"))
    if roster_id is not None:
        return roster_id
    draft_slot = _as_int(row.get("draft_slot"))
    if draft_slot is None:
        return None
    if slot_to_roster_id:
        mapped = _lookup_int_keyed(slot_to_roster_id, draft_slot)
        mapped_id = _as_int(mapped)
        if mapped_id is not None:
            return mapped_id
    return draft_slot


def _lookup_int_keyed(mapping: Mapping[Any, Any], key: int) -> Any:
    """Return a mapping value trying int and string keys."""
    if key in mapping:
        return mapping[key]
    string_key = str(key)
    if string_key in mapping:
        return mapping[string_key]
    return None


def _as_int(value: Any) -> Optional[int]:
    """Parse an integer from a DataFrame cell, or return None."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
