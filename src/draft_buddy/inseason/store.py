"""Persist in-season snapshots and weekly ROS projection files."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from draft_buddy.inseason.state import LeagueSnapshot


class InSeasonStore:
    """Read and write league-scoped in-season files.

    Parameters
    ----------
    state_path : str
        JSON path for ``LeagueSnapshot``.
    projections_path : str
        JSON path for weekly ROS vectors.
    """

    def __init__(self, state_path: str, projections_path: str) -> None:
        self._state_path = state_path
        self._projections_path = projections_path

    @property
    def state_path(self) -> str:
        """Return the snapshot file path."""
        return self._state_path

    @property
    def projections_path(self) -> str:
        """Return the ROS projections file path."""
        return self._projections_path

    def load_snapshot(self) -> Optional[LeagueSnapshot]:
        """Load a snapshot when the file exists.

        Returns
        -------
        LeagueSnapshot or None
            Stored snapshot, or ``None`` when missing.
        """
        if not os.path.isfile(self._state_path):
            return None
        with open(self._state_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        return LeagueSnapshot.from_dict(payload)

    def save_snapshot(self, snapshot: LeagueSnapshot) -> None:
        """Persist a league snapshot.

        Parameters
        ----------
        snapshot : LeagueSnapshot
            State to write.
        """
        os.makedirs(os.path.dirname(self._state_path) or ".", exist_ok=True)
        with open(self._state_path, "w", encoding="utf-8") as handle:
            json.dump(snapshot.to_dict(), handle, indent=2)

    def load_projections(self) -> Dict[int, Dict[str, Any]]:
        """Load weekly ROS vectors keyed by player id.

        Returns
        -------
        dict
            Simulator-shaped weekly projections.
        """
        if not os.path.isfile(self._projections_path):
            return {}
        with open(self._projections_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        players = payload.get("players", payload)
        result: Dict[int, Dict[str, Any]] = {}
        for raw_id, row in players.items():
            pts = [float(value) for value in row.get("pts") or []]
            result[int(raw_id)] = {"pos": row.get("pos") or row.get("position"), "pts": pts}
        return result

    def save_projections(
        self,
        weekly_projections: Dict[int, Dict[str, Any]],
        current_week: int,
    ) -> None:
        """Persist weekly ROS vectors.

        Parameters
        ----------
        weekly_projections : dict
            Simulator-shaped weekly projections.
        current_week : int
            Week used when the file was generated.
        """
        os.makedirs(os.path.dirname(self._projections_path) or ".", exist_ok=True)
        payload = {
            "current_week": int(current_week),
            "players": {
                str(player_id): {
                    "pos": row.get("pos") or row.get("position"),
                    "pts": list(row.get("pts") or []),
                }
                for player_id, row in weekly_projections.items()
            },
        }
        with open(self._projections_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
