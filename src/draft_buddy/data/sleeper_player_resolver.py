"""Resolve Sleeper pick identities onto local Player records."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from draft_buddy.core.entities import Player, PlayerCatalog

STRING_SLEEPER_ID_OFFSET = 2_000_000_000
SKILL_POSITIONS = frozenset({"QB", "RB", "WR", "TE"})
DISPLAY_ONLY_POSITIONS = frozenset({"K", "DEF", "DST", "?"})

_CHAR_BASE = 40


def encode_sleeper_player_id(sleeper_id: str) -> int:
    """Map a Sleeper player id onto a stable integer.

    Numeric Sleeper ids pass through as integers. Non-numeric ids such as
    DST abbreviations are packed into a reserved range starting at
    ``STRING_SLEEPER_ID_OFFSET``.

    Parameters
    ----------
    sleeper_id : str
        Raw Sleeper ``player_id`` from a pick or directory payload.

    Returns
    -------
    int
        Internal player id.
    """
    raw = str(sleeper_id).strip()
    if raw.isdigit():
        return int(raw)
    packed = 0
    for char in raw.upper()[:6]:
        packed = packed * _CHAR_BASE + (_encode_char(char) + 1)
    return STRING_SLEEPER_ID_OFFSET + packed


def is_skill_position(position: str) -> bool:
    """Return whether ``position`` counts toward skill roster slots.

    Parameters
    ----------
    position : str
        Position code.

    Returns
    -------
    bool
        True for QB/RB/WR/TE.
    """
    return position.upper() in SKILL_POSITIONS


def _encode_char(char: str) -> int:
    """Return a 0-based packing value for one identifier character."""
    if "A" <= char <= "Z":
        return ord(char) - ord("A")
    if "0" <= char <= "9":
        return 26 + (ord(char) - ord("0"))
    return 36


class PlayerResolver:
    """Resolve a Sleeper pick onto a catalog or placeholder Player.

    Lookup order is catalog, then the cached Sleeper directory, then pick
    metadata. The resolver never returns ``None``.
    """

    def __init__(
        self,
        catalog: PlayerCatalog,
        directory_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Parameters
        ----------
        catalog : PlayerCatalog
            Session catalog of fully projected players.
        directory_df : pd.DataFrame, optional
            Cached Sleeper player directory with ``sleeper_id`` column.
        """
        self._catalog = catalog
        self._directory_by_id = _index_directory(directory_df)

    @property
    def catalog(self) -> PlayerCatalog:
        """Return the current catalog, including runtime placeholders."""
        return self._catalog

    def replace_catalog(self, catalog: PlayerCatalog) -> None:
        """Replace the working catalog after a placeholder is added.

        Parameters
        ----------
        catalog : PlayerCatalog
            Updated catalog.
        """
        self._catalog = catalog

    def resolve(
        self,
        sleeper_id: str,
        pick_metadata: Optional[Mapping[str, Any]] = None,
    ) -> Player:
        """Return a Player for one Sleeper pick identity.

        Parameters
        ----------
        sleeper_id : str
            Raw Sleeper player id from the pick payload.
        pick_metadata : Mapping, optional
            Pick metadata fields (name, position, team).

        Returns
        -------
        Player
            Catalog player or a sleeper-only placeholder.
        """
        player_id = encode_sleeper_player_id(sleeper_id)
        catalog_player = self._catalog.get(player_id)
        if catalog_player is not None:
            return catalog_player

        directory_row = self._directory_by_id.get(str(sleeper_id))
        if directory_row is not None:
            return self._placeholder_from_directory(player_id, str(sleeper_id), directory_row)

        metadata = pick_metadata or {}
        first_name = str(metadata.get("first_name") or "").strip()
        last_name = str(metadata.get("last_name") or "").strip()
        name = " ".join(part for part in (first_name, last_name) if part) or "Unknown Player"
        position = _normalize_position(metadata.get("position"))
        team = metadata.get("team")
        return self._build_placeholder(
            player_id=player_id,
            sleeper_id=str(sleeper_id),
            name=name,
            position=position,
            team=None if team in (None, "") else str(team),
            sleeper_status=None,
            sleeper_injury_status=_optional_str(metadata.get("injury_status")),
            sleeper_depth_chart_position=None,
        )

    def _placeholder_from_directory(
        self, player_id: int, sleeper_id: str, row: Mapping[str, Any]
    ) -> Player:
        """Build a placeholder Player from a Sleeper directory row."""
        full_name = str(row.get("full_name") or "").strip() or "Unknown Player"
        return self._build_placeholder(
            player_id=player_id,
            sleeper_id=sleeper_id,
            name=full_name,
            position=_normalize_position(row.get("position")),
            team=_optional_str(row.get("team")),
            sleeper_status=_optional_str(row.get("status")),
            sleeper_injury_status=_optional_str(row.get("injury_status")),
            sleeper_depth_chart_position=_optional_str(row.get("depth_chart_position")),
        )

    @staticmethod
    def _build_placeholder(
        player_id: int,
        sleeper_id: str,
        name: str,
        position: str,
        team: Optional[str],
        sleeper_status: Optional[str],
        sleeper_injury_status: Optional[str],
        sleeper_depth_chart_position: Optional[str],
    ) -> Player:
        """Construct a sleeper-only placeholder Player."""
        return Player(
            player_id=player_id,
            name=name,
            position=position,
            projected_points=0.0,
            games_played_frac=1.0,
            adp=float(np.inf),
            bye_week=None,
            team=team,
            sleeper_id=sleeper_id,
            sleeper_status=sleeper_status,
            sleeper_injury_status=sleeper_injury_status,
            sleeper_depth_chart_position=sleeper_depth_chart_position,
            data_completeness="sleeper_only",
        )


def _index_directory(directory_df: Optional[pd.DataFrame]) -> dict[str, dict]:
    """Index a Sleeper directory DataFrame by string sleeper_id."""
    if directory_df is None or directory_df.empty:
        return {}
    indexed: dict[str, dict] = {}
    for row in directory_df.to_dict(orient="records"):
        sleeper_id = row.get("sleeper_id")
        if sleeper_id is None:
            continue
        indexed[str(sleeper_id)] = row
    return indexed


def _normalize_position(raw_position: Any) -> str:
    """Return an uppercased position, defaulting unknown values to ``?``."""
    if raw_position is None or str(raw_position).strip() == "":
        return "?"
    position = str(raw_position).strip().upper()
    if position == "DST":
        return "DEF"
    return position


def _optional_str(value: Any) -> Optional[str]:
    """Return a stripped string or None."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None
