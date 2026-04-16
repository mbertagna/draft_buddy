"""Core draft entities and value objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

import numpy as np


@dataclass(frozen=True, slots=True)
class Player:
    """Fantasy-relevant player record for draft and simulation.

    Parameters
    ----------
    player_id : int
        Stable player identifier.
    name : str
        Display name.
    position : str
        Position code (QB, RB, WR, TE).
    projected_points : float
        Season projection used for value.
    games_played_frac : float or str, optional
        Fraction of games played, or ``"R"`` for rookies.
    adp : float, optional
        Average draft position; infinity when unknown.
    bye_week : int, optional
        NFL bye week.
    team : str, optional
        Team abbreviation for stacking and display.
    """

    player_id: int
    name: str
    position: str
    projected_points: float
    games_played_frac: float | str = 1.0
    adp: float = field(default=np.inf)
    bye_week: Optional[int] = None
    team: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize the player to a JSON-friendly dictionary.

        Returns
        -------
        dict
            Field names mapped to primitive values.
        """
        return {
            "player_id": self.player_id,
            "name": self.name,
            "position": self.position,
            "projected_points": self.projected_points,
            "games_played_frac": self.games_played_frac,
            "adp": self.adp if np.isfinite(self.adp) else None,
            "bye_week": self.bye_week,
            "team": self.team,
        }


@dataclass(slots=True)
class PlayerCatalog:
    """Ordered collection and lookup index for draftable players.

    Parameters
    ----------
    players : Iterable[Player]
        Players in their canonical iteration order.
    """

    players: tuple[Player, ...]
    _players_by_id: dict[int, Player] = field(init=False, repr=False)

    def __init__(self, players: Iterable[Player]) -> None:
        ordered_players = tuple(players)
        self.players = ordered_players
        self._players_by_id = {player.player_id: player for player in ordered_players}

    def __iter__(self) -> Iterator[Player]:
        """Iterate through players in canonical order."""
        return iter(self.players)

    def __len__(self) -> int:
        """Return number of players in the catalog."""
        return len(self.players)

    def __contains__(self, player_id: object) -> bool:
        """Return whether the catalog contains a player id."""
        return player_id in self._players_by_id

    def __getitem__(self, player_id: int) -> Player:
        """Return a player by id.

        Parameters
        ----------
        player_id : int
            Player identifier.

        Returns
        -------
        Player
            Matching player.
        """
        return self._players_by_id[player_id]

    @property
    def player_ids(self) -> tuple[int, ...]:
        """Return all player ids in catalog order."""
        return tuple(player.player_id for player in self.players)

    def get(self, player_id: int) -> Optional[Player]:
        """Return a player by id when present."""
        return self._players_by_id.get(player_id)

    def require(self, player_id: int) -> Player:
        """Return a player by id or raise a descriptive error."""
        player = self.get(player_id)
        if player is None:
            raise KeyError(f"Unknown player id: {player_id}")
        return player

    def resolve(self, player_ids: Iterable[int]) -> list[Player]:
        """Resolve player ids to player objects.

        Parameters
        ----------
        player_ids : Iterable[int]
            Player ids to resolve.

        Returns
        -------
        list[Player]
            Matching players in the requested order.
        """
        return [self.require(player_id) for player_id in player_ids]

    def with_updated_player(self, updated_player: Player) -> "PlayerCatalog":
        """Return a new catalog with one player replaced by id."""
        return PlayerCatalog(
            updated_player if player.player_id == updated_player.player_id else player
            for player in self.players
        )

    def to_weekly_projections(self) -> dict[int, dict[str, object]]:
        """Build flat weekly projection data from season projections.

        Returns
        -------
        dict[int, dict[str, object]]
            Mapping of player id to position and 18 repeated point values.
        """
        return {
            player.player_id: {"pts": [player.projected_points] * 18, "pos": player.position}
            for player in self.players
        }


@dataclass(frozen=True, slots=True)
class Pick:
    """Represents a single draft pick event.

    Parameters
    ----------
    pick_number : int
        One-indexed global draft pick number before the pick is applied.
    team_id : int
        Team making the pick.
    player_id : int
        Identifier of the drafted player.
    is_manual_pick : bool, optional
        Whether the pick came from a user/manual interaction.
    previous_pick_index : int, optional
        Pick index before the pick was applied.
    previous_override_team_id : int, optional
        Team override active before the pick was applied.
    """

    pick_number: int
    team_id: int
    player_id: int
    is_manual_pick: bool = False
    previous_pick_index: int = 0
    previous_override_team_id: Optional[int] = None

    def to_dict(self) -> dict:
        """Serialize the pick for JSON storage."""
        return {
            "pick_number": self.pick_number,
            "team_id": self.team_id,
            "player_id": self.player_id,
            "is_manual_pick": self.is_manual_pick,
            "previous_pick_index": self.previous_pick_index,
            "previous_override_team_id": self.previous_override_team_id,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Pick":
        """Build a pick from serialized data."""
        return cls(
            pick_number=int(payload["pick_number"]),
            team_id=int(payload["team_id"]),
            player_id=int(payload["player_id"]),
            is_manual_pick=bool(payload.get("is_manual_pick", False)),
            previous_pick_index=int(payload.get("previous_pick_index", 0)),
            previous_override_team_id=payload.get("previous_override_team_id"),
        )


@dataclass(slots=True)
class TeamRoster:
    """Mutable team roster stored as player ids plus slot counts.

    Parameters
    ----------
    player_ids : Iterable[int], optional
        Player ids currently on the team.
    qb_count : int, optional
        Number of quarterbacks counted on roster.
    rb_count : int, optional
        Number of running backs counted on roster.
    wr_count : int, optional
        Number of wide receivers counted on roster.
    te_count : int, optional
        Number of tight ends counted on roster.
    flex_count : int, optional
        Number of flex slots occupied.
    """

    player_ids: list[int] = field(default_factory=list)
    qb_count: int = 0
    rb_count: int = 0
    wr_count: int = 0
    te_count: int = 0
    flex_count: int = 0

    @property
    def size(self) -> int:
        """Return number of players on the roster."""
        return len(self.player_ids)

    def position_count(self, position: str) -> int:
        """Return count for one position-style key.

        Parameters
        ----------
        position : str
            One of ``QB``, ``RB``, ``WR``, ``TE``, or ``FLEX``.

        Returns
        -------
        int
            Matching count.
        """
        if position == "QB":
            return self.qb_count
        if position == "RB":
            return self.rb_count
        if position == "WR":
            return self.wr_count
        if position == "TE":
            return self.te_count
        if position == "FLEX":
            return self.flex_count
        return 0

    def get(self, key: str, default: int = 0) -> int:
        """Provide dict-like access for position counts."""
        return self.position_count(key) if key in {"QB", "RB", "WR", "TE", "FLEX"} else default

    def set_position_count(self, position: str, value: int) -> None:
        """Set one position-style count field."""
        if position == "QB":
            self.qb_count = value
        elif position == "RB":
            self.rb_count = value
        elif position == "WR":
            self.wr_count = value
        elif position == "TE":
            self.te_count = value
        elif position == "FLEX":
            self.flex_count = value

    def to_dict(self) -> dict:
        """Serialize roster state to a JSON-friendly dictionary."""
        return {
            "player_ids": list(self.player_ids),
            "qb_count": self.qb_count,
            "rb_count": self.rb_count,
            "wr_count": self.wr_count,
            "te_count": self.te_count,
            "flex_count": self.flex_count,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TeamRoster":
        """Build a roster from serialized data."""
        return cls(
            player_ids=[int(player_id) for player_id in payload.get("player_ids", [])],
            qb_count=int(payload.get("qb_count", 0)),
            rb_count=int(payload.get("rb_count", 0)),
            wr_count=int(payload.get("wr_count", 0)),
            te_count=int(payload.get("te_count", 0)),
            flex_count=int(payload.get("flex_count", 0)),
        )
