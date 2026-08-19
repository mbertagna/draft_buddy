"""Helpers for episode draftable pools keyed by ADP and training regimes."""

from __future__ import annotations

import random
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Set

import numpy as np

from draft_buddy.core.entities import PlayerCatalog
from draft_buddy.data.player_filter import iter_inactive_player_ids

_FLEX_ELIGIBLE_POSITIONS = ("RB", "WR", "TE")
_DRAFTABLE_POSITIONS = ("QB", "RB", "WR", "TE")


def complete_draft_pool_size(num_teams: int, slots_per_team: int) -> int:
    """Return the number of players drafted in a complete league draft.

    Parameters
    ----------
    num_teams : int
        Number of teams in the draft.
    slots_per_team : int
        Roster slots filled per team (starters plus bench).

    Returns
    -------
    int
        ``num_teams * slots_per_team``.
    """
    return int(num_teams) * int(slots_per_team)


def position_draft_floors(
    num_teams: int,
    roster_structure: Mapping[str, int],
    bench_maxes: Mapping[str, int],
) -> Dict[str, int]:
    """Return worst-case per-position supply floors for a complete draft.

    Flex slots can be filled entirely by one eligible position, and flex
    occupants do not increment that position's roster counter. Floors
    therefore include flex for RB/WR/TE so an ADP-limited pool cannot be
    drained of a position while a team still has open roster room.

    Parameters
    ----------
    num_teams : int
        Number of teams in the draft.
    roster_structure : Mapping[str, int]
        Starter slots by position, including ``FLEX``.
    bench_maxes : Mapping[str, int]
        Simulated per-position bench maxima.

    Returns
    -------
    dict of str to int
        Minimum players to keep available per draftable position.
    """
    flex_slots = int(roster_structure.get("FLEX", 0))
    floors: Dict[str, int] = {}
    for position in _DRAFTABLE_POSITIONS:
        starter_slots = int(roster_structure.get(position, 0))
        bench_slots = int(bench_maxes.get(position, 0))
        flex_share = flex_slots if position in _FLEX_ELIGIBLE_POSITIONS else 0
        floors[position] = int(num_teams) * (starter_slots + bench_slots + flex_share)
    return floors


def select_top_adp_player_ids(catalog: PlayerCatalog, player_ids: Iterable[int], n: int) -> Set[int]:
    """Return the top-``n`` player ids by ascending FantasyPros ADP.

    Finite ADP values are ranked first (lower is better). Non-finite ADP
    values fill remaining slots after all finite-ADP candidates.

    Parameters
    ----------
    catalog : PlayerCatalog
        Player catalog used for ADP lookups.
    player_ids : Iterable[int]
        Candidate player ids to rank.
    n : int
        Maximum number of ids to keep.

    Returns
    -------
    set of int
        Selected player ids (at most ``n``, or all candidates when fewer).
    """
    if n <= 0:
        return set()

    candidates = [player_id for player_id in player_ids if player_id in catalog]
    finite: list[tuple[float, int]] = []
    infinite: list[int] = []
    for player_id in candidates:
        adp = catalog[player_id].adp
        if np.isfinite(adp):
            finite.append((float(adp), player_id))
        else:
            infinite.append(player_id)

    finite.sort(key=lambda item: (item[0], item[1]))
    ordered = [player_id for _, player_id in finite] + sorted(infinite)
    return set(ordered[:n])


def top_up_pool_to_position_floors(
    catalog: PlayerCatalog,
    pool_ids: Set[int],
    candidate_ids: Iterable[int],
    floors: Mapping[str, int],
) -> Set[int]:
    """Add best-ADP players until each position meets its supply floor.

    Parameters
    ----------
    catalog : PlayerCatalog
        Player catalog used for ADP and position lookups.
    pool_ids : set of int
        Current pool ids to extend.
    candidate_ids : Iterable[int]
        Ids allowed as top-up sources (typically the pre-ADP candidate set).
    floors : Mapping[str, int]
        Minimum counts by position.

    Returns
    -------
    set of int
        Pool ids after position top-ups (may equal ``pool_ids`` when already met).
    """
    expanded = set(pool_ids)
    candidate_set = {player_id for player_id in candidate_ids if player_id in catalog}
    for position, floor in floors.items():
        if floor <= 0:
            continue
        in_pool = [
            player_id
            for player_id in expanded
            if catalog[player_id].position == position
        ]
        shortfall = int(floor) - len(in_pool)
        if shortfall <= 0:
            continue
        outside = [
            player_id
            for player_id in candidate_set
            if player_id not in expanded and catalog[player_id].position == position
        ]
        additions = select_top_adp_player_ids(catalog, outside, shortfall)
        expanded.update(additions)
    return expanded


def choose_draft_pool_regime(regimes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Choose one draft-pool regime using optional sample weights.

    Parameters
    ----------
    regimes : Sequence[Mapping[str, Any]]
        Candidate regimes. Missing ``weight`` defaults to ``1.0``.

    Returns
    -------
    dict
        Selected regime mapping (shallow copy).

    Raises
    ------
    ValueError
        If ``regimes`` is empty or all weights are non-positive.
    """
    if not regimes:
        raise ValueError("At least one draft pool regime is required.")
    weights = [max(float(regime.get("weight", 1.0)), 0.0) for regime in regimes]
    total_weight = sum(weights)
    if total_weight <= 0.0:
        raise ValueError("Draft pool regime weights must sum to a positive value.")
    chosen = random.choices(list(regimes), weights=weights, k=1)[0]
    return dict(chosen)


def build_episode_pool_ids(
    catalog: PlayerCatalog,
    regime: Mapping[str, Any],
    *,
    min_n: int,
    default_extra_min: int,
    default_extra_max: int,
    roster_statuses: Iterable[str],
    injury_statuses: Iterable[str],
    num_teams: Optional[int] = None,
    roster_structure: Optional[Mapping[str, int]] = None,
    bench_maxes: Optional[Mapping[str, int]] = None,
) -> Set[int]:
    """Build the available player id set for one training episode.

    Parameters
    ----------
    catalog : PlayerCatalog
        Full player catalog for the environment.
    regime : Mapping[str, Any]
        Regime with optional ``prune_inactive``, ``limit_adp``, and
        ``adp_extra_min`` / ``adp_extra_max`` overrides.
    min_n : int
        Minimum ADP pool size when ADP limiting is enabled (full draft size).
    default_extra_min : int
        Default absolute ADP reach floor when the regime omits overrides.
    default_extra_max : int
        Default absolute ADP reach ceiling when the regime omits overrides.
    roster_statuses : Iterable[str]
        Inactive roster statuses used when ``prune_inactive`` is true.
    injury_statuses : Iterable[str]
        Inactive injury statuses used when ``prune_inactive`` is true.
    num_teams : int, optional
        League size used for position supply floors when ADP-limiting.
    roster_structure : Mapping[str, int], optional
        Starter slots used for position supply floors when ADP-limiting.
    bench_maxes : Mapping[str, int], optional
        Bench maxima used for position supply floors when ADP-limiting.

    Returns
    -------
    set of int
        Player ids that should start the episode as available.
    """
    candidate_ids = set(catalog.player_ids)
    if bool(regime.get("prune_inactive", False)):
        inactive_ids = set(
            iter_inactive_player_ids(catalog, roster_statuses, injury_statuses)
        )
        candidate_ids -= inactive_ids

    if not bool(regime.get("limit_adp", False)):
        return candidate_ids

    extra_min = int(regime.get("adp_extra_min", default_extra_min))
    extra_max = int(regime.get("adp_extra_max", default_extra_max))
    if extra_min > extra_max:
        extra_min, extra_max = extra_max, extra_min
    extra = random.randint(extra_min, extra_max)
    pool_size = min(len(candidate_ids), max(0, int(min_n) + extra))
    pool_ids = select_top_adp_player_ids(catalog, candidate_ids, pool_size)

    if num_teams is not None and roster_structure is not None and bench_maxes is not None:
        floors = position_draft_floors(num_teams, roster_structure, bench_maxes)
        pool_ids = top_up_pool_to_position_floors(
            catalog, pool_ids, candidate_ids, floors
        )
    return pool_ids
