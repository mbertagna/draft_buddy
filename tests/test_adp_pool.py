"""Tests for ADP pool helpers and training regime sampling."""

from __future__ import annotations

from collections import Counter

import numpy as np

from draft_buddy.core.entities import PlayerCatalog
from draft_buddy.data.adp_pool import (
    build_episode_pool_ids,
    choose_draft_pool_regime,
    complete_draft_pool_size,
    position_draft_floors,
    select_top_adp_player_ids,
)


def test_complete_draft_pool_size_multiplies_teams_and_slots() -> None:
    """Verify complete draft size is teams times slots."""
    assert complete_draft_pool_size(10, 14) == 140


def test_select_top_adp_player_ids_prefers_finite_ascending_adp(player_factory) -> None:
    """Verify finite ADP ranks first and non-finite fills last."""
    low = player_factory(1, "RB")
    mid = player_factory(2, "WR")
    high = player_factory(3, "QB")
    missing = player_factory(4, "TE")
    object.__setattr__(low, "adp", 1.0)
    object.__setattr__(mid, "adp", 5.0)
    object.__setattr__(high, "adp", 10.0)
    object.__setattr__(missing, "adp", np.inf)
    catalog = PlayerCatalog([low, mid, high, missing])

    selected = select_top_adp_player_ids(catalog, catalog.player_ids, 3)

    assert selected == {1, 2, 3}


def test_choose_draft_pool_regime_prefers_higher_weights() -> None:
    """Verify weighted regime sampling prefers higher weights."""
    regimes = [
        {"id": "rare", "weight": 0.01, "prune_inactive": False, "limit_adp": False},
        {"id": "common", "weight": 0.99, "prune_inactive": True, "limit_adp": False},
    ]
    draws = [choose_draft_pool_regime(regimes)["id"] for _ in range(3000)]
    assert draws.count("common") > draws.count("rare")


def test_build_episode_pool_ids_inactive_only_drops_inactive(player_factory) -> None:
    """Verify inactive-only regime removes matching status players."""
    active = player_factory(1, "RB")
    inactive = player_factory(2, "RB")
    object.__setattr__(active, "adp", 1.0)
    object.__setattr__(inactive, "adp", 2.0)
    object.__setattr__(inactive, "sleeper_status", "Inactive")
    catalog = PlayerCatalog([active, inactive])

    pool = build_episode_pool_ids(
        catalog,
        {"id": "inactive_only", "prune_inactive": True, "limit_adp": False},
        min_n=1,
        default_extra_min=0,
        default_extra_max=0,
        roster_statuses=["Inactive"],
        injury_statuses=[],
    )

    assert pool == {1}


def test_build_episode_pool_ids_adp_limit_respects_min_n(player_factory) -> None:
    """Verify ADP limiting keeps at least min_n when catalog allows."""
    players = []
    for index in range(1, 6):
        player = player_factory(index, "WR")
        object.__setattr__(player, "adp", float(index))
        players.append(player)
    catalog = PlayerCatalog(players)

    pool = build_episode_pool_ids(
        catalog,
        {"id": "adp_only", "prune_inactive": False, "limit_adp": True},
        min_n=3,
        default_extra_min=0,
        default_extra_max=0,
        roster_statuses=[],
        injury_statuses=[],
    )

    assert pool == {1, 2, 3}


def test_position_draft_floors_include_flex_for_skill_positions() -> None:
    """Verify flex-eligible floors include full flex absorption."""
    floors = position_draft_floors(
        num_teams=10,
        roster_structure={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 3},
        bench_maxes={"QB": 1, "RB": 3, "WR": 3, "TE": 2},
    )

    assert floors == {"QB": 20, "RB": 80, "WR": 80, "TE": 60}


def test_build_episode_pool_ids_adp_limit_tops_up_position_floors(player_factory) -> None:
    """Verify ADP pools add deeper players so position floors are met."""
    players = []
    next_id = 1
    # Skew top ADP toward WR so a raw top-8 cut would starve TE/QB.
    for adp, position, count in (
        (1.0, "WR", 8),
        (20.0, "RB", 4),
        (40.0, "QB", 3),
        (60.0, "TE", 3),
    ):
        for offset in range(count):
            player = player_factory(next_id, position)
            object.__setattr__(player, "adp", adp + offset)
            players.append(player)
            next_id += 1
    catalog = PlayerCatalog(players)

    pool = build_episode_pool_ids(
        catalog,
        {"id": "adp_only", "prune_inactive": False, "limit_adp": True},
        min_n=8,
        default_extra_min=0,
        default_extra_max=0,
        roster_statuses=[],
        injury_statuses=[],
        num_teams=1,
        roster_structure={"QB": 1, "RB": 1, "WR": 1, "TE": 1, "FLEX": 1},
        bench_maxes={"QB": 0, "RB": 0, "WR": 0, "TE": 0},
    )

    by_position = Counter(catalog.require(pid).position for pid in pool)
    assert by_position["QB"] >= 1
    assert by_position["RB"] >= 2  # starter + flex
    assert by_position["WR"] >= 2
    assert by_position["TE"] >= 2
