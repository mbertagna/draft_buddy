"""Tests for Sleeper player id encoding and three-tier resolution."""

from __future__ import annotations

import pandas as pd

from draft_buddy.core.entities import Player, PlayerCatalog
from draft_buddy.data.sleeper_player_resolver import (
    STRING_SLEEPER_ID_OFFSET,
    PlayerResolver,
    encode_sleeper_player_id,
)


def test_encode_sleeper_player_id_passes_numeric_ids_through() -> None:
    """Verify numeric Sleeper ids become the same integer."""
    assert encode_sleeper_player_id("4984") == 4984
    assert encode_sleeper_player_id("1001") == 1001


def test_encode_sleeper_player_id_packs_dst_abbreviations() -> None:
    """Verify DST string ids land in the reserved integer range."""
    encoded = encode_sleeper_player_id("DET")

    assert encoded >= STRING_SLEEPER_ID_OFFSET
    assert encode_sleeper_player_id("DET") == encoded
    assert encode_sleeper_player_id("PHI") != encoded


def test_resolver_returns_catalog_player_when_present() -> None:
    """Verify tier-1 catalog lookup wins over directory and metadata."""
    catalog_player = Player(
        player_id=1001,
        name="Josh Allen",
        position="QB",
        projected_points=350.0,
        sleeper_id="1001",
    )
    resolver = PlayerResolver(PlayerCatalog([catalog_player]))

    resolved = resolver.resolve("1001", {"first_name": "Ignored"})

    assert resolved is catalog_player
    assert resolved.data_completeness == "full"


def test_resolver_builds_placeholder_from_directory() -> None:
    """Verify tier-2 directory lookup fills identity without projections."""
    catalog = PlayerCatalog([])
    directory = pd.DataFrame(
        [
            {
                "sleeper_id": "8888",
                "full_name": "Directory Only",
                "position": "WR",
                "team": "SEA",
                "status": "Active",
                "injury_status": "",
                "depth_chart_position": "WR",
            }
        ]
    )
    resolver = PlayerResolver(catalog, directory)

    resolved = resolver.resolve("8888")

    assert resolved.name == "Directory Only"
    assert resolved.position == "WR"
    assert resolved.projected_points == 0.0
    assert resolved.data_completeness == "sleeper_only"
    assert resolved.sleeper_id == "8888"


def test_resolver_builds_placeholder_from_pick_metadata() -> None:
    """Verify tier-3 metadata is used when catalog and directory miss."""
    resolver = PlayerResolver(PlayerCatalog([]))

    resolved = resolver.resolve(
        "9999",
        {"first_name": "Off", "last_name": "Catalog", "position": "RB", "team": "CHI"},
    )

    assert resolved.name == "Off Catalog"
    assert resolved.position == "RB"
    assert resolved.team == "CHI"
    assert resolved.data_completeness == "sleeper_only"
    assert resolved.player_id == 9999


def test_resolver_unknown_player_uses_question_mark_position() -> None:
    """Verify a last-resort placeholder still has a stable id."""
    resolver = PlayerResolver(PlayerCatalog([]))

    resolved = resolver.resolve("mystery")

    assert resolved.name == "Unknown Player"
    assert resolved.position == "?"
    assert resolved.data_completeness == "sleeper_only"
    assert resolved.player_id >= STRING_SLEEPER_ID_OFFSET


def test_resolver_normalizes_dst_position_to_def() -> None:
    """Verify DST metadata is stored as DEF for display-only handling."""
    resolver = PlayerResolver(PlayerCatalog([]))

    resolved = resolver.resolve(
        "DET",
        {"first_name": "Detroit", "last_name": "Defense", "position": "DST", "team": "DET"},
    )

    assert resolved.position == "DEF"
    assert resolved.player_id == encode_sleeper_player_id("DET")
