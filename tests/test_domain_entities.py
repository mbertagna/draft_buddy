"""Tests for core domain entities."""


def test_player_dataclass_is_defined_in_domain_entities_module():
    """The Player entity lives under draft_buddy.domain.entities."""
    from draft_buddy.domain.entities import Player

    assert Player.__module__ == "draft_buddy.domain.entities"
