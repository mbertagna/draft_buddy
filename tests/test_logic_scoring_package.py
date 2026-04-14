"""Tests for the logic.scoring package layout."""


def test_scoring_service_class_is_defined_in_logic_scoring_service_module():
    """Scoring orchestration lives under draft_buddy.logic.scoring.service."""
    from draft_buddy.logic.scoring.service import ScoringService

    assert ScoringService.__module__ == "draft_buddy.logic.scoring.service"


def test_scoring_engine_class_is_defined_in_logic_scoring_engine_module():
    """Stat-level scoring math lives under draft_buddy.logic.scoring.engine."""
    from draft_buddy.logic.scoring.engine import ScoringEngine

    assert ScoringEngine.__module__ == "draft_buddy.logic.scoring.engine"
