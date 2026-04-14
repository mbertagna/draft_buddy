"""Tests for the fantasy data preparation pipeline package."""


def test_fantasy_data_processor_is_importable_from_data_pipeline_package():
    """Pipeline orchestration lives under draft_buddy.data_pipeline."""
    from draft_buddy.data_pipeline import FantasyDataProcessor

    assert FantasyDataProcessor.__module__ == "draft_buddy.data_pipeline.data_processor"
