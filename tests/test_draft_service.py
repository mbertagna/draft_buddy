"""Tests for the legacy draft service orchestration layer."""

from unittest.mock import MagicMock, patch

from draft_buddy.logic.draft_service import DraftService


def test_get_or_create_draft_returns_cached_env(mock_config):
    """Verify draft service caches environments by session ID."""
    env = MagicMock()
    factory = MagicMock(return_value=env)
    service = DraftService(mock_config, env_factory=factory)

    first = service.get_or_create_draft("session-1")
    second = service.get_or_create_draft("session-1")

    assert first is second


def test_get_or_create_draft_resets_and_saves_when_loaded_order_is_empty(mock_config):
    """Verify empty loaded drafts are reset before caching."""
    env = MagicMock()
    env.draft_order = []
    factory = MagicMock(return_value=env)
    service = DraftService(mock_config, env_factory=factory)

    service.get_or_create_draft("session-1")

    assert env.reset.called and env.save_state.called


def test_get_draft_returns_none_for_unknown_session(mock_config):
    """Verify lookup returns None when no draft was created."""
    service = DraftService(mock_config, env_factory=MagicMock())

    assert service.get_draft("missing") is None


def test_create_new_draft_archives_existing_progressed_draft(mock_config):
    """Verify creating a new draft archives in-progress sessions before replacement."""
    existing = MagicMock()
    existing.current_pick_number = 3
    new_env = MagicMock()
    service = DraftService(mock_config, env_factory=MagicMock(return_value=new_env))
    service._drafts["session-1"] = existing

    with patch("os.makedirs") as mock_makedirs:
        with patch("datetime.datetime") as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "2026-04-15_12-00-00"
            service.create_new_draft("session-1")

    assert existing.save_state.called and mock_makedirs.called


def test_create_new_draft_skips_archive_for_untouched_draft(mock_config):
    """Verify untouched drafts are replaced without archiving."""
    existing = MagicMock()
    existing.current_pick_number = 1
    new_env = MagicMock()
    service = DraftService(mock_config, env_factory=MagicMock(return_value=new_env))
    service._drafts["session-1"] = existing

    service.create_new_draft("session-1")

    assert existing.save_state.called is False
