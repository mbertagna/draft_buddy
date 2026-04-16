"""Tests for bulk draft-rest simulation helper."""

import pytest

from draft_buddy.logic.draft_rest_simulation import simulate_scheduled_picks_remaining


class _StubEnv:
    """Minimal env stub for rest-simulation tests."""

    def __init__(self, limit=3, should_raise=False):
        self.current_pick_idx = 0
        self.draft_order = list(range(limit))
        self.should_raise = should_raise

    def simulate_single_pick(self):
        """Advance or raise a validation error."""
        if self.should_raise:
            raise ValueError("blocked")
        self.current_pick_idx += 1


def test_simulate_scheduled_picks_remaining_loops_until_schedule_is_exhausted():
    """Verify repeated simulation stops at the end of the draft order."""
    env = _StubEnv(limit=4)

    simulate_scheduled_picks_remaining(env)

    assert env.current_pick_idx == 4


def test_simulate_scheduled_picks_remaining_propagates_simulation_errors():
    """Verify simulation errors are not swallowed."""
    env = _StubEnv(limit=2, should_raise=True)

    with pytest.raises(ValueError, match="blocked"):
        simulate_scheduled_picks_remaining(env)
