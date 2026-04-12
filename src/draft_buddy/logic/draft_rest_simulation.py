"""
Bulk simulation of scheduled snake picks for the web draft UI.

Keeps routing thin by isolating the loop that advances the draft environment.
"""


def simulate_scheduled_picks_remaining(draft_env) -> None:
    """
    Simulate every remaining pick in the snake schedule until it is exhausted.

    Parameters
    ----------
    draft_env
        Active ``FantasyFootballDraftEnv`` instance for the session.

    Raises
    ------
    ValueError
        Propagated from ``simulate_single_pick`` (e.g. draft already over,
        manual team on the clock).
    """
    while draft_env.current_pick_idx < len(draft_env.draft_order):
        draft_env.simulate_single_pick()
