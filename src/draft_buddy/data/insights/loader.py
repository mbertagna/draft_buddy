"""Load merged player insights JSON for runtime consumption."""

from __future__ import annotations

import json
from pathlib import Path

from draft_buddy.data.insights.schemas import PlayerInsight, PlayerInsightsFile


def load_player_insights(path: str) -> dict[int, PlayerInsight]:
    """Load player insights from a JSON file keyed by sleeper id.

    Parameters
    ----------
    path : str
        Path to ``player_insights_{year}.json``.

    Returns
    -------
    dict[int, PlayerInsight]
        Mapping of sleeper id to insight. Returns an empty dict when the file
        does not exist.
    """
    file_path = Path(path)
    if not file_path.exists():
        return {}

    with file_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    insights_file = PlayerInsightsFile.model_validate(payload)
    return {int(sleeper_id): insight for sleeper_id, insight in insights_file.players.items()}
