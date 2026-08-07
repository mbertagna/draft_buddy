"""Load league profiles and season overlays into a runtime Config."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

from draft_buddy.config.settings import Config, repository_root

DEFAULT_LEAGUE_ID = "red_league_10"
DEFAULT_SEASON = 2026
LEAGUE_ENV_VAR = "DRAFT_BUDDY_LEAGUE"
SEASON_ENV_VAR = "DRAFT_BUDDY_SEASON"


def _config_root() -> Path:
    """Return the repository ``config/`` directory."""
    return Path(repository_root()) / "config"


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overlay`` into a copy of ``base``."""
    merged = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON object from disk."""
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _resolve_player_data_csv(paths_config: Dict[str, Any], season: int, base_dir: Path) -> str:
    """Resolve the league-scoped player CSV path for a season."""
    template = paths_config.get("player_data_template")
    if not template:
        return str(base_dir / "data" / "generated_player_data.csv")
    relative_path = template.format(year=season)
    return str(base_dir / relative_path)


def load_runtime_config(
    league_id: Optional[str] = None,
    season: Optional[int] = None,
) -> Config:
    """Load the active league profile and season overlay into a Config.

    Parameters
    ----------
    league_id : Optional[str], optional
        League profile id. Defaults to ``DRAFT_BUDDY_LEAGUE`` env var.
    season : Optional[int], optional
        Draft season year. Defaults to ``DRAFT_BUDDY_SEASON`` env var.

    Returns
    -------
    Config
        Runtime configuration for the selected league and season.
    """
    resolved_league_id = league_id or os.environ.get(LEAGUE_ENV_VAR, DEFAULT_LEAGUE_ID)
    resolved_season = season or int(os.environ.get(SEASON_ENV_VAR, DEFAULT_SEASON))

    config_root = _config_root()
    league_path = config_root / "leagues" / f"{resolved_league_id}.json"
    season_path = config_root / "seasons" / f"{resolved_league_id}_{resolved_season}.json"

    if not league_path.is_file():
        raise FileNotFoundError(f"League profile not found: {league_path}")

    payload = _load_json(league_path)
    if season_path.is_file():
        payload = _deep_merge(payload, _load_json(season_path))

    payload.setdefault("league", {})
    payload["league"]["league_id"] = resolved_league_id
    payload.setdefault("season", {})
    payload["season"]["season"] = resolved_season

    config = Config.from_dict(payload)

    base_dir = Path(config.paths.BASE_DIR)
    paths_dict = payload.get("paths", {})
    player_csv = _resolve_player_data_csv(paths_dict, resolved_season, base_dir)
    config.paths.PLAYER_DATA_TEMPLATE = paths_dict.get("player_data_template", "")
    config.paths.PLAYER_DATA_CSV = player_csv
    os.makedirs(os.path.dirname(player_csv), exist_ok=True)

    return config
