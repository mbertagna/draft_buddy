"""Shared conventions for where raw external data sources are cached.

Every data source that downloads (or is manually placed as) raw input data
gets its own subdirectory under a common data root, keeping large,
re-downloadable inputs separate from generated outputs and from each other.
"""

import os


def nflverse_cache_dir(data_root: str) -> str:
    """Return the nflverse raw-data cache directory under a data root.

    Parameters
    ----------
    data_root : str
        Root data directory (e.g. ``./data``).

    Returns
    -------
    str
        Path to the nflverse cache directory.
    """
    return os.path.join(data_root, "cache", "nflverse")


def sleeper_cache_dir(data_root: str) -> str:
    """Return the Sleeper raw-data cache directory under a data root.

    Parameters
    ----------
    data_root : str
        Root data directory (e.g. ``./data``).

    Returns
    -------
    str
        Path to the Sleeper cache directory.
    """
    return os.path.join(data_root, "cache", "sleeper")


def adp_cache_dir(data_root: str) -> str:
    """Return the directory for manually-downloaded FantasyPros ADP files.

    Parameters
    ----------
    data_root : str
        Root data directory (e.g. ``./data``).

    Returns
    -------
    str
        Path to the ADP cache directory.
    """
    return os.path.join(data_root, "cache", "adp")


def insights_search_cache_dir(data_root: str) -> str:
    """Return the Google CSE search cache directory for player insights.

    Parameters
    ----------
    data_root : str
        Root data directory (e.g. ``./data``).

    Returns
    -------
    str
        Path to the insights search cache directory.
    """
    return os.path.join(data_root, "cache", "insights", "search")


def insights_synthesis_cache_dir(data_root: str) -> str:
    """Return the Gemini synthesis cache directory for player insights.

    Parameters
    ----------
    data_root : str
        Root data directory (e.g. ``./data``).

    Returns
    -------
    str
        Path to the insights synthesis cache directory.
    """
    return os.path.join(data_root, "cache", "insights", "synthesis")


def player_insights_output_path(data_root: str, year: int) -> str:
    """Return the path to the merged player insights JSON file for a draft year.

    Parameters
    ----------
    data_root : str
        Root data directory (e.g. ``./data``).
    year : int
        Draft year (e.g. ``2026``).

    Returns
    -------
    str
        Path to ``player_insights_{year}.json``.
    """
    return os.path.join(data_root, f"player_insights_{year}.json")
