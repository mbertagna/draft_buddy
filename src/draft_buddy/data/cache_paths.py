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
