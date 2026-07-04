"""Shared normalization for nflverse and Sleeper player identifiers."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def normalize_gsis_id(value) -> Optional[int]:
    """Normalize a GSIS or nflverse stats ``player_id`` to integer form.

    nflverse weekly stats use GSIS-style identifiers (e.g. ``00-0039040``)
    that are normalized to integers (``39040``) for aggregation joins.

    Parameters
    ----------
    value : Any
        Raw identifier from GSIS, roster, or stats exports.

    Returns
    -------
    int or None
        Integer nflverse player id, or None when missing or unparseable.
    """
    if pd.isna(value):
        return None
    digits_only = "".join(character for character in str(value) if character.isdigit())
    if not digits_only:
        return None
    return int(digits_only)


def normalize_sleeper_id(value) -> Optional[str]:
    """Normalize a Sleeper player id to a plain string.

    Parameters
    ----------
    value : Any
        Raw value, which may arrive as a string, int, or pandas-read float
        (e.g. ``4984.0``) when sourced from a roster CSV.

    Returns
    -------
    str or None
        Canonical string id, or None when the value is missing.
    """
    if pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()
