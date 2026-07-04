"""Shared player-name normalization for cross-source fuzzy matching.

Used by :mod:`draft_buddy.data.adp_matcher` so name standardization behaves
identically across every external data source.
"""

import re
import unicodedata
from typing import Optional

import pandas as pd


def standardize_name(name) -> Optional[str]:
    """Normalize a player name for consistent fuzzy matching.

    Parameters
    ----------
    name : str or Any
        Raw player name (may contain diacritics, suffixes, etc.).

    Returns
    -------
    str or None
        Standardized name string, or None if input is NaN.
    """
    if pd.isna(name):
        return None
    s = str(name)
    s = unicodedata.normalize('NFKC', s)
    s = ''.join(ch for ch in s if unicodedata.category(ch) not in ('Cf', 'Cc'))
    s = ''.join(ch for ch in unicodedata.normalize('NFKD', s) if not unicodedata.category(ch).startswith('M'))
    s = s.casefold()
    s = re.sub(r'\b(jr|sr|iv|iii|ii)\b\.?$', '', s).strip()
    s = re.sub(r'\s+', ' ', s).strip()
    return s
