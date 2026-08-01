"""NFL team abbreviation to full-name lookups for text matching."""

from __future__ import annotations

TEAM_FULL_NAMES: dict[str, str] = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LAC": "Los Angeles Chargers",
    "LAR": "Los Angeles Rams",
    "LV": "Las Vegas Raiders",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
}


def team_name_variants(team_abbr: str) -> list[str]:
    """Return name strings that plausibly refer to a team in free text.

    Used to decide whether a league-wide (multi-team) article mentions a
    specific team, so its content can be surfaced for that team's synthesis.

    Parameters
    ----------
    team_abbr : str
        NFL team abbreviation (e.g. ``SF``).

    Returns
    -------
    list[str]
        The team's full name, nickname, and abbreviation. Falls back to just
        the abbreviation when the team is not in the known lookup.
    """
    full_name = TEAM_FULL_NAMES.get(team_abbr)
    if full_name is None:
        return [team_abbr]

    nickname = full_name.rsplit(" ", 1)[-1]
    return [full_name, nickname, team_abbr]
