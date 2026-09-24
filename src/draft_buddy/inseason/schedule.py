"""Build remaining-week matchup tables for in-season simulation."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Mapping, Sequence

import pandas as pd

from draft_buddy.inseason.state import LeagueSnapshot, ScheduledMatchup
from draft_buddy.simulator.evaluator import generate_round_robin_schedule
from draft_buddy.simulator.team_identity import parse_team_id, team_label


def matchups_dataframe(matchups: Sequence[ScheduledMatchup]) -> pd.DataFrame:
    """Convert snapshot matchups into the evaluator schedule table.

    Parameters
    ----------
    matchups : sequence of ScheduledMatchup
        Remaining games.

    Returns
    -------
    pandas.DataFrame
        Evaluator matchup rows keyed by ``Team {id}`` labels.
    """
    rows = []
    matchup_index_by_week: Dict[int, int] = {}
    for item in matchups:
        matchup_index_by_week[item.week] = matchup_index_by_week.get(item.week, 0) + 1
        rows.append(
            {
                "Week": int(item.week),
                "Matchup": matchup_index_by_week[item.week],
                "Away Team": None,
                "Away Manager(s)": team_label(item.away_team_id),
                "Away Score": 0.0,
                "Home Score": 0.0,
                "Home Team": None,
                "Home Manager(s)": team_label(item.home_team_id),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "Week",
                "Matchup",
                "Away Team",
                "Away Manager(s)",
                "Away Score",
                "Home Score",
                "Home Team",
                "Home Manager(s)",
            ]
        )
    return pd.DataFrame(rows)


def fallback_remaining_matchups(
    team_ids: Iterable[int],
    start_week: int,
    last_week: int,
) -> List[ScheduledMatchup]:
    """Generate round-robin games for leftover regular-season weeks.

    Parameters
    ----------
    team_ids : iterable of int
        Compute team ids.
    start_week : int
        First remaining NFL week.
    last_week : int
        Last regular-season week.

    Returns
    -------
    list of ScheduledMatchup
        Synthetic remaining schedule.
    """
    ids = sorted(int(team_id) for team_id in team_ids)
    week_count = max(0, int(last_week) - int(start_week) + 1)
    if week_count <= 0 or len(ids) < 2:
        return []
    labels = [team_label(team_id) for team_id in ids]
    raw = generate_round_robin_schedule(labels, week_count)
    result: List[ScheduledMatchup] = []
    for _, row in raw.iterrows():
        home_id = parse_team_id(str(row["Home Manager(s)"]))
        away_id = parse_team_id(str(row["Away Manager(s)"]))
        if home_id is None or away_id is None:
            continue
        result.append(
            ScheduledMatchup(
                week=int(start_week) + int(row["Week"]) - 1,
                home_team_id=home_id,
                away_team_id=away_id,
                source="round_robin",
            )
        )
    return result


def fill_missing_weeks(
    provided: Sequence[ScheduledMatchup],
    team_ids: Iterable[int],
    start_week: int,
    last_week: int,
) -> List[ScheduledMatchup]:
    """Keep provided games and round-robin only weeks that have none.

    Parameters
    ----------
    provided : sequence of ScheduledMatchup
        Games already known for some remaining weeks.
    team_ids : iterable of int
        Compute team ids used for the generated weeks.
    start_week : int
        First remaining NFL week.
    last_week : int
        Last regular-season week.

    Returns
    -------
    list of ScheduledMatchup
        One slate covering every remaining week.
    """
    by_week: Dict[int, List[ScheduledMatchup]] = {}
    for item in provided:
        week = int(item.week)
        if week < int(start_week) or week > int(last_week):
            continue
        by_week.setdefault(week, []).append(item)
    filled: List[ScheduledMatchup] = []
    for week in range(int(start_week), int(last_week) + 1):
        if by_week.get(week):
            filled.extend(by_week[week])
            continue
        filled.extend(fallback_remaining_matchups(team_ids, week, week))
    return filled


def overlay_matchups(
    existing: Sequence[ScheduledMatchup],
    provided: Sequence[ScheduledMatchup],
    team_ids: Iterable[int],
    start_week: int,
    last_week: int,
) -> List[ScheduledMatchup]:
    """Replace weeks present in ``provided`` and fill any week that is still empty.

    Parameters
    ----------
    existing : sequence of ScheduledMatchup
        Slate already stored on the snapshot.
    provided : sequence of ScheduledMatchup
        Games from a file or a live feed.
    team_ids : iterable of int
        Compute team ids used when a week still has no game.
    start_week : int
        First remaining NFL week.
    last_week : int
        Last regular-season week.

    Returns
    -------
    list of ScheduledMatchup
        Merged remaining slate.
    """
    provided_weeks = {int(item.week) for item in provided}
    kept = [
        item
        for item in existing
        if int(item.week) not in provided_weeks
        and int(start_week) <= int(item.week) <= int(last_week)
    ]
    return fill_missing_weeks([*kept, *provided], team_ids, start_week, last_week)


def schedule_summary(matchups: Sequence[object]) -> str:
    """Describe how many games came from each slate source.

    Parameters
    ----------
    matchups : sequence
        ``ScheduledMatchup`` objects or serialized matchup mappings.

    Returns
    -------
    str
        Counts such as ``2 sleeper, 8 round robin``.
    """
    counts: Counter[str] = Counter()
    for item in matchups:
        if isinstance(item, Mapping):
            source = str(item.get("source") or "round_robin")
        else:
            source = str(getattr(item, "source", "") or "round_robin")
        counts[source.replace("_", " ")] += 1
    if not counts:
        return "no remaining matchups"
    return ", ".join(f"{counts[source]} {source}" for source in sorted(counts))


def snapshot_matchups_frame(snapshot: LeagueSnapshot) -> pd.DataFrame:
    """Return the remaining-week schedule for a snapshot.

    Parameters
    ----------
    snapshot : LeagueSnapshot
        Loaded league state.

    Returns
    -------
    pandas.DataFrame
        Evaluator matchup table.
    """
    return matchups_dataframe(snapshot.remaining_matchups)
