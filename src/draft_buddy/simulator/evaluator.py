"""Stateless season evaluator and schedule utilities."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


REQUIRED_STARTERS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
FLEX_ELIGIBLE = {"RB", "WR", "TE"}
FLEX_MAX = 1


def _optimal_lineup_points(pos_points_for_week: Dict[str, List[float]]) -> float:
    """Compute optimal single-week lineup points from positional lists.

    Parameters
    ----------
    pos_points_for_week : Dict[str, List[float]]
        Mapping of position to sorted weekly points.

    Returns
    -------
    float
        Optimal lineup points for the week.
    """
    by_pos_points: Dict[str, List[float]] = {
        "QB": list(pos_points_for_week.get("QB", [])),
        "RB": list(pos_points_for_week.get("RB", [])),
        "WR": list(pos_points_for_week.get("WR", [])),
        "TE": list(pos_points_for_week.get("TE", [])),
    }
    total_points = 0.0
    for position, needed in REQUIRED_STARTERS.items():
        candidates = by_pos_points.get(position, [])
        for index in range(min(needed, len(candidates))):
            total_points += candidates[index]
    flex_candidates: List[float] = []
    for position in FLEX_ELIGIBLE:
        starter_count = REQUIRED_STARTERS.get(position, 0)
        flex_candidates.extend(by_pos_points.get(position, [])[starter_count:])
    flex_candidates.sort(reverse=True)
    for index in range(min(FLEX_MAX, len(flex_candidates))):
        total_points += flex_candidates[index]
    return float(total_points)


def _solve_matchups_single_thread(
    precomputed_pos_points: Dict[str, List[Dict[str, List[float]]]], matchups_df: pd.DataFrame
) -> pd.DataFrame:
    """Resolve all matchup scores for one schedule table."""
    out_rows = []
    for _, row in matchups_df.iterrows():
        away_team = row["Away Manager(s)"]
        home_team = row["Home Manager(s)"]
        week = int(row["Week"])
        if pd.isna(away_team) or pd.isna(home_team):
            out_rows.append(row)
            continue
        home_week_points = precomputed_pos_points.get(str(home_team), [])
        away_week_points = precomputed_pos_points.get(str(away_team), [])
        home_score = (
            _optimal_lineup_points(home_week_points[week - 1]) if len(home_week_points) >= week else 0.0
        )
        away_score = (
            _optimal_lineup_points(away_week_points[week - 1]) if len(away_week_points) >= week else 0.0
        )
        row = row.copy()
        row["Away Score"] = away_score
        row["Home Score"] = home_score
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def _get_records_from_matchups_results_df(matchups_results_df: pd.DataFrame):
    """Compute regular season records table from matchup results."""
    dataframe = matchups_results_df.copy()
    dataframe["Away Manager(s)"] = dataframe["Away Manager(s)"].astype(str)
    dataframe["Home Manager(s)"] = dataframe["Home Manager(s)"].astype(str)
    names = np.unique(
        list(dataframe["Away Manager(s)"].unique()) + list(dataframe["Home Manager(s)"].unique())
    )
    records = {name: {"W": 0, "L": 0, "T": 0, "pts": 0.0} for name in names}
    for _, row in dataframe.iterrows():
        away_name = row["Away Manager(s)"]
        home_name = row["Home Manager(s)"]
        away_score = float(row["Away Score"])
        home_score = float(row["Home Score"])
        if away_score > home_score:
            records[away_name]["W"] += 1
            records[home_name]["L"] += 1
        elif away_score < home_score:
            records[away_name]["L"] += 1
            records[home_name]["W"] += 1
        else:
            records[away_name]["T"] += 1
            records[home_name]["T"] += 1
        records[away_name]["pts"] += away_score
        records[home_name]["pts"] += home_score
    for key in records:
        records[key]["pts"] = round(records[key]["pts"], 2)
    return sorted(records.items(), key=lambda item: (-item[1]["W"], -item[1]["T"], -item[1]["pts"]))


def generate_round_robin_schedule(team_names: List[str], num_weeks: int) -> pd.DataFrame:
    """Generate a round-robin schedule with optional extra sampled weeks."""
    import random as random_module

    names = list(team_names)
    team_count = len(names)
    if team_count < 2:
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
    has_bye = False
    if team_count % 2 == 1:
        names.append("__BYE__")
        team_count += 1
        has_bye = True
    half = team_count // 2
    rotation = names[1:]
    current = [names[0]] + rotation
    rounds: List[List[Tuple[str, str]]] = []
    for _ in range(team_count - 1):
        pairings = []
        for index in range(half):
            pairings.append((current[index], current[-(index + 1)]))
        rounds.append(pairings)
        rotation = [current[-1]] + current[1:-1]
        current = [current[0]] + rotation

    def _norm_pair(team_a, team_b):
        return tuple(sorted([team_a, team_b]))

    unique_pairs = set()
    rows = []
    week = 1
    for pairings in rounds:
        if week > num_weeks:
            break
        random_module.shuffle(pairings)
        matchup_index = 1
        for team_a, team_b in pairings:
            if has_bye and "__BYE__" in (team_a, team_b):
                continue
            home_team, away_team = (
                (team_a, team_b) if random_module.random() < 0.5 else (team_b, team_a)
            )
            unique_pairs.add(_norm_pair(team_a, team_b))
            rows.append(
                {
                    "Week": week,
                    "Matchup": matchup_index,
                    "Away Team": None,
                    "Away Manager(s)": away_team,
                    "Away Score": 0.0,
                    "Home Score": 0.0,
                    "Home Team": None,
                    "Home Manager(s)": home_team,
                }
            )
            matchup_index += 1
        week += 1

    all_pairs = []
    for first_index in range(len(team_names)):
        for second_index in range(first_index + 1, len(team_names)):
            all_pairs.append((team_names[first_index], team_names[second_index]))
    remaining_pairs = [pair for pair in all_pairs if _norm_pair(*pair) not in unique_pairs]
    while week <= num_weeks:
        candidate_pairs = list(remaining_pairs) if remaining_pairs else list(all_pairs)
        random_module.shuffle(candidate_pairs)
        used_teams = set()
        matchup_index = 1
        pairs_to_remove = []
        for team_a, team_b in candidate_pairs:
            if team_a in used_teams or team_b in used_teams:
                continue
            used_teams.add(team_a)
            used_teams.add(team_b)
            home_team, away_team = (
                (team_a, team_b) if random_module.random() < 0.5 else (team_b, team_a)
            )
            unique_pairs.add(_norm_pair(team_a, team_b))
            rows.append(
                {
                    "Week": week,
                    "Matchup": matchup_index,
                    "Away Team": None,
                    "Away Manager(s)": away_team,
                    "Away Score": 0.0,
                    "Home Score": 0.0,
                    "Home Team": None,
                    "Home Manager(s)": home_team,
                }
            )
            matchup_index += 1
            if (team_a, team_b) in remaining_pairs:
                pairs_to_remove.append((team_a, team_b))
        remaining_pairs = [pair for pair in remaining_pairs if pair not in pairs_to_remove]
        if matchup_index > 1:
            week += 1
        else:
            break
    return pd.DataFrame(rows)


def generate_and_resolve_playoffs(
    precomputed_pos_points: Dict[str, List[Dict[str, List[float]]]],
    regular_records: list,
    num_playoff_teams: int,
    start_week: int,
) -> pd.DataFrame:
    """Generate and resolve playoff bracket rounds."""
    all_seeds = [record[0] for record in regular_records]
    seeds = all_seeds[: max(2, int(num_playoff_teams))]
    seed_rank = {name: index + 1 for index, name in enumerate(all_seeds)}
    columns = ["Week", "Matchup", "Away Manager(s)", "Away Score", "Home Score", "Home Manager(s)"]
    playoff_rows = []
    if len(seeds) <= 1:
        return pd.DataFrame(columns=columns)
    next_power_of_two = 1 << (len(seeds) - 1).bit_length()
    byes = next_power_of_two - len(seeds)
    advancing = seeds[:byes]
    playing = seeds[byes:]

    def _pair_round(names: List[str]) -> List[Tuple[str, str]]:
        ordered = sorted(names, key=lambda name: seed_rank[name])
        pairs = []
        left, right = 0, len(ordered) - 1
        while left < right:
            pairs.append((ordered[left], ordered[right]))
            left += 1
            right -= 1
        return pairs

    def _play_round(pairs: List[Tuple[str, str]], current_week: int) -> Tuple[pd.DataFrame, List[str]]:
        if not pairs:
            return pd.DataFrame(columns=columns), []
        rows = {column: [] for column in columns}
        rows["Week"] += [current_week] * len(pairs)
        rows["Matchup"] += [index + 1 for index in range(len(pairs))]
        rows["Away Score"] += [0.0] * len(pairs)
        rows["Home Score"] += [0.0] * len(pairs)
        for home_team, away_team in pairs:
            rows["Away Manager(s)"].append(away_team)
            rows["Home Manager(s)"].append(home_team)
        round_dataframe = _solve_matchups_single_thread(precomputed_pos_points, pd.DataFrame(rows))
        winners = []
        for _, result_row in round_dataframe.iterrows():
            if float(result_row["Home Score"]) >= float(result_row["Away Score"]):
                winners.append(str(result_row["Home Manager(s)"]))
            else:
                winners.append(str(result_row["Away Manager(s)"]))
        return round_dataframe, winners

    week = int(start_week)
    round_dataframe, winners = _play_round(_pair_round(playing), week)
    if not round_dataframe.empty:
        playoff_rows.append(round_dataframe)
        week += 1
    participants = advancing + winners
    while len(participants) > 1:
        round_dataframe, winners = _play_round(_pair_round(participants), week)
        if not round_dataframe.empty:
            playoff_rows.append(round_dataframe)
        week += 1
        participants = winners
    if playoff_rows:
        return pd.concat(playoff_rows, ignore_index=True)
    return pd.DataFrame(columns=columns)


def format_playoff_tree_string(tree_array: list, lines=True):
    """Render a playoff tree list as a text bracket string."""
    n_elements = len(tree_array)
    n_layers = int(np.floor(np.log2(n_elements)) + 1)
    tree_str = ""
    tree_idx = 0
    max_length = max([len(value) for value in tree_array])
    max_length = max_length + 1 if lines and max_length % 2 != 0 else max_length
    for layer in range(1, n_layers + 1):
        if layer > 1:
            tree_str += "\n"
        if n_elements < tree_idx:
            break
        start_spaces = (2 ** (n_layers - layer) - 1) * " " * max_length
        sep_spaces = (2 ** (n_layers - layer + 1) - 1) * " " * max_length
        if lines and layer != 1:
            tree_str += start_spaces + max_length * " "
            for index in range(2 ** (layer - 1)):
                if index > n_elements - 1 - tree_idx:
                    break
                tree_str += "/" if index % 2 == 0 else "\\"
                tree_str += (
                    sep_spaces[2:] if index % 2 == 0 else " " * (len(sep_spaces) + 2 * max_length)
                )
            tree_str += "\n"
        tree_str += start_spaces
        for _ in range(2 ** (layer - 1)):
            tree_str += str(tree_array[tree_idx]).rjust(max_length)
            tree_idx += 1
            if n_elements <= tree_idx:
                break
            tree_str += sep_spaces
    return tree_str


def _get_playoffs_tree(playoffs: pd.DataFrame):
    """Convert playoff results into a printable bracket and winner."""
    tree_list = []
    for _, row in playoffs.iterrows():
        tree_list.append(f"{row['Home Manager(s)']} ({round(row['Home Score'], 2)})")
        tree_list.append(f"{row['Away Manager(s)']} ({round(row['Away Score'], 2)})")
    winner = (
        playoffs.iloc[len(playoffs) - 1]["Home Manager(s)"]
        if playoffs.iloc[len(playoffs) - 1]["Home Score"]
        >= playoffs.iloc[len(playoffs) - 1]["Away Score"]
        else playoffs.iloc[len(playoffs) - 1]["Away Manager(s)"]
    )
    tree_list.append(winner)
    tree_list.reverse()
    return format_playoff_tree_string([str(value) for value in tree_list]), winner


def _precompute_manager_weekly_points(
    weekly_projections: Dict[int, dict], rosters: Dict[str, List[int]], max_week: int
) -> Dict[str, List[Dict[str, List[float]]]]:
    """Precompute sorted weekly positional points for each manager."""
    managers_map: Dict[str, List[Dict[str, List[float]]]] = {}
    for manager, player_ids in rosters.items():
        weekly: List[Dict[str, List[float]]] = [{"QB": [], "RB": [], "WR": [], "TE": []} for _ in range(max_week)]
        for player_id in player_ids:
            projection = weekly_projections.get(player_id)
            if not projection:
                continue
            position = projection.get("pos")
            if position not in ["QB", "RB", "WR", "TE"]:
                continue
            points_list = projection.get("pts", [])
            for week_index in range(max_week):
                points = 0.0
                if 0 <= week_index < len(points_list):
                    try:
                        points = float(points_list[week_index])
                    except Exception:
                        points = 0.0
                weekly[week_index][position].append(points)
        for week_index in range(max_week):
            for position in ["QB", "RB", "WR", "TE"]:
                weekly[week_index][position].sort(reverse=True)
        managers_map[str(manager)] = weekly
    return managers_map


def simulate_season_fast(
    weekly_projections: Dict[int, dict],
    matchups: pd.DataFrame,
    rosters: Dict[str, List[int]],
    season: int,
    output_file_prefix: str = "",
    save_data: bool = False,
    num_playoff_teams: int = 6,
):
    """Run full regular season and playoffs with optimal weekly lineups.

    Parameters
    ----------
    weekly_projections : Dict[int, dict]
        Player projections keyed by player id.
    matchups : pd.DataFrame
        Regular season matchup table.
    rosters : Dict[str, List[int]]
        Manager rosters as player id lists.
    season : int
        Season year (kept for API compatibility).
    output_file_prefix : str, optional
        Prefix used when saving CSV outputs.
    save_data : bool, optional
        Whether to write simulation CSV files.
    num_playoff_teams : int, optional
        Number of playoff teams.

    Returns
    -------
    tuple
        Regular results, regular records, playoff results, playoff tree, winner.
    """
    regular_max_week = int(matchups["Week"].max()) if not matchups.empty else 14
    playoff_rounds = max(1, int(np.ceil(np.log2(max(2, int(num_playoff_teams))))))
    max_week_needed = min(18, regular_max_week + playoff_rounds + 1)
    precomputed = _precompute_manager_weekly_points(weekly_projections, rosters, max_week_needed)
    regular_results = _solve_matchups_single_thread(precomputed, matchups)
    regular_records = _get_records_from_matchups_results_df(regular_results)
    playoff_results = generate_and_resolve_playoffs(
        precomputed, regular_records, int(num_playoff_teams), start_week=regular_max_week + 1
    )
    playoffs_tree, winner = _get_playoffs_tree(playoff_results)
    if save_data and output_file_prefix:
        regular_results.to_csv(f"{output_file_prefix}_regular.csv", index=False)
        playoff_results.to_csv(f"{output_file_prefix}_playoff.csv", index=False)
    return regular_results, regular_records, playoff_results, playoffs_tree, winner
