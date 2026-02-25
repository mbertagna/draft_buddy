import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


# Lineup rules (match existing logic)
REQUIRED_STARTERS = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1}
FLEX_ELIGIBLE = {'RB', 'WR', 'TE'}
FLEX_MAX = 3


def _points_for_week(p_info: dict, week: int) -> float:
    """
    Returns projected points for a player in the given week, with bye handling:
    - If p_info has a 'bye' or 'bye_week' and it matches week, returns 0.0.
    - Otherwise returns p_info['pts'][week-1] if available, else 0.0.
    """
    bye_wk = p_info.get('bye', p_info.get('bye_week', None))
    if bye_wk is not None and int(bye_wk) == int(week):
        return 0.0
    pts_list = p_info.get('pts', [])
    idx = week - 1
    if 0 <= idx < len(pts_list):
        val = pts_list[idx]
        try:
            return float(val)
        except Exception:
            return 0.0
    return 0.0


def _optimal_lineup_points(pos_points_for_week: Dict[str, List[float]]) -> float:
    """
    Builds the best possible lineup for a single team in a given week and returns total starter points.
    - Picks best QB (1), best RB (2), best WR (2), best TE (1)
    - Then fills up to 3 FLEX from remaining eligible (RB/WR/TE)
    - Respects bye weeks by returning 0 points for players on bye (or excluding them implicitly)
    """
    # Input already provides sorted descending lists of points per position for this week
    by_pos_points: Dict[str, List[float]] = {
        'QB': list(pos_points_for_week.get('QB', [])),
        'RB': list(pos_points_for_week.get('RB', [])),
        'WR': list(pos_points_for_week.get('WR', [])),
        'TE': list(pos_points_for_week.get('TE', [])),
    }

    selected_ids = set()
    total_points = 0.0

    # Select required starters first
    for pos, need in REQUIRED_STARTERS.items():
        candidates = by_pos_points.get(pos, [])
        for i in range(min(need, len(candidates))):
            pts = candidates[i]
            total_points += pts

    # Build flex candidates from remaining RB/WR/TE
    flex_candidates: List[Tuple[float, dict]] = []
    for pos in FLEX_ELIGIBLE:
        # Remaining candidates after removing those used for starters are simply the tail of the list
        start_used = REQUIRED_STARTERS.get(pos, 0)
        flex_candidates.extend(by_pos_points.get(pos, [])[start_used:])
    flex_candidates.sort(reverse=True)

    for i in range(min(FLEX_MAX, len(flex_candidates))):
        total_points += flex_candidates[i]

    return float(total_points)


def _solve_matchups_single_thread(precomputed_pos_points: Dict[str, List[Dict[str, List[float]]]], matchups_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes weekly scores for all matchups in the given DataFrame in a single process.
    - Expects columns: 'Week', 'Away Manager(s)', 'Home Manager(s)', 'Away Score', 'Home Score'
    - For rows with NaN manager names (byes), returns row as-is.
    """
    out_rows = []
    for idx, row in matchups_df.iterrows():
        away_team = row['Away Manager(s)']
        home_team = row['Home Manager(s)']
        week = int(row['Week'])

        if pd.isna(away_team) or pd.isna(home_team):
            out_rows.append(row)
            continue

        # Weeks are 1-indexed; precomputed stored as list indexed by week-1
        home_week_points = precomputed_pos_points.get(str(home_team), [])
        away_week_points = precomputed_pos_points.get(str(away_team), [])
        home_score = _optimal_lineup_points(home_week_points[week - 1]) if len(home_week_points) >= week else 0.0
        away_score = _optimal_lineup_points(away_week_points[week - 1]) if len(away_week_points) >= week else 0.0

        row = row.copy()
        row['Away Score'] = away_score
        row['Home Score'] = home_score
        out_rows.append(row)

    return pd.DataFrame(out_rows)


def _get_records_from_matchups_results_df(matchups_results_df: pd.DataFrame):
    df = matchups_results_df.copy()
    df['Away Manager(s)'] = df['Away Manager(s)'].astype(str)
    df['Home Manager(s)'] = df['Home Manager(s)'].astype(str)

    names = np.unique(list(df['Away Manager(s)'].unique()) + list(df['Home Manager(s)'].unique()))
    records = {name: {'W': 0, 'L': 0, 'T': 0, 'pts': 0.0} for name in names}

    for _, row in df.iterrows():
        # All rows in this DataFrame are regular-season results
        away_name = row['Away Manager(s)']
        home_name = row['Home Manager(s)']
        away_score = float(row['Away Score'])
        home_score = float(row['Home Score'])

        if away_score > home_score:
            records[away_name]['W'] += 1
            records[home_name]['L'] += 1
        elif away_score < home_score:
            records[away_name]['L'] += 1
            records[home_name]['W'] += 1
        else:
            records[away_name]['T'] += 1
            records[home_name]['T'] += 1

        records[away_name]['pts'] += away_score
        records[home_name]['pts'] += home_score

    for k in records:
        records[k]['pts'] = round(records[k]['pts'], 2)

    return sorted(records.items(), key=lambda item: (-item[1]['W'], -item[1]['T'], -item[1]['pts']))


def _get_playoff_round_results(precomputed_pos_points: Dict[str, List[Dict[str, List[float]]]], matchups_df: pd.DataFrame) -> pd.DataFrame:
    return _solve_matchups_single_thread(precomputed_pos_points, matchups_df)


def generate_round_robin_schedule(team_names: List[str], num_weeks: int) -> pd.DataFrame:
    """
    Generates a realistic schedule:
    - First completes a single round-robin (each team plays each other once)
    - Then adds additional weeks (if needed) by sampling non-repeating matchups until num_weeks is reached.
    Ensures within-season uniqueness of pairings where possible.
    """
    import random as _rnd

    names = list(team_names)
    n = len(names)
    if n < 2:
        return pd.DataFrame(columns=['Week','Matchup','Away Team','Away Manager(s)','Away Score','Home Score','Home Team','Home Manager(s)'])

    # Add bye placeholder if odd
    has_bye = False
    if n % 2 == 1:
        names.append('__BYE__')
        n += 1
        has_bye = True

    half = n // 2
    rotation = names[1:]
    current = [names[0]] + rotation
    rounds: List[List[Tuple[str, str]]] = []
    for _ in range(n - 1):
        pairings = []
        for i in range(half):
            pair = (current[i], current[-(i + 1)])
            pairings.append(pair)
        rounds.append(pairings)
        rotation = [current[-1]] + current[1:-1]
        current = [current[0]] + rotation

    # Build a set of unique unordered pairs produced by RR
    def _norm_pair(a, b):
        return tuple(sorted([a, b]))

    unique_pairs = set()
    rows = []
    week = 1
    # First, schedule full round-robin
    for pairings in rounds:
        if week > num_weeks:
            break
        _rnd.shuffle(pairings)
        matchup_idx = 1
        for a, b in pairings:
            if has_bye and ('__BYE__' in (a, b)):
                continue
            # Randomize home/away
            home, away = (a, b) if _rnd.random() < 0.5 else (b, a)
            unique_pairs.add(_norm_pair(a, b))
            rows.append({
                'Week': week,
                'Matchup': matchup_idx,
                'Away Team': None,
                'Away Manager(s)': away,
                'Away Score': 0.0,
                'Home Score': 0.0,
                'Home Team': None,
                'Home Manager(s)': home,
            })
            matchup_idx += 1
        week += 1

    # If still need more weeks, sample additional non-repeating pairings from full pool
    all_pairs = []
    for i in range(len(team_names)):
        for j in range(i + 1, len(team_names)):
            all_pairs.append((team_names[i], team_names[j]))
    remaining_pairs = [p for p in all_pairs if _norm_pair(*p) not in unique_pairs]

    while week <= num_weeks and remaining_pairs:
        _rnd.shuffle(remaining_pairs)
        used = set()
        matchup_idx = 1
        to_remove = []
        for a, b in remaining_pairs:
            if a in used or b in used:
                continue
            used.add(a); used.add(b)
            home, away = (a, b) if _rnd.random() < 0.5 else (b, a)
            unique_pairs.add(_norm_pair(a, b))
            rows.append({
                'Week': week,
                'Matchup': matchup_idx,
                'Away Team': None,
                'Away Manager(s)': away,
                'Away Score': 0.0,
                'Home Score': 0.0,
                'Home Team': None,
                'Home Manager(s)': home,
            })
            matchup_idx += 1
            to_remove.append((a, b))
        # Remove scheduled pairs
        remaining_pairs = [p for p in remaining_pairs if p not in to_remove]
        if matchup_idx > 1:
            week += 1
        else:
            break

    return pd.DataFrame(rows)


def generate_and_resolve_playoffs(precomputed_pos_points: Dict[str, List[Dict[str, List[float]]]], regular_records: list, num_playoff_teams: int, start_week: int) -> pd.DataFrame:
    """
    Dynamic playoff generator & resolver using precomputed weekly positional points.
    """
    all_seeds = [r[0] for r in regular_records]
    seeds = all_seeds[:max(2, int(num_playoff_teams))]
    seed_rank = {name: i + 1 for i, name in enumerate(all_seeds)}

    cols = ['Week', 'Matchup', 'Away Manager(s)', 'Away Score', 'Home Score', 'Home Manager(s)']
    playoff_rows = []

    import math as _math
    if len(seeds) <= 1:
        return pd.DataFrame(columns=cols)
    next_pow2 = 1 << (len(seeds) - 1).bit_length()
    byes = next_pow2 - len(seeds)

    advancing = seeds[:byes]
    playing = seeds[byes:]

    def _pair_round(names: List[str]) -> List[Tuple[str, str]]:
        ordered = sorted(names, key=lambda n: seed_rank[n])
        pairs = []
        i, j = 0, len(ordered) - 1
        while i < j:
            high = ordered[i]
            low = ordered[j]
            pairs.append((high, low))
            i += 1
            j -= 1
        return pairs

    def _play_round(pairs: List[Tuple[str, str]], cur_week: int) -> Tuple[pd.DataFrame, List[str]]:
        if not pairs:
            return pd.DataFrame(columns=cols), []
        rows = {c: [] for c in cols}
        rows['Week'] += [cur_week] * len(pairs)
        rows['Matchup'] += [i + 1 for i in range(len(pairs))]
        rows['Away Score'] += [0.0] * len(pairs)
        rows['Home Score'] += [0.0] * len(pairs)
        for home, away in pairs:
            rows['Away Manager(s)'].append(away)
            rows['Home Manager(s)'].append(home)
        round_df = _get_playoff_round_results(precomputed_pos_points, pd.DataFrame(rows))
        winners = []
        for _, r in round_df.iterrows():
            if float(r['Home Score']) >= float(r['Away Score']):
                winners.append(str(r['Home Manager(s)']))
            else:
                winners.append(str(r['Away Manager(s)']))
        return round_df, winners

    week = int(start_week)
    first_pairs = _pair_round(playing)
    r_df, winners = _play_round(first_pairs, week)
    if not r_df.empty:
        playoff_rows.append(r_df)
        week += 1
    participants = advancing + winners

    while len(participants) > 1:
        pairs = _pair_round(participants)
        r_df, winners = _play_round(pairs, week)
        if not r_df.empty:
            playoff_rows.append(r_df)
        week += 1
        participants = winners

    if playoff_rows:
        return pd.concat(playoff_rows, ignore_index=True)
    return pd.DataFrame(columns=cols)


def _pad(s: str, max_length: int, pad_char='0', shift_neg=True):
    return '-' + (max_length - len(s)) * pad_char + s[1:] if shift_neg and s[0] == '-' else (max_length - len(s)) * pad_char + s


def _tree_string(tree_array: list, lines=True):
    n_elements = len(tree_array)
    n_layers = int(np.floor(np.log2(n_elements)) + 1)

    tree_str = ''
    tree_idx = 0

    max_length = max([len(v) for v in tree_array])
    max_length = max_length + 1 if lines and max_length % 2 != 0 else max_length

    for layer in range(1, n_layers + 1):
        if layer > 1:
            tree_str += '\n'

        if n_elements < tree_idx:
            break

        start_spaces = (2 ** (n_layers - layer) - 1) * ' ' * max_length
        sep_spaces = (2 ** (n_layers - layer + 1) - 1) * ' ' * max_length

        if lines and layer != 1:
            tree_str += start_spaces + max_length * ' '

            for e in range(2 ** (layer - 1)):
                if e > n_elements - 1 - tree_idx:
                    break
                tree_str += '/' if e % 2 == 0 else '\\'
                tree_str += sep_spaces[2:] if e % 2 == 0 else ' ' * (len(sep_spaces) + 2 * max_length)

            tree_str += '\n'

        tree_str += start_spaces

        for e in range(2 ** (layer - 1)):
            tree_str += _pad(str(tree_array[tree_idx]), max_length, pad_char=' ')
            tree_idx += 1
            if n_elements <= tree_idx:
                break
            tree_str += sep_spaces

    return tree_str


def _get_playoffs_tree(playoffs: pd.DataFrame):
    tree_list = []
    for _, row in playoffs.iterrows():
        tree_list.append(f"{row['Home Manager(s)']} ({round(row['Home Score'], 2)})")
        tree_list.append(f"{row['Away Manager(s)']} ({round(row['Away Score'], 2)})")

    winner = (playoffs.iloc[len(playoffs) - 1]['Home Manager(s)'] if playoffs.iloc[len(playoffs) - 1]['Home Score'] >=
              playoffs.iloc[len(playoffs) - 1]['Away Score'] else playoffs.iloc[len(playoffs) - 1]['Away Manager(s)'])

    tree_list.append(winner)
    tree_list.reverse()
    tree_list = [str(v) for v in tree_list]
    return _tree_string(tree_list), winner


def _precompute_manager_weekly_points(weekly_projections: Dict[int, dict], rosters: Dict[str, List[int]], max_week: int) -> Dict[str, List[Dict[str, List[float]]]]:
    """
    Precompute, for each manager and each week, sorted points list per position.
    Returns: manager -> [week_indexed (0-based) dict of pos->sorted_desc_points]
    """
    managers_map: Dict[str, List[Dict[str, List[float]]]] = {}
    for manager, player_ids in rosters.items():
        weekly: List[Dict[str, List[float]]] = []
        # Initialize structures for each week
        for w in range(max_week):
            weekly.append({'QB': [], 'RB': [], 'WR': [], 'TE': []})
        for pid in player_ids:
            p = weekly_projections.get(pid)
            if not p:
                continue
            pos = p.get('pos')
            if pos not in ['QB','RB','WR','TE']:
                continue
            pts_list = p.get('pts', [])
            for w in range(max_week):
                val = 0.0
                idx = w
                if 0 <= idx < len(pts_list):
                    try:
                        val = float(pts_list[idx])
                    except Exception:
                        val = 0.0
                weekly[w][pos].append(val)
        # Sort descending per week per position
        for w in range(max_week):
            for pos in ['QB','RB','WR','TE']:
                weekly[w][pos].sort(reverse=True)
        managers_map[str(manager)] = weekly
    return managers_map


def simulate_season_fast(weekly_projections: Dict[int, dict], matchups: pd.DataFrame, rosters: Dict[str, List[int]], season: int, output_file_prefix: str = '', save_data: bool = False, num_playoff_teams: int = 6):
    """
    Single-process season simulation with optimal weekly lineups and optional bye handling.

    Parameters
    ----------
    weekly_projections : dict
        Mapping player_id -> {'pos': str, 'pts': List[float], optional 'bye' or 'bye_week': int}
    matchups : pd.DataFrame
        DataFrame with columns ['Week','Matchup','Away Manager(s)','Away Score','Home Score','Home Manager(s)']
    rosters : dict
        Mapping manager name -> list of player_ids
    season : int
        Season year (unused but kept for API parity)
    output_file_prefix : str, optional
        If save_data=True, base path used for CSV outputs
    save_data : bool, optional
        Whether to write CSV outputs
    num_playoff_teams : int, optional
        Number of teams to advance to playoffs

    Returns
    -------
    tuple
        (regular_results_df, regular_records, playoff_results_df, playoffs_tree, winner)
    """
    # Determine how many weeks to precompute (regular max + playoff rounds)
    regular_max_week = int(matchups['Week'].max()) if not matchups.empty else 14
    import math as _math
    playoff_rounds = max(1, int(np.ceil(np.log2(max(2, int(num_playoff_teams))))))
    max_week_needed = min(18, regular_max_week + playoff_rounds + 1)

    precomp = _precompute_manager_weekly_points(weekly_projections, rosters, max_week_needed)

    regular_results = _solve_matchups_single_thread(precomp, matchups)
    regular_records = _get_records_from_matchups_results_df(regular_results)
    playoff_results = generate_and_resolve_playoffs(precomp, regular_records, int(num_playoff_teams), start_week=regular_max_week + 1)
    playoffs_tree, winner = _get_playoffs_tree(playoff_results)

    if save_data and output_file_prefix:
        regular_results.to_csv(f'{output_file_prefix}_regular.csv', index=False)
        playoff_results.to_csv(f'{output_file_prefix}_playoff.csv', index=False)

    return regular_results, regular_records, playoff_results, playoffs_tree, winner


