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


def _optimal_lineup_points(roster_player_ids: List[int], wtw_dict: Dict[int, dict], week: int) -> float:
    """
    Builds the best possible lineup for a single team in a given week and returns total starter points.
    - Picks best QB (1), best RB (2), best WR (2), best TE (1)
    - Then fills up to 3 FLEX from remaining eligible (RB/WR/TE)
    - Respects bye weeks by returning 0 points for players on bye (or excluding them implicitly)
    """
    # Partition players by position with week-specific points
    by_pos: Dict[str, List[Tuple[float, dict]]] = {'QB': [], 'RB': [], 'WR': [], 'TE': []}
    for pid in roster_player_ids:
        p_info = wtw_dict.get(pid)
        if not p_info:
            continue
        pos = p_info.get('pos')
        if pos not in by_pos:
            continue
        pts = _points_for_week(p_info, week)
        if pts > 0.0:  # zero-point players can still be selected if no others exist; keep them too
            by_pos[pos].append((pts, p_info))
        else:
            by_pos[pos].append((0.0, p_info))

    # Sort descending by points per position
    for pos in by_pos:
        by_pos[pos].sort(key=lambda t: t[0], reverse=True)

    selected_ids = set()
    total_points = 0.0

    # Select required starters first
    for pos, need in REQUIRED_STARTERS.items():
        candidates = by_pos.get(pos, [])
        for i in range(min(need, len(candidates))):
            pts, p_info = candidates[i]
            total_points += pts
            # avoid duplicate use in flex pool
            selected_ids.add(id(p_info))

    # Build flex candidates from remaining RB/WR/TE
    flex_candidates: List[Tuple[float, dict]] = []
    for pos in FLEX_ELIGIBLE:
        for pts, p_info in by_pos.get(pos, []):
            if id(p_info) not in selected_ids:
                flex_candidates.append((pts, p_info))
    flex_candidates.sort(key=lambda t: t[0], reverse=True)

    for i in range(min(FLEX_MAX, len(flex_candidates))):
        pts, _ = flex_candidates[i]
        total_points += pts

    return float(total_points)


def _solve_matchups_single_thread(wtw_dict: Dict[int, dict], matchups_df: pd.DataFrame, rosters: Dict[str, List[int]]) -> pd.DataFrame:
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

        home_roster = rosters.get(str(home_team), rosters.get(home_team, []))
        away_roster = rosters.get(str(away_team), rosters.get(away_team, []))

        home_score = _optimal_lineup_points(home_roster, wtw_dict, week)
        away_score = _optimal_lineup_points(away_roster, wtw_dict, week)

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
        week = int(row['Week'])
        if week not in range(1, 15):  # only regular season weeks
            continue
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


def _get_playoff_round_results(wtw_dict: Dict[int, dict], matchups_df: pd.DataFrame, rosters: Dict[str, List[int]]) -> pd.DataFrame:
    return _solve_matchups_single_thread(wtw_dict, matchups_df, rosters)


def _get_playoff_results_from_reg_records(wtw_dict: Dict[int, dict], regular_records: list, rosters: Dict[str, List[int]], season: int) -> pd.DataFrame:
    week = sum(list(regular_records[0][1].values())[:-1]) + 1
    seeding = [r[0] for r in regular_records]

    cols = ['Week', 'Matchup', 'Away Manager(s)', 'Away Score', 'Home Score', 'Home Manager(s)']

    # Round 1 (quarters with byes baked via None)
    r1 = {col: [] for col in cols}
    r1['Week'] += [week] * 4
    r1['Away Score'] += [0.0] * 4
    r1['Home Score'] += [0.0] * 4
    r1['Matchup'] += [i for i in range(1, 1 + 4)]
    r1['Away Manager(s)'] += [None, seeding[4], seeding[5], None]
    r1['Home Manager(s)'] += [seeding[0], seeding[3], seeding[2], seeding[1]]
    r1_df = _get_playoff_round_results(wtw_dict, pd.DataFrame(r1), rosters)
    week += 1

    # Round 2 (semis)
    r2 = {col: [] for col in cols}
    r2['Week'] += [week] * 2
    r2['Away Score'] += [0.0] * 2
    r2['Home Score'] += [0.0] * 2
    r2['Matchup'] += [i for i in range(1, 1 + 2)]

    r2['Away Manager(s)'] += [
        r1_df.iloc[1]['Home Manager(s)'] if r1_df.iloc[1]['Home Score'] >= r1_df.iloc[1]['Away Score'] else r1_df.iloc[1]['Away Manager(s)'],
        r1_df.iloc[3]['Home Manager(s)']
    ]

    r2['Home Manager(s)'] += [
        r1_df.iloc[0]['Home Manager(s)'],
        r1_df.iloc[2]['Home Manager(s)'] if r1_df.iloc[2]['Home Score'] >= r1_df.iloc[2]['Away Score'] else r1_df.iloc[2]['Away Manager(s)']
    ]

    r2_df = _get_playoff_round_results(wtw_dict, pd.DataFrame(r2), rosters)
    week += 1

    # Round 3 (final)
    r3 = {col: [] for col in cols}
    r3['Week'] += [week]
    r3['Away Score'] += [0.0]
    r3['Home Score'] += [0.0]
    r3['Matchup'] += [1]

    r3['Away Manager(s)'] += [
        r2_df.iloc[1]['Home Manager(s)'] if r2_df.iloc[1]['Home Score'] >= r2_df.iloc[1]['Away Score'] else r2_df.iloc[1]['Away Manager(s)']
    ]

    r3['Home Manager(s)'] += [
        r2_df.iloc[0]['Home Manager(s)'] if r2_df.iloc[0]['Home Score'] >= r2_df.iloc[0]['Away Score'] else r2_df.iloc[0]['Away Manager(s)']
    ]

    r3_df = _get_playoff_round_results(wtw_dict, pd.DataFrame(r3), rosters)

    return pd.concat([r1_df, r2_df, r3_df], ignore_index=True)


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


def simulate_season_fast(wtw_dict: Dict[int, dict], matchups: pd.DataFrame, rosters: Dict[str, List[int]], season: int, output_file_prefix: str = '', save_data: bool = False):
    """
    Single-process season simulation with optimal weekly lineups and optional bye handling.

    Args:
        wtw_dict: dict mapping player_id -> {'pos': str, 'pts': List[float], optional 'bye' or 'bye_week': int}
        matchups: DataFrame with columns ['Week','Matchup','Away Manager(s)','Away Score','Home Score','Home Manager(s)']
        rosters: dict mapping manager name -> list of player_ids
        season: season year (unused but kept for API parity)
        output_file_prefix: if save_data=True, base path used for CSV outputs
        save_data: whether to write CSV outputs

    Returns:
        regular_results_df, regular_records, playoff_results_df, playoffs_tree, winner
    """
    regular_results = _solve_matchups_single_thread(wtw_dict, matchups, rosters)
    regular_records = _get_records_from_matchups_results_df(regular_results)
    playoff_results = _get_playoff_results_from_reg_records(wtw_dict, regular_records, rosters, season)
    playoffs_tree, winner = _get_playoffs_tree(playoff_results)

    if save_data and output_file_prefix:
        regular_results.to_csv(f'{output_file_prefix}_regular.csv', index=False)
        playoff_results.to_csv(f'{output_file_prefix}_playoff.csv', index=False)

    return regular_results, regular_records, playoff_results, playoffs_tree, winner


