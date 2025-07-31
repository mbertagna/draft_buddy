import pickle
import subprocess
import sys
import pandas as pd
import numpy as np
import os

file_dir = os.path.dirname(os.path.abspath(__file__))


def get_records_from_matchups_results_df(matchups_results_df: pd.DataFrame) -> dict:
    matchups_results_df['Away Manager(s)'] = matchups_results_df['Away Manager(s)'].astype(str)
    matchups_results_df['Home Manager(s)'] = matchups_results_df['Home Manager(s)'].astype(str)

    records = {(name): {'W': 0, 'L': 0, 'T': 0, 'pts': 0} for name in np.unique(list(matchups_results_df['Away Manager(s)'].unique()) + list(matchups_results_df['Home Manager(s)'].unique()))}
    
    for idx, row in matchups_results_df.iterrows():
        away_name = row['Away Manager(s)']
        home_name = row['Home Manager(s)']

        away_score = round(row['Away Score'], 2)
        home_score = round(row['Home Score'], 2)

        if row['Week'] in range(1, 15):
            if away_score > home_score:
                records[away_name]['W'] += 1
                records[home_name]['L'] += 1
            elif away_score < home_score:
                records[away_name]['L'] += 1
                records[home_name]['W'] += 1
            elif away_score == home_score:
                records[away_name]['T'] += 1
                records[home_name]['T'] += 1
            else:
                raise(ValueError('MATH ERROR'))
            
            records[away_name]['pts'] += away_score
            records[home_name]['pts'] += home_score

    for k in records.keys():
        records[k]['pts'] = round(records[k]['pts'], 2)
            
    return sorted(records.items(), key=lambda item: (-item[1]['W'], -item[1]['T'], -item[1]['pts']))


def fast_solve_matchups(wtw_dict: dict, matchups_df: pd.DataFrame, rosters: dict, season: int) -> pd.DataFrame:

    arg_dict = {
                'matchups_df': matchups_df, 
                'rosters': rosters, 
                'wtw_dict': wtw_dict
                }

    data_bytes = pickle.dumps(arg_dict)

    # print(file_dir)

    process = subprocess.Popen([sys.executable, 'fast_solve_matchups.py'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=file_dir)
    stdout, _ = process.communicate(data_bytes)

    # print(stdout)

    sim_results = pickle.loads(stdout)

    return sim_results


def get_playoff_results_from_reg_records(wtw_dict: dict, records: list, rosters: dict, season: int):
    week = sum(list(records[0][1].values())[:-1]) + 1
    seeding = [r[0] for r in records]

    cols = ['Week', 'Matchup', 'Away Manager(s)', 'Away Score', 'Home Score', 'Home Manager(s)']

    round_1_playoff_matchups_dict = {col: [] for col in cols}
    round_1_playoff_matchups_dict['Week'] += [week] * 4
    round_1_playoff_matchups_dict['Away Score'] += [0.0] * 4
    round_1_playoff_matchups_dict['Home Score'] += [0.0] * 4
    round_1_playoff_matchups_dict['Matchup'] += [i for i in range(1, 1 + 4)]
    round_1_playoff_matchups_dict['Away Manager(s)'] += [None, seeding[4], seeding[5], None]
    round_1_playoff_matchups_dict['Home Manager(s)'] += [seeding[0], seeding[3], seeding[2], seeding[1]]

    round_1_playoff_matchups_df = fast_solve_matchups(wtw_dict, pd.DataFrame(round_1_playoff_matchups_dict), rosters, season)
    week += 1

    round_2_playoff_matchups_dict = {col: [] for col in cols}
    round_2_playoff_matchups_dict['Week'] += [week] * 2
    round_2_playoff_matchups_dict['Away Score'] += [0.0] * 2
    round_2_playoff_matchups_dict['Home Score'] += [0.0] * 2
    round_2_playoff_matchups_dict['Matchup'] += [i for i in range(1, 1 + 2)]

    round_2_playoff_matchups_dict['Away Manager(s)'] += [
        round_1_playoff_matchups_df.iloc[1]['Home Manager(s)'] if round_1_playoff_matchups_df.iloc[1]['Home Score'] >= 
        round_1_playoff_matchups_df.iloc[1]['Away Score'] else round_1_playoff_matchups_df.iloc[1]['Away Manager(s)'],
        round_1_playoff_matchups_df.iloc[3]['Home Manager(s)']
    ]

    round_2_playoff_matchups_dict['Home Manager(s)'] += [
        round_1_playoff_matchups_df.iloc[0]['Home Manager(s)'],
        round_1_playoff_matchups_df.iloc[2]['Home Manager(s)'] if round_1_playoff_matchups_df.iloc[2]['Home Score'] >= 
        round_1_playoff_matchups_df.iloc[2]['Away Score'] else round_1_playoff_matchups_df.iloc[2]['Away Manager(s)']
    ]

    round_2_playoff_matchups_df = fast_solve_matchups(wtw_dict, pd.DataFrame(round_2_playoff_matchups_dict), rosters, season)
    week += 1

    round_3_playoff_matchups_dict = {col: [] for col in cols}
    round_3_playoff_matchups_dict['Week'] += [week] * 1
    round_3_playoff_matchups_dict['Away Score'] += [0.0] * 1
    round_3_playoff_matchups_dict['Home Score'] += [0.0] * 1
    round_3_playoff_matchups_dict['Matchup'] += [i for i in range(1, 1 + 1)]

    round_3_playoff_matchups_dict['Away Manager(s)'] += [
        round_2_playoff_matchups_df.iloc[1]['Home Manager(s)'] if round_2_playoff_matchups_df.iloc[1]['Home Score'] >= 
        round_2_playoff_matchups_df.iloc[1]['Away Score'] else round_2_playoff_matchups_df.iloc[1]['Away Manager(s)']
    ]

    round_3_playoff_matchups_dict['Home Manager(s)'] += [
        round_2_playoff_matchups_df.iloc[0]['Home Manager(s)'] if round_2_playoff_matchups_df.iloc[0]['Home Score'] >= 
        round_2_playoff_matchups_df.iloc[0]['Away Score'] else round_2_playoff_matchups_df.iloc[0]['Away Manager(s)']
    ]

    round_3_playoff_matchups_df = fast_solve_matchups(wtw_dict, pd.DataFrame(round_3_playoff_matchups_dict), rosters, season)

    return pd.concat([round_1_playoff_matchups_df, round_2_playoff_matchups_df, round_3_playoff_matchups_df])


def pad(s: str, max_length: int, pad_char='0', shift_neg=True):
    return '-' + (max_length - len(s)) * pad_char + s[1:] if shift_neg and s[0] == '-' else (max_length - len(s)) * pad_char + s


def tree_string(tree_array: list, lines=True):
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

            tree_str += pad(str(tree_array[tree_idx]), max_length, pad_char=' ')
            tree_idx += 1

            if n_elements <= tree_idx:
                break
            
            tree_str += sep_spaces

    return tree_str


def get_playoffs_tree(playoffs: pd.DataFrame):

    tree_list = []

    for idx, row in playoffs.iterrows():
        tree_list.append(f"{row['Home Manager(s)']} ({round(row['Home Score'], 2)})")
        tree_list.append(f"{row['Away Manager(s)']} ({round(row['Away Score'], 2)})")

    winner = (playoffs.iloc[len(playoffs) - 1]['Home Manager(s)'] if playoffs.iloc[len(playoffs) - 1]['Home Score'] >= 
                playoffs.iloc[len(playoffs) - 1]['Away Score'] else playoffs.iloc[len(playoffs) - 1]['Away Manager(s)'])

    tree_list.append(winner)

    tree_list.reverse()
    tree_list = [str(v) for v in tree_list]
    
    return tree_string(tree_list), winner


def simulate_season(wtw_dict: dict, matchups: pd.DataFrame, rosters: dict, season: int, output_file_prefix: str, save_data=False):

    regular_results = fast_solve_matchups(wtw_dict, matchups, rosters, season)

    regular_records = get_records_from_matchups_results_df(regular_results)

    playoff_results = get_playoff_results_from_reg_records(wtw_dict, regular_records, rosters, season)

    playoffs_tree, winner = get_playoffs_tree(playoff_results)

    if save_data:

        regular_results.to_csv(f'{output_file_prefix}_regular.csv', index=False)

        playoff_results.to_csv(f'{output_file_prefix}_playoff.csv', index=False)

    return regular_results, regular_records, playoff_results, playoffs_tree, winner