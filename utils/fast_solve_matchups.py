import pandas as pd
import numpy as np
import pickle
import sys
import os
from multiprocessing import Pool

# from utils.roster_obj import Roster
# from utils.player_obj import Player
# from utils.player_data_utils import get_wtw_points_dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

wtw_dict = None
rosters = None

def get_sorted_players_info(r, week):
    players = r
    
    info = []
    for p_id in players:
        try:
            p_info = wtw_dict[p_id]
            info.append(p_info)
        except KeyError:
            pass

    return sorted(info, key=lambda x: x['pts'][week-1], reverse=True)


REQUIRED = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1}
FLEX_ELIGIBLE = {'RB', 'WR', 'TE'}
FLEX_MAX = 3

def get_pts_roster_from_sorted_players_info(sorted_players_info):
    roster = {'QB': [], 'RB': [], 'WR': [], 'TE': [], 'FLEX': [], 'BENCH': []}
    for p_info in sorted_players_info:
        pos = p_info['pos']
        if pos in REQUIRED and len(roster[pos]) < REQUIRED[pos]:
            roster[pos].append(p_info)
        elif pos in FLEX_ELIGIBLE and len(roster['FLEX']) < FLEX_MAX:
            roster['FLEX'].append(p_info)
        else:
            roster['BENCH'].append(p_info)
    return roster
            

def get_week_results(roster, week):
    
    sorted_players_info = get_sorted_players_info(roster, week)
    
    pts_roster = get_pts_roster_from_sorted_players_info(sorted_players_info)
    
    starters = (pts_roster['QB'] + pts_roster['RB'] + pts_roster['WR'] + pts_roster['TE'] + pts_roster['FLEX'])
    
    return sum([p['pts'][week-1] for p in starters])


def get_matchups_df_results(matchups_df_row_tuple):
    idx, matchups_df_row = matchups_df_row_tuple
    
    away_team = matchups_df_row['Away Manager(s)']
    home_team = matchups_df_row['Home Manager(s)']
    week = matchups_df_row['Week']
    
    if pd.isna(away_team) or pd.isna(home_team):
        return matchups_df_row
    
    home, away = get_week_results(rosters[home_team], week), get_week_results(rosters[away_team], week)
    
    matchups_df_row['Away Score'] = away
    matchups_df_row['Home Score'] = home
    
    return matchups_df_row


def init_worker(wtw_data, roster_data):
    global wtw_dict, rosters
    wtw_dict = wtw_data
    rosters = roster_data


if __name__ == '__main__':
    
    data = pickle.load(sys.stdin.buffer)
    matchups = data['matchups_df']
    
    with Pool(initializer=init_worker, initargs=(data['wtw_dict'], data['rosters'])) as pool:
        dfs = list(pool.imap(get_matchups_df_results, matchups.iterrows()))

    output = pd.DataFrame(dfs)
    
    sys.stdout.buffer.write(pickle.dumps(output))