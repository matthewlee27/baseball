from pybaseball import statcast
from pybaseball import  playerid_lookup
from pybaseball import  statcast_batter

import datetime

SEASON_START_DATE = '2026-03-25'
TODAY = str(datetime.date.today())

def pull_player_outcomes(player_id, desired_stats, start_date=SEASON_START_DATE, end_date=TODAY):
    '''
    returns: pandas dataframe with desired columns mapped to play-by-play
    outcomes. We only care about balls put in play for this version.
    '''
    player_stats_df = statcast_batter(start_date, end_date, player_id)
    player_stats_df = player_stats_df[player_stats_df['description'] == "hit_into_play"][desired_stats]
    player_stats_df = ob_to_bs(player_stats_df)
    return player_stats_df

def ob_to_bs(df):
    df['base_state'] = (
        df['on_1b'].notna().astype(int) * 1 +
        df['on_2b'].notna().astype(int) * 2 +
        df['on_3b'].notna().astype(int) * 4
    )
    return df

import torch
from torch.utils.data import Dataset, DataLoader

class PlayerDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]