import os
import sys

import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from playerdata import pull_player_outcomes, PlayerDataset


### Bregman's MLBAMID is 608324
BREGMAN_MLBAMID = 608324

## QUESTION: given some assembly of pitching metrics from an opposing pitcher,
## Can we predict the outcome of this Alex Bregman at-bat?

general_pitch_stats = ['pitch_type', 'release_speed', 'effective_speed', 'zone']
breaking_ball_stats = ['release_spin_rate', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
                       'sz_top', 'sz_bot']
situational_stats = ['balls', 'strikes', 'outs_when_up', 'inning',
                 'at_bat_number', 'on_1b', 'on_2b', 'on_3b']

feature_stats = general_pitch_stats + breaking_ball_stats + situational_stats
outcome_stats = ['events']
desired_stats = feature_stats + outcome_stats

bregman_df = pull_player_outcomes(BREGMAN_MLBAMID, desired_stats)

### FEATURE / OUTCOME SPLIT
feature_columns = list(bregman_df.columns)
feature_columns.remove('events')
outcome_columns = outcome_stats

X = bregman_df[feature_columns]
y = bregman_df[outcome_columns]

### NEURAL NETWORK
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

## train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## categorical + numerical scalar fits
scaler = StandardScaler()
X_train = pd.get_dummies(X_train, columns=['pitch_type', 'base_state'])
X_train_scaled = scaler.fit_transform(X)

X_test_scaled = scaler.transform(X_test)

## build tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

### initialize Dataset class, put into dataloader
train_dataset = PlayerDataset(X_train_tensor, y_train_tensor)
test_dataset = PlayerDataset(X_test_tensor, y_test_tensor)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)