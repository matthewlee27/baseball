from pybaseball import statcast
from pybaseball import  playerid_lookup
from pybaseball import  statcast_batter

### Bregman's MLBAMID is 608324

## type is pandas dataframe
bregman_stats_df = statcast_batter('2026-01-01', '2026-07-01', 608324)
desired_stats = ['pitch_type', 'description']
bregman_pitches_outcomes_df = bregman_stats_df[desired_stats]

print (bregman_pitches_outcomes_df.head())