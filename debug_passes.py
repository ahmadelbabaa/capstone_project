import json
import pandas as pd

print("Loading StatsBomb events... (this may take a moment)")
# Load events using pandas - more efficient for large files
events_df = pd.read_json('data/sb_data/sb_events.json')

# Filter for match 3925227
match_events = events_df[events_df['match_id'] == 3925227]
pass_events = match_events[match_events['type.name'] == 'Pass']

print(f"Total events for match 3925227: {len(match_events)}")
print(f"Pass events: {len(pass_events)}")
print(f"Period 1 passes: {len(pass_events[pass_events['period'] == 1])}")
print(f"Period 2 passes: {len(pass_events[pass_events['period'] == 2])}")

# Check pass distribution by player
period_1_passes = pass_events[pass_events['period'] == 1]
player_pass_counts = period_1_passes['player.id'].value_counts()

print(f"\nTop 10 players by passes in period 1:")
for player_id, count in player_pass_counts.head(10).items():
    print(f"  Player {player_id}: {count} passes")

print(f"\nPlayers with >= 10 passes in period 1: {len(player_pass_counts[player_pass_counts >= 10])}")
