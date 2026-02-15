import json
import sys
import os

# Change to toolkit directory
os.chdir('data_prep/data_merge/skillcorner-toolkit')
sys.path.insert(0, os.getcwd())

from event_synchronization.events_utils.statsbomb import StatsbombEvents
from event_synchronization.constants import PASS_EVENT

# Load data
print("Loading data...")
with open('../../../data/metadata_j1_2024/1410827_metadata.json', encoding='utf-8') as f:
    match_data = json.load(f)

with open('../../../data/sb_data/sb_events.json', encoding='utf-8') as f:
    statsbomb_events_raw = json.load(f)

# Filter for this match
statsbomb_match_id = 3925227
match_events = [e for e in statsbomb_events_raw if e.get('match_id') == statsbomb_match_id]

# Extract lineup
starting_xi_events = [e for e in match_events if e.get('type.name') == 'Starting XI']

statsbomb_lineup = []
for event in starting_xi_events:
    team_id = event.get('team.id')
    lineup = event.get('tactics.lineup', [])
    
    formatted_lineup = []
    for player in lineup:
        formatted_lineup.append({
            'player_id': player.get('player.id'),
            'player_name': player.get('player.name'),
            'jersey_number': player.get('jersey_number'),
            'positions': [{
                'id': player.get('position.id'),
                'name': player.get('position.name')
            }]
        })
    
    statsbomb_lineup.append({
        'team_id': team_id,
        'lineup': formatted_lineup
    })

# Initialize StatsbombEvents
sb_home_team_id = 1884
sb_events = StatsbombEvents(match_events, statsbomb_lineup, match_data, sb_home_team_id)

# Standardize events
standardized_events = sb_events.standardize_events()

print(f"Total standardized events: {len(standardized_events)}")

# Count passes by period and player
pass_events_by_period = {}
for event in standardized_events:
    if event.generic_event_type == PASS_EVENT:
        period = event.period
        if period not in pass_events_by_period:
            pass_events_by_period[period] = {}
        
        player_id = event.player_id
        if player_id is not None:
            if player_id not in pass_events_by_period[period]:
                pass_events_by_period[period][player_id] = 0
            pass_events_by_period[period][player_id] += 1

print(f"\nPass distribution by period:")
for period in sorted(pass_events_by_period.keys()):
    players = pass_events_by_period[period]
    total_passes = sum(players.values())
    players_with_10_plus = len([p for p, c in players.items() if c > 10])
    print(f"  Period {period}: {total_passes} total passes, {len(players)} players, {players_with_10_plus} players with >10 passes")
    
    if players_with_10_plus > 0:
        print(f"    Top 5 players:")
        sorted_players = sorted(players.items(), key=lambda x: x[1], reverse=True)
        for player_id, count in sorted_players[:5]:
            print(f"      Player {player_id}: {count} passes")

# Check for None player IDs
passes_with_none_player = [e for e in standardized_events if e.generic_event_type == PASS_EVENT and e.player_id is None]
print(f"\nPasses with None player_id: {len(passes_with_none_player)}")

# Check raw events
raw_pass_events_p1 = [e for e in match_events if e.get('type.name') == 'Pass' and e.get('period') == 1]
print(f"\nRaw StatsBomb pass events in period 1: {len(raw_pass_events_p1)}")
