import json
import sys
import os

# Change to toolkit directory
os.chdir('data_prep/data_merge/skillcorner-toolkit')
sys.path.insert(0, os.getcwd())

from event_synchronization.events_utils.statsbomb import StatsbombEvents

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

# Check player ID mapping
print(f"\nPlayer ID mapping created: {len(sb_events.stb_ply_id_to_skc_ply_id)} players")
print("First 5 mappings:")
for i, (stb_id, skc_id) in enumerate(list(sb_events.stb_ply_id_to_skc_ply_id.items())[:5]):
    print(f"  SB {stb_id} -> SKC {skc_id}")

# Get first pass event from raw data
first_pass = [e for e in match_events if e.get('type.name') == 'Pass'][0]
print(f"\nFirst raw pass event:")
print(f"  Player ID key: {first_pass.get('player.id')}")
print(f"  Player name: {first_pass.get('player.name')}")

# Check how StatsbombEvents extracts player ID
stb_player_id = sb_events.get_stb_player_id(first_pass)
print(f"\nExtracted SB player ID: {stb_player_id}")

skc_player_id = sb_events.get_player_id(stb_player_id)
print(f"Mapped to SKC player ID: {skc_player_id}")

# Check the event structure
print(f"\nFirst pass event keys: {list(first_pass.keys())[:20]}")
