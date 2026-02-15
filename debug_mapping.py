import json
import sys
import os

# Change to toolkit directory
os.chdir('data_prep/data_merge/skillcorner-toolkit')
sys.path.insert(0, os.getcwd())

from event_synchronization.events_utils.statsbomb import StatsbombEvents

# Load data - adjust paths since we changed directory
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

print(f"\nLineup structure:")
for i, team in enumerate(statsbomb_lineup):
    print(f"Team {i+1}: ID={team['team_id']}, Players={len(team['lineup'])}")
    print(f"  First player: {team['lineup'][0]}")

# Initialize StatsbombEvents
print("\nInitializing StatsbombEvents...")
sb_home_team_id = 1884
sb_events = StatsbombEvents(match_events, statsbomb_lineup, match_data, sb_home_team_id)

print(f"\nTeam ID mapping:")
print(f"  StatsBomb->SKC: {sb_events.stb_team_id_to_skc_team_id}")

print(f"\nPlayer ID mapping (first 10):")
for i, (stb_id, skc_id) in enumerate(list(sb_events.stb_ply_id_to_skc_ply_id.items())[:10]):
    print(f"  SB Player {stb_id} -> SKC Player {skc_id}")

print(f"\nTotal players mapped: {len(sb_events.stb_ply_id_to_skc_ply_id)}")

# Check if pass events have mapped player IDs
pass_events = [e for e in match_events if e.get('type.name') == 'Pass' and e.get('period') == 1]
print(f"\nPeriod 1 pass events: {len(pass_events)}")

mapped_passes = []
for event in pass_events[:20]:  # Check first 20
    stb_player_id = event.get('player.id')
    skc_player_id = sb_events.stb_ply_id_to_skc_ply_id.get(stb_player_id )
    if skc_player_id:
        mapped_passes.append((stb_player_id, skc_player_id))

print(f"Passes with mapped player IDs (first 20): {len(mapped_passes)}")
if mapped_passes:
    print(f"  Sample: SB Player {mapped_passes[0][0]} -> SKC Player {mapped_passes[0][1]}")
