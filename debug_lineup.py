import json

# Load match data (SkillCorner)
with open('data/metadata_j1_2024/1410827_metadata.json', encoding='utf-8') as f:
    match_data = json.load(f)

# Load StatsBomb events
with open('data/sb_data/sb_events.json', encoding='utf-8') as f:
    statsbomb_events = json.load(f)

# Extract lineup from StatsBomb events
statsbomb_match_id = 3925227
starting_xi_events = [e for e in statsbomb_events if e.get('type.name') == 'Starting XI' and e.get('match_id') == statsbomb_match_id]

print(f"Found {len(starting_xi_events)} Starting XI events for match {statsbomb_match_id}")

# Check the structure of one Starting XI event
if starting_xi_events:
    print("\nSample Starting XI event structure:")
    sample_event = starting_xi_events[0]
    print(f"Team ID: {sample_event.get('team.id')}")
    print(f"Team Name: {sample_event.get('team.name')}")
    lineup = sample_event.get('tactics.lineup', [])
    print(f"Number of players: {len(lineup)}")
    
    if lineup:
        print("\nFirst 3 players in lineup:")
        for player in lineup[:3]:
            print(f"  - {player}")

# Check SkillCorner match data structure 
print("\n" + "="*70)
print("SkillCorner Match Data:")
print("="*70)
print(f"Match ID: {match_data['id']}")
print(f"Home Team: {match_data['home_team']['name']} (ID: {match_data['home_team']['id']})")
print(f"Away Team: {match_data['away_team']['name']} (ID: {match_data['away_team']['id']})")

# Check if match_data has players
if 'players' in match_data:
    print(f"\nNumber of players in match_data: {len(match_data['players'])}")
    print("\nFirst 3 players:")
    for player in match_data['players'][:3]:
        print(f"  - ID: {player.get('id')}, Name: {player.get('name')}, Jersey: {player.get('jersey_number')}, Team ID: {player.get('team_id')}")
else:
    print("\nNo 'players' key in match_data!")
