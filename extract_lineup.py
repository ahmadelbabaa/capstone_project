import json

# Load events
print("Loading events...")
with open('data/sb_data/sb_events.json', encoding='utf-8') as f:
    events = json.load(f)

print(f"Total events: {len(events)}")

# Extract Starting XI events (using flattened keys)
starting_xi_events = [e for e in events if e.get('type.name') == 'Starting XI']
print(f"Found {len(starting_xi_events)} Starting XI events")

if starting_xi_events:
    # Create lineup structure expected by the toolkit
    lineup_by_match = {}
    
    for event in starting_xi_events:
        match_id = event.get('match_id')
        team_id = event.get('team.id')
        team_name = event.get('team.name')
        
        # The lineup might be in tactics.lineup as a JSON string, or as a nested field
        lineup = event.get('tactics.lineup')
        
        # If lineup is a string, parse it
        if isinstance(lineup, str):
            try:
                lineup = json.loads(lineup)
            except:
                lineup = []
        
        # If lineup is still None or not a list, skip
        if not lineup or not isinstance(lineup, list):
            # Try to access from nested structure
            tactics = {}
            for key in event:
                if key.startswith('tactics.lineup'):
                    # This is a flattened field from tactics.lineup array
                    pass
            lineup = []
        
        if match_id not in lineup_by_match:
            lineup_by_match[match_id] = []
        
        # Convert player keys from flattened to nested format expected by toolkit
        formatted_lineup = []
        for player in lineup:
            # Reformat player dictionary for toolkit
            formatted_player = {
                'player_id': player.get('player.id'),
                'player_name': player.get('player.name'),
                'jersey_number': player.get('jersey_number'),
                'position': {
                    'id': player.get('position.id'),
                    'name': player.get('position.name')
                }
            }
            formatted_lineup.append(formatted_player)
        
        lineup_by_match[match_id].append({
            'team_id': team_id,
            'team_name': team_name,
            'lineup': formatted_lineup
        })
    
    print(f"\nProcessed {len(lineup_by_match)} matches")
    
    # Show samples
    for match_id, lineup in list(lineup_by_match.items())[:3]:
        print(f"\nMatch {match_id}:")
        print(f"  Teams: {len(lineup)}")
        for team in lineup:
            print(f"    Team {team['team_id']} ({team['team_name']}): {len(team['lineup'])} players")
    
    # Save lineup file
    with open('data/sb_data/sb_lineups.json', 'w', encoding='utf-8') as f:
        json.dump(lineup_by_match, f, indent=2)
    
    print(f"\nSaved lineups to data/sb_data/sb_lineups.json")
else:
    print("No Starting XI events found!")

if starting_xi_events:
    # Create lineup structure expected by the toolkit
    lineup_by_match = {}
    
    for event in starting_xi_events:
        match_id = event.get('match_id')
        team_id = event.get('team', {}).get('id')
        team_name = event.get('team', {}).get('name')
        lineup = event.get('tactics', {}).get('lineup', [])
        
        if match_id not in lineup_by_match:
            lineup_by_match[match_id] = []
        
        lineup_by_match[match_id].append({
            'team_id': team_id,
            'team_name': team_name,
            'lineup': lineup
        })
    
    print(f"\nProcessed {len(lineup_by_match)} matches")
    
    # Show sample for match 3925227 (which might be SKC match 1410827)
    for match_id, lineup in list(lineup_by_match.items())[:3]:
        print(f"\nMatch {match_id}:")
        print(f"  Teams: {len(lineup)}")
        for team in lineup:
            print(f"    Team {team['team_id']} ({team['team_name']}): {len(team['lineup'])} players")
            if team['lineup']:
                print(f"      First player: {team['lineup'][0]}")
    
    # Save lineup for a specific match (we'll determine which one)
    # For now, save all matches
    with open('data/sb_data/sb_lineups_all.json', 'w', encoding='utf-8') as f:
        json.dump(lineup_by_match, f, indent=2)
    
    print(f"\nSaved all lineups to data/sb_data/sb_lineups_all.json")
else:
    print("No Starting XI events found!")
