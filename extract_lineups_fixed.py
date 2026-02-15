import json
import pandas as pd

# Load events
print("Loading events...")
events_df = pd.read_json('data/sb_data/sb_events.json')

print(f"Total events: {len(events_df)}")
print(f"Columns: {events_df.columns.tolist()[:20]}")

# Filter Starting XI events
starting_xi = events_df[events_df['type.name'] == 'Starting XI'].copy()
print(f"\nFound {len(starting_xi)} Starting XI events")

if len(starting_xi) > 0:
    # Group by match_id
    lineup_by_match = {}
    
    for _, event in starting_xi.iterrows():
        match_id = int(event['match_id'])
        team_id = int(event['team.id'])
        team_name = event['team.name']
        lineup = event['tactics.lineup']
        
        if match_id not in lineup_by_match:
            lineup_by_match[match_id] = []
        
        # Format the lineup for the toolkit
        formatted_lineup = []
        if isinstance(lineup, list):
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
        
        lineup_by_match[match_id].append({
            'team_id': team_id,
            'lineup': formatted_lineup
        })
    
    print(f"\nProcessed {len(lineup_by_match)} matches")
    
    # Show first few matches
    for match_id in list(lineup_by_match.keys())[:3]:
        lineup_data = lineup_by_match[match_id]
        print(f"\nMatch {match_id}: {len(lineup_data)} teams")
        for team in lineup_data:
            print(f"  Team {team['team_id']}: {len(team['lineup'])} players")
            if team['lineup']:
                print(f"    First player: {team['lineup'][0]}")
    
    # Save to file
    with open('data/sb_data/sb_lineups.json', 'w', encoding='utf-8') as f:
        json.dump(lineup_by_match, f, indent=2)
    
    print(f"\n✓ Saved lineups to data/sb_data/sb_lineups.json")
else:
    print("No Starting XI events found!")
