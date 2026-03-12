"""
Check if the failed matches have Starting XI events in StatsBomb data
"""
import json

# Load StatsBomb events
with open('data/sb_data/sb_events.json', 'r', encoding='utf-8') as f:
    sb_events = json.load(f)

# Failed match IDs from StatsBomb
failed_sb_match_ids = [3925596, 3925599, 3925604, 3925605]

for sb_match_id in failed_sb_match_ids:
    print(f"\n{'='*70}")
    print(f"StatsBomb Match ID: {sb_match_id}")
    print(f"{'='*70}")
    
    # Filter events for this match
    match_events = [e for e in sb_events if e.get('match_id') == sb_match_id]
    print(f"Total events: {len(match_events)}")
    
    # Check for Starting XI events - try both flattened and nested formats
    starting_xi_flattened = [e for e in match_events if e.get('type.name') == 'Starting XI']
    starting_xi_nested = [e for e in match_events if e.get('type', {}).get('name') == 'Starting XI']
    
    starting_xi_events = starting_xi_flattened if starting_xi_flattened else starting_xi_nested
    
    print(f"Starting XI events (flattened format): {len(starting_xi_flattened)}")
    print(f"Starting XI events (nested format): {len(starting_xi_nested)}")
    
    if starting_xi_events:
        for event in starting_xi_events:
            # Try both formats
            team_id = event.get('team.id') or event.get('team', {}).get('id')
            team_name = event.get('team.name') or event.get('team', {}).get('name')
            lineup = event.get('tactics.lineup') or event.get('tactics', {}).get('lineup', [])
            
            if lineup:
                print(f"  - Team: {team_name} (ID: {team_id}), Players: {len(lineup)}")
                # Show first player structure
                print(f"    First player keys: {list(lineup[0].keys())}")
            else:
                print(f"  - Team: {team_name} (ID: {team_id}), NO LINEUP DATA")
    else:
        print("  [!] NO STARTING XI EVENTS FOUND")
        
        # Check what event types exist (try both formats)
        event_types = set()
        for e in match_events:
            type_name = e.get('type.name') or e.get('type', {}).get('name')
            if type_name:
                event_types.add(type_name)
        
        print(f"\n  Available event types ({len(event_types)}):")
        for et in sorted(event_types):
            count = len([e for e in match_events if (e.get('type.name') == et or e.get('type', {}).get('name') == et)])
            print(f"    - {et}: {count}")
