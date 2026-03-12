"""
Direct check of sb_events.json for match 3925596 lineup
"""
import json

# Load StatsBomb events
with open('data/sb_data/sb_events.json', 'r', encoding='utf-8') as f:
    sb_events = json.load(f)

match_id = 3925596

print(f"Checking match {match_id} in sb_events.json...")
print(f"Total events in file: {len(sb_events)}")

# Filter for this match
match_events = [e for e in sb_events if e.get('match_id') == match_id]
print(f"Events for match {match_id}: {len(match_events)}")

if match_events:
    # Check data structure format
    first_event = match_events[0]
    print(f"\nFirst event keys (sample): {list(first_event.keys())[:10]}")
    
    # Check if flattened or nested
    if 'type.name' in first_event:
        print("Format: FLATTENED (using dot notation)")
        starting_xi = [e for e in match_events if e.get('type.name') == 'Starting XI']
    elif 'type' in first_event and isinstance(first_event['type'], dict):
        print("Format: NESTED (using dictionaries)")
        starting_xi = [e for e in match_events if e.get('type', {}).get('name') == 'Starting XI']
    else:
        print("Format: UNKNOWN")
        starting_xi = []
    
    print(f"\nStarting XI events found: {len(starting_xi)}")
    
    if starting_xi:
        for event in starting_xi:
            # Try both formats
            if 'team.name' in event:
                team_name = event.get('team.name')
                lineup = event.get('tactics.lineup')
            else:
                team_name = event.get('team', {}).get('name')
                lineup = event.get('tactics', {}).get('lineup')
            
            if lineup:
                print(f"  Team: {team_name}, Players in lineup: {len(lineup)}")
                print(f"  First player sample: {lineup[0]}")
            else:
                print(f"  Team: {team_name}, NO LINEUP DATA")
else:
    print("\nNo events found for this match in sb_events.json")
