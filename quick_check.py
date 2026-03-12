import json

# Quick check
with open('data/sb_data/sb_events.json', 'r', encoding='utf-8') as f:
    events = json.load(f)

m3925596 = [e for e in events if e.get('match_id') == 3925596]
print(f"Match 3925596 events: {len(m3925596)}")

if m3925596:
    # Check format
    sample = m3925596[0]
    if 'type.name' in sample:
        print("Format: Flattened")
        xi = [e for e in m3925596 if e.get('type.name') == 'Starting XI']
        print(f"Starting XI: {len(xi)}")
        if xi:
            lineup = xi[0].get('tactics.lineup')
            if lineup:
                print(f"Lineup length: {len(lineup)}")
                print(f"First player: {lineup[0]}")
