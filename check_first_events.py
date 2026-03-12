import json

# Load events
with open('data/sb_data/sb_events.json', 'r', encoding='utf-8') as f:
    events = json.load(f)

# Filter to match 3925601, period 1
match_events = [e for e in events if e['match_id'] == 3925601 and e['period'] == 1]

print(f"Total events in period 1: {len(match_events)}\n")
print("First 10 events:")
for i, e in enumerate(match_events[:10]):
    event_type = e.get('type', {}).get('name', 'Unknown')
    minute = e.get('minute', 0)
    second = e.get('second', 0)
    pass_type = e.get('pass', {}).get('type', {}).get('name', 'N/A')
    
    print(f"{i+1}. Min:{minute} Sec:{second:5.2f} | Type: {event_type:20s} | Pass Type: {pass_type}")
