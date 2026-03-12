#!/usr/bin/env python3
"""
Check StatsBomb event timestamp structure
"""
import json
import pandas as pd

with open('data/sb_data/test_match_3925601.json', 'r') as f:
    events = json.load(f)

print(f"Loaded {len(events)} events")
print("\nFirst event structure:")
first = events[0]
for key in sorted(first.keys()):
    print(f"  {key}: {first[key]}")

print("\n" + "=" * 70)
print("Timestamp field analysis:")
print("=" * 70)

# Check if events have 'timestamp' field
timestamps = [e.get('timestamp') for e in events[:10]]
minutes = [e.get('minute') for e in events[:10]]
seconds = [e.get('second') for e in events[:10]]
periods = [e.get('period') for e in events[:10]]

print("\nFirst 10 events:")
for i, (ts, min, sec, period) in enumerate(zip(timestamps, minutes, seconds, periods)):
    print(f"  Event {i}: period={period}, minute={min}, second={sec}, timestamp={type(ts).__name__}({ts})")
