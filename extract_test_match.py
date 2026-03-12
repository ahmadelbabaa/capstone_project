#!/usr/bin/env python3
"""
Extract single match events to test file for faster iteration
"""
import json
import sys

match_id = 3925601

print(f"Loading events file...")
with open('data/sb_data/sb_events.json', 'r', encoding='utf-8') as f:
    all_events = json.load(f)

print(f"Total events: {len(all_events)}")

# Filter to match
match_events = [e for e in all_events if e.get('match_id') == match_id]

print(f"Events for match {match_id}: {len(match_events)}")

# Save to test file
output_file = 'data/sb_data/test_match_3925601.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(match_events, f)

print(f"Saved to {output_file}")
