#!/usr/bin/env python3
"""
Inspect tracking data timestamp formats
"""
import json

with open('data/tracking_j1_2024/1901679_tracking_extrapolated.jsonl', 'r') as f:
    frames = json.load(f)

print("Inspecting first 100 frames for timestamp patterns...")
print("=" * 70)

timestamp_examples = []
valid_count = 0
none_count = 0
string_count = 0
float_count = 0

for i, frame in enumerate(frames[:100]):
    timestamp = frame.get('timestamp')
    period = frame.get('period')
    
    if timestamp is None:
        none_count += 1
    else:
        valid_count += 1
        if isinstance(timestamp, str):
            string_count += 1
            if len(timestamp_examples) < 5:
                timestamp_examples.append((i, timestamp, type(timestamp).__name__, period))
        elif isinstance(timestamp, (int, float)):
            float_count += 1
            if len(timestamp_examples) < 5:
                timestamp_examples.append((i, timestamp, type(timestamp).__name__, period))

print(f"First 100 frames:")
print(f"  Valid timestamps: {valid_count}")
print(f"  None timestamps: {none_count}")
print(f"  String format: {string_count}")
print(f"  Numeric format: {float_count}")

print(f"\nTimestamp examples:")
for frame_num, ts, ts_type, period in timestamp_examples:
    print(f"  Frame {frame_num}: '{ts}' (type: {ts_type}, period: {period})")

# Find first valid timestamp
print(f"\n" + "=" * 70)
print("Finding first frame with valid timestamp...")
for i, frame in enumerate(frames):
    timestamp = frame.get('timestamp')
    period = frame.get('period')
    if timestamp is not None and period is not None:
        print(f"First valid frame: {i}")
        print(f"  timestamp: {timestamp} (type: {type(timestamp).__name__})")
        print(f"  period: {period}")
        print(f"  frame: {frame.get('frame')}")
        print(f"  player_data length: {len(frame.get('player_data', []))}")
        break
