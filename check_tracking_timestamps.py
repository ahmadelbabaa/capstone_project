#!/usr/bin/env python3
"""
Check timestamps in the converted ELASTIC tracking data
"""
import sys
sys.path.insert(0, 'data_prep/elastic')

import pandas as pd
import json
from pathlib import Path

# Quick test - load and convert tracking like the merge script does
RAW_DATA_DIR = Path('data')

# Load tracking
tracking_file = RAW_DATA_DIR / 'tracking_j1_2024' / '1901679_tracking_extrapolated.jsonl'

with open(tracking_file, 'r') as f:
    content = f.read()
    frames_data = json.loads(content)

print(f"Loaded {len(frames_data)} frames")

# Extract player rows (simplified version of the merge script logic)
player_rows = []
valid_ts_count = 0
invalid_ts_count = 0

for frame in frames_data[:1000]:  # First 1000 frames
    frame_num = frame.get('frame')
    timestamp_raw = frame.get('timestamp')
    period = frame.get('period')
    
    if timestamp_raw is None or period is None:
        continue
    
    # Parse timestamp
    if isinstance(timestamp_raw, str):
        try:
            time_parts = timestamp_raw.split(':')
            if len(time_parts) == 3:
                hours = float(time_parts[0])
                minutes = float(time_parts[1])
                seconds = float(time_parts[2])
                timestamp = hours * 3600 + minutes * 60 + seconds
                valid_ts_count += 1
            else:
                invalid_ts_count += 1
                continue
        except (ValueError, AttributeError):
            invalid_ts_count += 1
            continue
    else:
        try:
            timestamp = float(timestamp_raw)
            valid_ts_count += 1
        except (ValueError, TypeError):
            invalid_ts_count += 1
            continue
    
    # Extract players
    for player in frame.get('player_data', []):
        if isinstance(player, dict) and 'x' in player and 'y' in player:
            player_rows.append({
                'frame_number': frame_num,
                'timestamp': timestamp,
                'period': int(period),
                'player_id': player.get('player_id'),
                'x': float(player['x']),
                'y': float(player['y']),
            })

print(f"\nTimestamp parsing (first 1000 frames):")
print(f"  Valid: {valid_ts_count}")
print(f"  Invalid: {invalid_ts_count}")

tracking_df = pd.DataFrame(player_rows)
print(f"\nCreated DataFrame with {len(tracking_df)} rows")
print(f"\nTimestamp column info:")
print(tracking_df['timestamp'].describe())
print(f"\nFirst 10 timestamps:")
print(tracking_df['timestamp'].head(10).tolist())
print(f"\nAny NaN timestamps? {tracking_df['timestamp'].isna().sum()}")
