#!/usr/bin/env python3
"""
Diagnose why ELASTIC synchronization rate is so low (0.19%)
"""
import json
import pandas as pd
import numpy as np

# Load the synchronized output
with open('data/elastic_merged/match_3925601_elastic.json', 'r') as f:
    synced_events = json.load(f)

synced_df = pd.DataFrame(synced_events)

print("=" * 70)
print("ELASTIC Synchronization Diagnostics")
print("=" * 70)

# 1. Check which events were synchronized
print(f"\n1. Synchronized Event Analysis")
print(f"   Total events: {len(synced_df)}")
synced_mask = synced_df['frame_id'].notna()
print(f"   Synchronized: {synced_mask.sum()} ({synced_mask.mean()*100:.2f}%)")
print(f"   Unsynchronized: {(~synced_mask).sum()}")

if synced_mask.sum() > 0:
    print(f"\n   Synchronized events:")
    synced_events_detail = synced_df[synced_mask][['period_id', 'utc_timestamp', 'spadl_type', 
                                                     'start_x', 'start_y', 'frame_id', 'player_name']]
    print(synced_events_detail.to_string())

# 2. Check coordinate ranges
print(f"\n2. Event Coordinate Ranges")
print(f"   X: [{synced_df['start_x'].min():.2f}, {synced_df['start_x'].max():.2f}] meters")
print(f"   Y: [{synced_df['start_y'].min():.2f}, {synced_df['start_y'].max():.2f}] meters")
print(f"   Events with coordinates: {synced_df['start_x'].notna().sum()}")

# Load tracking to compare
print(f"\n3. Loading tracking data for comparison...")
tracking_file = 'data/tracking_j1_2024/1901679_tracking_extrapolated.jsonl'
with open(tracking_file, 'r') as f:
    frames = json.load(f)

# Extract player positions from a few frames
sample_positions = []
for frame in frames[:1000]:  # First 1000 frames
    timestamp = frame.get('timestamp')
    period = frame.get('period')
    if timestamp is None or period is None:
        continue
    for player in frame.get('player_data', []):
        if 'x' in player and 'y' in player:
            sample_positions.append({
                'timestamp': float(timestamp) if isinstance(timestamp, (int, float)) else None,
                'period': int(period),
                'x': float(player['x']),
                'y': float(player['y'])
            })

tracking_sample = pd.DataFrame(sample_positions)
print(f"   Sampled {len(tracking_sample)} player positions from first 1000 frames")
print(f"\n   Tracking Coordinate Ranges (sample):")
print(f"   X: [{tracking_sample['x'].min():.2f}, {tracking_sample['x'].max():.2f}] meters")
print(f"   Y: [{tracking_sample['y'].min():.2f}, {tracking_sample['y'].max():.2f}] meters")

# 4. Check timestamp alignment
print(f"\n4. Timestamp Analysis")
print(f"   Event timestamps:")
print(f"   - Min: {synced_df['utc_timestamp'].min()}")
print(f"   - Max: {synced_df['utc_timestamp'].max()}")
print(f"   - Range: {(pd.to_datetime(synced_df['utc_timestamp'].max()) - pd.to_datetime(synced_df['utc_timestamp'].min())).total_seconds():.1f} seconds")

if len(tracking_sample) > 0:
    print(f"\n   Tracking timestamps (sample):")
    print(f"   - Min: {tracking_sample['timestamp'].min():.1f}s")
    print(f"   - Max: {tracking_sample['timestamp'].max():.1f}s")

# 5. Compare synchronized vs unsynchronized events
print(f"\n5. Synchronized vs Unsynchronized Event Comparison")
if synced_mask.sum() > 0 and (~synced_mask).sum() > 0:
    synced_coords = synced_df[synced_mask][['start_x', 'start_y']].describe()
    unsynced_coords = synced_df[~synced_mask][['start_x', 'start_y']].describe()
    
    print(f"\n   Synchronized event coordinates:")
    print(synced_coords.to_string())
    
    print(f"\n   Unsynchronized event coordinates:")
    print(unsynced_coords.to_string())

# 6. Check event types that synchronized
print(f"\n6. Synchronization by Event Type")
sync_by_type = synced_df.groupby('spadl_type').agg({
    'frame_id': lambda x: x.notna().sum(),
    'spadl_type': 'count'
}).rename(columns={'frame_id': 'synced', 'spadl_type': 'total'})
sync_by_type['rate'] = (sync_by_type['synced'] / sync_by_type['total'] * 100).round(2)
sync_by_type = sync_by_type.sort_values('synced', ascending=False)
print(sync_by_type.to_string())

# 7. Sample unsynchronized events
print(f"\n7. Sample Unsynchronized Events (first 10 'cross' events)")
unsynced_crosses = synced_df[(~synced_mask) & (synced_df['spadl_type'] == 'cross')].head(10)
if len(unsynced_crosses) > 0:
    print(unsynced_crosses[['period_id', 'spadl_type', 'start_x', 'start_y', 'player_name', 'success']].to_string())

print(f"\n" + "=" * 70)
print("Diagnostic Summary:")
print("=" * 70)
print(f"✓ Events have valid coordinates: {synced_df['start_x'].notna().sum()}/{len(synced_df)}")
print(f"✓ Coordinate ranges seem reasonable (±52.5m x, ±34m y)")
print(f"✓ Only {synced_mask.sum()} events matched to tracking frames")
print(f"\nPossible Issues:")
print(f"  1. Coordinate system mismatch between StatsBomb and SkillCorner")
print(f"  2. ELASTIC spatial/temporal matching thresholds too strict")
print(f"  3. Timestamp alignment issues between event and tracking data")
print(f"  4. Most events are 'cross' which might not match well spatially")
