#!/usr/bin/env python3
"""
Compare event and tracking timestamps in detail
"""
import json
import pandas as pd

# Load synchronized events
with open('data/elastic_merged/match_3925601_elastic.json', 'r') as f:
    events = json.load(f)
events_df = pd.DataFrame(events)

# Convert timestamp to datetime for comparison
events_df['utc_timestamp'] = pd.to_datetime(events_df['utc_timestamp'])

print("=" * 70)
print("Event vs Tracking Timestamp Comparison")
print("=" * 70)

# Show first/last events by period
for period in [1, 2]:
    period_events = events_df[events_df['period_id'] == period].sort_values('utc_timestamp')
    if len(period_events) > 0:
        first = period_events.iloc[0]
        last = period_events.iloc[-1]
        
        print(f"\nPeriod {period} Events:")
        print(f"  First event: {first['utc_timestamp']} ({first['spadl_type']})")
        print(f"  Last event:  {last['utc_timestamp']} ({last['spadl_type']})")
        print(f"  Total events: {len(period_events)}")
        
        # Calculate second offset
        base_time = pd.Timestamp('2024-01-01 00:00:00')
        first_seconds = (first['utc_timestamp'] - base_time).total_seconds()
        last_seconds = (last['utc_timestamp'] - base_time).total_seconds()
        print(f"  Time range: {first_seconds:.1f}s - {last_seconds:.1f}s ({last_seconds - first_seconds:.1f}s duration)")

# Now check tracking data
print(f"\n" + "=" * 70)
print("Tracking Timestamp Ranges")
print("=" * 70)

tracking_file = 'data/tracking_j1_2024/1901679_tracking_extrapolated.jsonl'
with open(tracking_file, 'r') as f:
    frames = json.load(f)

# Parse tracking timestamps by period
for period in [1, 2]:
    period_timestamps = []
    for frame in frames:
        if frame.get('period') == period and frame.get('timestamp'):
            ts_raw = frame.get('timestamp')
            if isinstance(ts_raw, str):
                try:
                    parts = ts_raw.split(':')
                    if len(parts) == 3:
                        hours = float(parts[0])
                        minutes = float(parts[1])
                        seconds = float(parts[2])
                        timestamp = hours * 3600 + minutes * 60 + seconds
                        period_timestamps.append(timestamp)
                except:
                    pass
    
    if period_timestamps:
        print(f"\nPeriod {period} Tracking:")
        print(f"  First frame: {min(period_timestamps):.1f}s")
        print(f"  Last frame: {max(period_timestamps):.1f}s")
        print(f"  Duration: {max(period_timestamps) - min(period_timestamps):.1f}s")
        print(f"  Total frames: {len(period_timestamps)}")

# Check if synchronized events match
print(f"\n" + "=" * 70)
print("Synchronized Event Analysis")
print("=" * 70)
synced = events_df[events_df['frame_id'].notna()]
print(f"\nSynchronized events: {len(synced)}")
for idx, event in synced.iterrows():
    base_time = pd.Timestamp('2024-01-01 00:00:00')
    event_seconds = (event['utc_timestamp'] - base_time).total_seconds()
    print(f"  Period {event['period_id']}: {event['spadl_type']} at {event_seconds:.1f}s → Frame {event['frame_id']}")

# Sample some unsynchronized events
print(f"\n" + "=" * 70)
print("Sample Unsynchronized Events (first 5 of each period)")
print("=" * 70)
unsynced = events_df[events_df['frame_id'].isna()]
for period in [1, 2]:
    period_unsynced = unsynced[unsynced['period_id'] == period].head(5)
    if len(period_unsynced) > 0:
        print(f"\nPeriod {period}:")
        for idx, event in period_unsynced.iterrows():
            base_time = pd.Timestamp('2024-01-01 00:00:00')
            event_seconds = (event['utc_timestamp'] - base_time).total_seconds()
            print(f"  {event['spadl_type']:15s} at {event_seconds:7.1f}s  Location: ({event['start_x']:6.2f}, {event['start_y']:6.2f})")
