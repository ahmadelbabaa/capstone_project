"""
Filter StatsBomb events for "From Goal Kick" play pattern

This script filters the original StatsBomb events to only include events
that are part of a "From Goal Kick" play pattern.

Author: Modified for J1 League 2024 data
"""
import pandas as pd
from pathlib import Path
from datetime import datetime

# Import paths
from paths import RAW_DATA_DIR


def main():
    """
    Filter original StatsBomb events for goal kick play patterns
    """
    print("\n" + "="*70)
    print("Filter StatsBomb Events: From Goal Kick Play Pattern")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load original events
    print("[STEP 1] Loading original StatsBomb events...")
    
    original_file = RAW_DATA_DIR / 'sb_data' / 'sb_events.json'
    
    if not original_file.exists():
        print(f"[ERROR] Original events file not found: {original_file}")
        return
    
    print(f"   Loading from: {original_file}")
    print("   This may take several minutes...")
    events_df = pd.read_json(original_file, convert_dates=False)
    
    print(f"   Loaded {len(events_df):,} events from {events_df['match_id'].nunique()} matches")
    input_file = original_file
    
    # Check if play_pattern.name column exists
    if 'play_pattern.name' not in events_df.columns:
        print("\n[ERROR] 'play_pattern.name' column not found in dataset!")
        print(f"   Available columns: {list(events_df.columns)}")
        return
    
    # Show play pattern distribution
    print("\n[STEP 2] Analyzing play patterns...")
    play_pattern_counts = events_df['play_pattern.name'].value_counts()
    print(f"   Total play patterns: {len(play_pattern_counts)}")
    print("\n   Top 10 play patterns:")
    for pattern, count in play_pattern_counts.head(10).items():
        pct = (count / len(events_df)) * 100
        print(f"      {pattern}: {count:,} ({pct:.1f}%)")
    
    # Filter for "From Goal Kick"
    print("\n[STEP 3] Filtering for 'From Goal Kick'...")
    
    # Check exact match first
    goal_kick_mask = events_df['play_pattern.name'] == 'From Goal Kick'
    filtered_df = events_df[goal_kick_mask].copy()
    
    if len(filtered_df) == 0:
        # Try case-insensitive search
        print("   No exact match found, trying case-insensitive...")
        goal_kick_mask = events_df['play_pattern.name'].str.lower() == 'from goal kick'
        filtered_df = events_df[goal_kick_mask].copy()
    
    if len(filtered_df) == 0:
        # Try partial match
        print("   No match found, trying partial match (contains 'goal kick')...")
        goal_kick_mask = events_df['play_pattern.name'].str.contains('goal kick', case=False, na=False)
        filtered_df = events_df[goal_kick_mask].copy()
    
    print(f"   Filtered to {len(filtered_df):,} events ({(len(filtered_df)/len(events_df)*100):.1f}%)")
    print(f"   Covering {filtered_df['match_id'].nunique()} matches")
    
    if len(filtered_df) == 0:
        print("\n[WARNING] No events found matching goal kick pattern!")
        print("   Available play patterns:")
        for pattern in play_pattern_counts.index:
            print(f"      - {pattern}")
        return
    
    # Show filtered data summary
    print("\n[STEP 4] Filtered data summary...")
    print(f"   Events per match (avg): {len(filtered_df) / filtered_df['match_id'].nunique():.1f}")
    print(f"   Event types in filtered data:")
    
    # Use type.name column (flattened from nested JSON)
    type_col = 'type.name' if 'type.name' in filtered_df.columns else 'type'
    for event_type, count in filtered_df[type_col].value_counts().head(10).items():
        print(f"      {event_type}: {count:,}")
    
    # Save filtered events
    print("\n[STEP 5] Saving filtered events...")
    
    # Save as JSON
    output_json = RAW_DATA_DIR / 'sb_data' / 'sb_events_goal_kick.json'
    print(f"   Saving JSON: {output_json}")
    filtered_df.to_json(output_json, orient='records')
    json_size = output_json.stat().st_size / 1024 / 1024
    print(f"   JSON size: {json_size:.1f} MB")
    
    # Summary
    print("\n" + "="*70)
    print("[SUMMARY]")
    print(f"  Input file: {input_file.name}")
    print(f"  Total events: {len(events_df):,}")
    print(f"  Filtered events: {len(filtered_df):,} ({(len(filtered_df)/len(events_df)*100):.1f}%)")
    print(f"  Matches: {filtered_df['match_id'].nunique()}")
    print(f"  Output JSON: {output_json.name} ({json_size:.1f} MB)")
    print("="*70)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    main()
