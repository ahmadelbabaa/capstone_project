"""
Merge J1 League 2024 tracking data with StatsBomb events data (JSON version)

Simplified version that works directly with JSON files without requiring Parquet conversion.

Author: Modified for J1 League 2024 data
"""
import pandas as pd
import numpy as np
import glob
import os
import json
from pathlib import Path
from datetime import datetime

# Import paths
from paths import MATCH_ID_MAPPING_FILE, RAW_DATA_DIR, PROJECT_ROOT

# Dictionary mapping role_id to role name
ROLE_MAPPING = {
    0: "Goalkeeper",
    2: "Center Back",
    3: "Left Center Back",
    4: "Right Center Back",
    5: "Left Wing Back",
    6: "Right Wing Back",
    7: "Defensive Midfield",
    9: "Left Midfield",
    10: "Right Midfield",
    11: "Attacking Midfield",
    12: "Left Winger",
    13: "Right Winger",
    14: "Left Forward",
    15: "Center Forward",
    16: "Right Forward",
    19: "Left Back",
    20: "Right Back",
    21: "Left Defensive Midfield",
    22: "Right Defensive Midfield"
}

def categorize_role(role_name):
    """Categorize player role into position lines"""
    if pd.isna(role_name) or role_name is None:
        return 'Unknown'
    
    defenders = ['Left Center Back', 'Right Center Back', 'Center Back', 'Left Back', 'Right Back']
    midfielders = ['Left Midfield', 'Right Midfield', 'Attacking Midfield', 
                   'Right Defensive Midfield', 'Left Defensive Midfield', 'Defensive Midfield',
                   'Left Wing Back', 'Right Wing Back']
    strikers = ['Right Forward', 'Left Winger', 'Right Winger', 'Center Forward', 'Left Forward']
    
    if role_name == 'Goalkeeper':
        return 'Goalkeeper'
    elif role_name in defenders:
        return 'Defender'
    elif role_name in midfielders:
        return 'Midfielder'
    elif role_name in strikers:
        return 'Striker'
    else:
        return 'Unknown'


def load_match_mapping():
    """Load match ID mapping"""
    print(f"[INFO] Loading match mapping from {MATCH_ID_MAPPING_FILE}")
    return pd.read_csv(MATCH_ID_MAPPING_FILE)


def load_events_data(use_goal_kick_filter=True):
    """
    Load StatsBomb events data (normalized coordinates if available)
    
    Args:
        use_goal_kick_filter (bool): If True, load only "From Goal Kick" events
    """
    import time
    
    # Determine which file to load
    if use_goal_kick_filter:
        # Try goal kick filtered files (JSON format)
        json_file = RAW_DATA_DIR / 'sb_data' / 'sb_events_goal_kick.json'
        parquet_file = RAW_DATA_DIR / 'sb_data' / 'sb_events_goal_kick.parquet'
        file_desc = "GOAL KICK FILTERED"
    else:
        # Try normalized files
        parquet_file = RAW_DATA_DIR / 'sb_data' / 'sb_events_normalized.parquet'
        json_file = RAW_DATA_DIR / 'sb_data' / 'sb_events_normalized.json'
        file_desc = "NORMALIZED"
    
    # Try JSON first for goal kick filter, Parquet first for normalized
    if use_goal_kick_filter:
        # Goal kick filter: JSON first
        if json_file.exists():
            events_file = json_file
            print(f"[INFO] Loading {file_desc} events from: {events_file}")
            print("   Loading JSON...")
            start_time = time.time()
            events_df = pd.read_json(events_file, convert_dates=False)
            elapsed = time.time() - start_time
            print(f"   Loaded {len(events_df):,} events from {events_df['match_id'].nunique()} matches ({elapsed:.1f}s)")
        elif parquet_file.exists():
            events_file = parquet_file
            print(f"[INFO] Loading {file_desc} events from: {events_file}")
            print("   Loading Parquet file (fast)...")
            start_time = time.time()
            events_df = pd.read_parquet(events_file)
            elapsed = time.time() - start_time
            print(f"   Loaded {len(events_df):,} events from {events_df['match_id'].nunique()} matches ({elapsed:.1f}s)")
        else:
            print(f"\n[ERROR] Goal kick filtered events not found!")
            print(f"   Expected: {json_file} or {parquet_file}")
            print(f"   Run filter_goal_kick_events.py first to create filtered dataset")
            return None
    else:
        # Normalized: Parquet first (faster)
        if parquet_file.exists():
            events_file = parquet_file
            print(f"[INFO] Loading {file_desc} events from: {events_file}")
            print("   Loading Parquet file (fast)...")
            start_time = time.time()
            events_df = pd.read_parquet(events_file)
            elapsed = time.time() - start_time
            print(f"   Loaded {len(events_df):,} events from {events_df['match_id'].nunique()} matches ({elapsed:.1f}s)")
        elif json_file.exists():
            events_file = json_file
            print(f"[INFO] Loading {file_desc} events from: {events_file}")
            print("   Loading JSON (this may take several minutes)...")
            start_time = time.time()
            events_df = pd.read_json(events_file, convert_dates=False)
            elapsed = time.time() - start_time
            print(f"   Loaded {len(events_df):,} events from {events_df['match_id'].nunique()} matches ({elapsed:.1f}s)")
        else:
            # Fall back to original events if neither normalized nor filtered exists
            original_file = RAW_DATA_DIR / 'sb_data' / 'sb_events.json'
            events_file = original_file
            print(f"[INFO] Filtered/normalized files not found, loading ORIGINAL events from: {events_file}")
            print(f"   [TIP] Run normalize_statsbomb_coordinates.py first for proper coordinate alignment")
            print("   Loading JSON (this may take several minutes)...")
            start_time = time.time()
            events_df = pd.read_json(events_file, convert_dates=False)
            elapsed = time.time() - start_time
            print(f"   Loaded {len(events_df):,} events from {events_df['match_id'].nunique()} matches ({elapsed:.1f}s)")
    
    return events_df


def load_pitch_dimensions(match_id):
    """Load pitch dimensions from metadata"""
    metadata_file = RAW_DATA_DIR / 'metadata_j1_2024' / f'{match_id}_metadata.json'
    
    if not metadata_file.exists():
        print(f"   Warning: Metadata not found for match {match_id}")
        return None, None
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    return metadata.get('pitch_length'), metadata.get('pitch_width')


def process_single_match(match_id, match_mapping, events_df):
    """
    Process a single match: load tracking data and merge with events
    
    Args:
        match_id (int): SkillCorner match ID
        match_mapping (pd.DataFrame): Match mapping dataframe  
        events_df (pd.DataFrame): StatsBomb events dataframe
        
    Returns:
        pd.DataFrame: Merged data or None if failed
    """
    print(f"\n[{match_id}] Processing match {match_id}...")
    
    # Get StatsBomb match ID
    match_info = match_mapping[match_mapping['skc_match_id'] == match_id]
    if len(match_info) == 0:
        print(f"   [SKIP] No mapping found")
        return None
    
    statsbomb_match_id = match_info['sb_match_id'].iloc[0]
    home_team = match_info['home_team'].iloc[0] if 'home_team' in match_info.columns else None
    away_team = match_info['away_team'].iloc[0] if 'away_team' in match_info.columns else None
    
    # Load tracking data
    tracking_file = RAW_DATA_DIR / 'tracking_j1_2024' / f'{match_id}_tracking_extrapolated.jsonl'
    if not tracking_file.exists():
        print(f"   [SKIP] Tracking file not found: {tracking_file}")
        return None
    
    print(f"   Loading tracking data...")
    try:
        tracking_df = pd.read_json(tracking_file, convert_dates=False)
        print(f"   Loaded {len(tracking_df):,} rows, {tracking_df['frame'].nunique():,} frames")
    except Exception as  e:
        print(f"   [ERROR] Failed to load tracking: {e}")
        return None
    
    # Add role information (if player_role_id column exists)
    if 'player_role_id' in tracking_df.columns:
        tracking_df['role_name'] = tracking_df['player_role_id'].map(ROLE_MAPPING)
        tracking_df['role_line'] = tracking_df['role_name'].apply(categorize_role)
    else:
        print(f"   Note: player_role_id not found, skipping role mapping")
    
    # Load pitch dimensions
    pitch_length, pitch_width = load_pitch_dimensions(match_id)
    
    # Get events for this match
    match_events = events_df[events_df['match_id'] == statsbomb_match_id].copy()
    print(f"   Found {len(match_events):,} events for StatsBomb match {statsbomb_match_id}")
    
    # Create timestamps
    # Tracking data may have timestamp as string "HH:MM:SS.SS" or numeric, events use minute/second
    timestamp_created = False
    
    # Try to use existing timestamp column first
    if 'timestamp' in tracking_df.columns:
        # Check if timestamp is string format (HH:MM:SS.SS)
        sample_ts = tracking_df['timestamp'].dropna().iloc[0] if len(tracking_df['timestamp'].dropna()) > 0 else None
        
        if sample_ts is not None and isinstance(sample_ts, str) and ':' in sample_ts:
            # Parse HH:MM:SS.SS format to seconds
            print(f"   Note: Converting timestamp from HH:MM:SS format to seconds")
            def parse_timestamp(ts_str):
                if pd.isna(ts_str) or ts_str is None:
                    return None
                try:
                    parts = ts_str.split(':')
                    hours = int(parts[0])
                    minutes = int(parts[1])
                    seconds = float(parts[2])
                    return hours * 3600 + minutes * 60 + seconds
                except:
                    return None
            
            tracking_df['timestamp'] = tracking_df['timestamp'].apply(parse_timestamp)
            timestamp_created = True
        else:
            # Try to convert to numeric
            tracking_df['timestamp'] = pd.to_numeric(tracking_df['timestamp'], errors='coerce')
            valid_timestamps = tracking_df['timestamp'].notna().sum()
            
            if valid_timestamps > 0:
                print(f"   Note: Using existing numeric timestamp column ({valid_timestamps} valid values)")
                timestamp_created = True
    
    # Remove rows with null timestamps
    if timestamp_created:
        null_count = tracking_df['timestamp'].isna().sum()
        if null_count > 0:
            print(f"   Warning: Removing {null_count} rows with null timestamps")
            tracking_df = tracking_df[tracking_df['timestamp'].notna()]
    
    # If timestamp doesn't exist or is all null, create from minute/second columns
    if not timestamp_created:
        if 'seconds' in tracking_df.columns and 'minute' in tracking_df.columns:
            tracking_df['timestamp'] = (tracking_df['minute'] * 60 + tracking_df['seconds']).astype(float)
            timestamp_created = True
        elif 'second' in tracking_df.columns and 'minute' in tracking_df.columns:
            tracking_df['timestamp'] = (tracking_df['minute'] * 60 + tracking_df['second']).astype(float)
            timestamp_created = True
    
    if not timestamp_created:
        print(f"   [ERROR] Cannot create timestamp - missing time columns")
        print(f"   Available columns: {list(tracking_df.columns)}")
        return None
    
    match_events['timestamp'] = (match_events['minute'] * 60 + match_events['second']).astype(float)
    
    # Merge tracking with events
    print(f"   Merging tracking + events...")
    
    # Check if both dataframes have 'period' column for grouping
    if 'period' in tracking_df.columns and 'period' in match_events.columns:
        # Ensure period columns have the same dtype, handling NaN values
        tracking_df['period'] = pd.to_numeric(tracking_df['period'], errors='coerce').fillna(1).astype(int)
        match_events['period'] = pd.to_numeric(match_events['period'], errors='coerce').fillna(1).astype(int)
        
        merged_df = pd.merge_asof(
            tracking_df.sort_values('timestamp'),
            match_events.sort_values('timestamp'),
            on='timestamp',
            by='period',
            direction='nearest',
            tolerance=1.0
        )
    else:
        # Merge without period grouping if column doesn't exist
        print(f"   Note: Merging without period grouping (column not found)")
        merged_df = pd.merge_asof(
            tracking_df.sort_values('timestamp'),
            match_events.sort_values('timestamp'),
            on='timestamp',
            direction='nearest',
            tolerance=1.0
        )
    
    # Add event columns
    merged_df['event_id'] = merged_df.get('id', None)
    merged_df['event_type'] = merged_df.get('type', None)
    merged_df['event_player'] = merged_df.get('player', None)
    merged_df['event_team'] = merged_df.get('team', None)
    merged_df['event_location'] = merged_df.get('location', None)
    merged_df['pass_outcome'] = merged_df.get('pass_outcome', None)
    merged_df['carry_outcome'] = merged_df.get('carry_outcome', None)
    merged_df['pass_recipient'] = merged_df.get('pass_recipient', None)
    
    # Add match info
    merged_df['skc_match_id'] = match_id
    merged_df['sb_match_id'] = statsbomb_match_id
    merged_df['pitch_length'] = pitch_length
    merged_df['pitch_width'] = pitch_width
    merged_df['home_team'] = home_team
    merged_df['away_team'] = away_team
    
    print(f"   [OK] Merged {len(merged_df):,} rows")
    return merged_df


def main(output_dir=None, max_matches=None, use_goal_kick_filter=True):
    """
    Main function to merge tracking and events data for J1 League 2024
    
    Args:
        output_dir (str/Path): Output directory (default: data/merged_j1_2024 or merged_j1_2024_goal_kick)
        max_matches (int): Maximum number of matches to process (for testing)
        use_goal_kick_filter (bool): If True, use only "From Goal Kick" events (default: True)
    """
    print("\n" + "="*70)
    print("J1 League 2024: Tracking + Events Merge (JSON version)")
    if use_goal_kick_filter:
        print("Filter: From Goal Kick events only")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup output directory
    if output_dir is None:
        if use_goal_kick_filter:
            output_dir = RAW_DATA_DIR / 'merged_j1_2024_goal_kick'
        else:
            output_dir = RAW_DATA_DIR / 'merged_j1_2024'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[INFO] Output directory: {output_dir}")
    
    # Load mappings and events
    print("\n[STEP 1] Loading data...")
    match_mapping = load_match_mapping()
    print(f"   Found {len(match_mapping)} match mappings")
    
    events_df = load_events_data(use_goal_kick_filter=use_goal_kick_filter)
    
    if events_df is None:
        print("\n[FATAL ERROR] Failed to load events data. Exiting.")
        return
    
    # Get list of tracking files
    tracking_files = sorted(glob.glob(str(RAW_DATA_DIR / 'tracking_j1_2024' / '*_tracking_extrapolated.jsonl')))
    print(f"\n[STEP 2] Found {len(tracking_files)} tracking files")
    
    if max_matches:
        tracking_files = tracking_files[:max_matches]
        print(f"   [TEST MODE] Processing only first {max_matches} matches")
    
    # Process each match
    print(f"\n[STEP 3] Processing matches...")
    successful = 0
    failed = 0
    
    for tracking_file in tracking_files:
        match_id = int(Path(tracking_file).stem.split('_')[0])
        
        try:
            merged_data = process_single_match(match_id, match_mapping, events_df)
            
            if merged_data is not None:
                # Save to JSON (handles complex nested structures)
                output_file = output_dir / f'{match_id}_merged.json'
                merged_data.to_json(output_file, orient='records', indent=2)
                print(f"   Saved to: {output_file.name}")
                successful += 1
            else:
                failed += 1
                
        except Exception as e:
            print(f"   [ERROR] Failed: {e}")
            failed += 1
    
    # Summary
    print("\n" + "="*70)
    print("[SUMMARY]")
    print(f"  Total files: {len(tracking_files)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    # For testing, process just a few matches
    # main(max_matches=3, use_goal_kick_filter=True)
    
    # For full processing with goal kick filter (default)
    main(use_goal_kick_filter=True)
    
    # For full processing without filter (all events)
    # main(use_goal_kick_filter=False)

