"""
Normalize StatsBomb coordinates (120x80) to SkillCorner coordinates (meters)

This script transforms StatsBomb event coordinates to match the actual pitch dimensions
used by SkillCorner tracking data, enabling proper spatial analysis.

Author: Modified for J1 League 2024 data
"""
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# Import paths
from paths import RAW_DATA_DIR, MATCH_ID_MAPPING_FILE

# StatsBomb standard pitch dimensions
SB_LENGTH = 120.0
SB_WIDTH = 80.0


def load_pitch_dimensions(match_id):
    """
    Load pitch dimensions from SkillCorner metadata
    
    Args:
        match_id (int): SkillCorner match ID
        
    Returns:
        tuple: (pitch_length, pitch_width) in meters, or (None, None) if not found
    """
    metadata_file = RAW_DATA_DIR / 'metadata_j1_2024' / f'{match_id}_metadata.json'
    
    if not metadata_file.exists():
        return None, None
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    return metadata.get('pitch_length'), metadata.get('pitch_width')


def normalize_coordinate(coord, scale_factor):
    """
    Normalize a single coordinate value
    
    Args:
        coord: Coordinate value (can be None/NaN)
        scale_factor: Scaling factor from SB to meters
        
    Returns:
        Normalized coordinate or None
    """
    if coord is None or pd.isna(coord):
        return None
    try:
        return float(coord) * scale_factor
    except (ValueError, TypeError):
        return None


def normalize_location(location, x_scale, y_scale):
    """
    Normalize a 2D location coordinate
    
    Args:
        location: List [x, y] or None
        x_scale: X-axis scaling factor
        y_scale: Y-axis scaling factor
        
    Returns:
        Normalized [x, y] or None
    """
    if not location or not isinstance(location, list):
        return None
    
    if len(location) >= 2:
        x = normalize_coordinate(location[0], x_scale)
        y = normalize_coordinate(location[1], y_scale)
        
        if x is not None and y is not None:
            return [x, y]
    
    return None


def normalize_location_3d(location, x_scale, y_scale):
    """
    Normalize a 3D location coordinate (e.g., shot with height)
    
    Args:
        location: List [x, y, z] or None
        x_scale: X-axis scaling factor
        y_scale: Y-axis scaling factor
        
    Returns:
        Normalized [x, y, z] or None (z unchanged)
    """
    if not location or not isinstance(location, list):
        return None
    
    if len(location) >= 3:
        x = normalize_coordinate(location[0], x_scale)
        y = normalize_coordinate(location[1], y_scale)
        z = location[2]  # Keep z-coordinate (height) unchanged
        
        if x is not None and y is not None:
            return [x, y, z]
    elif len(location) == 2:
        # Fall back to 2D if only 2 coordinates
        return normalize_location(location, x_scale, y_scale)
    
    return None


def normalize_statsbomb_events(events_df, pitch_length, pitch_width):
    """
    Normalize StatsBomb event coordinates to SkillCorner pitch dimensions
    
    Args:
        events_df (pd.DataFrame): StatsBomb events dataframe
        pitch_length (float): Actual pitch length in meters
        pitch_width (float): Actual pitch width in meters
        
    Returns:
        pd.DataFrame: Events with normalized coordinates
    """
    if pitch_length is None or pitch_width is None:
        print("   [WARNING] No pitch dimensions available, skipping normalization")
        return events_df
    
    # Calculate scale factors
    x_scale = pitch_length / SB_LENGTH
    y_scale = pitch_width / SB_WIDTH
    
    print(f"   Normalizing coordinates: SB(120x80) → SC({pitch_length:.1f}x{pitch_width:.1f})")
    print(f"   Scale factors: x={x_scale:.4f}, y={y_scale:.4f}")
    
    # Make a copy to avoid modifying original
    normalized_df = events_df.copy()
    
    # Normalize location (most events have this)
    if 'location' in normalized_df.columns:
        normalized_df['location'] = normalized_df['location'].apply(
            lambda loc: normalize_location(loc, x_scale, y_scale)
        )
        normalized_count = normalized_df['location'].notna().sum()
        print(f"   Normalized {normalized_count:,} event locations")
    
    # Normalize pass end location
    if 'pass_end_location' in normalized_df.columns:
        normalized_df['pass_end_location'] = normalized_df['pass_end_location'].apply(
            lambda loc: normalize_location(loc, x_scale, y_scale)
        )
    
    # Normalize carry end location
    if 'carry_end_location' in normalized_df.columns:
        normalized_df['carry_end_location'] = normalized_df['carry_end_location'].apply(
            lambda loc: normalize_location(loc, x_scale, y_scale)
        )
    
    # Normalize shot end location (includes z-coordinate)
    if 'shot_end_location' in normalized_df.columns:
        normalized_df['shot_end_location'] = normalized_df['shot_end_location'].apply(
            lambda loc: normalize_location_3d(loc, x_scale, y_scale)
        )
    
    # Normalize goalkeeper end location
    if 'goalkeeper_end_location' in normalized_df.columns:
        normalized_df['goalkeeper_end_location'] = normalized_df['goalkeeper_end_location'].apply(
            lambda loc: normalize_location(loc, x_scale, y_scale)
        )
    
    # Normalize shot freeze frame coordinates
    if 'shot_freeze_frame' in normalized_df.columns:
        def normalize_freeze_frame(freeze_frame):
            if not freeze_frame or not isinstance(freeze_frame, list):
                return None
            
            normalized_ff = []
            for player in freeze_frame:
                if isinstance(player, dict) and 'location' in player:
                    player_copy = player.copy()
                    player_copy['location'] = normalize_location(
                        player['location'], x_scale, y_scale
                    )
                    normalized_ff.append(player_copy)
                else:
                    normalized_ff.append(player)
            
            return normalized_ff
        
        normalized_df['shot_freeze_frame'] = normalized_df['shot_freeze_frame'].apply(
            normalize_freeze_frame
        )
    
    return normalized_df


def main():
    """
    Main function to normalize all StatsBomb events to SkillCorner coordinates
    """
    print("\n" + "="*70)
    print("StatsBomb → SkillCorner Coordinate Normalization")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load match mapping
    print("[STEP 1] Loading match mapping...")
    match_mapping = pd.read_csv(MATCH_ID_MAPPING_FILE)
    print(f"   Found {len(match_mapping)} match mappings\n")
    
    # Load StatsBomb events
    print("[STEP 2] Loading StatsBomb events...")
    events_file = RAW_DATA_DIR / 'sb_data' / 'sb_events.json'
    print(f"   File: {events_file}")
    print("   This may take several minutes for large files...")
    
    events_df = pd.read_json(events_file, convert_dates=False)
    print(f"   Loaded {len(events_df):,} events from {events_df['match_id'].nunique()} matches\n")
    
    # Process each match
    print("[STEP 3] Normalizing coordinates by match...")
    normalized_events = []
    matches_processed = 0
    matches_failed = 0
    
    for sb_match_id in events_df['match_id'].unique():
        # Get corresponding SkillCorner match ID
        mapping = match_mapping[match_mapping['sb_match_id'] == sb_match_id]
        
        if len(mapping) == 0:
            print(f"   [SKIP] SB match {sb_match_id}: No SkillCorner mapping")
            matches_failed += 1
            continue
        
        skc_match_id = mapping['skc_match_id'].iloc[0]
        
        # Load pitch dimensions
        pitch_length, pitch_width = load_pitch_dimensions(skc_match_id)
        
        if pitch_length is None or pitch_width is None:
            print(f"   [SKIP] SB match {sb_match_id}: No metadata found")
            matches_failed += 1
            # Keep original events without normalization
            match_events = events_df[events_df['match_id'] == sb_match_id].copy()
            normalized_events.append(match_events)
            continue
        
        # Get events for this match
        match_events = events_df[events_df['match_id'] == sb_match_id].copy()
        
        print(f"   [MATCH {skc_match_id}] SB:{sb_match_id} - {len(match_events):,} events")
        
        # Normalize coordinates
        normalized_match_events = normalize_statsbomb_events(
            match_events, pitch_length, pitch_width
        )
        
        normalized_events.append(normalized_match_events)
        matches_processed += 1
    
    # Combine all normalized events
    print(f"\n[STEP 4] Combining normalized events...")
    final_events_df = pd.concat(normalized_events, ignore_index=True)
    print(f"   Total events: {len(final_events_df):,}")
    
    # Save normalized events
    output_file = RAW_DATA_DIR / 'sb_data' / 'sb_events_normalized.parquet'
    print(f"\n[STEP 5] Saving normalized events...")
    print(f"   Output: {output_file}")
    print(f"   Saving as Parquet (compressed, fast loading)...")
    
    final_events_df.to_parquet(output_file, compression='snappy', index=False)
    
    # Summary
    print("\n" + "="*70)
    print("[SUMMARY]")
    print(f"  Matches processed: {matches_processed}")
    print(f"  Matches skipped: {matches_failed}")
    print(f"  Total events: {len(final_events_df):,}")
    print(f"  Output file: {output_file}")
    print(f"  File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    print("="*70)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    main()
