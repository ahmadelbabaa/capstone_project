"""
Merge tracking data with events data

Author: Raimundo Oyarce - Aztec DH
Email: royarce@aztechdh.com
"""
import pandas as pd
import numpy as np
import glob
import os
import sys
import json
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time

# Import for coordinate transformations
try:
    from mplsoccer import Standardizer
    HAS_MPLSOCCER = True
except ImportError:
    HAS_MPLSOCCER = False
    print("Warning: mplsoccer not available, coordinate transformations will use proportional scaling")

# Try to import tqdm, fallback to basic progress if not available
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: tqdm not available, using basic progress reporting")

# Handle imports for both direct execution and module import
try:
    from .paths import MATCH_ID_MAPPING_FILE, RAW_DATA_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, USL_TRACKING_DIR, USL_DATA_DIR
    from .match_id_mapping import main as update_match_mappings
except ImportError:
    # Add parent directory to path for direct execution
    sys.path.append(str(Path(__file__).parent))
    from paths import MATCH_ID_MAPPING_FILE, RAW_DATA_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, USL_TRACKING_DIR, USL_DATA_DIR
    from match_id_mapping import main as update_match_mappings

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
    """
    Categorize player role into defensive line, midfield line, or attacking line
    
    Args:
        role_name (str): The role name from ROLE_MAPPING
        
    Returns:
        str: Category of the role ('Defender', 'Midfielder', 'Striker', 'Goalkeeper', or 'Unknown')
    """
    if pd.isna(role_name) or role_name is None:
        return 'Unknown'
    
    defenders = [
        'Left Center Back', 'Right Center Back', 'Center Back',
        'Left Back', 'Right Back'
    ]
    
    midfielders = [
        'Left Midfield', 'Right Midfield', 'Attacking Midfield', 
        'Right Defensive Midfield', 'Left Defensive Midfield', 'Defensive Midfield',
        'Left Wing Back', 'Right Wing Back'
    ]
    
    strikers = [
        'Right Forward', 'Left Winger', 'Right Winger',
        'Center Forward', 'Left Forward'
    ]
    
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

def get_raw_tracking_files():
    """
    Get all raw tracking JSON files
    
    Returns:
        list: List of raw tracking file paths
    """
    tracking_pattern = str(USL_TRACKING_DIR / 'tracking_*.json')
    raw_files = glob.glob(tracking_pattern)
    return raw_files

def get_processed_tracking_files():
    """
    Get all processed tracking parquet files with velocity
    
    Returns:
        set: Set of match IDs that have been processed
    """
    tracking_pattern = str(PROCESSED_DATA_DIR / 'tracking_*_with_velocity.parquet')
    processed_files = glob.glob(tracking_pattern)
    
    processed_match_ids = set()
    for file_path in processed_files:
        # Extract match_id from filename
        filename = os.path.basename(file_path)
        match_id = int(filename.split('_')[1])
        processed_match_ids.add(match_id)
    
    return processed_match_ids

def get_unprocessed_raw_files():
    """
    Get raw tracking files that haven't been processed to parquet with velocity yet
    
    Returns:
        list: List of unprocessed raw tracking file paths
    """
    raw_files = get_raw_tracking_files()
    processed_match_ids = get_processed_tracking_files()
    
    unprocessed_files = []
    for raw_file in raw_files:
        # Extract match_id from filename
        filename = os.path.basename(raw_file)
        match_id = int(filename.split('_')[1].split('.')[0])
        
        if match_id not in processed_match_ids:
            unprocessed_files.append(raw_file)
    
    print(f"Found {len(unprocessed_files)} unprocessed raw tracking files out of {len(raw_files)} total")
    return unprocessed_files

def process_raw_tracking_file(match_id, show_progress=False):
    """
    Process a single raw tracking file to create parquet with velocity
    
    Args:
        match_id (int): Match ID to process
        show_progress (bool): Whether to show detailed progress output
        
    Returns:
        tuple: (match_id, success, error_message)
    """
    try:
        # Path to the process_match.py script
        script_path = PROJECT_ROOT / "src" / "data" / "process_match.py"
        
        cmd = [
            sys.executable, str(script_path), str(match_id),
            '--frame-gap', '25',
            '--output-dir', str(PROCESSED_DATA_DIR)
        ]
        
        if show_progress:
            # Don't capture output so we can see the detailed progress
            print(f"  📋 Loading raw tracking data for match {match_id}...")
            print(f"  🔧 Command: {' '.join(cmd[-3:])}")  # Show relevant parts of command
            print(f"  🔄 Starting detailed processing (this may take several minutes):")
            print("     ↳ Loading JSON data...")
            print("     ↳ Processing tracking frames...")
            print("     ↳ Calculating velocities...")
            print("     ↳ Saving to parquet format...")
            print()
            
            result = subprocess.run(cmd, check=True)
            print(f"\n  ✅ Match {match_id} processing completed successfully!")
        else:
            # Capture output for parallel processing (cleaner logs)
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        return match_id, True, None
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Subprocess error: {str(e)}"
        if hasattr(e, 'stderr') and e.stderr:
            error_msg += f" - {e.stderr}"
        return match_id, False, error_msg
    except Exception as e:
        return match_id, False, f"Unexpected error: {str(e)}"

def process_raw_tracking_file_wrapper(args):
    """
    Wrapper function for multiprocessing compatibility
    
    Args:
        args: Tuple containing (match_id, raw_file_path, show_progress)
        
    Returns:
        tuple: (match_id, success, error_message)
    """
    match_id, raw_file_path, show_progress = args
    return process_raw_tracking_file(match_id, show_progress=show_progress)

def process_all_unprocessed_raw_files(max_workers=None, use_parallel=True):
    """
    Process all raw tracking files that haven't been converted to parquet with velocity yet
    
    Args:
        max_workers (int): Maximum number of parallel workers. If None, uses CPU count
        use_parallel (bool): Whether to use parallel processing. If False, processes sequentially
    
    Returns:
        tuple: (total_files, successfully_processed)
    """
    print("🔄 Checking for unprocessed raw tracking files...")
    
    unprocessed_files = get_unprocessed_raw_files()
    
    if not unprocessed_files:
        print("✅ All raw tracking files have already been processed!")
        return 0, 0
    
    # Determine number of workers
    if max_workers is None:
        max_workers = min(4, len(unprocessed_files))
    else:
        max_workers = min(max_workers, len(unprocessed_files))
    
    print(f"🔄 Processing {len(unprocessed_files)} raw tracking files...")
    
    if use_parallel and len(unprocessed_files) > 1:
        print(f"🚀 Using parallel processing with {max_workers} workers")
        print("💡 If the process is interrupted by memory, try: --sequential")
        return _process_files_parallel(unprocessed_files, max_workers)
    else:
        print("🔄 Using sequential processing")
        return _process_files_sequential(unprocessed_files)

def _process_files_sequential(unprocessed_files):
    """
    Process files sequentially with detailed progress for each match
    
    Args:
        unprocessed_files (list): List of unprocessed file paths
        
    Returns:
        tuple: (total_files, successfully_processed)
    """
    successfully_processed = 0
    total_processing_time = 0
    
    print("🔄 Sequential processing - showing detailed progress for each match:")
    print("=" * 70)
    
    for i, raw_file in enumerate(unprocessed_files, 1):
        # Extract match_id from filename
        filename = os.path.basename(raw_file)
        match_id = int(filename.split('_')[1].split('.')[0])
        
        print(f"\n📊 [{i}/{len(unprocessed_files)}] Starting Match {match_id}")
        print("-" * 50)
        
        # Show file size information
        file_size_mb = os.path.getsize(raw_file) / (1024 * 1024)
        print(f"📁 File size: {file_size_mb:.1f} MB")
        print(f"⏱️  Estimated time: {file_size_mb/50:.1f}-{file_size_mb/30:.1f} minutes")
        
        # Track processing time
        start_time = time.time()
        
        # Use show_progress=True for detailed output
        match_id_result, success, error_msg = process_raw_tracking_file(match_id, show_progress=True)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        elapsed_minutes = elapsed_time / 60
        
        if success:
            print(f"✅ Match {match_id} completed successfully!")
            print(f"⏱️  Actual processing time: {elapsed_minutes:.1f} minutes")
            successfully_processed += 1
            total_processing_time += elapsed_time
        else:
            print(f"❌ Error processing match {match_id}: {error_msg}")
            print(f"⏱️  Time before error: {elapsed_minutes:.1f} minutes")
        
        print("-" * 50)
        
        # Show overall progress and time estimates
        remaining = len(unprocessed_files) - i
        print(f"📈 Progress: {i}/{len(unprocessed_files)} completed, {remaining} remaining")
        
        if i > 0 and successfully_processed > 0:
            avg_time_per_file = total_processing_time / successfully_processed
            estimated_remaining_time = avg_time_per_file * remaining / 60  # in minutes
            print(f"⏱️  Average time per file: {avg_time_per_file/60:.1f} minutes")
            print(f"🕐 Estimated time remaining: {estimated_remaining_time:.1f} minutes ({estimated_remaining_time/60:.1f} hours)")
    
    print(f"\n🎉 Sequential processing complete!")
    print(f"✅ Successfully processed: {successfully_processed}/{len(unprocessed_files)} files")
    
    if successfully_processed > 0:
        total_time_hours = total_processing_time / 3600
        avg_time_minutes = (total_processing_time / successfully_processed) / 60
        print(f"⏱️  Total processing time: {total_time_hours:.1f} hours")
        print(f"📊 Average time per file: {avg_time_minutes:.1f} minutes")
    
    return len(unprocessed_files), successfully_processed

def _process_files_parallel(unprocessed_files, max_workers):
    """
    Process files in parallel using ProcessPoolExecutor
    
    Args:
        unprocessed_files (list): List of unprocessed file paths
        max_workers (int): Maximum number of parallel workers
        
    Returns:
        tuple: (total_files, successfully_processed)
    """
    successfully_processed = 0
    failed_files = []
    
    # Prepare arguments for parallel processing
    processing_args = []
    for raw_file in unprocessed_files:
        filename = os.path.basename(raw_file)
        match_id = int(filename.split('_')[1].split('.')[0])
        processing_args.append((match_id, raw_file, False))  # show_progress=False for parallel
    
    print(f"🚀 Starting parallel processing of {len(processing_args)} files...")
    
    # Use ProcessPoolExecutor for CPU-bound tasks
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_match = {
            executor.submit(process_raw_tracking_file_wrapper, args): args[0] 
            for args in processing_args
        }
        
        # Use tqdm for progress bar if available, otherwise basic progress
        if HAS_TQDM:
            with tqdm(total=len(future_to_match), desc="Processing matches", unit="match") as pbar:
                for future in as_completed(future_to_match):
                    match_id = future_to_match[future]
                    try:
                        result_match_id, success, error_msg = future.result()
                        
                        if success:
                            pbar.set_postfix_str(f"✅ Match {result_match_id} completed")
                            successfully_processed += 1
                        else:
                            pbar.set_postfix_str(f"❌ Match {result_match_id} failed")
                            failed_files.append((result_match_id, error_msg))
                            
                    except Exception as e:
                        pbar.set_postfix_str(f"❌ Match {match_id} exception")
                        failed_files.append((match_id, f"Future exception: {str(e)}"))
                    
                    pbar.update(1)
        else:
            # Basic progress without tqdm
            completed = 0
            total_tasks = len(future_to_match)
            for future in as_completed(future_to_match):
                match_id = future_to_match[future]
                completed += 1
                try:
                    result_match_id, success, error_msg = future.result()
                    
                    if success:
                        print(f"  ✅ ({completed}/{total_tasks}) Match {result_match_id} completed")
                        successfully_processed += 1
                    else:
                        print(f"  ❌ ({completed}/{total_tasks}) Match {result_match_id} failed: {error_msg}")
                        failed_files.append((result_match_id, error_msg))
                        
                except Exception as e:
                    print(f"  ❌ ({completed}/{total_tasks}) Match {match_id} exception: {str(e)}")
                    failed_files.append((match_id, f"Future exception: {str(e)}"))
    
    # Print results summary
    print(f"\n✅ Parallel processing complete!")
    print(f"   Successfully processed: {successfully_processed}/{len(unprocessed_files)} files")
    
    if failed_files:
        print(f"   Failed files ({len(failed_files)}):")
        for match_id, error_msg in failed_files:
            print(f"     ❌ Match {match_id}: {error_msg}")
    
    return len(unprocessed_files), successfully_processed

def update_match_id_mappings_automatically():
    """
    Automatically update match ID mappings by finding unmapped matches
    and matching them based on team names and dates (with 1-day tolerance)
    
    Returns:
        bool: True if update was successful, False otherwise
    """
    try:
        print("🔄 Automatically updating match ID mappings...")
        
        # Load existing mapping
        if MATCH_ID_MAPPING_FILE.exists():
            existing_mapping = pd.read_csv(MATCH_ID_MAPPING_FILE)
            existing_skillcorner_ids = set(existing_mapping['skillcorner_match_id'])
            print(f"📋 Found {len(existing_mapping)} existing mappings")
        else:
            existing_mapping = pd.DataFrame()
            existing_skillcorner_ids = set()
            print("📋 No existing mapping file found, creating new one")
        
        # Load team mapping
        team_mapping_file = PROJECT_ROOT / 'config' / 'team_id_mapping.csv'
        team_mapping = pd.read_csv(team_mapping_file)
        
        # Create team name mappings
        sc_to_sb_team = dict(zip(team_mapping['skillcorner_team_name'], team_mapping['statsbomb_team_name']))
        sc_to_sb_id = dict(zip(team_mapping['skillcorner_team_name'], team_mapping['statsbomb_team_id']))
        
        # Load SkillCorner matches
        print("📊 Loading SkillCorner matches...")
        sc_matches_file = USL_DATA_DIR / 'all_matches.json'
        if not sc_matches_file.exists():
            print(f"❌ SkillCorner matches file not found: {sc_matches_file}")
            return False
            
        with open(sc_matches_file, 'r') as f:
            sc_matches = json.load(f)
        
        # Load StatsBomb matches
        print("📊 Loading StatsBomb matches...")
        sb_matches_file = RAW_DATA_DIR / 'USLChampionship_2025_matches.parquet'
        if not sb_matches_file.exists():
            print(f"❌ StatsBomb matches file not found: {sb_matches_file}")
            return False
            
        sb_matches = pd.read_parquet(sb_matches_file)
        
        # Convert StatsBomb dates to string format for comparison
        sb_matches['date'] = pd.to_datetime(sb_matches['match_date']).dt.strftime('%Y-%m-%d')
        
        # Find unmapped SkillCorner matches
        unmapped_matches = []
        for match in sc_matches:
            sc_match_id = match['id']
            if sc_match_id not in existing_skillcorner_ids:
                unmapped_matches.append(match)
        
        print(f"🔍 Found {len(unmapped_matches)} unmapped matches")
        
        if not unmapped_matches:
            print("✅ All matches are already mapped!")
            return True
        
        # Find new mappings
        new_mappings = []
        mapped_count = 0
        
        for match in unmapped_matches:
            sc_match_id = match['id']
            sc_date = pd.to_datetime(match['date_time']).strftime('%Y-%m-%d')
            sc_home_team = match['home_team']['short_name']
            sc_away_team = match['away_team']['short_name']
            
            # Check if teams exist in mapping
            if sc_home_team not in sc_to_sb_team or sc_away_team not in sc_to_sb_team:
                continue
            
            sb_home_team = sc_to_sb_team[sc_home_team]
            sb_away_team = sc_to_sb_team[sc_away_team]
            
            # Find matching StatsBomb match with date tolerance
            potential_matches = sb_matches[
                (sb_matches['home_team'] == sb_home_team) &
                (sb_matches['away_team'] == sb_away_team)
            ]
            
            if len(potential_matches) == 0:
                continue
            
            # Check dates with 1-day tolerance
            best_match = None
            min_date_diff = float('inf')
            
            for _, sb_match in potential_matches.iterrows():
                sb_date = sb_match['date']
                date_diff = abs((pd.to_datetime(sc_date) - pd.to_datetime(sb_date)).days)
                
                if date_diff <= 1 and date_diff < min_date_diff:
                    min_date_diff = date_diff
                    best_match = sb_match
            
            if best_match is not None:
                # Get team IDs
                sb_home_id = sc_to_sb_id[sc_home_team]
                sb_away_id = sc_to_sb_id[sc_away_team]
                
                new_mapping = {
                    'skillcorner_match_id': sc_match_id,
                    'statsbomb_match_id': best_match['match_id'],
                    'date': sc_date,
                    'skillcorner_home_team_id': match['home_team']['id'],
                    'skillcorner_home_team_name': sc_home_team,
                    'statsbomb_home_team_id': sb_home_id,
                    'statsbomb_home_team_name': sb_home_team,
                    'skillcorner_away_team_id': match['away_team']['id'],
                    'skillcorner_away_team_name': sc_away_team,
                    'statsbomb_away_team_id': sb_away_id,
                    'statsbomb_away_team_name': sb_away_team
                }
                
                new_mappings.append(new_mapping)
                mapped_count += 1
                
                print(f"   ✅ Mapped match {sc_match_id} → {best_match['match_id']} ({sc_home_team} vs {sc_away_team}) - Date diff: {min_date_diff} days")
        
        if new_mappings:
            print(f"🎯 Found {len(new_mappings)} new mappings")
            
            # Create new mappings DataFrame
            new_mappings_df = pd.DataFrame(new_mappings)
            
            # Combine with existing mappings
            if len(existing_mapping) > 0:
                combined_mapping = pd.concat([existing_mapping, new_mappings_df], ignore_index=True)
            else:
                combined_mapping = new_mappings_df
            
            # Sort by date and match ID
            combined_mapping['date'] = pd.to_datetime(combined_mapping['date'])
            combined_mapping = combined_mapping.sort_values(['date', 'skillcorner_match_id'], ascending=[False, True])
            combined_mapping['date'] = combined_mapping['date'].dt.strftime('%Y-%m-%d')
            
            # Save updated mapping
            combined_mapping.to_csv(MATCH_ID_MAPPING_FILE, index=False)
            print(f"💾 Updated mapping file with {len(combined_mapping)} total mappings")
            
            # Show summary of new mappings
            print(f"\n📊 New mappings summary:")
            for mapping in new_mappings:
                print(f"   Match {mapping['skillcorner_match_id']}: {mapping['skillcorner_home_team_name']} vs {mapping['skillcorner_away_team_name']}")
                print(f"     → StatsBomb {mapping['statsbomb_match_id']}: {mapping['statsbomb_home_team_name']} vs {mapping['statsbomb_away_team_name']}")
                print(f"     Date: {mapping['date']}")
            
            return True
        else:
            print("⚠️  No new mappings found")
            return True
            
    except Exception as e:
        print(f"❌ Error updating match mappings automatically: {e}")
        print("Proceeding with existing mapping file...")
        return False

def update_match_id_mappings():
    """
    Update match ID mappings by running the mapping update process
    
    Returns:
        bool: True if update was successful, False otherwise
    """
    try:
        print("🔄 Updating match ID mappings...")
        
        # First try automatic update
        if update_match_id_mappings_automatically():
            print("✅ Automatic match ID mapping update completed!")
            return True
        else:
            # Fallback to manual update
            print("🔄 Falling back to manual update...")
            update_match_mappings()
            print("✅ Manual match ID mappings updated successfully!")
            return True
    except Exception as e:
        print(f"❌ Error updating match mappings: {e}")
        print("Proceeding with existing mapping file...")
        return False

def load_match_mapping():
    """
    Load match ID mapping after ensuring it's updated
    
    Returns:
        pd.DataFrame: Match mapping dataframe
    """
    # Load match ID mapping (don't auto-update here since we do it in main)
    return pd.read_csv(MATCH_ID_MAPPING_FILE)

def load_events_data():
    """
    Load events data from parquet file
    
    Returns:
        pd.DataFrame: Events dataframe
    """
    # Load events data
    events_file = RAW_DATA_DIR / 'USLChampionship_2025.parquet'
    events_df = pd.read_parquet(events_file)
    return events_df

def load_pitch_dimensions(skillcorner_match_id):
    """
    Load pitch dimensions from match JSON file
    
    Args:
        skillcorner_match_id (int): SkillCorner match ID
        
    Returns:
        tuple: (pitch_length, pitch_width) or (None, None) if not found
    """
    try:
        match_file = RAW_DATA_DIR / 'usl_championship_2025' / 'matches' / f'match_{skillcorner_match_id}.json'
        
        if not match_file.exists():
            print(f"Warning: Match file not found for match {skillcorner_match_id}")
            return None, None
            
        with open(match_file, 'r') as f:
            match_data = json.load(f)
            
        pitch_length = match_data.get('pitch_length')
        pitch_width = match_data.get('pitch_width')
        
        return pitch_length, pitch_width
        
    except Exception as e:
        print(f"Error loading pitch dimensions for match {skillcorner_match_id}: {e}")
        return None, None

def subsample_tracking_data(tracking_df, subsample_seconds=0.2):
    """
    Subsample tracking data to keep frames with timestamps divisible by subsample_seconds.
    Examples:
    - subsample_seconds=0.1: keeps frames at 121.0, 121.1, 121.2, 121.3, etc.
    - subsample_seconds=0.2: keeps frames at 121.0, 121.2, 121.4, 121.6, etc.
    - subsample_seconds=0.5: keeps frames at 121.0, 121.5, 122.0, 122.5, etc.
    
    Args:
        tracking_df (pd.DataFrame): Full tracking data
        subsample_seconds (float): Target interval in seconds (default: 0.2s)
        
    Returns:
        pd.DataFrame: Subsampled tracking data with frames at exact intervals
    """
    print(f"Subsampling tracking data every {subsample_seconds} seconds...")
    
    # Create timestamp if not available
    if 'timestamp' not in tracking_df.columns and 'seconds' in tracking_df.columns:
        tracking_df = tracking_df.copy()
        tracking_df['timestamp'] = tracking_df['minute'] * 60 + tracking_df['seconds']
        print("Created timestamp from minute and seconds columns")
    
    if 'timestamp' not in tracking_df.columns:
        print("Warning: No timestamp column found, cannot perform time-based subsampling")
        return tracking_df
    
    # Group by period and process each period separately
    subsampled_data = []
    
    for period in tracking_df['period'].unique():
        period_data = tracking_df[tracking_df['period'] == period].copy()
        
        # Sort by timestamp
        period_data = period_data.sort_values('timestamp')
        
        # Get unique timestamps
        unique_timestamps = sorted(period_data['timestamp'].unique())
        
        # Select timestamps that are divisible by subsample_seconds
        selected_timestamps = []
        for timestamp in unique_timestamps:
            # Check if timestamp is divisible by subsample_seconds (with small tolerance for floating point)
            remainder = timestamp % subsample_seconds
            if remainder < 0.001 or abs(remainder - subsample_seconds) < 0.001:
                selected_timestamps.append(timestamp)
        
        # Filter data to selected timestamps only
        subsampled_period = period_data[period_data['timestamp'].isin(selected_timestamps)]
        subsampled_data.append(subsampled_period)
        
        # Show examples of selected timestamps
        if selected_timestamps:
            examples = selected_timestamps[:5]
            print(f"  Period {period}: {len(unique_timestamps)} → {len(selected_timestamps)} frames")
            print(f"    Examples: {[f'{t:.1f}s' for t in examples]}")
            
            # Show preservation rate
            preservation_rate = len(selected_timestamps) / len(unique_timestamps) * 100
            print(f"    Preservation rate: {preservation_rate:.1f}%")
        else:
            print(f"  Period {period}: No frames found at {subsample_seconds}s intervals")
    
    # Combine all periods
    result_df = pd.concat(subsampled_data, ignore_index=True)
    
    # Show overall statistics
    original_frames = len(tracking_df['frame_number'].unique())
    result_frames = len(result_df['frame_number'].unique())
    total_preservation = len(result_df) / len(tracking_df) * 100
    
    print(f"Total frames: {original_frames} → {result_frames} ({total_preservation:.1f}% of original data)")
    
    # Show some examples of the selected timestamps
    if len(result_df) > 0:
        all_timestamps = sorted(result_df['timestamp'].unique())
        if len(all_timestamps) >= 5:
            print(f"Sample selected timestamps: {[f'{t:.1f}s' for t in all_timestamps[:5]]}")
    
    return result_df

def filter_priority_events(events_df, priority_types=['Pass', 'Carry', 'Shot']):
    """
    Filter events intelligently: apply priority filtering only when multiple events compete 
    for the same time window, otherwise keep all events
    
    Args:
        events_df (pd.DataFrame): Full events data
        priority_types (list): List of event types to prioritize when there are conflicts
        
    Returns:
        pd.DataFrame: Intelligently filtered events data
    """
    print(f"Applying intelligent event filtering with priority types: {priority_types}")
    
    # Show original event distribution
    original_count = len(events_df)
    event_counts = events_df['type'].value_counts()
    print(f"Original events: {original_count:,}")
    
    # Create timestamp for temporal analysis
    events_df['timestamp'] = (events_df['minute'] * 60 + events_df['second']).astype(float)
    
    # Group events by period and timestamp windows (1-second tolerance)
    # to identify when multiple events compete for the same time window
    events_df['time_window'] = (events_df['timestamp']).round(0)  # Round to nearest second
    
    # Identify time windows with multiple events
    conflicts_by_period = {}
    total_conflicts = 0
    
    for period in events_df['period'].unique():
        period_events = events_df[events_df['period'] == period]
        time_window_counts = period_events.groupby('time_window').size()
        conflict_windows = time_window_counts[time_window_counts > 1].index
        conflicts_by_period[period] = set(conflict_windows)
        total_conflicts += len(conflict_windows)
        
        if len(conflict_windows) > 0:
            print(f"  Period {period}: {len(conflict_windows)} time windows with multiple events")
    
    print(f"Total conflicting time windows: {total_conflicts}")
    
    if total_conflicts == 0:
        print("✅ No event conflicts detected - keeping all events")
        # Remove temporary columns
        result_df = events_df.drop(['time_window'], axis=1)
        return result_df
    
    # Apply intelligent filtering: only filter events in conflicting time windows
    filtered_events = []
    priority_filtered_count = 0
    
    for period in events_df['period'].unique():
        period_events = events_df[events_df['period'] == period].copy()
        conflict_windows = conflicts_by_period[period]
        
        for time_window in period_events['time_window'].unique():
            window_events = period_events[period_events['time_window'] == time_window]
            
            if time_window in conflict_windows:
                # Multiple events in this window - apply priority filtering
                priority_events_in_window = window_events[window_events['type'].isin(priority_types)]
                
                if len(priority_events_in_window) > 0:
                    # Keep priority events only
                    filtered_events.append(priority_events_in_window)
                    priority_filtered_count += len(window_events) - len(priority_events_in_window)
                else:
                    # No priority events in window, keep first event (by timestamp)
                    filtered_events.append(window_events.iloc[:1])
                    priority_filtered_count += len(window_events) - 1
            else:
                # Single event in this window - keep regardless of type
                filtered_events.append(window_events)
    
    # Combine all filtered events
    if filtered_events:
        # Filter out empty DataFrames before concatenation
        valid_events = [df for df in filtered_events if not df.empty and len(df) > 0]
        
        if valid_events:
            try:
                result_df = pd.concat(valid_events, ignore_index=True)
            except Exception as e:
                print(f"⚠️  Error in pd.concat, trying alternative approach: {e}")
                # Fallback: combine manually
                result_df = pd.concat(valid_events, ignore_index=True, sort=False)
        else:
            result_df = events_df.head(0).copy()  # Empty dataframe with same structure
    else:
        result_df = events_df.head(0).copy()  # Empty dataframe with same structure
    
    # Remove temporary columns
    result_df = result_df.drop(['time_window'], axis=1)
    
    # Show filtering results
    filtered_count = len(result_df)
    filtered_event_counts = result_df['type'].value_counts()
    
    print(f"Filtered events: {filtered_count:,} ({filtered_count/original_count*100:.1f}% of original)")
    print(f"Events filtered due to conflicts: {priority_filtered_count:,}")
    print(f"Event distribution after intelligent filtering:")
    for event_type in priority_types:
        if event_type in filtered_event_counts:
            print(f"  {event_type}: {filtered_event_counts[event_type]:,}")
    
    # Show other event types that were preserved
    other_types = filtered_event_counts[~filtered_event_counts.index.isin(priority_types)]
    if len(other_types) > 0:
        print(f"Other preserved event types:")
        for event_type, count in other_types.head(5).items():
            print(f"  {event_type}: {count:,}")
    
    return result_df

def transform_statsbomb_to_skillcorner(x_sb, y_sb, pitch_length, pitch_width):
    """
    Transform coordinates from StatsBomb to SkillCorner using proper standardization.
    
    This function uses mplsoccer's Standardizer to maintain relative positions 
    to pitch markings (penalty areas, center circle, etc.) when converting 
    between data providers, avoiding the problems of naive linear scaling.
    
    Args:
        x_sb (float): StatsBomb x-coordinate (0-120)
        y_sb (float): StatsBomb y-coordinate (0-80)  
        pitch_length (float): Real pitch length in meters (SkillCorner)
        pitch_width (float): Real pitch width in meters (SkillCorner)
        
    Returns:
        tuple: (x_sc, y_sc) in SkillCorner coordinates
    """
    if pd.isna(x_sb) or pd.isna(y_sb) or pd.isna(pitch_length) or pd.isna(pitch_width):
        return None, None
    
    try:
        if HAS_MPLSOCCER:
            # Create standardizer from StatsBomb to SkillCorner
            # StatsBomb uses 120x80 dimensions, SkillCorner uses real pitch dimensions
            standardizer = Standardizer(pitch_from='statsbomb', pitch_to='skillcorner',
                                       length_to=pitch_length, width_to=pitch_width)
            
            # Transform coordinates - standardizer expects arrays
            x_transformed, y_transformed = standardizer.transform(
                np.array([x_sb]), np.array([y_sb])
            )
            
            return float(x_transformed[0]), float(y_transformed[0])
        else:
            # Fallback to proportional scaling if mplsoccer not available
            x_sc = (x_sb / 120.0) * pitch_length
            y_sc = (y_sb / 80.0) * pitch_width
            return x_sc, y_sc
        
    except Exception as e:
        # Fallback to proportional scaling if standardizer fails
        print(f"⚠️ Standardizer failed, using proportional scaling: {e}")
        x_sc = (x_sb / 120.0) * pitch_length
        y_sc = (y_sb / 80.0) * pitch_width
        return x_sc, y_sc

def flip_statsbomb_coordinates(x_sb, y_sb):
    """
    Flip StatsBomb coordinates when team is attacking towards left (attacking_half = 'left').
    
    StatsBomb coordinates are always oriented so teams attack from left to right (0->120).
    But when a team's attacking_half is 'left', they are actually attacking towards x=0,
    so we need to flip the coordinates.
    
    Args:
        x_sb (float): Original StatsBomb x-coordinate (0-120)
        y_sb (float): Original StatsBomb y-coordinate (0-80)
        
    Returns:
        tuple: (flipped_x, flipped_y) in StatsBomb coordinates
    """
    if pd.isna(x_sb) or pd.isna(y_sb):
        return None, None
    
    # Flip x-coordinate: 120 - x (so 0 becomes 120, 120 becomes 0)
    flipped_x = 120.0 - float(x_sb)
    # Flip y-coordinate: 80 - y (so 0 becomes 80, 80 becomes 0)  
    flipped_y = 80.0 - float(y_sb)
    
    return flipped_x, flipped_y

def extract_coordinates_from_location(location):
    """
    Extract x, y coordinates from various location formats.
    
    Args:
        location: Can be list, tuple, numpy array, or string representation
        
    Returns:
        tuple: (x, y) or (None, None) if extraction fails
    """
    if location is None or (hasattr(location, 'size') and location.size == 0):
        return None, None
    
    try:
        # Handle different data types
        if hasattr(location, 'shape'):  # numpy array
            if location.shape[0] >= 2:
                return float(location[0]), float(location[1])
        elif isinstance(location, str):
            # If string, try to parse as list
            import ast
            location = ast.literal_eval(location)
            if len(location) >= 2:
                return float(location[0]), float(location[1])
        elif isinstance(location, (list, tuple)):
            if len(location) >= 2:
                return float(location[0]), float(location[1])
    except (ValueError, TypeError, SyntaxError, IndexError, AttributeError):
        pass
    
    return None, None

def process_tracking_file(tracking_file, events_df, match_mapping, subsample_seconds=0.2, priority_events=['Pass', 'Carry', 'Shot'], preserve_all_frames=False):
    """
    Process a single tracking file and merge it with events data (optimized for memory)
    Uses intelligent event filtering: only applies priority filtering when multiple events 
    compete for the same time window, otherwise preserves all events.
    
    Args:
        tracking_file (str): Path to tracking file
        events_df (pd.DataFrame): Events dataframe
        match_mapping (pd.DataFrame): Match mapping dataframe
        subsample_seconds (float): Interval for subsampling tracking data (default: 0.2s)
        priority_events (list): List of priority event types when resolving conflicts
        
    Returns:
        pd.DataFrame: Merged tracking and events data (memory optimized)
    """
    # Extract match_id from filename
    match_id = int(os.path.basename(tracking_file).split('_')[1])
    
    # Get corresponding StatsBomb match_id
    match_info = match_mapping[match_mapping['skillcorner_match_id'] == match_id]
    if len(match_info) == 0:
        print(f"No mapping found for match {match_id}")
        return None
    
    statsbomb_match_id = match_info['statsbomb_match_id'].iloc[0]
    
    # Load tracking data
    print(f"Loading tracking data from {tracking_file}")
    tracking_df = pd.read_parquet(tracking_file)
    print(f"Original tracking data: {len(tracking_df):,} rows, {len(tracking_df['frame_number'].unique()):,} unique frames")
    
    # Apply frame preservation logic based on requirements
    if preserve_all_frames:
        print("🔧 PRESERVE_ALL_FRAMES mode: Keeping all tracking data for each frame")
    else:
        # Apply intelligent frame preservation to maintain target frequency
        tracking_df = subsample_tracking_data(tracking_df, subsample_seconds=subsample_seconds)
    
    # Add role name based on player_role_id
    tracking_df['role_name'] = tracking_df['player_role_id'].map(ROLE_MAPPING)
    
    # Add line category based on role name
    tracking_df['role_line'] = tracking_df['role_name'].apply(categorize_role)
    
    # Load pitch dimensions
    pitch_length, pitch_width = load_pitch_dimensions(match_id)
    
    # Get events for this match and apply intelligent filtering
    match_events = events_df[events_df['match_id'] == statsbomb_match_id].copy()
    print(f"Processing events for teams: {match_events.team.unique()}")
    
    # Apply intelligent filtering: only filter when there are event conflicts
    match_events = filter_priority_events(match_events, priority_types=priority_events)
    
    # Create timestamp in tracking data
    tracking_df['timestamp'] = (tracking_df['minute'] * 60 + tracking_df['seconds']).astype(float)
    
    # Create timestamp in events data (assuming events have minute and second columns)
    match_events['timestamp'] = (match_events['minute'] * 60 + match_events['second']).astype(float)
    
    # Check for column conflicts and handle them before merge
    conflict_columns = set(tracking_df.columns) & set(match_events.columns)
    print(f"   🔍 Column conflicts detected: {conflict_columns}")
    
    # Rename conflicting columns in events data to avoid loss
    events_rename_map = {}
    for col in conflict_columns:
        if col not in ['timestamp', 'period']:  # Keep merge keys as is
            events_rename_map[col] = f'event_{col}'
    
    if events_rename_map:
        print(f"   🔧 Renaming event columns to avoid conflicts: {events_rename_map}")
        match_events = match_events.rename(columns=events_rename_map)
    
    # Merge tracking data with events
    print(f"   🔍 Pre-merge: tracking_df has player_id: {'player_id' in tracking_df.columns}")
    print(f"   🔍 Pre-merge: events has player_id: {'player_id' in match_events.columns}")
    
    merged_df = pd.merge_asof(
        tracking_df.sort_values('timestamp'),
        match_events.sort_values('timestamp'),
        on='timestamp',
        by='period',
        direction='nearest',
        tolerance=1.0  # 1 second tolerance for matching
    )
    
    print(f"   🔍 Post-merge: merged_df has player_id: {'player_id' in merged_df.columns}")
    
    # Add event information from events data (handle renamed columns)
    merged_df['event_id'] = merged_df.get('id', None)  # Event ID from events data
    
    # Handle event_player - check for renamed version first
    if 'event_player' in merged_df.columns:
        merged_df['event_player'] = merged_df['event_player']
    else:
        merged_df['event_player'] = merged_df.get('player_y', merged_df.get('player', None))
    
    # Handle event_team - check for renamed version first  
    if 'event_team' in merged_df.columns:
        merged_df['event_team'] = merged_df['event_team']
    else:
        merged_df['event_team'] = merged_df.get('team_y', None)
    
    merged_df['event_type'] = merged_df.get('type', None)
    merged_df['event_location'] = merged_df.get('location', None)
    
    # Add new columns for pass and carry analysis
    merged_df['pass_outcome'] = merged_df.get('pass_outcome', None)
    merged_df['event_duration'] = merged_df.get('duration', None)  # Use 'duration' from StatsBomb events
    merged_df['carry_outcome'] = merged_df.get('carry_outcome', None)
    merged_df['pass_recipient'] = merged_df.get('pass_recipient', None)  # Add pass recipient information
    
    # Add team name columns for both SkillCorner and StatsBomb
    # Load team mapping for this match
    team_mapping_file = PROJECT_ROOT / 'config' / 'team_id_mapping.csv'
    team_mapping_df = pd.read_csv(team_mapping_file)
    
    # Get team names for this match
    home_team_info = match_info.iloc[0]
    away_team_info = match_info.iloc[0]  # We'll get the away team from the match data
    
    # Get home and away team IDs from match info
    home_team_id = home_team_info['skillcorner_home_team_id']
    away_team_id = home_team_info['skillcorner_away_team_id']
    
    # Get team names from team mapping
    home_team_skillcorner_name = team_mapping_df[team_mapping_df['skillcorner_team_id'] == home_team_id]['skillcorner_team_name'].iloc[0] if len(team_mapping_df[team_mapping_df['skillcorner_team_id'] == home_team_id]) > 0 else None
    away_team_skillcorner_name = team_mapping_df[team_mapping_df['skillcorner_team_id'] == away_team_id]['skillcorner_team_name'].iloc[0] if len(team_mapping_df[team_mapping_df['skillcorner_team_id'] == away_team_id]) > 0 else None
    
    home_team_statsbomb_name = team_mapping_df[team_mapping_df['skillcorner_team_id'] == home_team_id]['statsbomb_team_name'].iloc[0] if len(team_mapping_df[team_mapping_df['skillcorner_team_id'] == home_team_id]) > 0 else None
    away_team_statsbomb_name = team_mapping_df[team_mapping_df['skillcorner_team_id'] == away_team_id]['statsbomb_team_name'].iloc[0] if len(team_mapping_df[team_mapping_df['skillcorner_team_id'] == away_team_id]) > 0 else None
    
    # Create team name mapping based on team ID
    team_name_mapping = {
        home_team_id: {
            'skillcorner_name': home_team_skillcorner_name,
            'statsbomb_name': home_team_statsbomb_name
        },
        away_team_id: {
            'skillcorner_name': away_team_skillcorner_name,
            'statsbomb_name': away_team_statsbomb_name
        }
    }
    
    # Add team name columns to merged_df
    merged_df['skillcorner_team_name'] = merged_df['team'].map(lambda x: team_name_mapping.get(x, {}).get('skillcorner_name') if x in team_name_mapping else None)
    merged_df['statsbomb_team_name'] = merged_df['team'].map(lambda x: team_name_mapping.get(x, {}).get('statsbomb_name') if x in team_name_mapping else None)
    
    # Create combined event_end_location from pass_end_location and carry_end_location
    # Initialize with None values
    merged_df['event_end_location'] = None
    
    # For Pass events, use pass_end_location if available
    if 'pass_end_location' in merged_df.columns:
        pass_mask = (merged_df['event_type'] == 'Pass') & (merged_df['pass_end_location'].notna())
        merged_df.loc[pass_mask, 'event_end_location'] = merged_df.loc[pass_mask, 'pass_end_location']
    
    # For Carry events, use carry_end_location if available
    if 'carry_end_location' in merged_df.columns:
        carry_mask = (merged_df['event_type'] == 'Carry') & (merged_df['carry_end_location'].notna())
        merged_df.loc[carry_mask, 'event_end_location'] = merged_df.loc[carry_mask, 'carry_end_location']
    
    # Add match IDs and pitch dimensions
    merged_df['skillcorner_match_id'] = match_id
    merged_df['statsbomb_match_id'] = statsbomb_match_id
    merged_df['pitch_length'] = pitch_length
    merged_df['pitch_width'] = pitch_width
    
    # ===== OPTIMIZED: COORDINATE PROCESSING MOVED TO BATCH SAVE =====
    # Coordinate processing is now done just before saving the batch for better performance
    # This eliminates the slow row-by-row processing during merge
    print(f"   ⚡ Skipping coordinate processing during merge (will be done before save)")
    
    # Initialize empty coordinate columns for compatibility
    coord_columns = [
        'event_location_x_sb', 'event_location_y_sb',
        'event_location_x_sb_flipped', 'event_location_y_sb_flipped',
        'event_end_location_x_sb', 'event_end_location_y_sb',
        'event_end_location_x_sb_flipped', 'event_end_location_y_sb_flipped'
    ]
    
    for col in coord_columns:
        merged_df[col] = None
    
    # Preserve ALL tracking columns and add event columns
    # First, get all original tracking columns
    tracking_columns = tracking_df.columns.tolist()
    
    # Define essential event columns to add
    event_columns = [
        'event_id', 'event_player', 'event_team', 'event_type', 'event_location',
        'pass_outcome', 'event_duration', 'carry_outcome', 'event_end_location', 'pass_recipient',
        # New coordinate columns (StatsBomb only - SkillCorner transformation on-demand)
        'event_location_x_sb', 'event_location_y_sb',
        'event_location_x_sb_flipped', 'event_location_y_sb_flipped',
        'event_end_location_x_sb', 'event_end_location_y_sb',
        'event_end_location_x_sb_flipped', 'event_end_location_y_sb_flipped'
    ]
    
    # Define additional match info columns
    match_info_columns = ['skillcorner_match_id', 'statsbomb_match_id', 'pitch_length', 'pitch_width', 'skillcorner_team_name', 'statsbomb_team_name']
    
    # Build list of columns to keep, prioritizing tracking columns and avoiding duplicates
    columns_to_keep = []
    
    # Add all tracking columns first (but handle naming conflicts from merge)
    for col in tracking_columns:
        if col in merged_df.columns:
            columns_to_keep.append(col)
        elif col == 'minute' and 'minute_x' in merged_df.columns:
            # Handle minute conflict - after merge, tracking minute becomes minute_x
            columns_to_keep.append('minute_x')
        elif col == 'seconds' and 'seconds' in merged_df.columns:
            # Handle seconds column
            columns_to_keep.append('seconds')
        else:
            # Debug: Track missing columns from tracking data
            if col == 'player_id':
                print(f"⚠️  Warning: {col} from tracking data not found in merged_df.columns")
                print(f"   Available columns: {sorted(merged_df.columns.tolist())}")
            elif col not in ['seconds']:  # seconds gets renamed, so ignore it
                print(f"⚠️  Warning: tracking column '{col}' not found in merged dataframe")
    
    # Add event columns that don't conflict with tracking columns
    for col in event_columns:
        if col in merged_df.columns and col not in columns_to_keep:
            columns_to_keep.append(col)
    
    # Add match info columns
    for col in match_info_columns:
        if col not in columns_to_keep:
            columns_to_keep.append(col)
    
    # Select only available columns from the merged dataframe (remove duplicates)
    available_columns = []
    seen_columns = set()
    for col in columns_to_keep:
        if col in merged_df.columns and col not in seen_columns:
            available_columns.append(col)
            seen_columns.add(col)
    
    print(f"   📋 Selected {len(available_columns)} unique columns (removed {len(columns_to_keep) - len(available_columns)} duplicates)")
    result_df = merged_df[available_columns].copy()
    
    # Rename conflicting columns for clarity (only if they exist and target doesn't exist)
    column_renames = {}
    if 'player_short_name' in result_df.columns and 'player' not in result_df.columns:
        column_renames['player_short_name'] = 'player'
    if 'team_name' in result_df.columns and 'team' not in result_df.columns:
        column_renames['team_name'] = 'team'
    if 'seconds' in result_df.columns and 'second' not in result_df.columns:
        column_renames['seconds'] = 'second'
    if 'minute_x' in result_df.columns and 'minute' not in result_df.columns:
        column_renames['minute_x'] = 'minute'
    
    # Apply renames only if safe to do so
    if column_renames:
        print(f"   🔄 Applying column renames: {column_renames}")
        result_df = result_df.rename(columns=column_renames)
    
    # Final check for duplicate columns
    if len(result_df.columns) != len(set(result_df.columns)):
        duplicates = [col for col in result_df.columns if list(result_df.columns).count(col) > 1]
        print(f"   ⚠️  Found duplicate columns after processing: {list(set(duplicates))}")
        # Keep only the first occurrence of each column
        result_df = result_df.loc[:, ~result_df.columns.duplicated()]
        print(f"   ✅ Removed duplicates, final columns: {len(result_df.columns)}")
    
    print(f"✅ Preserved {len(result_df.columns)} columns from tracking data ({len(tracking_columns)} original) + events")
    
    return result_df

def check_output_file_columns(output_file):
    """
    Check if the output file has all required columns for behind defense analysis
    
    Args:
        output_file (Path): Path to the output file
        
    Returns:
        tuple: (has_all_columns, missing_columns, needs_reprocessing)
    """
    # Define all columns that should be preserved from tracking data with velocities
    # Note: 'minute' gets renamed to 'minute_x' during merge, then back to 'minute' in final output
    required_tracking_columns = [
        'frame_number', 'minute', 'second', 'period', 'team', 'player_id', 'player_role_id',
        'x', 'y', 'is_detected', 'player_in_possession', 'team_in_possession', 
        'interpolated', 'player_number', 'player', 'team_color',
        'defending_half', 'attacking_half', 'offside', 
        'velocity_x', 'velocity_y', 'velocity_magnitude', 'velocity_capped',
        'role_name', 'role_line'
    ]
    
    # Event-related columns
    required_event_columns = [
        'event_id', 'event_player', 'event_team', 'event_type', 'event_location',
        'pass_outcome', 'event_duration', 'carry_outcome', 'event_end_location', 'pass_recipient',
        # Coordinate columns (StatsBomb only - SkillCorner transformation on-demand)
        'event_location_x_sb', 'event_location_y_sb',
        'event_location_x_sb_flipped', 'event_location_y_sb_flipped',
        'event_end_location_x_sb', 'event_end_location_y_sb',
        'event_end_location_x_sb_flipped', 'event_end_location_y_sb_flipped'
    ]
    
    # Match info columns
    required_match_columns = ['skillcorner_match_id', 'statsbomb_match_id', 'pitch_length', 'pitch_width']
    
    # Team name columns
    required_team_columns = ['skillcorner_team_name', 'statsbomb_team_name']
    
    # Combine all required columns
    required_columns = required_tracking_columns + required_event_columns + required_match_columns + required_team_columns
    
    if not output_file.exists():
        print("Output file does not exist - will create new file")
        return False, required_columns, True
    
    try:
        # Read just the first few rows to check columns
        existing_df = pd.read_parquet(output_file)
        # Take only first row to minimize memory usage
        existing_df = existing_df.head(1)
        existing_columns = set(existing_df.columns)
        required_columns_set = set(required_columns)
        
        # Handle special cases where columns might have been renamed during processing
        # Check for alternative column names that are equivalent
        actual_missing = []
        for col in required_columns:
            if col not in existing_columns:
                # Check for known alternatives
                found_alternative = False
                if col == 'minute' and 'minute' in existing_columns:
                    found_alternative = True  # minute should exist as is in final output
                elif col == 'second' and 'seconds' in existing_columns:
                    found_alternative = True  # seconds can be renamed to second
                elif col == 'player' and 'player_short_name' in existing_columns:
                    found_alternative = True  # player_short_name can be renamed to player
                elif col == 'team' and 'team_name' in existing_columns:
                    found_alternative = True  # team_name can be renamed to team
                
                if not found_alternative:
                    actual_missing.append(col)
        
        missing_columns = actual_missing
        has_all_columns = len(missing_columns) == 0
        
        if missing_columns:
            print(f"⚠️  Missing columns in existing file: {sorted(missing_columns)}")
            print("🔄 Need to reprocess all data to include new columns")
            return False, list(missing_columns), True
        else:
            print("✅ Existing file has all required columns")
            return True, [], False
            
    except Exception as e:
        print(f"❌ Error reading existing output file: {e}")
        print("🔄 Will reprocess from scratch")
        return False, required_columns, True

def get_already_processed_matches(output_dir, force_reprocess=False):
    """
    Get list of match IDs that have already been processed
    NEW APPROACH: Reads individual match files instead of consolidated file
    
    Args:
        output_dir (Path): Path to the output directory containing individual match files
        force_reprocess (bool): If True, return empty set to force reprocessing
        
    Returns:
        set: Set of skillcorner_match_ids that have already been processed
    """
    if force_reprocess:
        print("🔄 Force reprocessing enabled - treating all matches as unprocessed")
        return set()
    
    # Look for individual match files
    match_files = []
    try:
        import glob
        # Pattern for individual match files: match_*.parquet
        match_pattern = str(output_dir / 'match_*.parquet')
        match_files = glob.glob(match_pattern)
        
        if match_files:
            print(f"🔍 Found {len(match_files)} individual match files")
            
            # Extract match IDs from filenames
            processed_matches = set()
            for match_file in match_files:
                try:
                    # Extract match_id from filename: match_2006551.parquet -> 2006551
                    filename = os.path.basename(match_file)
                    if filename.startswith('match_') and filename.endswith('.parquet'):
                        match_id = int(filename.replace('match_', '').replace('.parquet', ''))
                        processed_matches.add(match_id)
                except Exception as e:
                    print(f"⚠️  Error parsing filename {filename}: {e}")
                    continue
            
            print(f"✅ Found {len(processed_matches)} processed matches from individual files")
            return processed_matches
        else:
            print("📄 No individual match files found")
            return set()
            
    except Exception as e:
        print(f"❌ Error searching for match files: {e}")
        return set()

def get_unprocessed_tracking_files(tracking_files, processed_matches):
    """
    Filter tracking files to only include those not yet processed
    ULTRA-OPTIMIZED: Fast filtering with detailed reporting
    
    Args:
        tracking_files (list): List of all tracking files
        processed_matches (set): Set of already processed match IDs
        
    Returns:
        list: List of unprocessed tracking files
    """
    unprocessed_files = []
    skipped_files = []
    
    print(f"🔍 Filtering {len(tracking_files)} tracking files against {len(processed_matches)} processed matches...")
    
    for tracking_file in tracking_files:
        try:
            # Extract match_id from filename (more robust parsing)
            filename = os.path.basename(tracking_file)
            
            # Handle different filename patterns
            if filename.startswith('tracking_') and '_with_velocity.parquet' in filename:
                # Pattern: tracking_2006551_with_velocity.parquet
                match_id = int(filename.split('_')[1])
            elif filename.startswith('tracking_') and '.parquet' in filename:
                # Pattern: tracking_2006551.parquet
                match_id = int(filename.split('_')[1].split('.')[0])
            else:
                print(f"⚠️  Unknown filename pattern: {filename}")
                continue
            
            if match_id not in processed_matches:
                unprocessed_files.append(tracking_file)
            else:
                skipped_files.append(match_id)
                
        except (ValueError, IndexError) as e:
            print(f"⚠️  Error parsing filename {tracking_file}: {e}")
            continue
    
    # Detailed reporting
    print(f"📊 File filtering results:")
    print(f"   Total files found: {len(tracking_files)}")
    print(f"   Already processed: {len(skipped_files)}")
    print(f"   To be processed: {len(unprocessed_files)}")
    
    if skipped_files:
        print(f"   Skipped match IDs: {sorted(skipped_files)[:10]}{'...' if len(skipped_files) > 10 else ''}")
    
    if unprocessed_files:
        # Show first few unprocessed files
        unprocessed_ids = []
        for f in unprocessed_files[:5]:
            try:
                match_id = int(os.path.basename(f).split('_')[1])
                unprocessed_ids.append(match_id)
            except:
                pass
        print(f"   Next to process: {unprocessed_ids}")
    
    return unprocessed_files

def process_coordinates_before_save(batch_data, tracking_df_dict):
    """
    Process coordinates in ULTRA-OPTIMIZED vectorized manner just before saving the batch
    MAXIMUM PERFORMANCE with numpy operations and minimal pandas overhead
    
    Args:
        batch_data (pd.DataFrame): Batch of data to process
        tracking_df_dict (dict): Dictionary {match_id: tracking_df} to get attacking_half
        
    Returns:
        pd.DataFrame: Batch with processed coordinates
    """
    print(f"🔄 Processing coordinates for {len(batch_data):,} events in batch...")
    
    # Filter only events with coordinates
    events_mask = batch_data['event_type'].notna()
    events_df = batch_data[events_mask].copy()
    
    if len(events_df) == 0:
        print(f"   ⚠️ No events found for coordinate processing")
        return batch_data
    
    print(f"   📊 Processing coordinates for {len(events_df):,} events...")
    
    # Initialize coordinate columns
    coord_columns = [
        'event_location_x_sb', 'event_location_y_sb',
        'event_location_x_sb_flipped', 'event_location_y_sb_flipped',
        'event_end_location_x_sb', 'event_end_location_y_sb',
        'event_end_location_x_sb_flipped', 'event_end_location_y_sb_flipped'
    ]
    
    for col in coord_columns:
        events_df[col] = None
    
    # ULTRA-OPTIMIZED: Extract coordinates using vectorized operations
    print(f"   🔍 Extracting coordinates (ultra-optimized)...")
    
    # Start coordinates - ULTRA-OPTIMIZED: Direct numpy operations
    start_coords = events_df['event_location'].apply(extract_coordinates_from_location)
    events_df['event_location_x_sb'] = start_coords.apply(lambda x: x[0] if x[0] is not None else None)
    events_df['event_location_y_sb'] = start_coords.apply(lambda x: x[1] if x[1] is not None else None)
    
    # End coordinates - ULTRA-OPTIMIZED: Direct numpy operations
    end_coords = events_df['event_end_location'].apply(extract_coordinates_from_location)
    events_df['event_end_location_x_sb'] = end_coords.apply(lambda x: x[0] if x[0] is not None else None)
    events_df['event_end_location_y_sb'] = end_coords.apply(lambda x: x[1] if x[1] is not None else None)
    
    # ULTRA-OPTIMIZED: Get attacking_half using vectorized operations
    print(f"   🎯 Determining attacking directions (ultra-optimized)...")
    
    # Create a more efficient team mapping
    team_attacking_half = {}
    
    # Use groupby for better performance
    for match_id in events_df['skillcorner_match_id'].unique():
        if match_id in tracking_df_dict:
            tracking_df = tracking_df_dict[match_id]
            # Get unique teams for this match more efficiently
            match_teams = events_df[events_df['skillcorner_match_id'] == match_id]['event_team'].dropna().unique()
            
            for team in match_teams:
                team_tracking = tracking_df[tracking_df['team'] == team]
                if not team_tracking.empty:
                    team_attacking_half[team] = team_tracking['attacking_half'].iloc[0]
    
    # ULTRA-OPTIMIZED: Apply flip using pure numpy operations
    print(f"   🔄 Applying coordinate flips (ultra-optimized)...")
    
    # Create a mapping series for attacking_half
    events_df['team_attacking_half'] = events_df['event_team'].map(team_attacking_half)
    
    # Create masks for teams that need flipping
    needs_flip_mask = events_df['team_attacking_half'] == 'left'
    
    # ULTRA-OPTIMIZED: Initialize flipped columns with original values (numpy arrays)
    events_df['event_location_x_sb_flipped'] = events_df['event_location_x_sb'].values
    events_df['event_location_y_sb_flipped'] = events_df['event_location_y_sb'].values
    events_df['event_end_location_x_sb_flipped'] = events_df['event_end_location_x_sb'].values
    events_df['event_end_location_y_sb_flipped'] = events_df['event_end_location_y_sb'].values
    
    # ULTRA-OPTIMIZED: Apply vectorized flip only where needed using pure numpy
    if needs_flip_mask.any():
        # Get numpy arrays directly for maximum speed
        start_x = events_df['event_location_x_sb'].values
        start_y = events_df['event_location_y_sb'].values
        end_x = events_df['event_end_location_x_sb'].values
        end_y = events_df['event_end_location_y_sb'].values
        
        # Create flip mask as numpy array
        flip_mask = needs_flip_mask.values
        
        # ULTRA-OPTIMIZED: Pure numpy operations for maximum speed
        # Vectorized flip: 120 - x, 80 - y (only where flip_mask is True)
        start_x_flipped = np.where(flip_mask & pd.notna(start_x), 120.0 - start_x, start_x)
        start_y_flipped = np.where(flip_mask & pd.notna(start_y), 80.0 - start_y, start_y)
        end_x_flipped = np.where(flip_mask & pd.notna(end_x), 120.0 - end_x, end_x)
        end_y_flipped = np.where(flip_mask & pd.notna(end_y), 80.0 - end_y, end_y)
        
        # ULTRA-OPTIMIZED: Direct assignment using numpy arrays
        events_df['event_location_x_sb_flipped'] = start_x_flipped
        events_df['event_location_y_sb_flipped'] = start_y_flipped
        events_df['event_end_location_x_sb_flipped'] = end_x_flipped
        events_df['event_end_location_y_sb_flipped'] = end_y_flipped
        
        print(f"   ✅ Applied ultra-optimized flip to {needs_flip_mask.sum():,} events")
    
    # ULTRA-OPTIMIZED: Update original batch using numpy operations
    # First, ensure all columns exist in batch_data
    missing_cols = set(events_df.columns) - set(batch_data.columns)
    if missing_cols:
        print(f"   📋 Adding {len(missing_cols)} missing columns to batch_data")
        for col in missing_cols:
            batch_data[col] = None
    
    # ULTRA-OPTIMIZED: Update only the rows that have events using numpy indexing
    for col in events_df.columns:
        if col in batch_data.columns:
            batch_data.loc[events_mask, col] = events_df[col].values
    
    print(f"   ✅ Ultra-optimized coordinate processing completed!")
    return batch_data

def append_to_output_file(new_data, output_file, tracking_df_dict=None, process_coordinates=True):
    """
    Append new data to existing output file efficiently (memory-safe)
    NOW WITH COORDINATE PROCESSING BEFORE SAVING AND ULTRA LOW MEMORY MODE
    
    Args:
        new_data (pd.DataFrame): New data to append
        output_file (Path): Path to output file
        tracking_df_dict (dict): Dictionary of tracking dataframes for coordinate processing
        process_coordinates (bool): Whether to process coordinates before saving
    """
    if process_coordinates and tracking_df_dict is not None:
        print(f"🔄 Processing coordinates before saving...")
        new_data = process_coordinates_before_save(new_data, tracking_df_dict)
    
    # Check available memory before proceeding
    import psutil
    memory_info = psutil.virtual_memory()
    available_gb = memory_info.available / (1024**3)
    print(f"💾 Available memory: {available_gb:.1f} GB")
    
    if output_file.exists():
        print(f"Appending {len(new_data):,} new rows to existing file...")
        
        # FOR ULTRA-LARGE FILES: Use chunked append strategy
        if available_gb < 4.0:  # Less than 4GB available
            print(f"  ⚠️  Low memory detected! Using ultra-safe append mode...")
            return append_with_ultra_low_memory(new_data, output_file)
        
        # Use chunked processing for very large files to avoid memory issues
        try:
            existing_df = pd.read_parquet(output_file)
            existing_count = len(existing_df)
            
            print(f"  Existing rows: {existing_count:,}")
            print(f"  New rows: {len(new_data):,}")
            
            # Estimate memory usage
            total_rows = existing_count + len(new_data)
            estimated_memory_gb = (total_rows * len(new_data.columns) * 8) / (1024**3)  # Rough estimate
            
            if estimated_memory_gb > available_gb * 0.5:  # Use more than 50% of available memory
                print(f"  ⚠️  Large dataset detected! Estimated memory: {estimated_memory_gb:.1f}GB")
                print(f"  🔄 Using chunked processing...")
                return append_with_chunked_processing(new_data, output_file, existing_df)
            
            # Standard processing for smaller datasets
            return append_standard_processing(new_data, output_file, existing_df)
            
        except (MemoryError, Exception) as e:
            print(f"  💥 Error during standard processing: {str(e)}")
            print(f"  🔄 Falling back to ultra-safe mode...")
            return append_with_ultra_low_memory(new_data, output_file)
            
    else:
        combined_df = new_data
        print(f"Creating new output file with {len(new_data):,} rows")
    
    # Save with maximum compression for efficiency
    try:
        combined_df.to_parquet(
            output_file, 
            index=False,
            compression='snappy',  # Fast compression
            engine='pyarrow'       # Most efficient engine
        )
        
        # Clear memory
        del combined_df
        
        print(f"✅ Data saved successfully to {output_file}")
        return None  # Don't return large dataframe to save memory
        
    except Exception as e:
        print(f"💥 Error saving to parquet: {str(e)}")
        print(f"🔄 Trying with different compression...")
        
        # Try with different settings for ultra-large files
        combined_df.to_parquet(
            output_file, 
            index=False,
            compression='gzip',    # Higher compression ratio
            engine='pyarrow',
            row_group_size=50000   # Smaller row groups for memory efficiency
        )
        
        del combined_df
        print(f"✅ Data saved successfully with fallback settings to {output_file}")
        return None

def save_match_data_separately(new_data, output_dir, tracking_df_dict=None, process_coordinates=True):
    """
    Save match data to individual files (one per match_id)
    NEW APPROACH: Much more efficient than appending to a single large file
    
    Args:
        new_data (DataFrame): New data to save
        output_dir (Path): Directory to save individual match files
        tracking_df_dict (dict): Dictionary of tracking dataframes for coordinate processing
        process_coordinates (bool): Whether to process coordinates before saving
    """
    if new_data.empty:
        print("⚠️  No new data to save")
        return
    
    print(f"💾 Saving {len(new_data):,} rows to individual match files...")
    
    # Process coordinates if requested
    if process_coordinates and tracking_df_dict is not None:
        print("🔄 Processing coordinates before saving...")
        new_data = process_coordinates_before_save(new_data, tracking_df_dict)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group data by match_id and save each group separately
    match_groups = new_data.groupby('skillcorner_match_id')
    total_matches = len(match_groups)
    
    print(f"📊 Saving {total_matches} matches to individual files...")
    
    saved_matches = []
    for match_id, match_data in match_groups:
        try:
            # Create filename for this match
            match_filename = f"match_{match_id}.parquet"
            match_filepath = output_dir / match_filename
            
            # Save match data
            match_data.to_parquet(match_filepath, index=False, compression='snappy')
            
            saved_matches.append(match_id)
            print(f"   ✅ Saved match {match_id}: {len(match_data):,} rows -> {match_filename}")
            
        except Exception as e:
            print(f"   ❌ Error saving match {match_id}: {e}")
            continue
    
    print(f"📊 Successfully saved {len(saved_matches)} matches")
    return saved_matches

def append_with_ultra_low_memory(new_data, output_file):
    """
    Ultra-safe append method for very low memory systems
    Saves new data to a separate file to avoid memory issues
    """
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    temp_output = output_file.with_suffix(f'.append_{timestamp}.parquet')
    
    try:
        new_data.to_parquet(
            temp_output,
            index=False,
            compression='gzip',  # Better compression for storage
            engine='pyarrow',
            row_group_size=25000  # Small row groups for memory efficiency
        )
        
        print(f"  💾 New data saved to temporary file: {temp_output}")
        print(f"  ⚠️  Please manually combine files when system has sufficient memory:")
        print(f"     1. Original file: {output_file}")
        print(f"     2. New data file: {temp_output}")
        print(f"  💡 Suggested merge command:")
        print(f"     python -c \"import pandas as pd; pd.concat([pd.read_parquet('{output_file}'), pd.read_parquet('{temp_output}')]).to_parquet('{output_file}', index=False)\"")
        
        return None
        
    except Exception as e:
        print(f"💥 Critical error even in ultra-safe mode: {str(e)}")
        print(f"  Saving as CSV instead...")
        csv_output = temp_output.with_suffix('.csv')
        new_data.to_csv(csv_output, index=False)
        print(f"  📄 Data saved as CSV: {csv_output}")
        return None

def append_with_chunked_processing(new_data, output_file, existing_df):
    """
    Process large datasets using chunked approach to manage memory
    """
    try:
        # Process column alignment first
        existing_cols = set(existing_df.columns)
        new_cols = set(new_data.columns)
        
        if existing_cols != new_cols:
            print(f"  🔧 Aligning columns...")
            missing_in_new = existing_cols - new_cols
            missing_in_existing = new_cols - existing_cols
            
            for col in missing_in_new:
                new_data[col] = None
            for col in missing_in_existing:
                existing_df[col] = None
                
            # Reorder columns
            all_columns = list(new_data.columns)
            existing_df = existing_df[all_columns]
        
        # Create temporary backup
        backup_file = output_file.with_suffix('.backup.parquet')
        print(f"  💾 Creating backup: {backup_file.name}")
        existing_df.to_parquet(backup_file, index=False, compression='snappy')
        
        # Clear existing_df from memory
        del existing_df
        
        # Append new data directly to original file
        print(f"  🔄 Writing combined data in chunks...")
        
        # Read backup and new data in chunks
        import pyarrow.parquet as pq
        
        # Write to temporary file first
        temp_output = output_file.with_suffix('.temp.parquet')
        
        # Combine data using pyarrow for better memory efficiency
        backup_table = pq.read_table(backup_file)
        new_table = pq.Table.from_pandas(new_data)
        
        # Concatenate tables (more memory efficient than pandas)
        combined_table = pq.concat_tables([backup_table, new_table])
        
        # Write with optimized settings
        pq.write_table(
            combined_table, 
            temp_output,
            compression='snappy',
            row_group_size=100000  # Optimize for memory
        )
        
        # Replace original file
        import shutil
        shutil.move(str(temp_output), str(output_file))
        
        # Clean up
        backup_file.unlink()
        
        print(f"  ✅ Chunked processing completed successfully!")
        return None
        
    except Exception as e:
        print(f"  💥 Error in chunked processing: {str(e)}")
        print(f"  🔄 Falling back to ultra-safe mode...")
        return append_with_ultra_low_memory(new_data, output_file)

def append_standard_processing(new_data, output_file, existing_df):
    """
    Standard processing for datasets that fit comfortably in memory
    """
    try:
        # Check for column compatibility
        existing_cols = set(existing_df.columns)
        new_cols = set(new_data.columns)
        
        if existing_cols != new_cols:
            print(f"  ⚠️  Column mismatch detected:")
            print(f"     Existing columns: {len(existing_cols)}")
            print(f"     New columns: {len(new_cols)}")
            
            # Find differences
            missing_in_new = existing_cols - new_cols
            missing_in_existing = new_cols - existing_cols
            
            if missing_in_new:
                print(f"     Missing in new data: {sorted(list(missing_in_new)[:5])}{'...' if len(missing_in_new) > 5 else ''}")
                for col in missing_in_new:
                    new_data[col] = None
            
            if missing_in_existing:
                print(f"     Missing in existing data: {sorted(list(missing_in_existing)[:5])}{'...' if len(missing_in_existing) > 5 else ''}")
                for col in missing_in_existing:
                    existing_df[col] = None
            
            # Reorder columns to match
            all_columns = list(new_data.columns)
            existing_df = existing_df[all_columns]
            
            print(f"  ✅ Column alignment completed")
        
        # Combine with new data
        combined_df = pd.concat([existing_df, new_data], ignore_index=True)
        print(f"  Combined successfully: {len(combined_df):,} total rows")
        
        # Save with optimized settings
        combined_df.to_parquet(
            output_file, 
            index=False,
            compression='snappy',
            engine='pyarrow'
        )
        
        # Clear memory
        del combined_df
        del existing_df
        
        print(f"  ✅ Standard processing completed successfully!")
        return None
        
    except MemoryError as e:
        print(f"  💥 Memory error in standard processing: {str(e)}")
        del existing_df  # Clean up
        return append_with_ultra_low_memory(new_data, output_file)

def analyze_processing_status(output_dir, tracking_files):
    """
    Analyze the current processing status and provide detailed information
    about what has been processed and what remains to be done
    NEW APPROACH: Works with individual match files instead of consolidated file
    
    Args:
        output_dir (Path): Path to the output directory containing individual match files
        tracking_files (list): List of all tracking files
    """
    print("\n📊 PROCESSING STATUS ANALYSIS")
    print("=" * 50)
    
    # Get all processed matches from individual files
    all_processed_matches = get_already_processed_matches(output_dir, force_reprocess=False)
    
    # Check individual match files status
    try:
        import glob
        match_pattern = str(output_dir / 'match_*.parquet')
        match_files = glob.glob(match_pattern)
        
        if match_files:
            print(f"📄 Individual match files status:")
            print(f"   Directory: {output_dir}")
            print(f"   Total match files: {len(match_files)}")
            
            # Calculate total size
            total_size_mb = 0
            total_rows = 0
            
            for match_file in match_files:
                try:
                    file_size_mb = Path(match_file).stat().st_size / (1024**2)
                    total_size_mb += file_size_mb
                    
                    # Count rows in this file
                    import pyarrow.parquet as pq
                    parquet_file = pq.ParquetFile(match_file)
                    file_rows = parquet_file.metadata.num_rows
                    total_rows += file_rows
                    
                except Exception as e:
                    print(f"   ⚠️  Error reading {os.path.basename(match_file)}: {e}")
            
            print(f"   Total size: {total_size_mb:.1f} MB ({total_size_mb/1024:.2f} GB)")
            print(f"   Total rows: {total_rows:,}")
            print(f"   Processed matches: {len(all_processed_matches)}")
            
        else:
            print(f"📄 No individual match files found in: {output_dir}")
            total_rows = 0
            
    except Exception as e:
        print(f"❌ Error analyzing match files: {e}")
        total_rows = 0
    
    # Analyze tracking files
    print(f"\n📁 Tracking files analysis:")
    print(f"   Total tracking files: {len(tracking_files)}")
    
    # Count processed vs unprocessed
    unprocessed_count = 0
    processed_count = 0
    
    for tracking_file in tracking_files:
        try:
            filename = os.path.basename(tracking_file)
            if filename.startswith('tracking_') and '_with_velocity.parquet' in filename:
                match_id = int(filename.split('_')[1])
            elif filename.startswith('tracking_') and '.parquet' in filename:
                match_id = int(filename.split('_')[1].split('.')[0])
            else:
                continue
                
            if match_id in all_processed_matches:
                processed_count += 1
            else:
                unprocessed_count += 1
                
        except:
            continue
    
    print(f"   Already processed: {processed_count}")
    print(f"   Remaining to process: {unprocessed_count}")
    
    if unprocessed_count > 0:
        progress_percent = (processed_count / (processed_count + unprocessed_count)) * 100
        print(f"   Progress: {progress_percent:.1f}% complete")
        
        # Estimate remaining processing time
        if processed_count > 0 and total_rows > 0:
            avg_rows_per_match = total_rows / len(all_processed_matches) if all_processed_matches else 0
            if avg_rows_per_match > 0:
                estimated_remaining_rows = unprocessed_count * avg_rows_per_match
                print(f"   Estimated remaining rows: {estimated_remaining_rows:,.0f}")
    
    # Summary
    print(f"\n📊 SUMMARY:")
    print(f"   Individual match files: {len(match_files) if 'match_files' in locals() else 0}")
    print(f"   Total unique matches: {len(all_processed_matches)}")
    print(f"   Files to process: {unprocessed_count}")
    
    print("=" * 50)

def suggest_sequential_processing():
    """
    Print suggestion to use sequential processing if parallel processing fails
    """
    print("\n⚠️  Parallel processing may be causing memory issues.")
    print("💡 Suggestions to resolve this problem:")
    print("   1. Use sequential processing (slower but more stable):")
    print("      python src/utils/merge_tracking_events.py --sequential")
    print("   2. Reduce the number of workers:")
    print("      python src/utils/merge_tracking_events.py --max-workers 1")
    print("   3. Process files one by one for debugging:")
    print("      python src/utils/merge_tracking_events.py --sequential --batch-size 1")
    print("\n🔄 Sequential processing will show detailed progress for each file.")

def main(max_workers=None, use_parallel=True, subsample_seconds=0.2, priority_events=['Pass', 'Carry', 'Shot'], force_reprocess=False, batch_size=20, skip_coordinates=False, preserve_all_frames=False):
    """
    Main function to process all tracking files and merge with events data (memory optimized)
    First processes all raw tracking files that haven't been converted to parquet yet,
    then updates match mappings, and finally merges with events data
    Only processes files that haven't been processed yet and appends to existing output
    Uses intelligent frame preservation and event filtering to optimize memory usage
    Saves every 'batch_size' matches to prevent data loss and manage memory
    
    Args:
        max_workers (int): Maximum number of parallel workers for raw file processing
        use_parallel (bool): Whether to use parallel processing (default: True)
        subsample_seconds (float): Target interval for frame preservation (default: 0.2s)
        priority_events (list): List of priority event types when resolving conflicts
        force_reprocess (bool): Force reprocessing of all matches
        batch_size (int): Number of matches to process before saving (default: 20)
        skip_coordinates (bool): Skip coordinate transformation (default: False)
        preserve_all_frames (bool): Keep all frames regardless of frequency (default: False)
    """
    import glob  # Import glob for file pattern matching
    
    print("🚀 Starting comprehensive tracking data processing pipeline...")
    print("=" * 70)
    
    # Show processing configuration
    if use_parallel:
        workers = max_workers if max_workers else 4
        print(f"🔧 Configuration: Parallel processing with up to {workers} workers (DEFAULT)")
    else:
        print(f"🔧 Configuration: Sequential processing")
    
    print(f"💾 Batch processing: Saving every {batch_size} matches")
    
    # Show frame preservation mode
    if preserve_all_frames:
        print(f"🔧 Frame mode: PRESERVE ALL FRAMES (maintaining original frequency)")
    else:
        print(f"🔧 Frame mode: Intelligent preservation (target: {subsample_seconds}s intervals)")
    
    # Step 1: Process all raw tracking files that haven't been converted to parquet yet
    print("\n📊 STEP 1: Processing raw tracking files...")
    print("-" * 50)
    total_raw, processed_raw = process_all_unprocessed_raw_files(max_workers=max_workers, use_parallel=use_parallel)
    
    if processed_raw > 0:
        print(f"✅ Successfully processed {processed_raw} new raw tracking files")
    
    # Step 2: Update match mappings
    print("\n🔄 STEP 2: Updating match mappings...")
    print("-" * 50)
    match_mapping = load_match_mapping()
    update_match_id_mappings()
    
    # Reload mapping after update
    match_mapping = load_match_mapping()
    
    # Step 3: Load events data
    print("\n📋 STEP 3: Loading events data...")
    print("-" * 50)
    events_df = load_events_data()
    print(f"✅ Loaded {len(events_df)} events from {len(events_df['match_id'].unique())} matches")
    print(f"📊 Memory optimization settings:")
    print(f"   Subsample interval: {subsample_seconds}s")
    print(f"   Priority events (conflict resolution): {priority_events}")
    print(f"   Intelligent filtering: Only applies priority filter when events conflict")
    print(f"   Expected memory reduction: ~{(1-subsample_seconds*2)*100:.0f}% tracking + variable % events")
    
    # Step 4: Setup output directory and check for unprocessed tracking files
    print("\n🔧 STEP 4: Checking tracking files for merge...")
    print("-" * 50)
    
    # Define output directory for individual match files
    output_dir = PROJECT_ROOT / 'data' / 'merged' / 'individual_matches'
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    print("💡 NEW APPROACH: Saving individual match files instead of consolidated file")
    print("   Benefits:")
    print("   • Much more memory efficient")
    print("   • Faster processing (no large file appends)")
    print("   • Easier to manage and analyze individual matches")
    print("   • No risk of file corruption during processing")
    
    # Get already processed matches from individual files
    processed_matches = get_already_processed_matches(output_dir, force_reprocess=force_reprocess)
    
    # Get all tracking files
    tracking_pattern = str(PROCESSED_DATA_DIR / 'tracking_*_with_velocity.parquet')
    all_tracking_files = glob.glob(tracking_pattern)
    
    if not all_tracking_files:
        print(f"❌ No tracking files found with pattern: {tracking_pattern}")
        return
    
    print(f"Found {len(all_tracking_files)} total tracking files with velocity")
    
    # Analyze current processing status
    analyze_processing_status(output_dir, all_tracking_files)
    
    # Filter to only unprocessed files for the merge
    unprocessed_files = get_unprocessed_tracking_files(all_tracking_files, processed_matches)
    
    if not unprocessed_files:
        print("✅ All files have already been merged with events!")
        print(f"Output directory: {output_dir}")
        
        # Show current statistics
        import glob
        match_files = glob.glob(str(output_dir / 'match_*.parquet'))
        if match_files:
            total_size_mb = sum(Path(f).stat().st_size / (1024**2) for f in match_files)
            print(f"\n📊 Current output statistics:")
            print(f"Total match files: {len(match_files)}")
            print(f"Total size: {total_size_mb:.1f} MB ({total_size_mb/1024:.2f} GB)")
        return
    
    # Step 5: Merge tracking files with events data in batches
    print(f"\n🔗 STEP 5: Merging {len(unprocessed_files)} tracking files with events (batch size: {batch_size})...")
    print("-" * 50)
    
    # Process files in batches to save periodically
    total_processed = 0
    total_new_rows = 0
    
    for batch_start in range(0, len(unprocessed_files), batch_size):
        batch_end = min(batch_start + batch_size, len(unprocessed_files))
        batch_files = unprocessed_files[batch_start:batch_end]
        batch_num = (batch_start // batch_size) + 1
        total_batches = (len(unprocessed_files) + batch_size - 1) // batch_size
        
        print(f"\n📦 BATCH {batch_num}/{total_batches}: Processing {len(batch_files)} files ({batch_start + 1}-{batch_end})...")
        print("-" * 40)
        
        # Process current batch
        batch_results = []
        tracking_df_dict = {}  # Store tracking data for coordinate processing
        
        for i, tracking_file in enumerate(batch_files, 1):
            file_idx = batch_start + i
            print(f"🔄 Processing {file_idx}/{len(unprocessed_files)}: {os.path.basename(tracking_file)}")
            
            # Extract match_id for tracking data storage
            match_id = int(os.path.basename(tracking_file).split('_')[1])
            
            result_df = process_tracking_file(
                tracking_file, 
                events_df, 
                match_mapping, 
                subsample_seconds=subsample_seconds, 
                priority_events=priority_events,
                preserve_all_frames=preserve_all_frames
            )
            
            if result_df is not None:
                batch_results.append(result_df)
                # Store tracking data for coordinate processing
                tracking_df = pd.read_parquet(tracking_file)
                tracking_df_dict[match_id] = tracking_df
        
        # Save current batch if we have results
        if batch_results:
            print(f"\n💾 Saving batch {batch_num}/{total_batches}...")
            
            # Combine batch results
            batch_data = pd.concat(batch_results, ignore_index=True)
            batch_rows = len(batch_data)
            batch_matches = batch_data['skillcorner_match_id'].nunique()
            
            print(f"📊 Batch {batch_num} statistics:")
            print(f"   Files processed: {len(batch_results)}")
            print(f"   Matches: {batch_matches}")
            print(f"   Rows: {batch_rows:,}")
            
            # Save batch data to individual match files
            save_match_data_separately(
                batch_data, 
                output_dir, 
                tracking_df_dict=tracking_df_dict,
                process_coordinates=not skip_coordinates  # Enable/disable coordinate processing
            )
            
            # Update counters
            total_processed += len(batch_results)
            total_new_rows += batch_rows
            
            print(f"✅ Batch {batch_num} saved successfully!")
            print(f"📈 Progress: {total_processed}/{len(unprocessed_files)} files processed")
            
            # Clear batch memory
            del batch_data
            del batch_results
            
            # Show overall progress
            progress_percent = (total_processed / len(unprocessed_files)) * 100
            print(f"🎯 Overall progress: {progress_percent:.1f}% complete")
            
            if batch_num < total_batches:
                print(f"⏳ Continuing to next batch...")
        else:
            print(f"⚠️  No valid results in batch {batch_num}")
    
    # Final summary
    print(f"\n🎉 PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"✅ Successfully processed {total_processed} tracking files in {total_batches} batches")
    print(f"✅ Added {total_new_rows:,} total new rows to the dataset")
    print(f"✅ Results saved to individual match files in: {output_dir}")
    
    # Show final statistics
    import glob
    match_files = glob.glob(str(output_dir / 'match_*.parquet'))
    if match_files:
        total_size_mb = sum(Path(f).stat().st_size / (1024**2) for f in match_files)
        print(f"✅ Total match files: {len(match_files)}")
        print(f"✅ Total size: {total_size_mb:.1f} MB ({total_size_mb/1024:.2f} GB)")
    
    # Print final statistics
    print(f"\n📊 Final processing summary:")
    print(f"Batch size: {batch_size} files per save")
    print(f"Total batches: {total_batches}")
    print(f"Files processed: {total_processed}")
    print(f"New rows added: {total_new_rows:,}")
    
    # Show current output statistics
    try:
        # Read a sample of match files to show stats
        import glob
        match_files = glob.glob(str(output_dir / 'match_*.parquet'))
        if match_files:
            # Read first few match files as sample
            sample_files = match_files[:min(5, len(match_files))]
            sample_dfs = []
            
            for sample_file in sample_files:
                try:
                    sample_df = pd.read_parquet(sample_file)
                    sample_dfs.append(sample_df)
                except Exception as e:
                    print(f"⚠️  Error reading {os.path.basename(sample_file)}: {e}")
                    continue
            
            if sample_dfs:
                combined_sample = pd.concat(sample_dfs, ignore_index=True)
                print(f"\n📋 Output sample statistics (from {len(sample_dfs)} match files):")
                print(f"Sample matches: {combined_sample['skillcorner_match_id'].nunique()}")
                print(f"Columns: {len(combined_sample.columns)} ({', '.join(combined_sample.columns[:8])}...)")
                print(f"Sample rows: {len(combined_sample):,}")
                
                # Sample event distribution
                if 'event_type' in combined_sample.columns:
                    sample_events = combined_sample[combined_sample['event_type'].notna()]['event_type'].value_counts().head(5)
                    print(f"Top event types (sample):")
                    for event_type, count in sample_events.items():
                        print(f"  {event_type}: {count:,}")
                
                # Clean up memory
                del combined_sample, sample_dfs
                    
    except Exception as e:
        print(f"Could not read sample statistics: {e}")
    
    print(f"\n💾 Memory optimization results:")
    print(f"Batch processing: Data saved every {batch_size} matches")
    print(f"Subsample interval: {subsample_seconds}s → ~{(1/subsample_seconds)*100:.0f}% data reduction")
    print(f"Intelligent filtering: Priority events used only for conflict resolution")
    print(f"Compression: Snappy compression applied")
    
    print(f"🧹 Processing completed successfully!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Comprehensive tracking data processing pipeline with parallel support'
    )
    parser.add_argument(
        '--max-workers', 
        type=int, 
        default=4,
        help='Maximum number of parallel workers for raw file processing (default: 4)'
    )
    parser.add_argument(
        '--sequential', 
        action='store_true',
        help='Use sequential processing instead of parallel (useful for debugging)'
    )
    parser.add_argument(
        '--workers', 
        type=int, 
        default=None,
        help='Alias for --max-workers'
    )
    parser.add_argument(
        '--subsample',
        type=float,
        default=0.2,
        help='Subsample interval in seconds for tracking data (default: 0.2s)'
    )
    parser.add_argument(
        '--priority-events',
        nargs='+',
        default=['Pass', 'Carry', 'Shot'],
        help='Priority event types for conflict resolution (default: Pass Carry Shot)'
    )
    parser.add_argument(
        '--force-reprocess',
        action='store_true',
        help='Force reprocessing of all matches even if output file exists'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=20,
        help='Number of matches to process before saving (default: 20)'
    )
    parser.add_argument(
        '--skip-coordinates',
        action='store_true',
        help='Skip coordinate processing to speed up merge (coordinates will be empty)'
    )
    parser.add_argument(
        '--preserve-all-frames',
        action='store_true',
        help='Preserve ALL tracking frames regardless of frequency (disables subsampling)'
    )
    
    args = parser.parse_args()
    
    # Determine max_workers (--workers takes precedence over --max-workers)
    max_workers = args.workers if args.workers is not None else args.max_workers
    
    # Determine if using parallel processing
    use_parallel = not args.sequential
    
    # Show configuration
    if use_parallel:
        workers = max_workers if max_workers else 2
        print(f"🚀 Starting with parallel processing ({workers} workers)")
    else:
        print(f"🔄 Starting with sequential processing")
    
    print(f"📊 Memory optimization settings:")
    print(f"   Frame preservation: {'ALL FRAMES (no subsampling)' if args.preserve_all_frames else f'Target interval: {args.subsample}s'}")
    print(f"   Priority events: {args.priority_events}")
    print(f"   Batch size: {args.batch_size} matches per save")
    print(f"   Coordinate processing: {'Disabled' if args.skip_coordinates else 'Enabled (before save)'}")
    
    main(
        max_workers=max_workers, 
        use_parallel=use_parallel,
        subsample_seconds=args.subsample,
        priority_events=args.priority_events,
        force_reprocess=args.force_reprocess,
        batch_size=args.batch_size,
        skip_coordinates=args.skip_coordinates,
        preserve_all_frames=args.preserve_all_frames
    ) 