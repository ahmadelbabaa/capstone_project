"""
Convert JSON/JSONL files to Parquet format for efficient processing

This script converts:
1. StatsBomb events (JSON) → Parquet
2. SkillCorner tracking data (JSONL) → Parquet with velocity calculations

Author: Data Processing Script
"""

import pandas as pd
import numpy as np
import glob
import json
from pathlib import Path
from datetime import datetime

def convert_statsbomb_events():
    """
    Convert StatsBomb events from JSON to Parquet format
    """
    print("\n" + "="*70)
    print("[STEP 1] Converting StatsBomb Events (JSON -> Parquet)")
    print("="*70)
    
    events_json = Path('data/sb_data/sb_events.json')
    events_parquet = Path('data/sb_data/sb_events.parquet')
    
    if not events_json.exists():
        print(f"[ERROR] File not found: {events_json}")
        return False
    
    try:
        print(f"[INFO] Reading JSON file: {events_json}")
        print(f"   File size: {events_json.stat().st_size / 1024**2:.1f} MB")
        
        # Read JSON - this might take a moment for large files
        print("   Loading data (this may take a minute for large files)...")
        # Disable automatic date conversion for faster loading
        df = pd.read_json(events_json, convert_dates=False)
        
        print(f"[SUCCESS] Loaded {len(df):,} events")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Matches: {df['match_id'].nunique() if 'match_id' in df.columns else 'N/A'}")
        
        # Convert list/dict columns to JSON strings for Parquet compatibility
        print("   Converting complex columns to JSON strings...")
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column contains lists or dicts
                sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                if isinstance(sample, (list, dict)):
                    df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) else None)
        
        # Save as Parquet
        print(f"[INFO] Saving as Parquet: {events_parquet}")
        df.to_parquet(events_parquet, compression='snappy', index=False)
        
        parquet_size = events_parquet.stat().st_size / 1024**2
        json_size = events_json.stat().st_size / 1024**2
        reduction = (1 - parquet_size/json_size) * 100
        
        print(f"[SUCCESS] Successfully converted!")
        print(f"   JSON size: {json_size:.1f} MB")
        print(f"   Parquet size: {parquet_size:.1f} MB")
        print(f"   Space saved: {reduction:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error converting events: {e}")
        return False


def convert_tracking_files():
    """
    Convert SkillCorner tracking JSONL files to Parquet format
    """
    print("\n" + "="*70)
    print("[STEP 2] Converting Tracking Data (JSONL -> Parquet)")
    print("="*70)
    
    tracking_dir = Path('data/tracking_j1_2024')
    output_dir = Path('data/tracking_j1_2024_parquet')
    
    if not tracking_dir.exists():
        print(f"[ERROR] Directory not found: {tracking_dir}")
        return False
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")
    
    # Find all JSONL files
    jsonl_files = glob.glob(str(tracking_dir / '*_tracking_extrapolated.jsonl'))
    
    if not jsonl_files:
        print(f"[ERROR] No tracking JSONL files found in {tracking_dir}")
        return False
    
    print(f"[INFO] Found {len(jsonl_files)} tracking files to convert")
    print("")
    
    successful = 0
    failed = 0
    
    for i, jsonl_file in enumerate(jsonl_files, 1):
        try:
            # Extract match ID from filename
            match_id = Path(jsonl_file).stem.split('_')[0]
            
            print(f"[{i}/{len(jsonl_files)}] Processing match {match_id}...")
            
            # Read JSON (note: files have .jsonl extension but are actually JSON arrays)
            print(f"      Reading JSON...")
            df = pd.read_json(jsonl_file, convert_dates=False)
            
            original_rows = len(df)
            frames = df['frame_number'].nunique() if 'frame_number' in df.columns else 0
            
            # Create match-specific subdirectory
            match_output_dir = output_dir / match_id
            match_output_dir.mkdir(exist_ok=True)
            
            # Save as Parquet with match ID in path
            output_file = match_output_dir / f'{match_id}_tracking.parquet'
            print(f"      Saving to Parquet...")
            df.to_parquet(output_file, compression='snappy', index=False)
            
            # Get file sizes
            jsonl_size = Path(jsonl_file).stat().st_size / 1024**2
            parquet_size = output_file.stat().st_size / 1024**2
            
            print(f"      [OK] {original_rows:,} rows, {frames:,} frames")
            print(f"      [SIZE] {jsonl_size:.1f} MB -> {parquet_size:.1f} MB ({(1-parquet_size/jsonl_size)*100:.0f}% reduction)")
            print("")
            
            successful += 1
            
        except Exception as e:
            print(f"      [FAILED] {e}")
            print("")
            failed += 1
    
    print("="*70)
    print(f"[COMPLETE] Conversion complete!")
    print(f"   Successful: {successful}/{len(jsonl_files)}")
    if failed > 0:
        print(f"   Failed: {failed}/{len(jsonl_files)}")
    
    return successful > 0


def verify_conversions():
    """
    Verify that converted files can be read and have expected structure
    """
    print("\n" + "="*70)
    print("[STEP 3] Verifying Conversions")
    print("="*70)
    
    all_good = True
    
    # Verify events
    events_parquet = Path('data/sb_data/sb_events.parquet')
    if events_parquet.exists():
        try:
            df = pd.read_parquet(events_parquet)
            print(f"[OK] Events Parquet: {len(df):,} rows, {len(df.columns)} columns")
            
            # Show sample columns
            print(f"   Sample columns: {', '.join(df.columns[:10].tolist())}...")
            
        except Exception as e:
            print(f"[ERROR] Error reading events Parquet: {e}")
            all_good = False
    else:
        print(f"[WARNING] Events Parquet not found")
        all_good = False
    
    # Verify tracking (sample a few files)
    tracking_dir = Path('data/tracking_j1_2024_parquet')
    if tracking_dir.exists():
        parquet_files = list(tracking_dir.glob('tracking_*_with_velocity.parquet'))
        
        # Look for parquet files in subdirectories
        parquet_files = list(tracking_dir.glob('*/*_tracking.parquet'))
        
        if parquet_files:
            print(f"[OK] Tracking Parquet: {len(parquet_files)} files found")
            
            # Verify a sample file
            sample_file = parquet_files[0]
            try:
                df = pd.read_parquet(sample_file)
                print(f"   Sample file: {sample_file.parent.name}/{sample_file.name}")
                print(f"   Structure: {len(df):,} rows, {len(df.columns)} columns")
                print(f"   Sample columns: {', '.join(df.columns[:10].tolist())}...")
                    
            except Exception as e:
                print(f"   [ERROR] Error reading sample tracking file: {e}")
                all_good = False
        else:
            print(f"[WARNING] No tracking Parquet files found")
            all_good = False
    else:
        print(f"[WARNING] Tracking Parquet directory not found")
        all_good = False
    
    print("")
    if all_good:
        print("[SUCCESS] All conversions verified successfully!")
    else:
        print("[WARNING] Some issues detected, please review above")
    
    return all_good


def main():
    """
    Main conversion workflow
    """
    print("\n" + "#"*70)
    print("# JSON/JSONL to Parquet Conversion Script")
    print("# " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("#"*70)
    
    # Step 1: Convert StatsBomb events
    events_success = convert_statsbomb_events()
    
    # Step 2: Convert tracking data
    tracking_success = convert_tracking_files()
    
    # Step 3: Verify conversions
    verify_conversions()
    
    print("\n" + "="*70)
    print("[SUMMARY]")
    print("="*70)
    print(f"StatsBomb Events: {'[OK] Converted' if events_success else '[FAILED]'}")
    print(f"Tracking Data: {'[OK] Converted' if tracking_success else '[FAILED]'}")
    print("")
    
    if events_success and tracking_success:
        print("[SUCCESS] All conversions completed successfully!")
        print("")
        print("[OUTPUT] Output files:")
        print("   * data/sb_data/sb_events.parquet")
        print("   * data/tracking_j1_2024_parquet/{match_id}/{match_id}_tracking.parquet")
        print("")
        print("[NEXT STEPS]")
        print("   1. Create or update paths.py to point to the new Parquet files")
        print("   2. Update merge_tracking_events.py to match your J1 data structure")
        print("   3. Run the merge script to combine tracking + events data")
    else:
        print("[WARNING] Some conversions failed. Please review error messages above.")
    
    print("="*70)


if __name__ == "__main__":
    main()
