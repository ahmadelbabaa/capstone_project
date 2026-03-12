"""
Convert JSON/JSONL files to Parquet format for efficient processing
Converts:
1. StatsBomb events (sb_events.json) -> parquet
2. All tracking extrapolated files (*.jsonl) -> parquet
"""

import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

# Define paths
DATA_DIR = Path(__file__).parent / "data"
SB_DATA_DIR = DATA_DIR / "sb_data"
TRACKING_DIR = DATA_DIR / "tracking_j1_2024"
METADATA_DIR = DATA_DIR / "metadata_j1_2024"
OUTPUT_DIR = DATA_DIR / "processed"

# Create output directories
(OUTPUT_DIR / "events").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "tracking").mkdir(parents=True, exist_ok=True)


def convert_sb_events_to_parquet():
    """Convert StatsBomb events JSON to Parquet format"""
    print("Converting StatsBomb events to Parquet...")
    
    input_file = SB_DATA_DIR / "sb_events.json"
    output_file = OUTPUT_DIR / "events" / "sb_events.parquet"
    
    if not input_file.exists():
        print(f"❌ File not found: {input_file}")
        return
    
    print(f"  Reading {input_file.name}...")
    # Load JSON with pandas
    df = pd.read_json(input_file)
    
    print(f"  Loaded {len(df):,} events")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Convert complex columns (lists, dicts) to JSON strings for Parquet compatibility
    print(f"  Converting complex data types...")
    for col in df.columns:
        # Check if column contains lists or dicts
        if df[col].dtype == 'object':
            sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
            if isinstance(sample, (list, dict)):
                print(f"    Converting column '{col}' (type: {type(sample).__name__})")
                df[col] = df[col].apply(lambda x: json.dumps(x) if pd.notna(x) else None)
    
    # Save as parquet with compression
    print(f"  Writing to {output_file}...")
    df.to_parquet(
        output_file,
        engine='pyarrow',
        compression='snappy',
        index=False
    )
    
    # Verify file size
    file_size_mb = output_file.stat().st_size / 1024**2
    print(f"  ✅ Saved: {output_file.name} ({file_size_mb:.2f} MB)")
    
    # Load and verify
    df_verify = pd.read_parquet(output_file)
    print(f"  ✅ Verified: {len(df_verify):,} events loaded from parquet")
    print()


def convert_tracking_jsonl_to_parquet():
    """Convert all tracking JSONL files to Parquet format"""
    print("Converting tracking JSONL files to Parquet...")
    
    # Find all tracking files
    tracking_files = sorted(TRACKING_DIR.glob("*_tracking_extrapolated.jsonl"))
    
    if not tracking_files:
        print(f"❌ No tracking files found in {TRACKING_DIR}")
        return
    
    print(f"  Found {len(tracking_files)} tracking files\n")
    
    total_size_json = 0
    total_size_parquet = 0
    failed_files = []
    
    # Process each file with progress bar
    for tracking_file in tqdm(tracking_files, desc="Converting tracking files"):
        try:
            # Extract match ID from filename
            # Format: {match_id}_tracking_extrapolated.jsonl
            match_id = tracking_file.stem.replace("_tracking_extrapolated", "")
            output_file = OUTPUT_DIR / "tracking" / f"{match_id}_tracking.parquet"
            
            # Read JSONL file (one JSON object per line)
            records = []
            with open(tracking_file, 'r') as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        records.append(json.loads(line))
            
            # Convert to DataFrame
            df = pd.DataFrame(records)
            
            # Convert complex columns (lists, dicts) to JSON strings for Parquet compatibility
            for col in df.columns:
                if df[col].dtype == 'object':
                    sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                    if isinstance(sample, (list, dict)):
                        df[col] = df[col].apply(lambda x: json.dumps(x) if pd.notna(x) else None)
            
            # Save as parquet
            df.to_parquet(
                output_file,
                engine='pyarrow',
                compression='snappy',
                index=False
            )
            
            # Track sizes
            total_size_json += tracking_file.stat().st_size
            total_size_parquet += output_file.stat().st_size
            
        except Exception as e:
            failed_files.append((tracking_file.name, str(e)))
            tqdm.write(f"  ❌ Failed: {tracking_file.name} - {e}")
    
    # Summary
    print(f"\n✅ Successfully converted {len(tracking_files) - len(failed_files)}/{len(tracking_files)} tracking files")
    print(f"  Original JSON size: {total_size_json / 1024**3:.2f} GB")
    print(f"  Parquet size: {total_size_parquet / 1024**3:.2f} GB")
    print(f"  Compression ratio: {total_size_json / total_size_parquet:.2f}x")
    
    if failed_files:
        print(f"\n❌ Failed files ({len(failed_files)}):")
        for filename, error in failed_files[:10]:  # Show first 10
            print(f"  - {filename}: {error}")
    print()





def create_summary():
    """Create summary of converted files"""
    print("=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    
    # Count files
    event_files = list((OUTPUT_DIR / "events").glob("*.parquet"))
    tracking_files = list((OUTPUT_DIR / "tracking").glob("*.parquet"))
    
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nConverted files:")
    print(f"  Events:   {len(event_files)} files")
    print(f"  Tracking: {len(tracking_files)} files")
    
    # Calculate total sizes
    total_size = sum(
        f.stat().st_size 
        for f in (OUTPUT_DIR / "events").rglob("*.parquet")
    ) + sum(
        f.stat().st_size 
        for f in (OUTPUT_DIR / "tracking").rglob("*.parquet")
    )
    
    print(f"\nTotal parquet size: {total_size / 1024**3:.2f} GB")
    print("\n✅ Conversion complete! Files are ready for merge_tracking_events_v2.py")
    print("=" * 60)


def main():
    """Main conversion pipeline"""
    print("\n" + "=" * 60)
    print("JSON to Parquet Conversion Pipeline")
    print("=" * 60 + "\n")
    
    # Convert each data type
    convert_sb_events_to_parquet()
    convert_tracking_jsonl_to_parquet()
    
    # Print summary
    create_summary()


if __name__ == "__main__":
    main()
