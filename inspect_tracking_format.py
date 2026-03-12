"""
Quick utility to inspect SkillCorner tracking data format
"""

import json
from pathlib import Path
import pprint

# Find first tracking file
tracking_dir = Path('data/tracking_j1_2024')
tracking_files = list(tracking_dir.glob('*_tracking_extrapolated.jsonl'))

if not tracking_files:
    print("No tracking files found!")
else:
    # Load first file
    sample_file = tracking_files[0]
    print(f"Inspecting: {sample_file.name}")
    print("=" * 70)
    
    try:
        with open(sample_file, 'r', encoding='utf-8') as f:
            print("Loading JSON array (may take a moment)...")
            data = json.load(f)
        
        print(f"\n✅ Loaded successfully!")
        print(f"Type: {type(data)}")
        print(f"Total frames: {len(data):,}")
        
        if isinstance(data, list) and len(data) > 0:
            print("\n" + "=" * 70)
            print("First frame structure:")
            print("-" * 70)
            first_frame = data[0]
            pprint.pprint(first_frame, depth=2, width=100)
            
            print("\n" + "=" * 70)
            print("First frame keys:")
            print(list(first_frame.keys()))
            
            print("\n" + "=" * 70)
            print("Sample of first 3 frames (key info):")
            for i, frame in enumerate(data[:3]):
                print(f"\nFrame {i}:")
                for key in ['timestamp', 'period', 'x', 'y', 'player_id', 'team', 'frame_number']:
                    if key in frame:
                        print(f"  {key}: {frame[key]}")
            
            print("\n" + "=" * 70)
            print("Column summary from first 100 frames:")
            import pandas as pd
            sample_df = pd.DataFrame(data[:100])
            print(f"Columns: {list(sample_df.columns)}")
            print(f"\nData types:")
            print(sample_df.dtypes)
            print(f"\nSample data:")
            print(sample_df.head())
        
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON: {e}")
