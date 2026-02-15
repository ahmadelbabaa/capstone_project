"""
Batch script to synchronize all SkillCorner tracking data with StatsBomb events
"""
import pandas as pd
import subprocess
import os
from pathlib import Path

# Load mapping
mapping_df = pd.read_csv('data/mapping_ids/skc_sb_match_mapping.csv')

print(f"="*70)
print(f"BATCH SYNCHRONIZATION: {len(mapping_df)} matches")
print(f"="*70)

# Setup paths
toolkit_dir = Path('data_prep/data_merge/skillcorner-toolkit')
metadata_dir = Path('data/metadata_j1_2024')
tracking_dir = Path('data/tracking_j1_2024')
events_path = Path('data/sb_data/sb_events.json')
output_dir = Path('data/merged_data')

# Create output directory
output_dir.mkdir(parents=True, exist_ok=True)

# Track progress
successful = []
failed = []

for idx, row in mapping_df.iterrows():
    skc_match_id = row['skc_match_id']
    sb_match_id = row['sb_match_id']
    sb_home_team_id = row['sb_home_team_id']
    
    print(f"\n[{idx+1}/{len(mapping_df)}] Processing match {skc_match_id} (SB: {sb_match_id})...")
    
    # Build file paths
    match_data_path = metadata_dir / f"{skc_match_id}_metadata.json"
    tracking_data_path = tracking_dir / f"{skc_match_id}_tracking_extrapolated.jsonl"
    match_output_dir = output_dir / str(skc_match_id)
    
    # Create output directory
    match_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Skip if already processed
    if (match_output_dir / "freeze_frame_format.json").exists():
        print(f"  ⊘ Already processed, skipping")
        successful.append(skc_match_id)
        continue
    
    # Check if files exist
    if not match_data_path.exists():
        print(f"  ✗ Metadata file missing")
        failed.append((skc_match_id, "Metadata missing"))
        continue
        
    if not tracking_data_path.exists():
        print(f"  ✗ Tracking file missing")
        failed.append((skc_match_id, "Tracking missing"))
        continue
    
    # Run synchronization
    try:
        cmd = [
            'py', '-m', 'tools.with_tracking.run_statsbomb',
            '--match_data_path', str(match_data_path.absolute()),
            '--tracking_data_path', str(tracking_data_path.absolute()),
            '--statsbomb_events_path', str(events_path.absolute()),
            '--statsbomb_match_id', str(sb_match_id),
            '--statsbomb_home_team_id', str(sb_home_team_id),
            '--save_outputs_dir', str(match_output_dir.absolute())
        ]
        
        result = subprocess.run(
            cmd,
            cwd=toolkit_dir,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per match
        )
        
        if result.returncode == 0:
            print(f"  ✓ Success")
            successful.append(skc_match_id)
        else:
            print(f" ✗ Failed with exit code {result.returncode}")
            print(f"    Error: {result.stderr[:200]}")
            failed.append((skc_match_id, f"Exit code {result.returncode}"))
    
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout (>5 minutes)")
        failed.append((skc_match_id, "Timeout"))
    except Exception as e:
        print(f"  ✗ Error: {str(e)[:100]}")
        failed.append((skc_match_id, str(e)[:100]))

print(f"\n" + "="*70)
print(f"BATCH COMPLETE")
print(f"="*70)
print(f"Successful: {len(successful)}/{len(mapping_df)}")
print(f"Failed: {len(failed)}/{len(mapping_df)}")

if failed:
    print(f"\nFailed matches:")
    for match_id, reason in failed[:10]:
        print(f"  - {match_id}: {reason}")
    if len(failed) > 10:
        print(f"  ... and {len(failed) - 10} more")
