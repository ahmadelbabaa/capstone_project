"""
Retry the 4 matches that failed due to Unicode encoding errors
"""
import pandas as pd
import subprocess
from pathlib import Path

# Load failed matches
failed_df = pd.read_csv('failed_matches.csv')

print(f"="*70)
print(f"RETRYING {len(failed_df)} FAILED MATCHES")
print(f"="*70)

# Setup paths
toolkit_dir = Path('data_prep/data_merge/skillcorner-toolkit')
metadata_dir = Path('data/metadata_j1_2024')
tracking_dir = Path('data/tracking_j1_2024')
events_path = Path('data/sb_data/sb_events.json')
output_dir = Path('data/merged_data')

# Track progress
successful = []
failed = []

for idx, row in failed_df.iterrows():
    skc_match_id = row['skc_match_id']
    sb_match_id = row['sb_match_id']
    sb_home_team_id = row['sb_home_team_id']
    
    print(f"\n[{idx+1}/{len(failed_df)}] Processing match {skc_match_id} (SB: {sb_match_id})...")
    
    # Build file paths
    match_data_path = metadata_dir / f"{skc_match_id}_metadata.json"
    tracking_data_path = tracking_dir / f"{skc_match_id}_tracking_extrapolated.jsonl"
    match_output_dir = output_dir / str(skc_match_id)
    
    # Create output directory
    match_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if files exist
    if not match_data_path.exists():
        print(f"  [X] Metadata file missing")
        failed.append((skc_match_id, "Metadata missing"))
        continue
        
    if not tracking_data_path.exists():
        print(f"  [X] Tracking file missing")
        failed.append((skc_match_id, "Tracking missing"))
        continue
    
    # Run synchronization with UTF-8 environment
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
        
        # Set UTF-8 encoding for the subprocess
        import os
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(
            cmd,
            cwd=toolkit_dir,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout per match
            env=env
        )
        
        if result.returncode == 0:
            print(f"  [OK] Success")
            successful.append(skc_match_id)
        else:
            print(f"  [X] Failed with exit code {result.returncode}")
            print(f"\n--- STDOUT ---")
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            print(f"\n--- STDERR ---")
            print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            failed.append((skc_match_id, f"Exit code {result.returncode}"))
    
    except subprocess.TimeoutExpired:
        print(f"  [X] Timeout (>5 minutes)")
        failed.append((skc_match_id, "Timeout"))
    except Exception as e:
        print(f"  [X] Error: {str(e)[:100]}")
        failed.append((skc_match_id, str(e)[:100]))

print(f"\n" + "="*70)
print(f"RETRY COMPLETE")
print(f"="*70)
print(f"Successful: {len(successful)}/{len(failed_df)}")
print(f"Failed: {len(failed)}/{len(failed_df)}")

if failed:
    print(f"\nStill failed:")
    for match_id, reason in failed:
        print(f"  - {match_id}: {reason}")
else:
    print(f"\n[OK] All matches successfully processed!")
