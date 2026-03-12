"""
Identify which matches failed during batch synchronization
"""
import pandas as pd
from pathlib import Path

# Load mapping
mapping_df = pd.read_csv('data/mapping_ids/skc_sb_match_mapping.csv')
output_dir = Path('data/merged_data')

print(f"Checking {len(mapping_df)} matches for processing status...\n")

failed_matches = []
successful_matches = []

for idx, row in mapping_df.iterrows():
    skc_match_id = row['skc_match_id']
    sb_match_id = row['sb_match_id']
    match_output_dir = output_dir / str(skc_match_id)
    
    # Check if successfully processed
    if (match_output_dir / "freeze_frame_format.json").exists():
        successful_matches.append({
            'index': idx,
            'skc_match_id': skc_match_id,
            'sb_match_id': sb_match_id
        })
    else:
        failed_matches.append({
            'index': idx,
            'skc_match_id': skc_match_id,
            'sb_match_id': sb_match_id,
            'sb_home_team_id': row['sb_home_team_id']
        })

print(f"={'='*70}")
print(f"BATCH SYNCHRONIZATION STATUS")
print(f"={'='*70}")
print(f"✓ Successful: {len(successful_matches)}/{len(mapping_df)}")
print(f"✗ Failed: {len(failed_matches)}/{len(mapping_df)}")

if failed_matches:
    print(f"\n{'='*70}")
    print(f"FAILED MATCHES DETAILS:")
    print(f"{'='*70}")
    for match in failed_matches:
        print(f"\n[{match['index']+1}] Match {match['skc_match_id']}")
        print(f"    StatsBomb Match ID: {match['sb_match_id']}")
        print(f"    StatsBomb Home Team ID: {match['sb_home_team_id']}")
        
        # Try to get match details from metadata
        metadata_file = Path(f"data/metadata_j1_2024/{match['skc_match_id']}_metadata.json")
        if metadata_file.exists():
            import json
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                home_team = metadata.get('home_team', {}).get('short_name', 'Unknown')
                away_team = metadata.get('away_team', {}).get('short_name', 'Unknown')
                print(f"    Teams: {home_team} vs {away_team}")
        
        # Check if files exist
        tracking_file = Path(f"data/tracking_j1_2024/{match['skc_match_id']}_tracking_extrapolated.jsonl")
        print(f"    Metadata exists: {metadata_file.exists()}")
        print(f"    Tracking exists: {tracking_file.exists()}")
    
    # Save failed matches to CSV for easy re-processing
    failed_df = pd.DataFrame(failed_matches)
    failed_df.to_csv('failed_matches.csv', index=False)
    print(f"\n✓ Failed matches saved to 'failed_matches.csv'")

print(f"\n{'='*70}\n")
