import json
import pandas as pd
import glob
import os

print("="*70)
print("CREATING SKILLCORNER <-> STATSBOMB MATCH MAPPING")
print("="*70)

# Load all SkillCorner match metadata
print("\nLoading SkillCorner metadata files...")
metadata_files = glob.glob('data/metadata_j1_2024/*.json')
print(f"Found {len(metadata_files)} SkillCorner metadata files")

skc_matches = []
for filepath in metadata_files:
    with open(filepath, 'r', encoding='utf-8') as f:
        match_data = json.load(f)
        skc_matches.append({
            'skc_match_id': match_data['id'],
            'skc_date': match_data['date_time'][:10],  # Extract date (YYYY-MM-DD)
            'skc_home_team': match_data['home_team']['name'],
            'skc_away_team': match_data['away_team']['name'],
            'skc_home_score': match_data.get('home_team_score'),
            'skc_away_score': match_data.get('away_team_score')
        })

skc_df = pd.DataFrame(skc_matches)
print(f"✓ Loaded {len(skc_df)} SkillCorner matches")

# Load StatsBomb matches
print("\nLoading StatsBomb matches...")
with open('data/sb_data/sb_matches.json', 'r', encoding='utf-8') as f:
    sb_matches_raw = json.load(f)

sb_matches = []
for match in sb_matches_raw:
    sb_matches.append({
        'sb_match_id': match['match_id'],
        'sb_date': match['match_date'],
        'sb_home_team_id': match['home_team.home_team_id'],
        'sb_home_team': match['home_team.home_team_name'],
        'sb_away_team': match['away_team.away_team_name'],
        'sb_home_score': match['home_score'],
        'sb_away_score': match['away_score']
    })

sb_df = pd.DataFrame(sb_matches)
print(f"✓ Loaded {len(sb_df)} StatsBomb matches")

# Team name normalization: Map StatsBomb names to SkillCorner names
print("\nApplying team name normalization...")
sb_to_skc_names = {
    'Tokyo': 'FC Tokyo',
    'Júbilo Iwata': 'Jubilo Iwata',
    'Urawa Reds': 'Urawa Red Diamonds',
    'Consadole Sapporo': 'Hokkaido Consadole Sapporo'
}

def normalize_team_name(sb_name):
    """Convert StatsBomb team name to SkillCorner format"""
    return sb_to_skc_names.get(sb_name, sb_name)

sb_df['sb_home_team_normalized'] = sb_df['sb_home_team'].apply(normalize_team_name)
sb_df['sb_away_team_normalized'] = sb_df['sb_away_team'].apply(normalize_team_name)
print(f"✓ Normalized {len(sb_to_skc_names)} team name variations")

# Match on date + home team + away team
print("\n" + "="*70)
print("MATCHING STRATEGY: Date + Home Team + Away Team (with normalization)")
print("="*70)

mapping = []
unmatched_skc = []

for _, skc_match in skc_df.iterrows():
    # Try to find matching StatsBomb match using normalized team names
    sb_match = sb_df[
        (sb_df['sb_date'] == skc_match['skc_date']) &
        (sb_df['sb_home_team_normalized'] == skc_match['skc_home_team']) &
        (sb_df['sb_away_team_normalized'] == skc_match['skc_away_team'])
    ]
    
    if len(sb_match) == 1:
        # Found exact match
        mapping.append({
            'skc_match_id': skc_match['skc_match_id'],
            'sb_match_id': sb_match.iloc[0]['sb_match_id'],
            'sb_home_team_id': sb_match.iloc[0]['sb_home_team_id'],
            'date': skc_match['skc_date'],
            'home_team': skc_match['skc_home_team'],
            'away_team': skc_match['skc_away_team'],
            'skc_home_score': skc_match['skc_home_score'],
            'skc_away_score': skc_match['skc_away_score'],
            'sb_home_score': sb_match.iloc[0]['sb_home_score'],
            'sb_away_score': sb_match.iloc[0]['sb_away_score'],
        })
    elif len(sb_match) > 1:
        print(f"⚠️  Multiple matches found for SKC {skc_match['skc_match_id']}")
    else:
        unmatched_skc.append(skc_match)

mapping_df = pd.DataFrame(mapping)

print(f"\n✓ Successfully matched {len(mapping_df)} matches")
print(f"✗ {len(unmatched_skc)} SkillCorner matches without StatsBomb match")

# Validate score matches
if len(mapping_df) > 0:
    score_matches = (
        (mapping_df['skc_home_score'] == mapping_df['sb_home_score']) & 
        (mapping_df['skc_away_score'] == mapping_df['sb_away_score'])
    ).sum()
    print(f"📊 Score validation: {score_matches}/{len(mapping_df)} matches have matching scores ({score_matches/len(mapping_df)*100:.1f}%)")

# Show sample mappings
print("\n" + "="*70)
print("SAMPLE MAPPINGS (first 10)")
print("="*70)
if len(mapping_df) > 0:
    for idx, row in mapping_df.head(10).iterrows():
        score_match = "✓" if (row['skc_home_score'] == row['sb_home_score'] and row['skc_away_score'] == row['sb_away_score']) else "✗"
        print(f"\nSKC {row['skc_match_id']} → SB {row['sb_match_id']} (Team ID: {row['sb_home_team_id']})")
        print(f"  {row['date']}: {row['home_team']} vs {row['away_team']}")
        print(f"  Scores: SKC {row['skc_home_score']}-{row['skc_away_score']} | SB {row['sb_home_score']}-{row['sb_away_score']} {score_match}")

# Prepare output with specified columns
output_df = mapping_df[['skc_match_id', 'sb_match_id', 'sb_home_team_id', 'date', 'home_team', 'away_team']].copy()

# Save mapping
output_path = 'data/mapping_ids/skc_sb_match_mapping.csv'
os.makedirs('data/mapping_ids', exist_ok=True)
output_df.to_csv(output_path, index=False)
print(f"\n✓ Saved mapping to {output_path}")

# Show unmatched matches if any
if len(unmatched_skc) > 0:
    print(f"\n" + "="*70)
    print(f"UNMATCHED SKILLCORNER MATCHES ({len(unmatched_skc)})")
    print("="*70)
    for match in unmatched_skc[:5]:
        print(f"  SKC {match['skc_match_id']}: {match['skc_date']} - {match['skc_home_team']} vs {match['skc_away_team']}")
    if len(unmatched_skc) > 5:
        print(f"  ... and {len(unmatched_skc) - 5} more")

print("\n" + "="*70)
print("MAPPING COMPLETE")
print("="*70)
