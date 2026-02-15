import json
import pandas as pd
import glob
from difflib import get_close_matches

print("Analyzing unmatched matches...\n")

# Load SkillCorner matches
metadata_files = glob.glob('data/metadata_j1_2024/*.json')
skc_matches = []
for filepath in metadata_files:
    with open(filepath, 'r', encoding='utf-8') as f:
        match_data = json.load(f)
        skc_matches.append({
            'skc_match_id': match_data['id'],
            'skc_date': match_data['date_time'][:10],
            'skc_home_team': match_data['home_team']['name'],
            'skc_away_team': match_data['away_team']['name'],
        })
skc_df = pd.DataFrame(skc_matches)

# Load StatsBomb matches
with open('data/sb_data/sb_matches.json', 'r', encoding='utf-8') as f:
    sb_matches_raw = json.load(f)

sb_matches = []
for match in sb_matches_raw:
    sb_matches.append({
        'sb_match_id': match['match_id'],
        'sb_date': match['match_date'],
        'sb_home_team': match['home_team.home_team_name'],
        'sb_away_team': match['away_team.away_team_name'],
    })
sb_df = pd.DataFrame(sb_matches)

# Load existing mapping
mapping_df = pd.read_csv('data/mapping_ids/skc_sb_match_mapping.csv')
matched_skc_ids = mapping_df['skc_match_id'].tolist()

# Find unmatched
unmatched_skc = skc_df[~skc_df['skc_match_id'].isin(matched_skc_ids)]

print(f"Total unmatched: {len(unmatched_skc)}\n")
print("="*80)
print("ANALYZING FIRST 10 UNMATCHED MATCHES")
print("="*80)

# Get all unique team names from both datasets
sb_teams = set(sb_df['sb_home_team'].unique()) | set(sb_df['sb_away_team'].unique())

for idx, (_, match) in enumerate(unmatched_skc.head(10).iterrows()):
    print(f"\n{idx+1}. SKC Match {match['skc_match_id']} - {match['skc_date']}")
    print(f"   SKC: {match['skc_home_team']} vs {match['skc_away_team']}")
    
    # Find potential StatsBomb matches on same date
    sb_same_date = sb_df[sb_df['sb_date'] == match['skc_date']]
    
    if len(sb_same_date) > 0:
        print(f"   StatsBomb matches on same date:")
        for _, sb_match in sb_same_date.iterrows():
            print(f"     - {sb_match['sb_home_team']} vs {sb_match['sb_away_team']}")
        
        # Try to find close team name matches
        home_close = get_close_matches(match['skc_home_team'], sb_teams, n=1, cutoff=0.6)
        away_close = get_close_matches(match['skc_away_team'], sb_teams, n=1, cutoff=0.6)
        
        if home_close or away_close:
            print(f"   Potential name variations:")
            if home_close:
                print(f"     Home: '{match['skc_home_team']}' → '{home_close[0]}'")
            if away_close:
                print(f"     Away: '{match['skc_away_team']}' → '{away_close[0]}'")

print("\n" + "="*80)
print("UNIQUE TEAM NAME DIFFERENCES")
print("="*80)

# Collect all unique team names from unmatched
unmatched_home_teams = set(unmatched_skc['skc_home_team'].unique())
unmatched_away_teams = set(unmatched_skc['skc_away_team'].unique())
unmatched_teams = unmatched_home_teams | unmatched_away_teams

print("\nTeam names in unmatched SKC matches that might need mapping:")
for team in sorted(unmatched_teams):
    close = get_close_matches(team, sb_teams, n=1, cutoff=0.6)
    if close:
        print(f"  '{team}' → '{close[0]}'")
    else:
        print(f"  '{team}' → NOT FOUND in StatsBomb")
