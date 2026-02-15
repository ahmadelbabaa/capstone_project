import json
import os

# Get base directory
base_dir = r"C:\Users\User\OneDrive\Capstone Project"

# Load data
with open(os.path.join(base_dir, r'data\sb_data\sb_events.json'), encoding='utf-8') as f:
    events = json.load(f)

with open(os.path.join(base_dir, r'data\metadata_j1_2024\1410827_metadata.json'), encoding='utf-8') as f:
    metadata = json.load(f)

# Filter for this match
match_events = [e for e in events if e.get('match_id') == 3925227]

# Extract Starting XI
starting_xi = [e for e in match_events if e.get('type.name') == 'Starting XI']
starting_xi_ids = set()
for event in starting_xi:
    lineup = event.get('tactics.lineup', [])
    for player in lineup:
        starting_xi_ids.add(player.get('player.id'))

print(f"Starting XI players: {len(starting_xi_ids)}")

# Extract all unique player IDs from events
all_player_ids = set()
for event in match_events:
    pid = event.get('player.id')
    if pid:
        all_player_ids.add(pid)

print(f"Total unique player IDs in events: {len(all_player_ids)}")
print(f"Player IDs not in Starting XI: {len(all_player_ids - starting_xi_ids)}")

# Extract substitutes
substitution_events = [e for e in match_events if e.get('type.name') == 'Substitution']
print(f"\nSubstitution events: {len(substitution_events)}")

substitute_ids = set()
substitute_names = []
for sub_event in substitution_events:
    # Player coming ON
    sub_id = sub_event.get('substitution.replacement.id')
    sub_name = sub_event.get('substitution.replacement.name')
    if sub_id:
        substitute_ids.add(sub_id)
        substitute_names.append((sub_id, sub_name))
    
    # Player coming OFF (also need to track these!)
    off_id = sub_event.get('player.id')
    if off_id:
        substitute_ids.add(off_id)

print(f"Unique substitute player IDs (ON + OFF): {len(substitute_ids)}")
print(f"\nSubstitutes coming ON:")
for pid, name in substitute_names[:10]:  # Show first 10
    print(f"  {pid}: {name}")

# Build SkillCorner name lookup
skc_players = metadata.get('players', [])
print(f"\nSkillCorner players: {len(skc_players)}")

skc_name_to_player = {}
for player in skc_players:
    full_name = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()
    short_name = player.get('short_name', '')
    
    if full_name:
        skc_name_to_player[full_name.lower()] = player
    if short_name:
        skc_name_to_player[short_name.lower()] = player

print(f"SkillCorner name lookup entries: {len(skc_name_to_player)}")

# Try to match substitutes
print(f"\nMatching substitutes by name:")
matched = 0
unmatched = []
for sub_id, sub_name in substitute_names:
    sub_name_lower = sub_name.lower() if sub_name else ''
    if sub_name_lower and sub_name_lower in skc_name_to_player:
        matched += 1
        skc_p = skc_name_to_player[sub_name_lower]
        print(f"  ✓ {sub_name} → Jersey {skc_p.get('number')}")
    else:
        unmatched.append((sub_id, sub_name))

print(f"\nMatched: {matched}/{len(substitute_names)}")

# Show what names exist in SkillCorner for comparison
print(f"\nSkillCorner player names (sample):")
for i, (name_key, player) in enumerate(list(skc_name_to_player.items())[:20]):
    print(f"  '{name_key}' → #{player.get('number')} {player.get('first_name')} {player.get('last_name')}")

print(f"\nUnmatched StatsBomb substitutes (trying to match against SkillCorner):")
for pid, name in unmatched[:10]:
    print(f"  StatsBomb: '{name.lower()}' (ID: {pid})")
    # Check for partial matches
    partial_matches = [k for k in skc_name_to_player.keys() if name.lower().split()[-1] in k or k.split()[-1] in name.lower()]
    if partial_matches:
        print(f"    Possible SKC matches: {partial_matches[:3]}")

# Check what player IDs are in events but not in Starting XI or Substitutes
other_ids = all_player_ids - starting_xi_ids - substitute_ids
print(f"\nPlayer IDs in events but NOT in Starting XI or Substitutes: {len(other_ids)}")
if other_ids:
    print(f"Sample IDs: {list(other_ids)[:20]}")
