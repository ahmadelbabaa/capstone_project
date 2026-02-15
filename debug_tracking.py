import json

# Load one frame from tracking data to see if it has jersey numbers
print("Checking tracking data structure...")
with open('data/tracking_j1_2024/1410827_tracking_extrapolated.jsonl', 'r', encoding='utf-8') as f:
    # It's actually a JSON array, not JSONL
    tracking_data = json.load(f)

# Find a frame with players
frame_with_players = None
for frame in tracking_data[:100]:  # Check first 100 frames
    if frame.get('player_data') and len(frame['player_data']) > 0:
        frame_with_players = frame
        break

if frame_with_players:
    print(f"Found frame {frame_with_players['frame']} with {len(frame_with_players['player_data'])} players")
    print(f"\nKeys in tracking frame: {frame_with_players.keys()}")
    
    print("\nFirst 3 players in tracking data:")
    for i, player in enumerate(frame_with_players['player_data'][:3]):
        print(f"\n  Player {i+1}:")
        print(f"    Keys: {list(player.keys())}")
        for key, value in player.items():
            print(f"    {key}: {value}")
else:
    print("No frames with player data found in first 100 frames")
