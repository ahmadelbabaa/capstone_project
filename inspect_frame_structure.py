import json

# Load first few frames to inspect structure
with open('data/tracking_j1_2024/1901679_tracking_extrapolated.jsonl', 'r') as f:
    content = f.read()
    
# Parse as JSON array
frames = json.loads(content)

if frames and len(frames) > 0:
    first_frame = frames[0]
    print("First frame keys:")
    print(first_frame.keys())
    print("\nFirst frame structure:")
    for key, value in first_frame.items():
        if isinstance(value, list):
            print(f"  {key}: list with {len(value)} items")
            if len(value) > 0:
                print(f"    First item: {value[0]}")
        else:
            print(f"  {key}: {value}")
