"""
Merge StatsBomb events with SkillCorner tracking using ELASTIC framework

This script adapts the ELASTIC spatiotemporal matching framework for:
- StatsBomb event data (full events dataset)
- SkillCorner tracking data
- J1 League 2024 matches

Author: Adapted for J1 League analysis
Date: March 2026
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add ELASTIC to Python path
PROJECT_ROOT = Path(__file__).parent
ELASTIC_PATH = PROJECT_ROOT / 'data_prep' / 'elastic'
sys.path.insert(0, str(ELASTIC_PATH))

# Try to import ELASTIC framework
try:
    from sync.elastic import ELASTIC
    from sync import schema, config
    ELASTIC_AVAILABLE = True
    print("✅ ELASTIC framework loaded successfully")
except ImportError as e:
    ELASTIC_AVAILABLE = False
    print("⚠️  ELASTIC framework import failed:", e)
    print("💡 Check that data_prep/elastic exists and has sync/ module")

# Try to import kloppy for coordinate normalization
try:
    from kloppy import statsbomb as kloppy_statsbomb
    from kloppy.domain import Provider
    KLOPPY_AVAILABLE = True
    print("✅ kloppy library loaded")
except ImportError:
    KLOPPY_AVAILABLE = False
    print("⚠️  kloppy not available - install with: pip install kloppy")

# Project paths
RAW_DATA_DIR = PROJECT_ROOT / 'data'
MAPPING_FILE = RAW_DATA_DIR / 'mapping_ids' / 'skc_sb_match_mapping.csv'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'elastic_merged'


class StatsBombToSPADL:
    """
    Convert StatsBomb events to SPADL format for ELASTIC
    
    SPADL (Soccer Player Action Description Language) is a unified event taxonomy
    from Decroos et al., 2019: https://arxiv.org/abs/1802.07127
    """
    
    # StatsBomb pitch dimensions (yards)
    STB_LENGTH = 120.0
    STB_WIDTH = 80.0
    
    @staticmethod
    def convert_to_spadl(events_df, pitch_length=105.0, pitch_width=68.0):
        """
        Convert StatsBomb events to SPADL format with ELASTIC schema
        
        Uses coordinate transformation compatible with kloppy:
        - Shifts from corner origin (0,0) to center origin
        - Scales from 120x80 yards to real pitch dimensions in meters
        - Flips Y-axis for proper orientation
        
        Args:
            events_df (pd.DataFrame): Raw StatsBomb events
            pitch_length (float): Pitch length in meters (default: 105.0)
            pitch_width (float): Pitch width in meters (default: 68.0)
            
        Returns:
            pd.DataFrame: Events in ELASTIC-compatible SPADL format
        """
        print(f"   Converting {len(events_df)} StatsBomb events to SPADL format...")
        print(f"   Target pitch: {pitch_length}m × {pitch_width}m")
        
        elastic_events = pd.DataFrame()
        
        # ELASTIC required fields
        elastic_events['period_id'] = events_df['period'].astype('int64')
        
        # Create UTC timestamps (ELASTIC requires datetime64)
        # StatsBomb events have 'timestamp' field in "HH:MM:SS.ms" format
        # Convert to seconds from period start, then add base_time
        base_time = pd.Timestamp('2024-01-01 00:00:00')  # Common reference for events and tracking
        
        if 'timestamp' in events_df.columns:
            # Try parsing timestamp field (HH:MM:SS.ms format)
            try:
                def parse_sb_timestamp(ts_str):
                    """Parse StatsBomb timestamp string to seconds"""
                    if pd.isna(ts_str) or not isinstance(ts_str, str):
                        return None
                    try:
                        parts = ts_str.split(':')
                        if len(parts) == 3:
                            hours = float(parts[0])
                            minutes = float(parts[1])
                            seconds = float(parts[2])
                            return hours * 3600 + minutes * 60 + seconds
                    except:
                        pass
                    return None
                
                timestamp_seconds = events_df['timestamp'].apply(parse_sb_timestamp)
                # If parsing failed, fall back to minute/second
                if timestamp_seconds.isna().all():
                    timestamp_seconds = events_df['minute'] * 60 + events_df['second']
                else:
                    timestamp_seconds = timestamp_seconds.fillna(events_df['minute'] * 60 + events_df['second'])
                
                elastic_events['utc_timestamp'] = base_time + pd.to_timedelta(timestamp_seconds, unit='s')
            except Exception as e:
                print(f"      ⚠️  Error parsing timestamp field: {e}, using minute/second")
                total_seconds = events_df['minute'] * 60 + events_df['second']
                elastic_events['utc_timestamp'] = base_time + pd.to_timedelta(total_seconds, unit='s')
        else:
            # No timestamp field, use minute/second
            total_seconds = events_df['minute'] * 60 + events_df['second']
            elastic_events['utc_timestamp'] = base_time + pd.to_timedelta(total_seconds, unit='s')
        
        # Player identification (must be string type for ELASTIC .str operations)
        player_ids = events_df.get('player.id', None)
        if player_ids is not None:
            # Convert to string, replace 'nan' string with empty string for consistency
            elastic_events['player_id'] = player_ids.astype(str).replace('nan', '')
        else:
            elastic_events['player_id'] = ''
        
        # Convert to SPADL event types
        elastic_events['spadl_type'] = events_df.apply(
            StatsBombToSPADL._map_to_spadl_type, axis=1
        )
        
        # Determine success (ELASTIC required)
        elastic_events['success'] = events_df.apply(
            StatsBombToSPADL._determine_success, axis=1
        )
        
        # Extract and transform coordinates
        coords = events_df['location'].apply(StatsBombToSPADL._extract_coordinates)
        x_raw = coords.apply(lambda c: c[0] if c[0] is not None else None)
        y_raw = coords.apply(lambda c: c[1] if c[1] is not None else None)
        
        # Transform coordinates: StatsBomb (120x80 yards, corner origin) → meters (center origin)
        # Formula from kloppy/SkillCorner toolkit:
        # x = (x_raw - 60) * pitch_length / 120
        # y = -(y_raw - 40) * pitch_width / 80
        elastic_events['start_x'] = (x_raw - (StatsBombToSPADL.STB_LENGTH / 2)) * pitch_length / StatsBombToSPADL.STB_LENGTH
        elastic_events['start_y'] = -(y_raw - (StatsBombToSPADL.STB_WIDTH / 2)) * pitch_width / StatsBombToSPADL.STB_WIDTH
        
        # Keep metadata for reference
        elastic_events['event_id'] = events_df.get('id', range(len(events_df)))
        elastic_events['team_id'] = events_df['team.id']
        elastic_events['team_name'] = events_df['team.name']
        elastic_events['player_name'] = events_df.get('player.name', None)
        elastic_events['statsbomb_type'] = events_df['type.name']
        
        # Filter to only valid SPADL events
        valid_mask = elastic_events['spadl_type'].notna()
        elastic_events = elastic_events[valid_mask].copy()
        
        # CRITICAL: Sort again chronologically after filtering
        # ELASTIC detect_kickoff expects first event of each period to be the kickoff
        elastic_events = elastic_events.sort_values(['period_id', 'utc_timestamp']).reset_index(drop=True)
        
        # Ensure correct dtypes for ELASTIC schema validation
        elastic_events['period_id'] = elastic_events['period_id'].astype('int64')
        elastic_events['player_id'] = elastic_events['player_id'].astype(str)  # Must be string for .str operations
        elastic_events['success'] = elastic_events['success'].astype(bool)
        
        print(f"   ✅ Converted: {len(elastic_events)} valid SPADL events")
        print(f"      - SPADL types: {elastic_events['spadl_type'].nunique()} unique")
        print(f"      - Periods: {sorted(elastic_events['period_id'].unique())}")
        print(f"      - Events with coordinates: {elastic_events['start_x'].notna().sum()}")
        
        # Debug: Check first event of each period for kickoff detection
        for period in sorted(elastic_events['period_id'].unique()):
            period_events = elastic_events[elastic_events['period_id'] == period]
            if len(period_events) > 0:
                first = period_events.iloc[0]
                print(f"      - Period {period} first event: {first['spadl_type']} (StatsBomb: {first.get('statsbomb_type', 'Unknown')})")
        
        return elastic_events
    
    @staticmethod
    def _extract_coordinates(location):
        """Extract x, y from StatsBomb location field"""
        if location is None or isinstance(location, str):
            return (None, None)
        
        try:
            if hasattr(location, '__iter__') and len(location) >= 2:
                return (float(location[0]), float(location[1]))
        except (TypeError, ValueError, IndexError):
            pass
        
        return (None, None)
    
    @staticmethod
    def _map_to_spadl_type(event):
        """
        Map StatsBomb event type to SPADL type
        
        Based on ELASTIC config.SPADL_TYPES
        """
        event_type = event.get('type.name', '')
        pass_type = event.get('pass.type.name', '')
        
        # Pass-based events
        if event_type == 'Pass':
            # Kick-Off: Must map to 'pass' for ELASTIC detect_kickoff
            # Period 1 kickoff: minute 0, Period 2 kickoff: minute 45
            minute = event.get('minute', 999)
            second = event.get('second', 999)
            period = event.get('period', 0)
            # First 10 seconds of each period is likely kickoff
            if (period == 1 and minute == 0 and second < 10.0) or \
               (period == 2 and minute == 45 and second < 10.0):
                return 'pass'  # Kickoff
            
            if pass_type == 'Goal Kick':
                return 'goalkick'
            elif pass_type == 'Throw-in':
                return 'throw_in'
            elif pass_type == 'Corner':
                # Check if it's a cross
                if event.get('pass.cross', False):
                    return 'corner_crossed'
                return 'corner_short'
            elif pass_type == 'Free Kick':
                if event.get('pass.cross', False):
                    return 'freekick_crossed'
                return 'freekick_short'
            elif event.get('pass.cross', False):
                return 'cross'
            else:
                return 'pass'
        
        # Shot events
        elif event_type == 'Shot':
            if event.get('shot.type.name') == 'Penalty':
                return 'shot_penalty'
            elif event.get('shot.type.name') == 'Free Kick':
                return 'shot_freekick'
            else:
                return 'shot'
        
        # Defensive events
        elif event_type == 'Interception':
            return 'interception'
        elif event_type == 'Block':
            return 'shot_block'
        elif event_type == 'Clearance':
            return 'clearance'
        
        # Duels and tackles
        elif event_type == 'Duel':
            duel_type = event.get('duel.type.name', '')
            if 'Tackle' in duel_type:
                return 'tackle'
            # Aerial duels often result in clearances/interceptions
            return None  # Skip generic duels
        
        elif event_type == 'Foul Committed':
            return 'foul'
        
        # Ball control
        elif event_type == 'Dribble':
            return 'take_on'
        elif event_type == 'Miscontrol':
            return 'bad_touch'
        elif event_type == 'Dispossessed':
            return 'dispossessed'
        elif event_type == 'Ball Recovery':
            return 'ball_recovery'
        
        # Goalkeeper events
        elif event_type == 'Goal Keeper':
            gk_type = event.get('goalkeeper.type.name', '')
            if 'Save' in gk_type:
                return 'keeper_save'
            elif 'Claim' in gk_type or 'Collected' in gk_type:
                return 'keeper_claim'
            elif 'Punch' in gk_type:
                return 'keeper_punch'
            elif 'Smother' in gk_type or 'Keeper Sweeper' in gk_type:
                return 'keeper_sweeper'
            else:
                return 'keeper_pick_up'
        
        # Events not in SPADL taxonomy (skip these)
        # Ball Receipt*, Carry, Pressure, Shield, etc.
        return None
    
    @staticmethod
    def _determine_success(event):
        """Determine if event was successful"""
        event_type = event.get('type.name', '')
        
        # Pass success
        if event_type == 'Pass':
            outcome = event.get('pass.outcome.name', None)
            return outcome is None  # None means successful
        
        # Shot success (goal)
        elif event_type == 'Shot':
            outcome = event.get('shot.outcome.name', '')
            return outcome == 'Goal'
        
        # Dribble success
        elif event_type == 'Dribble':
            outcome = event.get('dribble.outcome.name', None)
            if outcome is None:
                return True
            outcome_str = str(outcome) if pd.notna(outcome) else ''
            return 'Complete' in outcome_str
        
        # Duel success
        elif event_type == 'Duel':
            outcome = event.get('duel.outcome.name', '')
            outcome_str = str(outcome) if pd.notna(outcome) else ''
            return 'Won' in outcome_str or 'Success' in outcome_str
        
        # Defensive actions are usually successful if recorded
        elif event_type in ['Interception', 'Block', 'Clearance', 'Ball Recovery']:
            return True
        
        # Goalkeeper saves are successful
        elif event_type == 'Goal Keeper':
            outcome = event.get('goalkeeper.outcome.name', '')
            outcome_str = str(outcome) if pd.notna(outcome) else ''
            return 'Saved' in outcome_str or outcome_str == ''
        
        # Fouls, miscontrols, dispossessed are failures
        elif event_type in ['Foul Committed', 'Miscontrol', 'Dispossessed']:
            return False
        
        # Default: assume success
        return True


class SkillCornerToElasticTracking:
    """
    Convert SkillCorner tracking to ELASTIC expected format
    
    ELASTIC requires tracking schema with:
    - frame_id, period_id, timestamp, utc_timestamp
    - player_id, ball (boolean)
    - x, y, z (coordinates)
    - speed, accel_s, accel_v (velocity/acceleration)
    """
    
    @staticmethod
    def convert_tracking(tracking_df, fps=25):
        """
        Convert SkillCorner tracking DataFrame to ELASTIC format
        
        Args:
            tracking_df (pd.DataFrame): SkillCorner tracking data
            fps (int): Frames per second (default: 25)
            
        Returns:
            pd.DataFrame: Tracking in ELASTIC schema format
        """
        if tracking_df is None or len(tracking_df) == 0:
            print(f"   ⚠️  No tracking data to convert")
            return pd.DataFrame()
        
        print(f"   Converting {len(tracking_df)} tracking frames to ELASTIC format...")
        
        elastic_tracking = pd.DataFrame()
        
        # Frame and period identification
        elastic_tracking['frame_id'] = tracking_df['frame_number'].astype('int64')
        elastic_tracking['period_id'] = tracking_df['period'].astype('int64')
        
        # Time information
        elastic_tracking['timestamp'] = tracking_df['timestamp'].astype(float)
        
        # Create UTC timestamps (ELASTIC requires datetime64)
        # Use base time + offset from timestamp
        base_time = pd.Timestamp('2024-01-01 00:00:00')
        elastic_tracking['utc_timestamp'] = base_time + pd.to_timedelta(
            elastic_tracking['timestamp'], unit='s'
        )
        
        # Player identification (must be string type for ELASTIC .str operations)
        elastic_tracking['player_id'] = tracking_df['player_id'].astype(str)
        elastic_tracking['ball'] = tracking_df['player_id'] == 'ball'  # True for ball rows
        
        # Coordinates (already in meters with center origin)
        elastic_tracking['x'] = tracking_df['x'].astype(float)
        elastic_tracking['y'] = tracking_df['y'].astype(float)
        # Z-coordinate: use z from tracking if available (for ball), otherwise NaN
        if 'z' in tracking_df.columns:
            elastic_tracking['z'] = tracking_df['z'].astype(float)
            # ELASTIC schema requires z >= 0 (ball can't be below ground)
            elastic_tracking['z'] = elastic_tracking['z'].clip(lower=0.0)
        else:
            elastic_tracking['z'] = np.nan
        
        # Calculate speed and acceleration
        # Group by player and period to calculate derivatives
        print(f"      Calculating velocity and acceleration...")
        elastic_tracking['speed'] = 0.0
        elastic_tracking['accel_s'] = 0.0  # Scalar acceleration
        elastic_tracking['accel_v'] = 0.0  # Vector acceleration
        
        for (player_id, period_id), group in tracking_df.groupby(['player_id', 'period']):
            if len(group) < 3:
                continue
            
            indices = group.index
            
            # Calculate velocity
            dx = np.diff(group['x'].values)
            dy = np.diff(group['y'].values)
            dt = np.diff(group['timestamp'].values)
            
            # Avoid division by zero
            dt = np.where(dt == 0, 0.001, dt)
            
            vx = dx / dt
            vy = dy / dt
            speed = np.sqrt(vx**2 + vy**2)
            
            # Pad with first value to match length
            speed = np.concatenate([[speed[0]], speed])
            elastic_tracking.loc[indices, 'speed'] = speed
            
            # Calculate acceleration (derivative of speed)
            if len(speed) >= 3:
                accel = np.diff(speed) / dt
                accel = np.concatenate([[accel[0]], accel])
                elastic_tracking.loc[indices, 'accel_s'] = accel
                
                # Vector acceleration (derivative of velocity)
                ax = np.diff(vx) / dt[:-1]
                ay = np.diff(vy) / dt[:-1]
                accel_v = np.sqrt(ax**2 + ay**2)
                accel_v = np.concatenate([[accel_v[0]], [accel_v[0]], accel_v])
                elastic_tracking.loc[indices, 'accel_v'] = accel_v
        
        # Reset index for ELASTIC (requires sequential integer index)
        elastic_tracking = elastic_tracking.reset_index(drop=True)
        
        # Ensure correct dtypes for ELASTIC schema validation
        elastic_tracking['frame_id'] = elastic_tracking['frame_id'].astype('int64')
        elastic_tracking['period_id'] = elastic_tracking['period_id'].astype('int64')
        elastic_tracking['player_id'] = elastic_tracking['player_id'].astype(str)
        elastic_tracking['ball'] = elastic_tracking['ball'].astype(bool)
        
        print(f"   ✅ Converted: {len(elastic_tracking)} tracking frames")
        print(f"      - Unique players: {elastic_tracking[elastic_tracking['ball']==False]['player_id'].nunique()}")
        print(f"      - Ball tracking rows: {elastic_tracking['ball'].sum()}")
        print(f"      - Periods: {sorted(elastic_tracking['period_id'].unique())}")
        print(f"      - Time range: {elastic_tracking['timestamp'].min():.1f}s - {elastic_tracking['timestamp'].max():.1f}s")
        
        return elastic_tracking


# Note: elastic_match() was removed - now using actual ELASTIC framework
# The ELASTIC.run() method performs 4-stage spatiotemporal synchronization


def load_goal_kick_events(match_id=None):
    """Load filtered goal kick events (DEPRECATED - use load_all_sb_events)"""
    events_file = RAW_DATA_DIR / 'sb_data' / 'sb_events_goal_kick.json'
    
    if not events_file.exists():
        print(f"❌ Goal kick events file not found: {events_file}")
        print("💡 Run filter_goal_kick_events.py first")
        return None
    
    print(f"📋 Loading goal kick events from {events_file.name}")
    events_df = pd.read_json(events_file, convert_dates=False)
    
    if match_id:
        events_df = events_df[events_df['match_id'] == match_id]
    
    print(f"   ✅ Loaded {len(events_df):,} goal kick events")
    
    return events_df


def load_all_sb_events(match_id=None):
    """Load full StatsBomb events dataset"""
    
    # First check for test file if match_id is specified
    if match_id:
        test_file = RAW_DATA_DIR / 'sb_data' / f'test_match_{match_id}.json'
        if test_file.exists():
            print(f"📋 Loading from test file: {test_file.name}")
            import json
            with open(test_file, 'r', encoding='utf-8') as f:
                events = json.load(f)
            events_df = pd.json_normalize(events)
            print(f"   ✅ Loaded {len(events_df):,} events for match {match_id}")
            return events_df
    
    # Try to find StatsBomb events file
    possible_files = [
        RAW_DATA_DIR / 'sb_data' / 'sb_events.json',
        RAW_DATA_DIR / 'sb_data' / 'all_events.json',
        RAW_DATA_DIR / 'sb_data' / 'statsbomb_events.json',
    ]
    
    events_file = None
    for f in possible_files:
        if f.exists():
            events_file = f
            break
    
    if events_file is None:
        print(f"❌ StatsBomb events file not found. Tried:")
        for f in possible_files:
            print(f"   - {f}")
        return None
    
    print(f"📋 Loading full StatsBomb events from {events_file.name}")
    
    # Use json.load + manual filtering for speed when querying single match
    if match_id:
        import json
        with open(events_file, 'r', encoding='utf-8') as f:
            all_events = json.load(f)
        # Filter to match_id
        events = [e for e in all_events if e.get('match_id') == match_id]
        events_df = pd.json_normalize(events)
        print(f"   ✅ Loaded {len(events_df):,} events for match {match_id}")
    else:
        # Load all events (slow)
        events_df = pd.read_json(events_file, convert_dates=False)
        print(f"   ✅ Loaded {len(events_df):,} total events")
        print(f"      Unique matches: {events_df['match_id'].nunique()}")
        print(f"      Event types: {events_df['type.name'].nunique()}")
    
    return events_df


def load_tracking_data(skc_match_id):
    """Load SkillCorner tracking data for a match"""
    tracking_file = RAW_DATA_DIR / 'tracking_j1_2024' / f'{skc_match_id}_tracking_extrapolated.jsonl'
    
    if not tracking_file.exists():
        print(f"   ⚠️  Tracking file not found: {tracking_file}")
        return None, None, None
    
    print(f"   📊 Loading tracking: {tracking_file.name}")
    
    # Load tracking data - file is a JSON array
    try:
        with open(tracking_file, 'r', encoding='utf-8') as f:
            print(f"      Reading JSON array (this may take a moment)...")
            frames_data = json.load(f)
        
        if not isinstance(frames_data, list):
            print(f"      ⚠️  Expected JSON array, got {type(frames_data)}")
            return None, None, None
        
        print(f"      Loaded {len(frames_data):,} frames from JSON")
        
        # Extract player data from nested structure
        # Each frame has 'player_data' as a list of players
        print(f"      Extracting player positions from frames...")
        player_rows = []
        
        for frame in frames_data:
            frame_num = frame.get('frame')
            timestamp_raw = frame.get('timestamp')
            period = frame.get('period')
            
            # Skip frames without valid data
            if timestamp_raw is None or period is None:
                continue
            
            # Parse timestamp - can be string 'HH:MM:SS.ms' or float
            if isinstance(timestamp_raw, str):
                try:
                    # Parse HH:MM:SS.ms format
                    time_parts = timestamp_raw.split(':')
                    if len(time_parts) == 3:
                        hours = float(time_parts[0])
                        minutes = float(time_parts[1])
                        seconds = float(time_parts[2])
                        timestamp = hours * 3600 + minutes * 60 + seconds
                    else:
                        continue
                except (ValueError, AttributeError):
                    continue
            else:
                try:
                    timestamp = float(timestamp_raw)
                except (ValueError, TypeError):
                    continue
            
            # Extract each player from player_data list
            player_data = frame.get('player_data', [])
            for player in player_data:
                if isinstance(player, dict) and 'x' in player and 'y' in player:
                    player_row = {
                        'frame_number': frame_num,
                        'timestamp': timestamp,
                        'period': int(period),
                        'player_id': player.get('player_id'),
                        'team': player.get('team_id', player.get('team')),
                        'jersey_number': player.get('jersey_number'),
                        'x': float(player['x']),
                        'y': float(player['y']),
                    }
                    player_rows.append(player_row)
            
            # Extract ball data
            ball_data = frame.get('ball_data', {})
            if isinstance(ball_data, dict) and ball_data.get('x') is not None and ball_data.get('y') is not None:
                ball_row = {
                    'frame_number': frame_num,
                    'timestamp': timestamp,
                    'period': int(period),
                    'player_id': 'ball',  # Ball ID
                    'team': None,
                    'jersey_number': None,
                    'x': float(ball_data['x']),
                    'y': float(ball_data['y']),
                    'z': float(ball_data.get('z', 0.0)) if ball_data.get('z') is not None else 0.0,  # Ball height
                }
                player_rows.append(ball_row)
        
        if not player_rows:
            print(f"      ⚠️  No player positions found in tracking data")
            return None, None, None
        
        tracking_df = pd.DataFrame(player_rows)
        print(f"      Extracted {len(tracking_df):,} tracking rows")
        ball_rows = len([r for r in player_rows if r.get('player_id') == 'ball'])
        print(f"      Ball tracking rows: {ball_rows}")
        
    except json.JSONDecodeError as e:
        print(f"      ❌ Failed to parse JSON: {e}")
        return None, None, None
    except Exception as e:
        print(f"      ❌ Error loading tracking: {e}")
        return None, None, None
    
    # Calculate minute and seconds from timestamp (timestamp is in seconds)
    tracking_df['minute'] = (tracking_df['timestamp'] // 60).astype(int)
    tracking_df['seconds'] = (tracking_df['timestamp'] % 60).astype(float)
    
    print(f"      Time range: {tracking_df['timestamp'].min():.1f}s - {tracking_df['timestamp'].max():.1f}s")
    print(f"      Periods: {sorted(tracking_df['period'].unique())}")
    print(f"      Unique players: {tracking_df['player_id'].nunique()}")
    
    # Load pitch dimensions from metadata
    metadata_file = RAW_DATA_DIR / 'metadata_j1_2024' / f'{skc_match_id}_metadata.json'
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        pitch_length = metadata.get('pitch_length', 105.0)
        pitch_width = metadata.get('pitch_width', 68.0)
    else:
        pitch_length = 105.0
        pitch_width = 68.0
    
    print(f"      Pitch: {pitch_length}m × {pitch_width}m")
    
    return tracking_df, pitch_length, pitch_width


def merge_single_match_with_elastic(sb_match_id, mapping_df):
    """
    Merge one match using actual ELASTIC framework
    
    This function:
    1. Loads StatsBomb events and converts to SPADL format
    2. Loads SkillCorner tracking and formats to ELASTIC schema
    3. Runs ELASTIC synchronization (4-stage spatiotemporal matching)
    4. Returns synchronized events with tracking frame_ids
    
    Args:
        sb_match_id (int): StatsBomb match ID
        mapping_df (pd.DataFrame): Match ID mapping
        
    Returns:
        pd.DataFrame: Synchronized events or None if failed
    """
    print(f"\n{'='*70}")
    print(f"📊 Processing Match {sb_match_id}")
    print(f"{'='*70}")
    
    # Get SkillCorner match ID
    match_info = mapping_df[mapping_df['sb_match_id'] == sb_match_id]
    if len(match_info) == 0:
        print(f"❌ No mapping found for match {sb_match_id}")
        return None
    
    skc_match_id = match_info['skc_match_id'].iloc[0]
    print(f"SkillCorner Match ID: {skc_match_id}")
    print(f"Date: {match_info['date'].iloc[0]}")
    print(f"Match: {match_info['home_team'].iloc[0]} vs {match_info['away_team'].iloc[0]}")
    
    # Load ALL StatsBomb events for this match
    events_df = load_all_sb_events(match_id=sb_match_id)
    if events_df is None or len(events_df) == 0:
        print(f"⚠️  No events found for this match")
        return None
    
    # CRITICAL: Sort events chronologically before conversion
    # ELASTIC expects first event of each period to be the kickoff pass
    events_df = events_df.sort_values(['period', 'minute', 'second']).reset_index(drop=True)
    
    print(f"\n📋 Raw Events: {len(events_df)} total (sorted chronologically)")
    print(f"   Event types: {events_df['type.name'].nunique()} unique StatsBomb types")
    
    # Load tracking data
    tracking_df, pitch_length, pitch_width = load_tracking_data(skc_match_id)
    if tracking_df is None or len(tracking_df) == 0:
        print(f"❌ Tracking data not available or empty")
        return None
    
    # Convert to ELASTIC format
    print(f"\n🔄 Converting to ELASTIC-compatible format...")
    
    # Convert events to SPADL
    elastic_events = StatsBombToSPADL.convert_to_spadl(
        events_df, pitch_length, pitch_width
    )
    
    if len(elastic_events) == 0:
        print(f"⚠️  No valid SPADL events after conversion")
        return None
    
    # Convert tracking to ELASTIC schema
    elastic_tracking = SkillCornerToElasticTracking.convert_tracking(tracking_df, fps=25)
    
    if len(elastic_tracking) == 0:
        print(f"⚠️  No valid tracking data after conversion")
        return None
    
    if not ELASTIC_AVAILABLE:
        print(f"\n⚠️  ELASTIC framework not available - cannot perform synchronization")
        print(f"💡 Install requirements: cd data_prep/elastic && pip install -r requirements.txt")
        return None
    
    # Run ELASTIC synchronization
    print(f"\n🔬 Running ELASTIC synchronization...")
    print(f"   This performs 4-stage spatiotemporal matching:")
    print(f"   1. Kick-off synchronization")
    print(f"   2. Major event sync (pass, shot, cross, clearance)")
    print(f"   3. Receive detection")
    print(f"   4. Minor event sync (tackle, foul, dispossessed)")
    
    try:
        # Create ELASTIC syncer instance
        syncer = ELASTIC(elastic_events, elastic_tracking)
        
        # Run synchronization
        syncer.run()
        
        # Get synchronized events
        synced_events = syncer.events.copy()
        
        print(f"\n✅ Synchronization complete!")
        print(f"   Total events: {len(synced_events)}")
        print(f"   Synchronized: {synced_events['frame_id'].notna().sum()}")
        print(f"   Unsynchronized: {synced_events['frame_id'].isna().sum()}")
        
        # Add match identifiers
        synced_events['statsbomb_match_id'] = sb_match_id
        synced_events['skillcorner_match_id'] = skc_match_id
        
        # Show synchronization quality by event type
        print(f"\n📊 Synchronization by SPADL Type:")
        sync_by_type = synced_events.groupby('spadl_type').agg({
            'frame_id': lambda x: f"{x.notna().sum()}/{len(x)}"
        }).sort_index()
        for spadl_type, row in sync_by_type.iterrows():
            print(f"   {spadl_type:20s}: {row['frame_id']}")
        
        return synced_events
        
    except Exception as e:
        print(f"\n❌ ELASTIC synchronization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_all_matches():
    """Process all matches with StatsBomb events using ELASTIC framework"""
    print("🚀 ELASTIC-based Event Synchronization Pipeline")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    if not ELASTIC_AVAILABLE:
        print("❌ ELASTIC framework not available - cannot proceed")
        return
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load match mapping
    print(f"🗺️  Loading match mapping...")
    mapping_df = pd.read_csv(MAPPING_FILE)
    print(f"   ✅ {len(mapping_df)} match mappings loaded\n")
    
    # Load full StatsBomb events to see which matches we have
    all_events = load_all_sb_events()
    if all_events is None:
        return
    
    unique_matches = all_events['match_id'].unique()
    print(f"\n🎯 Target: {len(unique_matches)} matches with StatsBomb events")
    print(f"   Total events to synchronize: {len(all_events):,}\n")
    
    # Process each match
    all_synced = []
    successful = 0
    failed = []
    
    for idx, sb_match_id in enumerate(unique_matches, 1):
        print(f"\n[{idx}/{len(unique_matches)}] ", end='')
        
        try:
            synced_df = merge_single_match_with_elastic(sb_match_id, mapping_df)
            
            if synced_df is not None and len(synced_df) > 0:
                all_synced.append(synced_df)
                successful += 1
                
                # Save individual match file
                output_file = OUTPUT_DIR / f'match_{sb_match_id}_elastic.json'
                synced_df.to_json(output_file, orient='records', indent=2)
                print(f"💾 Saved: {output_file.name}")
            else:
                failed.append((sb_match_id, "No synchronization"))
                
        except Exception as e:
            print(f"❌ Error: {e}")
            failed.append((sb_match_id, str(e)))
    
    # Combine all matches
    if all_synced:
        print(f"\n{'='*70}")
        print(f"🎉 PROCESSING COMPLETE")
        print(f"{'='*70}")
        
        combined_df = pd.concat(all_synced, ignore_index=True)
        
        # Save combined file
        combined_file = OUTPUT_DIR / 'all_events_elastic_synced.json'
        combined_df.to_json(combined_file, orient='records', indent=2)
        
        print(f"\n📊 Final Statistics:")
        print(f"   ✅ Successful matches: {successful}/{len(unique_matches)}")
        print(f"   Total events: {len(combined_df):,}")
        print(f"   Synchronized events: {combined_df['frame_id'].notna().sum():,}")
        print(f"   Unsynchronized events: {combined_df['frame_id'].isna().sum():,}")
        print(f"   Synchronization rate: {(combined_df['frame_id'].notna().sum() / len(combined_df) * 100):.1f}%")
        
        print(f"\n💾 Outputs:")
        print(f"   Individual files: {OUTPUT_DIR / 'match_*_elastic.json'}")
        print(f"   Combined file: {combined_file}")
        
        # Overall SPADL type summary
        print(f"\n📊 Events by SPADL Type:")
        type_summary = combined_df['spadl_type'].value_counts().head(10)
        for spadl_type, count in type_summary.items():
            pct = (count / len(combined_df)) * 100
            synced = combined_df[combined_df['spadl_type'] == spadl_type]['frame_id'].notna().sum()
            sync_rate = (synced / count * 100) if count > 0 else 0
            print(f"   {spadl_type:20s}: {count:6d} ({pct:5.1f}%) - {sync_rate:5.1f}%  synced")
        
    else:
        print(f"\n❌ No data synchronized successfully")
    
    if failed:
        print(f"\n⚠️  Failed matches: {len(failed)}")
        for sb_id, reason in failed[:5]:
            print(f"   Match {sb_id}: {reason}")
        if len(failed) > 5:
            print(f"   ...and {len(failed) - 5} more")
    
    print(f"\n🏁 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def test_single_match(match_id=None):
    """Test merge on a single match for quick evaluation"""
    print("🧪 ELASTIC Single Match Test")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load match mapping
    print(f"🗺️  Loading match mapping...")
    mapping_df = pd.read_csv(MAPPING_FILE)
    print(f"   ✅ {len(mapping_df)} match mappings loaded\n")
    
    # If no match specified, use test match 3925601
    if match_id is None:
        match_id = 3925601  # Default test match
        print(f"🎯 Testing with default test match: {match_id}\n")
    
    try:
        synced_df = merge_single_match_with_elastic(match_id, mapping_df)
        
        if synced_df is not None and len(synced_df) > 0:
            # Save output
            output_file = OUTPUT_DIR / f'match_{match_id}_elastic.json'
            synced_df.to_json(output_file, orient='records', indent=2)
            
            print(f"\n{'='*70}")
            print(f"✅ TEST SUCCESSFUL")
            print(f"{'='*70}")
            print(f"\n📊 Results:")
            print(f"   Total events: {len(synced_df):,}")
            print(f"   Synchronized events: {synced_df['frame_id'].notna().sum():,}")
            print(f"   Output file: {output_file}")
            
            # Show sample synchronized event
            synced_sample = synced_df[synced_df['frame_id'].notna()].iloc[0] if len(synced_df[synced_df['frame_id'].notna()]) > 0 else None
            if synced_sample is not None:
                print(f"\n📋 Sample synchronized event:")
                print(f"   SPADL type: {synced_sample['spadl_type']}")
                print(f"   Player: {synced_sample.get('player_name', 'N/A')}")
                print(f"   Location: ({synced_sample['start_x']:.1f}, {synced_sample['start_y']:.1f}) meters")
                print(f"   Frame ID: {synced_sample['frame_id']}")
                print(f"   Period: {synced_sample['period_id']}")
                print(f"   Success: {synced_sample['success']}")
            
            # SPADL type breakdown
            print(f"\n📊 SPADL Event Type Breakdown:")
            event_counts = synced_df['spadl_type'].value_counts().head(10)
            for event_type, count in event_counts.items():
                pct = (count / len(synced_df)) * 100
                synced = synced_df[synced_df['spadl_type'] == event_type]['frame_id'].notna().sum()
                print(f"   {event_type:20s}: {count:4d} ({pct:5.1f}%) - {synced} synced")
            
            return synced_df
        else:
            print(f"\n❌ TEST FAILED: No events synchronized")
            return None
            
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main entry point"""
    import sys
    
    if not ELASTIC_AVAILABLE:
        print("\n❌ ELASTIC framework not available!")
        print("💡 Make sure data_prep/elastic exists and has the sync/ module")
        print("💡 Install requirements: cd data_prep/elastic && pip install -r requirements.txt")
        return
    
    if not KLOPPY_AVAILABLE:
        print("\n⚠️  kloppy not available (optional but recommended)")
        print("💡 Install with: pip install kloppy")
        print()
    
    # Check for test mode
    if len(sys.argv) > 1:
        if sys.argv[1] == '--test':
            match_id = int(sys.argv[2]) if len(sys.argv) > 2 else None
            test_single_match(match_id)
        elif sys.argv[1] == '--match':
            match_id = int(sys.argv[2])
            test_single_match(match_id)
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Usage:")
            print("  python merge_goal_kicks_with_elastic.py          # Process all matches")
            print("  python merge_goal_kicks_with_elastic.py --test   # Test with first match")
            print("  python merge_goal_kicks_with_elastic.py --test <match_id>  # Test specific match")
    else:
        # Default: process all matches
        process_all_matches()


if __name__ == "__main__":
    main()
