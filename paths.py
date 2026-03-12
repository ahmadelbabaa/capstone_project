"""
Path configurations for J1 League 2024 data processing

This file defines all the directory and file paths used in the data processing pipeline.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Data directories
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MERGED_DATA_DIR = DATA_DIR / 'merged_data'

# J1 2024 specific directories
J1_TRACKING_DIR = DATA_DIR / 'tracking_j1_2024'
J1_METADATA_DIR = DATA_DIR / 'metadata_j1_2024'
J1_SB_DATA_DIR = DATA_DIR / 'sb_data'

# Legacy naming for compatibility with merge script
USL_TRACKING_DIR = J1_TRACKING_DIR
USL_DATA_DIR = J1_SB_DATA_DIR

# Mapping files
MAPPING_IDS_DIR = DATA_DIR / 'mapping_ids'
MATCH_ID_MAPPING_FILE = MAPPING_IDS_DIR / 'skc_sb_match_mapping.csv'

# Create directories if they don't exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MERGED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Print paths for verification when module is loaded
if __name__ == "__main__":
    print("Data Processing Paths:")
    print(f"  PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"  RAW_DATA_DIR: {RAW_DATA_DIR}")
    print(f"  PROCESSED_DATA_DIR: {PROCESSED_DATA_DIR}")
    print(f"  MERGED_DATA_DIR: {MERGED_DATA_DIR}")
    print(f"  J1_TRACKING_DIR: {J1_TRACKING_DIR}")
    print(f"  J1_METADATA_DIR: {J1_METADATA_DIR}")
    print(f"  J1_SB_DATA_DIR: {J1_SB_DATA_DIR}")
    print(f"  MATCH_ID_MAPPING_FILE: {MATCH_ID_MAPPING_FILE}")
