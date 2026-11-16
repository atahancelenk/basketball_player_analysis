"""Keep GitHub source files separate from local data and training results."""

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_STORE = PROJECT_DIR.parent / "staj4"
RAW_DATASET_NAME = "basketball-player-detection-3.v18i.yolov8"

# In this workspace, staj4 holds local assets. A standalone clone can place
# the downloaded data beside these scripts; both layouts work without edits.
ASSET_DIR = LOCAL_STORE if (LOCAL_STORE / RAW_DATASET_NAME).is_dir() else PROJECT_DIR
RAW_DATASET = ASSET_DIR / RAW_DATASET_NAME
PREPARED_DATASET = ASSET_DIR / "player_dataset"
OUTPUTS = ASSET_DIR / "outputs"
PRETRAINED_WEIGHTS = ASSET_DIR / "yolov8n.pt"
