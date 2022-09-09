from pathlib import Path

CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"
LOGS_DIR = Path(__file__).parent / "logs"

MAX_BOBBER_LIFETIME_SEC = 20
BOBBER_SHOW_UP_SECONDS = 2.0
LOOT_OFFSET_PIXELS = 75
MAX_LOOT_ITEMS = 1
LOOTING_GAP_TIME = 0.5
WORLD_OF_WARCRAFT_NAME = "World of Warcraft"
SLEEP_NOT_ACTIVE_SECONDS = 5
MAX_START_WAIT_TIME = 600
FISHING_WAIT_TIME = 4.5
ENVIRON_SAVE_DATA_VARIABLE_NAME = "SAVEDATA"
BOBBER_DETECTOR_CHECKPOINT_NAME = "bobber_detector.ckpt"
SPLASH_DETECTOR_CHECKPOINT_NAME = "splash_classifier.ckpt"
