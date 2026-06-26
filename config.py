# config.py
import os

# Data Paths
DATA_DIR = "data"
PRED_DIR = os.path.join(DATA_DIR, "predictions")
MODEL_DIR = "models"

RAW_DATA_PATH = os.path.join(DATA_DIR, "raw_m5_data.csv")
TREE_PRED_PATH = os.path.join(PRED_DIR, "tree_expert_preds.csv")
SEQ_PRED_PATH = os.path.join(PRED_DIR, "seq_expert_preds.csv")
FUSION_PRED_PATH = os.path.join(PRED_DIR, "fusion_preds.csv")

# Ensure environment directories exist
for path in [DATA_DIR, PRED_DIR, MODEL_DIR]:
    os.makedirs(path, exist_ok=True)

# Hyperparameters & Split Horizons
WINDOW = 30
VAL_DAYS = 90
TEST_DAYS = 90
SEED = 42

# Explicit Test Window Boundary matching your Frontend Dashboard
TEST_START_DATE = "2016-03-28"