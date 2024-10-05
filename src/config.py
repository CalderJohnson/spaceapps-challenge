"""Constants, locators, and hyperparameters for the project."""

# Locators
LUNAR_TRAINING_CATALOG = "./space_apps_2024_seismic_detection/data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv"
LUNAR_TRAINING_DATA = "./space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA"

# Preprocessing hyperparameters
VELOCITY_THRESHOLD = (-10 ** -15, 10 ** -15)  # Velocity bounds to consider

# Transformer hyperparameters
INPUT_SIZE = 2   # (time_rel, velocity)
D_MODEL = 64     # Feature count
NUM_HEADS = 8    # Multi-head attention heads
NUM_LAYERS = 4   # Transformer encoder layers
OUTPUT_SIZE = 1  # Regression (relative time to quake)
MAX_SEQ_LEN = 4096 # Maximum sequence length

# Training hyperparameters
N_EPOCHS = 16
BATCH_SIZE = 32
SEQ_LEN = 4096
SPLIT = 0.9
