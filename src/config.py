"""Constants, locators, and hyperparameters for the project."""

# Locators
LUNAR_TRAINING_CATALOG = "./space_apps_2024_seismic_detection/data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv"
LUNAR_TRAINING_DATA = "./space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA"
LUNAR_TESTING_DATA = "./space_apps_2024_seismic_detection/data/lunar/test/data"
LUNAR_TESTING_CATEGORIES = [
    "S12_GradeB",
    "S15_GradeA",
    "S15_GradeB",
    "S16_GradeA",
    "S16_GradeB",
]

# Preprocessing hyperparameters
VELOCITY_THRESHOLD = (-10 ** -14, 10 ** -14)  # Velocity bounds to consider

# Transformer hyperparameters
INPUT_SIZE = 2     # (time_rel, velocity)
D_MODEL = 256      # Embedding dimensionality
NUM_HEADS = 8      # Multi-head attention heads
NUM_LAYERS = 4     # Transformer encoder layers
OUTPUT_SIZE = 1    # Regression (relative time to quake)
MAX_SEQ_LEN = 4096 # Maximum sequence length

# Training hyperparameters
N_EPOCHS = 10
BATCH_SIZE = 8
SEQ_LEN = 4096
SPLIT = 0.9
