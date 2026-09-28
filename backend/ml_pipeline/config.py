"""
PhytoSense AI - ML Pipeline Configuration

Central configuration for dataset preparation, preprocessing,
training, and experiment management.
"""

from pathlib import Path
import os

from dotenv import load_dotenv


# ============================================================================
# Environment
# ============================================================================

BASE_DIR = Path(__file__).resolve().parents[2]

ENV_FILE = BASE_DIR / ".env"

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)


# ============================================================================
# Dataset
# ============================================================================

DATASET_VERSION = "PlantVillage Tomato v1"

DATASETS_ROOT = Path(
    os.getenv(
        "DATASETS_ROOT",
        r"D:\Code\_Archieve\Minor Project\Datasets",
    )
)

PLANTVILLAGE_ROOT = (
    DATASETS_ROOT
    / "PlantVillage-Dataset"
)

COLOR_ROOT = (
    PLANTVILLAGE_ROOT
    / "raw"
    / "color"
)

MANIFEST_PATH = (
    BASE_DIR
    / "backend"
    / "ml_pipeline"
    / "manifests"
    / "plantvillage_tomato_split.csv"
)


# ============================================================================
# Experiment
# ============================================================================

EXPERIMENT_ID = "EXP-001"

EXPERIMENT_NAME = "EfficientNetV2-B0 Baseline"

EXPERIMENT_ROOT = (
    BASE_DIR
    / "backend"
    / "ml_pipeline"
    / "experiments"
    / EXPERIMENT_ID
)

CHECKPOINT_DIR = EXPERIMENT_ROOT / "checkpoints"

MODEL_DIR = EXPERIMENT_ROOT / "models"

HISTORY_DIR = EXPERIMENT_ROOT / "history"

METRICS_DIR = EXPERIMENT_ROOT / "metrics"

PLOTS_DIR = EXPERIMENT_ROOT / "plots"

LOG_DIR = EXPERIMENT_ROOT / "logs"


# ============================================================================
# Experiment artifacts
# ============================================================================

CONFIG_JSON = EXPERIMENT_ROOT / "config.json"

BEST_CHECKPOINT = (
    CHECKPOINT_DIR / "checkpoint_best.pt"
)

LAST_CHECKPOINT = (
    CHECKPOINT_DIR / "checkpoint_last.pt"
)

TRAINING_HISTORY_JSON = (
    HISTORY_DIR / "training_history.json"
)

METRICS_JSON = (
    METRICS_DIR / "metrics.json"
)

PREDICTIONS_CSV = (
    METRICS_DIR / "predictions.csv"
)

CLASSIFICATION_REPORT_CSV = (
    METRICS_DIR / "classification_report.csv"
)

ACCURACY_CURVE = (
    PLOTS_DIR / "accuracy_curve.png"
)

LOSS_CURVE = (
    PLOTS_DIR / "loss_curve.png"
)

CONFUSION_MATRIX = (
    PLOTS_DIR / "confusion_matrix.png"
)

CONFUSION_MATRIX_NORMALIZED = (
    PLOTS_DIR / "confusion_matrix_normalized.png"
)

TRAINING_LOG = (
    LOG_DIR / "training.log"
)


# ============================================================================
# Model
# ============================================================================

MODEL_NAME = "tf_efficientnetv2_b0"

MODEL_DISPLAY_NAME = "EfficientNetV2-B0"

PRETRAINED = True

NUM_CLASSES = 10


# ============================================================================
# Class definitions
# ============================================================================

CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

CLASS_TO_INDEX = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}

INDEX_TO_CLASS = {
    index: class_name
    for class_name, index in CLASS_TO_INDEX.items()
}


# ============================================================================
# Dataset split
# ============================================================================

TRAIN_SPLIT = "train"

VAL_SPLIT = "val"

TEST_SPLIT = "test"

PROJECT_SPLITS = (
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
)

TRAIN_RATIO = 0.70

VAL_RATIO = 0.15

TEST_RATIO = 0.15

RANDOM_SEED = 42


# ============================================================================
# Image preprocessing
# ============================================================================

IMAGE_SIZE = 224

RESIZE_SIZE = 256

IMAGE_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGE_STD = (
    0.229,
    0.224,
    0.225,
)


# ============================================================================
# Training
# ============================================================================

BATCH_SIZE = 32

MAX_EPOCHS = 150

EARLY_STOPPING_ENABLED = True

EARLY_STOPPING_PATIENCE = 10

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

# Research terminology:
# For EXP-001, lambda is represented by AdamW weight decay.
REGULARIZATION_LAMBDA = WEIGHT_DECAY

OPTIMIZER_NAME = "AdamW"


# ============================================================================
# Learning-rate scheduler
# ============================================================================

SCHEDULER_NAME = "ReduceLROnPlateau"

SCHEDULER_MODE = "max"

SCHEDULER_FACTOR = 0.5

SCHEDULER_PATIENCE = 3

SCHEDULER_MIN_LR = 0.0


# ============================================================================
# Loss
# ============================================================================

LOSS_NAME = "WeightedCrossEntropy"

CLASS_WEIGHT_METHOD = "sqrt_inverse_frequency"


# ============================================================================
# Model selection
# ============================================================================

MODEL_SELECTION_METRIC = "val_macro_f1"

MODEL_SELECTION_MODE = "max"


# ============================================================================
# Evaluation metrics
# ============================================================================

PRIMARY_EVALUATION_METRICS = [
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
    "cohen_kappa",
]

CLASS_WISE_METRICS = [
    "class_accuracy",
    "precision",
    "recall",
    "f1",
    "support",
]


# ============================================================================
# DataLoader
# ============================================================================

NUM_WORKERS = 0

PIN_MEMORY = True


# ============================================================================
# Runtime
# ============================================================================

DEVICE_PREFERENCE = "auto"


# ============================================================================
# Utility
# ============================================================================

def ensure_experiment_directories() -> None:
    """
    Create all directories required for an experiment.
    """

    directories = [
        EXPERIMENT_ROOT,
        CHECKPOINT_DIR,
        MODEL_DIR,
        HISTORY_DIR,
        METRICS_DIR,
        PLOTS_DIR,
        LOG_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )