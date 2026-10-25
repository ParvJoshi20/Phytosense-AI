"""
PhytoSense AI - ML Pipeline Configuration

Central configuration for dataset preparation, preprocessing,
training, and experiment management.
"""

from pathlib import Path
import json
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
# Experiment selection
# ============================================================================

EXPERIMENTS_ROOT = BASE_DIR / "backend" / "ml_pipeline" / "experiments"
EXPERIMENT_ID = os.getenv("PHYTOSENSE_EXPERIMENT_ID", "EXP-001").strip().upper()
if not EXPERIMENT_ID:
    raise ValueError("PHYTOSENSE_EXPERIMENT_ID cannot be empty.")

EXPERIMENT_SPEC_PATH = EXPERIMENTS_ROOT / EXPERIMENT_ID / "config.json"
_EXPERIMENT_SPEC = {}
if EXPERIMENT_ID != "EXP-001":
    if not EXPERIMENT_SPEC_PATH.exists():
        raise FileNotFoundError(
            f"Experiment config not found for {EXPERIMENT_ID}: {EXPERIMENT_SPEC_PATH}"
        )
    with EXPERIMENT_SPEC_PATH.open("r", encoding="utf-8") as _spec_file:
        _EXPERIMENT_SPEC = json.load(_spec_file)

EXPERIMENT_NAME = _EXPERIMENT_SPEC.get("experiment", {}).get(
    "name", "EfficientNetV2-B0 Baseline"
)

_MODEL_CONFIG = _EXPERIMENT_SPEC.get("model", {})
if EXPERIMENT_ID == "EXP-001":
    # EXP-001 is the frozen baseline; never redirect its output directory to a
    # different architecture via an environment override.
    MODEL_NAME = "tf_efficientnetv2_b0"
else:
    MODEL_NAME = os.getenv(
        "PHYTOSENSE_MODEL_NAME", _MODEL_CONFIG.get("name", "tf_efficientnetv2_b0")
    ).strip()
_MODEL_DISPLAY_NAMES = {
    "tf_efficientnetv2_b0": "EfficientNetV2-B0",
    "resnet50": "ResNet50",
    "mobilenetv3_large_100": "MobileNetV3-Large",
    "densenet121": "DenseNet121",
    "convnext_tiny": "ConvNeXt-Tiny",
}
if MODEL_NAME not in _MODEL_DISPLAY_NAMES:
    raise ValueError(
        f"Unsupported PHYTOSENSE_MODEL_NAME '{MODEL_NAME}'. "
        f"Choose one of: {', '.join(_MODEL_DISPLAY_NAMES)}"
    )
MODEL_DISPLAY_NAME = _MODEL_DISPLAY_NAMES[MODEL_NAME]
PRETRAINED = bool(_MODEL_CONFIG.get("pretrained", True))

# EXP-001 keeps its historical directory layout. EXP-002 stores each model in
# its own subdirectory so sequential runs cannot overwrite another model's
# checkpoint, metrics, plots, or history.
EXPERIMENT_ROOT = EXPERIMENTS_ROOT / EXPERIMENT_ID
if EXPERIMENT_ID != "EXP-001":
    EXPERIMENT_ROOT = EXPERIMENT_ROOT / MODEL_NAME

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

# MODEL_NAME, MODEL_DISPLAY_NAME, and PRETRAINED are resolved above from the
# selected experiment and optional PHYTOSENSE_MODEL_NAME override.
NUM_CLASSES = int(_MODEL_CONFIG.get("num_classes", 10))


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
# Experiment-specific settings loaded from config.json (EXP-002 onward)
# ============================================================================

if _EXPERIMENT_SPEC:
    _dataset_cfg = _EXPERIMENT_SPEC.get("dataset", {})
    _split_cfg = _dataset_cfg.get("split", {})
    _preprocess_cfg = _EXPERIMENT_SPEC.get("preprocessing", {})
    _training_cfg = _EXPERIMENT_SPEC.get("training", {})
    _scheduler_cfg = _EXPERIMENT_SPEC.get("scheduler", {})

    _manifest_rel = _dataset_cfg.get("manifest_path")
    if _manifest_rel:
        MANIFEST_PATH = (EXPERIMENTS_ROOT / EXPERIMENT_ID / _manifest_rel).resolve()
    _image_root_env = _dataset_cfg.get("image_root_env", "DATASETS_ROOT")
    _image_subpath = _dataset_cfg.get(
        "image_root_subpath", "PlantVillage-Dataset/raw/color"
    )
    COLOR_ROOT = Path(os.getenv(_image_root_env, str(DATASETS_ROOT))) / _image_subpath

    CLASS_NAMES = list(_dataset_cfg.get("class_names", CLASS_NAMES))
    CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
    INDEX_TO_CLASS = {index: name for name, index in CLASS_TO_INDEX.items()}
    NUM_CLASSES = int(_dataset_cfg.get("num_classes", NUM_CLASSES))

    IMAGE_SIZE = int(_preprocess_cfg.get("image_size", IMAGE_SIZE))
    RESIZE_SIZE = int(_preprocess_cfg.get("resize_size", RESIZE_SIZE))
    IMAGE_MEAN = tuple(_preprocess_cfg.get("mean", IMAGE_MEAN))
    IMAGE_STD = tuple(_preprocess_cfg.get("std", IMAGE_STD))

    BATCH_SIZE = int(_training_cfg.get("batch_size", BATCH_SIZE))
    MAX_EPOCHS = int(_training_cfg.get("max_epochs", MAX_EPOCHS))
    EARLY_STOPPING_ENABLED = bool(
        _training_cfg.get("early_stopping_enabled", EARLY_STOPPING_ENABLED)
    )
    EARLY_STOPPING_PATIENCE = int(
        _training_cfg.get("early_stopping_patience", EARLY_STOPPING_PATIENCE)
    )
    LEARNING_RATE = float(_training_cfg.get("learning_rate", LEARNING_RATE))
    WEIGHT_DECAY = float(_training_cfg.get("weight_decay", WEIGHT_DECAY))
    REGULARIZATION_LAMBDA = WEIGHT_DECAY
    RANDOM_SEED = int(_training_cfg.get("random_seed", RANDOM_SEED))
    CLASS_WEIGHT_METHOD = _training_cfg.get("class_weight_method", CLASS_WEIGHT_METHOD)
    MODEL_SELECTION_METRIC = _training_cfg.get(
        "model_selection_metric", MODEL_SELECTION_METRIC
    )
    SCHEDULER_NAME = _scheduler_cfg.get("name", SCHEDULER_NAME)
    SCHEDULER_MODE = _scheduler_cfg.get("mode", SCHEDULER_MODE)
    SCHEDULER_FACTOR = float(_scheduler_cfg.get("factor", SCHEDULER_FACTOR))
    SCHEDULER_PATIENCE = int(_scheduler_cfg.get("patience", SCHEDULER_PATIENCE))
    SCHEDULER_MIN_LR = float(_scheduler_cfg.get("min_lr", SCHEDULER_MIN_LR))

    # Recompute artifact paths after the experiment/model-specific root is set.
    CONFIG_JSON = EXPERIMENT_ROOT / "config.json"
    BEST_CHECKPOINT = CHECKPOINT_DIR / "checkpoint_best.pt"
    LAST_CHECKPOINT = CHECKPOINT_DIR / "checkpoint_last.pt"
    TRAINING_HISTORY_JSON = HISTORY_DIR / "training_history.json"
    METRICS_JSON = METRICS_DIR / "metrics.json"
    PREDICTIONS_CSV = METRICS_DIR / "predictions.csv"
    CLASSIFICATION_REPORT_CSV = METRICS_DIR / "classification_report.csv"
    ACCURACY_CURVE = PLOTS_DIR / "accuracy_curve.png"
    LOSS_CURVE = PLOTS_DIR / "loss_curve.png"
    CONFUSION_MATRIX = PLOTS_DIR / "confusion_matrix.png"
    CONFUSION_MATRIX_NORMALIZED = PLOTS_DIR / "confusion_matrix_normalized.png"
    TRAINING_LOG = LOG_DIR / "training.log"

# Optional runtime overrides are useful on GPUs with limited VRAM. They do not
# alter the frozen dataset manifest or class mapping.
BATCH_SIZE = int(os.getenv("PHYTOSENSE_BATCH_SIZE", BATCH_SIZE))
MAX_EPOCHS = int(os.getenv("PHYTOSENSE_MAX_EPOCHS", MAX_EPOCHS))


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