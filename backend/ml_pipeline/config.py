"""
PhytoSense AI
Central Project Configuration.

This module contains project-wide configuration shared by
the dataset, training, evaluation, and XAI pipelines.

Machine-specific paths are loaded from the local .env file.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

ML_PIPELINE_ROOT = Path(__file__).resolve().parent

load_dotenv(
    ML_PIPELINE_ROOT / ".env"
)


# ============================================================
# PROJECT ROOTS
# ============================================================

plantvillage_root = os.getenv(
    "PLANTVILLAGE_ROOT"
)

if not plantvillage_root:
    raise EnvironmentError(
        "PLANTVILLAGE_ROOT is not configured. "
        "Create backend/ml_pipeline/.env and set "
        "PLANTVILLAGE_ROOT to the local PlantVillage "
        "dataset path."
    )


PLANTVILLAGE_ROOT = Path(
    plantvillage_root
).expanduser()


COLOR_ROOT = (
    PLANTVILLAGE_ROOT
    / "raw"
    / "color"
)


MANIFEST_ROOT = (
    ML_PIPELINE_ROOT
    / "manifests"
)


TOMATO_MANIFEST = (
    MANIFEST_ROOT
    / "plantvillage_tomato_split.csv"
)


# ============================================================
# DATASET
# ============================================================

NUM_CLASSES = 10

TRAIN_RATIO = 0.70

VAL_RATIO = 0.15

TEST_RATIO = 0.15

RANDOM_SEED = 42


# ============================================================
# TOMATO CLASSES
# ============================================================

TOMATO_CLASSES = [
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
    for index, class_name
    in enumerate(TOMATO_CLASSES)
}


# ============================================================
# VALIDATION
# ============================================================

if not abs(
    TRAIN_RATIO
    + VAL_RATIO
    + TEST_RATIO
    - 1.0
) < 1e-9:

    raise ValueError(
        "Train/validation/test ratios "
        "must sum to 1.0."
    )


if len(TOMATO_CLASSES) != NUM_CLASSES:

    raise ValueError(
        "NUM_CLASSES does not match "
        "TOMATO_CLASSES."
    )


if not PLANTVILLAGE_ROOT.exists():

    raise FileNotFoundError(
        "PlantVillage dataset directory "
        f"does not exist:\n"
        f"{PLANTVILLAGE_ROOT}"
    )


if not COLOR_ROOT.exists():

    raise FileNotFoundError(
        "PlantVillage color directory "
        f"does not exist:\n"
        f"{COLOR_ROOT}"
    )