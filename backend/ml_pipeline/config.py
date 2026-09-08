from pathlib import Path


# ============================================================
# PROJECT ROOTS
# ============================================================

ML_PIPELINE_ROOT = Path(__file__).resolve().parent

PLANTVILLAGE_ROOT = Path(
    r"D:\Code\_Archieve\Minor Project\Datasets\PlantVillage-Dataset"
)

COLOR_ROOT = PLANTVILLAGE_ROOT / "raw" / "color"

MANIFEST_ROOT = ML_PIPELINE_ROOT / "manifests"

TOMATO_MANIFEST = (
    MANIFEST_ROOT / "plantvillage_tomato_split.csv"
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
    for index, class_name in enumerate(TOMATO_CLASSES)
}


# ============================================================
# VALIDATION
# ============================================================

if not abs(
    TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0
) < 1e-9:
    raise ValueError(
        "Train/validation/test ratios must sum to 1.0."
    )

if len(TOMATO_CLASSES) != NUM_CLASSES:
    raise ValueError(
        "NUM_CLASSES does not match TOMATO_CLASSES."
    )