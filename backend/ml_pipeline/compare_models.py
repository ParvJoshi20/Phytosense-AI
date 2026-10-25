"""Aggregate completed EXP-001/EXP-002 test metrics into a comparison CSV.

Run from the repository root:
    python backend/ml_pipeline/compare_models.py

Only metrics files that already exist are included. This script does not train
models, load checkpoints, or evaluate the test set.
"""

import csv
import json
from pathlib import Path

ML_PIPELINE_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = ML_PIPELINE_ROOT / "experiments"

CANDIDATES = [
    ("EXP-001", "tf_efficientnetv2_b0"),
    ("EXP-002", "resnet50"),
    ("EXP-002", "mobilenetv3_large_100"),
    ("EXP-002", "densenet121"),
    ("EXP-002", "convnext_tiny"),
]

FIELDS = [
    "experiment_id",
    "model_name",
    "model_display_name",
    "split_evaluated",
    "test_samples",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "cohen_kappa",
    "test_loss",
    "best_validation_macro_f1",
    "checkpoint_epoch",
    "metrics_file",
]


def main():
    rows = []
    missing = []

    for experiment_id, model_name in CANDIDATES:
        if experiment_id == "EXP-001":
            metrics_path = (
                EXPERIMENTS_ROOT / experiment_id / "metrics" / "metrics.json"
            )
        else:
            metrics_path = (
                EXPERIMENTS_ROOT / experiment_id / model_name
                / "metrics" / "metrics.json"
            )

        if not metrics_path.exists():
            missing.append(model_name)
            continue

        with metrics_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)

        if metrics.get("split_evaluated") != "test":
            raise ValueError(
                f"Expected final test metrics in {metrics_path}; "
                f"found split_evaluated={metrics.get('split_evaluated')!r}."
            )

        row = {field: metrics.get(field) for field in FIELDS}
        row["metrics_file"] = str(metrics_path.relative_to(ML_PIPELINE_ROOT))
        rows.append(row)

    output_path = (
        EXPERIMENTS_ROOT / "EXP-002" / "comparision" / "model_comparision.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} model result(s) to: {output_path}")
    if missing:
        print("Metrics not available yet for:")
        for model_name in missing:
            print(f"  - {model_name}")
    print("No training or test-set evaluation was performed by this script.")


if __name__ == "__main__":
    main()
