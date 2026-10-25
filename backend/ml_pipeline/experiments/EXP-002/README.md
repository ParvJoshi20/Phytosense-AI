# EXP-002 — Architecture comparison

EXP-002 compares ResNet50, MobileNetV3-Large, DenseNet121, and ConvNeXt-Tiny against the already-recorded EXP-001 EfficientNetV2-B0 baseline. EXP-001 is a reference point; its checkpoint and metrics are not overwritten by this experiment.

The frozen manifest is reused. Training selects checkpoints using validation Macro F1; the test split is reserved for final evaluation.

## PowerShell commands

Run these from the repository root. Each candidate writes to its own directory under `backend/ml_pipeline/experiments/EXP-002/`.

```powershell
$env:PHYTOSENSE_EXPERIMENT_ID = "EXP-002"
$env:PHYTOSENSE_MODEL_NAME = "resnet50"
python backend/ml_pipeline/train.py
python backend/ml_pipeline/evaluate.py
```

Repeat the two Python commands with each model identifier:

- `resnet50`
- `mobilenetv3_large_100`
- `densenet121`
- `convnext_tiny`

EfficientNetV2-B0 is already available from EXP-001 and should be reused as the baseline reference. To reduce GPU memory use, optionally set `$env:PHYTOSENSE_BATCH_SIZE = "16"` before training. An optional `$env:PHYTOSENSE_MAX_EPOCHS` override is also supported for a pilot run; record any such deviation when reporting results.

Clear the environment variables when returning to the EXP-001 default:

```powershell
Remove-Item Env:PHYTOSENSE_EXPERIMENT_ID -ErrorAction SilentlyContinue
Remove-Item Env:PHYTOSENSE_MODEL_NAME -ErrorAction SilentlyContinue
Remove-Item Env:PHYTOSENSE_BATCH_SIZE -ErrorAction SilentlyContinue
Remove-Item Env:PHYTOSENSE_MAX_EPOCHS -ErrorAction SilentlyContinue
```

## Build the comparison table

After evaluating the candidates, run this from the repository root:

```powershell
python backend/ml_pipeline/compare_models.py
```

It collects existing test metrics only; missing model results are reported and skipped. It does not retrain models or run evaluation. The existing output location spelling is retained for compatibility: `experiments/EXP-002/comparision/model_comparision.csv`.
