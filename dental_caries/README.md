# Dental Caries Classification (DENTEX-ready)

This module is a clean, reproducible training pipeline for your paper:

- Baselines: `resnet50`, `efficientnet_b0`
- Proposed: `efficientnet_b0_cbam`
- Metrics: AUC, F1, sensitivity, specificity, precision, recall, PR-AUC
- Explainability: Grad-CAM (and SHAP-ready environment)

## 1) Setup

```bash
pip install -r dental_caries/requirements.txt
```

## 2) Prepare `manifest.csv`

Create a CSV with at least these columns:

- `image_path`: absolute path or path relative to workspace
- `label`: integer class id (`0` = non-caries, `1` = caries/deep-caries)
- `split`: one of `train`, `val`, `test`

Optional:

- `patient_id`: useful for leakage-safe grouping before split

Example:

```csv
image_path,label,split,patient_id
data/dentex/tooth_rois/img_0001.png,0,train,p001
data/dentex/tooth_rois/img_0002.png,1,train,p001
data/dentex/tooth_rois/img_0003.png,1,val,p010
```

## 3) Configure

Edit `dental_caries/config.yaml` paths and hyperparameters.

## 4) Train

```bash
python dental_caries/src/train.py --config dental_caries/config.yaml
```

## 5) Evaluate

```bash
python dental_caries/src/evaluate.py --config dental_caries/config.yaml
```

## 6) Grad-CAM

```bash
python dental_caries/src/gradcam.py --config dental_caries/config.yaml --image "ABS_OR_REL_IMAGE_PATH"
```

## Notes

- Keep split creation patient-safe (no same patient in train and test).
- Start with binary labels, then extend to 3-class once baseline is stable.
