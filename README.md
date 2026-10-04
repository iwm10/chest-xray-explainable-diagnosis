# Explainable Multi-Label Chest X-Ray Diagnosis

Multi-label classification of 14 thoracic pathologies from chest radiographs (NIH ChestX-ray14), with Grad-CAM explanations evaluated against radiologist-drawn bounding boxes.

Samsung Innovation Campus AI Capstone Project, Team Core (6 members).

> Research prototype. Not intended, validated, or suitable for clinical use.

**Final model:** ConvNeXt-Tiny 320×320. **Test macro AUROC 0.8158**, evaluated on a patient-level split with zero patient leakage.

---

## My contribution: Data Engineer

**My notebook:** [01_data_preparation on Kaggle](https://www.kaggle.com/code/iwmm10/01-data-preparation)

I owned the data foundation the whole project was trained and evaluated on: dataset construction, the split design, and the integrity of the evaluation protocol.

- **Rarity-aware subset construction.** I built a 31,077-image working subset from the full NIH ChestX-ray14 dataset, constructed so that rare pathologies keep enough examples to be learnable and measurable across all 14 classes.
- **Patient-level stratified splitting with zero leakage.** I split 11,907 patients into train/val/test so that no patient appears in more than one split. Image-level splitting on ChestX-ray14 leaks patients across splits and inflates AUROC.
- **Isolated localization splits.** I separated every bounding-box-annotated patient into two dedicated splits, `loc_tune` and `loc_report`. This lets the Grad-CAM heatmap threshold be selected on `loc_tune` and frozen before evaluation on `loc_report`, so the reported localization numbers are never tuned on the data they are reported on.
- **Preprocessing.** I wrote `notebooks/01_data_preparation.ipynb`, which covers extraction, label parsing, subset construction, splitting, and split export.
- **Evaluation-protocol review.** I identified that the original evaluation plan specified IoU for Grad-CAM localization, while IoBB is the standard metric for this benchmark (Wang et al., ChestX-ray8). The final evaluation uses IoBB.
- **Code QA.** I caught a silent class-drop bug in the Grad-CAM code before final evaluation.
- **Infrastructure.** I migrated the team workflow from Colab and Google Drive to Kaggle with private datasets.

The model-development notebooks, backend, frontend, and other team-level artifacts in this repository are shared Team Core work, kept here to show the project end-to-end.

---

## Problem

Chest X-ray classifiers return probabilities, but probabilities alone do not show which image regions drove a prediction, and they are only trustworthy if the evaluation data is clean. This project pairs multi-label classification with Grad-CAM explainability, and evaluates both on splits built to prevent patient leakage and threshold overfitting.

---

## Data

The working subset is NIH ChestX-ray14: **31,077 X-rays from 11,907 patients**. 25,895 images form the classification subset, and the rest form the dedicated localization splits. All splits are defined at the patient level.

| Split        | Images | Patients | Purpose                                           |
| ------------ | ------ | -------- | ------------------------------------------------- |
| `train`      | 18,013 | 7,930    | Model training                                    |
| `val`        | 3,978  | 1,699    | Hyperparameters and per-class decision thresholds |
| `test`       | 3,904  | 1,700    | Final classification evaluation                   |
| `loc_tune`   | 2,606  | 289      | Grad-CAM threshold selection                      |
| `loc_report` | 2,576  | 289      | Final localization evaluation                     |

The images are not redistributed here. ChestX-ray14 is available from the NIH.

The model has 14 outputs: Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema, Emphysema, Fibrosis, Effusion, Pneumonia, Pleural_Thickening, Cardiomegaly, Nodule, Mass, and Hernia. `No Finding` is not a 15th output. It is derived when no class crosses its threshold.

---

## Model (team)

| Model             | Input   | Test Macro AUROC |
| ----------------- | ------- | ---------------- |
| ResNet50          | 224×224 | ~0.783           |
| DenseNet-121      | 224×224 | ~0.796           |
| **ConvNeXt-Tiny** | 320×320 | **0.8158**       |

The final ConvNeXt-Tiny model:

|                             |                               |
| --------------------------- | ----------------------------- |
| Loss                        | `BCEWithLogitsLoss`           |
| Optimizer                   | AdamW                         |
| Decision thresholds         | Tuned per class on validation |
| Macro F1 @ 0.50             | 0.3034                        |
| Macro F1 @ tuned thresholds | **0.4236**                    |
| Macro ECE                   | **0.0258**                    |

The checkpoint is not committed because of its file size.

---

## Explainability (team)

Grad-CAM heatmaps were thresholded at T = 0.90, a value selected on `loc_tune` only and frozen before evaluation on `loc_report`. Localization is scored with IoBB (intersection area divided by predicted box area) on the 8 pathologies with bounding-box annotations.

| Metric      | Value |
| ----------- | ----- |
| Mean IoBB   | 0.417 |
| IoBB ≥ 0.50 | 39.4% |

Results range from Cardiomegaly (mean IoBB 0.886) down to Nodule (0.065). Full per-class results, the Grad-CAM implementation, and example heatmaps are in Layan Alazwari's XAI repository:
→ https://github.com/l136758/explainable-multilabel-chest-xray

---

## Serving (team)

The project includes a FastAPI backend (`backend/`) and an HTML/CSS/JavaScript frontend (`frontend/`) that serve predictions for all 14 outputs, together with Grad-CAM heatmaps.

```bash
pip install -r requirements.txt
# place the checkpoint at backend/model/convnext_tiny_320_best.pth
cd backend && uvicorn main:app --reload
# in a second terminal:
cd frontend && python -m http.server 5500
```

Then open `http://127.0.0.1:5500/`.

---

## Repository

```
notebooks/
  01_data_preparation.ipynb             Subset construction + patient-level splits   [my work]
  02_densenet_baseline.ipynb            DenseNet-121 baseline                         [team]
  03_evaluation.ipynb                   DenseNet evaluation                           [team]
  04_resnet50_baseline.ipynb            ResNet50 baseline                             [team]
  05_resnet_evaluation.ipynb            ResNet50 evaluation                           [team]
  convnext-tiny-320-training.ipynb      Final ConvNeXt-Tiny model                     [team]
  convnext-gradcam-localization.ipynb   Grad-CAM + IoBB evaluation                    [team]
backend/                                FastAPI backend                               [team]
frontend/                               HTML/CSS/JavaScript frontend                  [team]
results/                                Classification and localization outputs       [team]
docs/                                   Project documentation                         [team]
demo_images/                            Sample X-rays for the demo                    [team]
requirements.txt
```

---

## Team Core

| Team member          | Primary contribution                                                                   |
| -------------------- | -------------------------------------------------------------------------------------- |
| Rabeh Almutairi      | Team leadership, final ConvNeXt-Tiny model development                                 |
| **Mohammed Almalki** | **Data engineering, subset construction, patient-level and localization split design** |
| Abdullah Alsalhi     | ResNet50 baseline, GitHub workflow/setup                                               |
| Layan Alazwari       | Grad-CAM integration and IoBB localization evaluation                                  |
| Deema Omar Alquwaei  | FastAPI backend                                                                        |
| Layan Allhidean      | HTML/CSS/JavaScript frontend and API integration                                       |

Original team repository: https://github.com/ABDULLHALSALHI/team-core-chest-xray

---

## Limitations

- This is a research and educational prototype, not a clinically validated diagnostic system.
- ChestX-ray14 labels were NLP-mined from radiology reports and are noisy, which caps achievable accuracy.
- Quantitative localization covers only the 8 pathologies with bounding-box annotations.
- Grad-CAM shows which regions influenced a prediction. It does not confirm where the disease is.
- No external-dataset validation (e.g. CheXpert, MIMIC-CXR).labels were NLP-mined from radiology reports and are noisy, which caps achievable accuracy.
- Quantitative localization covers only the 8 pathologies with bounding-box annotations.
- Grad-CAM shows which regions influenced a prediction. It does not confirm where the disease is.
- No external-dataset validation (e.g. CheXpert, MIMIC-CXR).
