# PathQC-Triage

The project now includes a reusable implementation in
[`pathqc_pipeline.py`](./pathqc_pipeline.py). The notebook remains an
exploration/demo surface; new experiments should import the module so that
training and inference are reproducible and can be tested independently.

> A notebook-based proof of concept for patch-level histopathology quality
> classification and normal-tissue anomaly scoring.

## What this project actually does

PathQC-Triage is an experimental two-stage computer-vision pipeline for
colorectal histopathology images. It works with **PathMNIST**, a dataset of
small labelled image patches derived from the NCT-CRC-HE-100K benchmark.

It does **not** currently read native whole-slide-image files, perform clinical
diagnosis, or replace a pathologist. The “slide” examples in the notebook are
synthetic mosaics assembled from PathMNIST patches to demonstrate how a
whole-slide workflow could be extended in the future.

The notebook contains two separate experiments:

1. **Patch quality/tissue classification:** a supervised MobileNetV3-Small
   classifier predicts one of nine PathMNIST classes for each 64 × 64 patch.
2. **Normal-tissue anomaly scoring:** a SimCLR-style ResNet-18 encoder learns
   representations from normal-colon patches only. PCA and an Isolation Forest
   then assign an anomaly score to new patches.

The result is a research demonstrator that produces model metrics, diagnostic
plots, and spatial heatmaps. It is not a validated clinical QC system.

## Pipeline at a glance

```text
PathMNIST patches
        |
        +--> Stage 1: supervised MobileNetV3-Small
        |       |
        |       +--> predicted class for every patch
        |       +--> confusion matrix and class metrics
        |
        +--> Stage 2: normal patches only
                |
                +--> SimCLR-style ResNet-18 representation learning
                +--> PCA feature reduction
                +--> Isolation Forest anomaly score
                +--> ROC/PR curves and anomaly heatmap
```

## Stage 1: patch classification and QC demonstration

The first model is trained on a balanced subset of the PathMNIST training
split. It predicts these nine classes:

| ID | Short name | Meaning |
|---:|---|---|
| 0 | ADI | Adipose tissue |
| 1 | BACK | Background |
| 2 | DEB | Debris |
| 3 | LYM | Lymphocytes |
| 4 | MUC | Mucus |
| 5 | MUS | Smooth muscle |
| 6 | NORM | Normal colon mucosa |
| 7 | STR | Cancer-associated stroma |
| 8 | TUM | Colorectal adenocarcinoma epithelium |

The classifier is used to label each tile in a synthetic 512 × 512 image
region. The notebook counts BACK and DEB predictions and converts their
fraction into a simple demonstration verdict:

- **Below 10%:** `SLIDE APPROVED`
- **10% to 30%:** `SLIDE: REVIEW RECOMMENDED`
- **Above 30%:** `SLIDE REJECTED`

These thresholds are project-defined heuristics. They are not clinical
acceptance criteria, and the synthetic mosaic is not a real WSI.

The notebook reports overall accuracy, balanced accuracy, macro precision,
macro recall, macro F1, a normalized confusion matrix, training curves, and
per-class F1/recall plots. Because the model predicts all nine classes, this
stage is broader than a binary artifact detector: it classifies tissue types
as well as background and debris.

## Stage 2: self-supervised anomaly scoring

The second experiment uses only training patches labelled `NORM` during
representation learning. Their labels are discarded by the SimCLR dataset
wrapper, so the encoder is not trained with tumor labels.

For each patch:

1. Two augmented views are created.
2. A ResNet-18 encoder produces a 512-dimensional feature vector.
3. An MLP projection head is trained with NT-Xent contrastive loss.
4. The projection head is discarded for anomaly detection.
5. PCA reduces the normal feature vectors to 50 dimensions.
6. An Isolation Forest is fitted to the reduced normal feature distribution.
7. The negative Isolation Forest score is reported as the anomaly score.

For evaluation only, the notebook compares scores for held-out `NORM` and
`TUM` test patches. ROC-AUC, average precision, score distributions, and a
precision-recall plot are generated. Tumor labels are therefore **not used
to fit the anomaly detector**, but they are used to calculate the evaluation
metrics. This is a one-class/anomaly-detection experiment, not proof of
unsupervised clinical tumor detection.

## Outputs

Running the notebook can create:

- `best_artifact_model.pth` — best Stage 1 model checkpoint.
- `simclr_encoder.pth` — Stage 2 encoder weights.
- `anomaly_detector.pkl` — fitted PCA and Isolation Forest objects.
- `stage1_confusion_matrix.png` — normalized nine-class confusion matrix.
- `stage1_training_curves.png` — loss and macro-F1 by epoch.
- `stage1_class_metrics.png` — per-class F1 and recall.
- `dataset_distribution_and_samples.png` — class counts and example patches.
- `slide_artifact_heatmap.png` — synthetic slide-region QC overlay.
- `roc_curve.png` — normal-versus-tumor anomaly ROC curve.
- `score_distribution.png` — anomaly-score histograms.
- `anomaly_precision_recall_boxplot.png` — PR curve and score spread.
- `anomaly_heatmap.png` — anomaly scores over a synthetic patch grid.
- `test_set_showcase.png` — random held-out test patches with true labels,
  predictions, confidence, and sample accuracy.

## Resume-ready project highlights

- Built a two-stage computational pathology pipeline using MobileNetV3-Small
  for nine-class tissue/artifact classification and SimCLR with Isolation
  Forest for normal-tissue anomaly scoring.
- Improved Stage 1 training with partial backbone fine-tuning, balanced
  sampling, AdamW, mixed precision, label smoothing, early stopping, and
  macro-F1 checkpointing; achieved **0.7607 validation macro-F1** on the
  balanced PathMNIST evaluation subset.
- Implemented batched tile inference and confidence-aware spatial triage that
  converts BACK/DEB probabilities into artifact heatmaps and slide review or
  rejection recommendations.

## Showing the project in a presentation

After running the notebook from top to bottom, use the final
**Showcase: random held-out test patches** cell. It samples unseen images from
the PathMNIST test split and displays:

- the original histopathology patch;
- the ground-truth class;
- the Stage 1 predicted class;
- the model confidence; and
- whether the prediction is correct.

Green titles indicate correct predictions and red titles indicate incorrect
predictions. The cell also reports the accuracy of the displayed random sample
and saves the grid as `test_set_showcase.png`. Use this figure to demonstrate
that the project is making patch-level predictions on held-out data, rather
than only showing training curves or synthetic examples.

## Running the notebook

Open [`PathQC_Triage.ipynb`](PathQC_Triage.ipynb) in Jupyter, VS Code, or
Google Colab and run the cells in order. The first code cell installs the
Python packages into the currently selected environment using `%pip`; it does
not create a virtual environment.

For a local installation, install the dependencies into an environment you
already manage:

```bash
pip install -r requirements.txt
```

The notebook's first cell still supports hosted notebook environments, while
[`requirements.txt`](./requirements.txt) is the reproducible local install
for the reusable pipeline.

A GPU is recommended for training, but the notebook can fall back to CPU.
The notebook uses `num_workers=0` for the main data loaders to be more
portable across Windows, local Jupyter, and hosted notebook runtimes.

## Improvements in the reusable pipeline

- **Partial fine-tuning:** the final MobileNetV3 feature blocks adapt to
  histology instead of training only a randomly initialized classifier head.
- **Better model selection:** checkpoints use macro-F1, which is safer than
  overall accuracy for imbalanced tissue classes, and include model,
  optimizer, epoch, and configuration state.
- **More efficient training:** AdamW parameter groups, mixed precision on CUDA,
  pinned/non-blocking transfers, early stopping, and a plateau scheduler reduce
  wasted compute.
- **Faster and more stable detection:** tile inference is batched and supports
  overlap. Triage uses the summed BACK/DEB probability rather than a brittle
  argmax count, reports uncertainty, and escalates low-confidence regions for
  review.
- **Reproducible experiments:** `QCConfig`, `seed_everything`, and persisted
  JSON metadata make runs comparable.

Example:

```python
from pathlib import Path
import torch
from pathqc_pipeline import QCConfig, build_classifier, make_transforms

config = QCConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_classifier(unfreeze_last_blocks=config.unfreeze_last_blocks)
train_transform, validation_transform = make_transforms(config.image_size)
```

The module does not claim clinical validity. Thresholds still need to be
calibrated on representative, slide-level validation data.

## Limitations and next steps

- PathMNIST patches are not native WSIs; WSI reading, tissue detection, and
  multiresolution tiling still need to be implemented.
- The QC heatmap is a patch-classification visualization, not a validated
  artifact segmentation model.
- The synthetic slide contains pasted patches and blank background, so its
  usability verdict should not be interpreted as real-slide performance.
- The anomaly detector is trained on one tissue class and evaluated on one
  tumour class. Broader external validation is required.
- The reported metrics depend on the selected subset sizes, random seed,
  runtime, pretrained weights, and number of training epochs. Results should
  be regenerated rather than copied as fixed claims.
- No clinical deployment, diagnostic decision, or patient-level conclusion
  should be made from this notebook.

Useful extensions would include real WSI ingestion, tissue masking, overlap
aware tiling, slide-level aggregation, calibration, external validation,
artifact-specific labels, and review by a domain expert.

## Technology

- **Deep learning:** PyTorch and Torchvision
- **Dataset:** MedMNIST PathMNIST
- **Representation learning:** SimCLR-style augmentations and NT-Xent loss
- **Anomaly detection:** scikit-learn PCA and Isolation Forest
- **Visualization:** Matplotlib, Seaborn, OpenCV, and PIL
