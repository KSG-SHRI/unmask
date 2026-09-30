# Unmask: leakage-aware deepfake face-crop benchmark

This repository tests whether simple visual cues distinguish the included real and manipulated **face crops**. It is a research baseline, not a production detector or a claim of generalization to unseen generators, videos, or platforms.

## Why the evaluation changed

The original script concatenated three unrelated images into one forward pass, used a label from only the third image, had no held-out test set, and required CUDA. Its loss output was not a defensible accuracy metric. The current experiment keeps images from the same filename source together across train, validation, and test. It runs on CPU and reports the confusion matrix, fake precision/recall/F1, balanced accuracy, and ROC AUC.

## Data and limitations

The repository currently contains 8,005 `Real` and 8,428 `Fake` 128×128 PNG face crops. Filename patterns indicate related frames and manipulated source pairs. We group by the first filename token, which avoids adjacent frames from that source crossing the split. A source can still appear as the *target* identity of a fake image in another group, so this is **not** a fully identity-disjoint benchmark. The original dataset provenance, consent, and license are not documented; do not redistribute or deploy this dataset without resolving them. Results only describe the supplied images.

## Reproduce

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python model.py --data dataset --output results/baseline.json
```

Two logistic-regression baselines are compared: 16×16 RGB pixels and hand-crafted color/edge texture statistics. Hyperparameter `C` is selected on a grouped validation set. The grouped test set is evaluated once per baseline. The random seed and exact split sizes are saved in `results/baseline.json`.

Running the training command also saves a texture baseline to `results/texture_model.joblib`; the binary model is not included in the repository. To score one face crop after training:

```bash
.venv/bin/python predict.py path/to/face.png
```

Only load a model file you trust; joblib uses Python pickle. The score is not calibrated for use in moderation, journalism, or identity decisions.

## Measured baseline (seed 42)

The source-grouped training, validation, and test sets contain 8,898, 3,705, and 3,830 face crops respectively. Class counts shift substantially across groups, so balanced accuracy is more informative than raw accuracy.

| Features | Test balanced accuracy | Test ROC AUC | Fake recall | Fake F1 |
| --- | ---: | ---: | ---: | ---: |
| 16×16 RGB pixels | 0.506 | 0.512 | 0.257 | 0.369 |
| Color and edge texture | 0.549 | 0.593 | 0.296 | 0.422 |

The texture baseline is better on this held-out group split, but far from a reliable detector. Its 1,730 false negatives on the test set are a strong reason to avoid deployment. The validation AUC was lower still (0.443), indicating unstable generalization across source groups. These are useful negative results, not evidence of robust deepfake detection.

## Next research steps

1. Identify and cite the original dataset and license.
2. Add a truly identity-disjoint external test set from a different manipulation method and compression level.
3. Evaluate at the video level and report uncertainty and confidence intervals.
4. Only then compare a pretrained CNN with the simple baselines on identical splits.
