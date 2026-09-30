"""Reproducible, source-grouped face-crop deepfake baseline."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


def records(root):
    paths = sorted((root / "Real").glob("*.png")) + sorted((root / "Fake").glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNGs in {root}/Real and {root}/Fake")
    labels = np.array([int(p.parent.name == "Fake") for p in paths], dtype=np.int8)
    groups = np.array([p.stem.split("_")[0] for p in paths])
    return paths, labels, groups


def split_indices(labels, groups):
    all_idx = np.arange(len(labels))
    outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    train_val, test = next(outer.split(all_idx, labels, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.17647, random_state=SEED + 1)
    train_pos, val_pos = next(inner.split(train_val, labels[train_val], groups[train_val]))
    split = {"train": train_val[train_pos], "validation": train_val[val_pos], "test": test}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert not (set(groups[split[a]]) & set(groups[split[b]]))
    for name, ids in split.items():
        if len(np.unique(labels[ids])) != 2:
            raise ValueError(f"{name} contains only one class")
    return split


def features(path, kind):
    with Image.open(path) as image:
        image = image.convert("RGB")
        if kind == "pixels":
            return np.asarray(image.resize((16, 16), Image.Resampling.BILINEAR), dtype=np.float32).ravel() / 255
        pixels = np.asarray(image.resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32) / 255
    gray = pixels.mean(axis=2)
    out = []
    for channel in range(3):
        values = pixels[:, :, channel]
        out.extend((float(values.mean()), float(values.std())))
        out.extend(np.histogram(values, bins=16, range=(0, 1), density=True)[0].tolist())
    dx, dy = np.diff(gray, axis=1), np.diff(gray, axis=0)
    lap = gray[1:-1, :-2] + gray[1:-1, 2:] + gray[:-2, 1:-1] + gray[2:, 1:-1] - 4 * gray[1:-1, 1:-1]
    out.extend((float(np.mean(np.abs(dx))), float(np.std(dx)), float(np.mean(np.abs(dy))),
                float(np.std(dy)), float(np.var(lap))))
    return np.asarray(out, dtype=np.float32)


def score(y, probability):
    prediction = (probability >= 0.5).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {"n": int(len(y)), "real": int((y == 0).sum()), "fake": int((y == 1).sum()),
            "accuracy": float(accuracy_score(y, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            "precision_fake": float(precision_score(y, prediction, zero_division=0)),
            "recall_fake": float(recall_score(y, prediction, zero_division=0)),
            "f1_fake": float(f1_score(y, prediction, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, probability)),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("results/baseline.json"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths, labels, groups = records(args.data)
    split = split_indices(labels, groups)
    result = {"seed": SEED, "unit": "face crop", "group_key": "first filename token",
              "split": {name: {"images": len(ids), "groups": len(set(groups[ids])),
                               "real": int((labels[ids] == 0).sum()), "fake": int((labels[ids] == 1).sum())}
                        for name, ids in split.items()}, "models": {}}
    for kind in ("pixels", "texture"):
        print(f"Extracting {kind} features from {len(paths)} images...", flush=True)
        x = np.stack([features(p, kind) for p in paths])
        candidates = []
        for c in (0.01, 0.1, 1.0):
            model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000, random_state=SEED))
            model.fit(x[split["train"]], labels[split["train"]])
            val_prob = model.predict_proba(x[split["validation"]])[:, 1]
            candidates.append((score(labels[split["validation"]], val_prob)["roc_auc"], c, model))
        best_auc, best_c, best_model = max(candidates, key=lambda item: item[0])
        test_prob = best_model.predict_proba(x[split["test"]])[:, 1]
        result["models"][kind] = {"selected_C": best_c, "validation_roc_auc": best_auc,
                                  "test": score(labels[split["test"]], test_prob)}
        print(kind, result["models"][kind]["test"], flush=True)
        if kind == "texture":
            artifact = args.output.parent / "texture_model.joblib"
            joblib.dump({"model": best_model, "feature_kind": kind,
                         "training_split": "source-grouped train only", "seed": SEED}, artifact)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
