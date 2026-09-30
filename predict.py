"""Research-only single-face-crop inference using a locally trained baseline."""

import argparse
from pathlib import Path

import joblib

from model import features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="A cropped face image")
    parser.add_argument("--model", type=Path, default=Path("results/texture_model.joblib"))
    args = parser.parse_args()
    # Only load models you trained or trust: joblib uses pickle internally.
    bundle = joblib.load(args.model)
    probability = bundle["model"].predict_proba(features(args.image, bundle["feature_kind"])[None, :])[0, 1]
    print(f"Fake-class score: {probability:.3f}")
    print("Research baseline only. This score is not calibrated for real-world deepfake detection.")


if __name__ == "__main__":
    main()
