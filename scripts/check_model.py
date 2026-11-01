"""Pre-deployment model compatibility and optional real-sample smoke test."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qtagger.inference import QTaggerPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate trusted QTagger+ Kaggle artifacts")
    parser.add_argument("--model", default="models/qtagger_v3_1.joblib")
    parser.add_argument("--sample", default=None, help="Optional real-feature demo CSV")
    args = parser.parse_args()

    predictor = QTaggerPredictor.from_file(args.model)
    print("Model format:", predictor.bundle["format_version"])
    print("Task:", predictor.bundle.get("task", "unknown"))
    print("Classes:", predictor.classes)
    print("Raw feature count:", len(predictor.raw_columns))
    print("Quantum qubits:", predictor.n_qubits)
    print("Landmarks:", len(predictor.landmarks))
    print("Simulator:", predictor.device_name)
    if args.sample:
        frame = pd.read_csv(args.sample, low_memory=False)
        predictions = predictor.predict(frame.head(5))
        print("Sample predictions:", predictions.tolist())
        print("Inference smoke test: PASS")
    else:
        print("Model structure check: PASS (provide --sample to test predictions)")


if __name__ == "__main__":
    main()
