"""Evaluate the trained player detector on the untouched test split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from project_paths import OUTPUTS, PREPARED_DATASET


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PREPARED_DATASET / "data.yaml")
    parser.add_argument("--weights", type=Path, default=OUTPUTS / "train" / "weights" / "best.pt")
    parser.add_argument("--device", default=None, help="cpu or CUDA device number, e.g. 0")
    parser.add_argument("--imgsz", type=int, default=320)
    args = parser.parse_args()
    if not args.data.is_file() or not args.weights.is_file():
        parser.error("Prepared data or trained weights missing. Run prepare_dataset.py and train.py first.")

    os.environ.setdefault("YOLO_CONFIG_DIR", str(OUTPUTS / "ultralytics_config"))
    from ultralytics import YOLO

    model = YOLO(str(args.weights.resolve()))
    metrics = model.val(
        data=str(args.data.resolve()),
        split="test",
        imgsz=args.imgsz,
        device=args.device,
        workers=0,
        plots=True,
        project=str(OUTPUTS),
        name="test_evaluation",
    )
    results = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
    }
    p, r = results["precision"], results["recall"]
    results["F1"] = 2 * p * r / (p + r) if p + r else 0.0
    output = OUTPUTS / "test_metrics.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))

    from analyze_errors import analyze
    analyze(model, args.data.parent.resolve(), "test", OUTPUTS / "test_errors",
            imgsz=args.imgsz, device=args.device)
    print(f"Test metrics: {output}")


if __name__ == "__main__":
    main()

