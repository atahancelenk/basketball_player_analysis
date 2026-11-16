"""Fine-tune a pretrained YOLOv8 nano detector on basketball players."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from project_paths import OUTPUTS, PREPARED_DATASET, PRETRAINED_WEIGHTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PREPARED_DATASET / "data.yaml")
    parser.add_argument("--model", default=str(PRETRAINED_WEIGHTS) if PRETRAINED_WEIGHTS.is_file() else "yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="cpu or CUDA device number, e.g. 0")
    parser.add_argument("--name", default="train", help="Output run name")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Dataset not found: {args.data}. Run prepare_dataset.py first.")

    os.environ.setdefault("YOLO_CONFIG_DIR", str(OUTPUTS / "ultralytics_config"))
    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,  # Stable Windows multiprocessing behavior.
        seed=42,
        patience=8,
        plots=True,
        project=str(OUTPUTS),
        name=args.name,
    )
    print(f"Training finished. Run directory: {model.trainer.save_dir}")


if __name__ == "__main__":
    main()

