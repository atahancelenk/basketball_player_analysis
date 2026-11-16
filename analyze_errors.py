"""List missed players and false detections at a fixed threshold."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

from PIL import Image, ImageDraw
from project_paths import OUTPUTS, PREPARED_DATASET

def box_iou(a: list[float], b: list[float]) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def read_truth(label: Path, width: int, height: int) -> list[list[float]]:
    boxes = []
    for line in label.read_text(encoding="utf-8").splitlines():
        cls, x, y, w, h = line.split()
        if cls != "0":
            raise ValueError(f"Unexpected class in {label}")
        x, y, w, h = map(float, (x, y, w, h))
        boxes.append([(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height])
    return boxes


def save_example(record: dict, path: Path) -> None:
    with Image.open(record["path"]) as original:
        image = original.convert("RGB")
    draw = ImageDraw.Draw(image)
    for index, box in enumerate(record["truth"]):
        if index not in record["matched"]:
            draw.rectangle(box, outline="#ed3434", width=6)  # Missed player
    for box, confidence, correct in record["predictions"]:
        color = "#29cb62" if correct else "#ffa127"
        draw.rectangle(box, outline=color, width=6)
        draw.text((box[0], max(0, box[1]-18)), f"{confidence:.2f}", fill=color)
    draw.rectangle((0, 0, 850, 34), fill="black")
    draw.text((12, 10), "Green=correct  Orange=false detection  Red=missed player", fill="white")
    image.save(path, quality=90)


def analyze(model, data: Path, split: str, output: Path, conf: float = 0.25,
            match_iou: float = 0.5, imgsz: int = 320, device: str | None = None) -> dict:
    images = sorted((data / split / "images").glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No images in {data / split / 'images'}")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    results = model.predict(source=[str(p) for p in images], stream=True, batch=8,
                            conf=conf, imgsz=imgsz, device=device, verbose=False)
    for result in results:
        path = Path(result.path)
        height, width = result.orig_shape
        truth = read_truth(data / split / "labels" / f"{path.stem}.txt", width, height)
        candidates = []
        if result.boxes is not None:
            candidates = list(zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()))
        candidates.sort(key=lambda item: item[1], reverse=True)
        matched, predictions = set(), []
        for box, score in candidates:
            best_iou, best_index = max(
                ((box_iou(box, gt), j) for j, gt in enumerate(truth) if j not in matched),
                default=(0.0, -1),
            )
            correct = best_iou >= match_iou
            if correct:
                matched.add(best_index)
            predictions.append((box, float(score), correct))
        tp = len(matched)
        records.append({"path": path, "truth": truth, "matched": matched,
                        "predictions": predictions, "gt": len(truth), "tp": tp,
                        "fp": len(predictions)-tp, "fn": len(truth)-tp})

    with (output / "per_image.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("image", "ground_truth", "correct", "false_detection", "missed_player"))
        for r in records:
            writer.writerow((r["path"].name, r["gt"], r["tp"], r["fp"], r["fn"]))
    totals = {key: sum(r[key] for r in records) for key in ("gt", "tp", "fp", "fn")}
    summary = {"split": split, "confidence_threshold": conf, "match_iou": match_iou,
               "images": len(records), "ground_truth_players": totals["gt"],
               "correct_detections": totals["tp"], "false_detections": totals["fp"],
               "missed_players": totals["fn"]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    ranked = sorted(records, key=lambda r: (r["fn"]+r["fp"], r["fn"]), reverse=True)
    selected, clips = [], set()
    for r in ranked:
        if r["fn"]+r["fp"] == 0:
            break
        clip = re.sub(r"-\d{4}_png\.rf\.[^.]+$", "", r["path"].stem)
        if clip not in clips:
            selected.append(r)
            clips.add(clip)
        if len(selected) == 6:
            break
    for r in ranked:
        if len(selected) == 6:
            break
        if r not in selected and r["fn"]+r["fp"] > 0:
            selected.append(r)
    for i, r in enumerate(selected, 1):
        save_example(r, output / f"error_{i:02d}.jpg")
    print(json.dumps(summary, indent=2))
    print(f"Error examples and counts: {output}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PREPARED_DATASET)
    parser.add_argument("--weights", type=Path, default=OUTPUTS / "train" / "weights" / "best.pt")
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=320)
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f"Weights not found: {args.weights}")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(OUTPUTS / "ultralytics_config"))
    from ultralytics import YOLO
    analyze(YOLO(str(args.weights.resolve())), args.data.resolve(), args.split,
            OUTPUTS / f"{args.split}_errors", imgsz=args.imgsz)


if __name__ == "__main__":
    main()
