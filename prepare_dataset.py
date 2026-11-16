"""Convert the Roboflow basketball dataset into a single player class."""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

import yaml
from project_paths import PREPARED_DATASET, RAW_DATASET


DEFAULT_SOURCE = RAW_DATASET
DEFAULT_OUTPUT = PREPARED_DATASET
SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def convert(source: Path, output: Path) -> None:
    source = source.resolve()
    output = output.resolve()
    if not (source / "data.yaml").is_file():
        raise FileNotFoundError(f"Source data.yaml not found: {source}")
    if output == source or source in output.parents:
        raise ValueError("Output must be outside the source dataset directory.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}")

    config = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8"))
    names = config["names"]
    if isinstance(names, dict):
        names = [names[i] if i in names else names[str(i)] for i in range(len(names))]
    player_ids = {
        i for i, name in enumerate(names)
        if name == "player" or name.startswith("player-")
    }
    if "player" not in names or not player_ids:
        raise ValueError("Player classes were not found in data.yaml.")
    print("Classes merged into player:", {i: names[i] for i in sorted(player_ids)})

    summary = {}
    for split in SPLITS:
        image_dir = source / split / "images"
        label_dir = source / split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing images/labels directory for {split}")
        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        labels = list(label_dir.glob("*.txt"))
        if {p.stem for p in images} != {p.stem for p in labels}:
            raise ValueError(f"Image/label filename mismatch in {split}")

        out_images = output / split / "images"
        out_labels = output / split / "labels"
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)
        boxes = 0
        for image in images:
            kept = []
            for line_number, line in enumerate(
                (label_dir / f"{image.stem}.txt").read_text(encoding="utf-8").splitlines(), 1
            ):
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"Invalid label: {split}/{image.stem}.txt:{line_number}")
                class_id = int(fields[0])
                x, y, w, h = map(float, fields[1:])
                if class_id < 0 or class_id >= len(names) or not all(
                    0 <= value <= 1 for value in (x, y, w, h)
                ) or w <= 0 or h <= 0:
                    raise ValueError(f"Invalid label: {split}/{image.stem}.txt:{line_number}")
                if class_id in player_ids:
                    kept.append("0 " + " ".join(fields[1:]))
            shutil.copy2(image, out_images / image.name)
            (out_labels / f"{image.stem}.txt").write_text(
                "\n".join(kept) + ("\n" if kept else ""), encoding="utf-8"
            )
            boxes += len(kept)
        summary[split] = Counter(images=len(images), player_boxes=boxes)

    # Absolute root avoids Roboflow's ../train path ambiguity on Windows.
    dataset_yaml = {
        "path": output.as_posix(),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": {0: "player"},
    }
    (output / "data.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    for split, counts in summary.items():
        print(f"{split}: {counts['images']} images, {counts['player_boxes']} player boxes")
    print(f"Prepared dataset: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    convert(args.source, args.output)

