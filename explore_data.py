"""Create report-ready figures from the prepared player dataset."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from project_paths import OUTPUTS, PREPARED_DATASET


SPLITS = ("train", "valid", "test")


def boxes_for(label_file: Path) -> list[tuple[float, float, float, float]]:
    boxes = []
    for line in label_file.read_text(encoding="utf-8").splitlines():
        cls, x, y, w, h = line.split()
        if cls != "0":
            raise ValueError(f"Unexpected class in {label_file}")
        boxes.append(tuple(map(float, (x, y, w, h))))
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PREPARED_DATASET)
    parser.add_argument("--output", type=Path, default=OUTPUTS / "eda")
    args = parser.parse_args()
    if not (args.data / "data.yaml").is_file():
        parser.error("Prepared dataset missing. Run prepare_dataset.py first.")
    args.output.mkdir(parents=True, exist_ok=True)

    images_by_split = {}
    counts_by_split = {}
    areas = []
    for split in SPLITS:
        images = sorted((args.data / split / "images").glob("*.jpg"))
        images_by_split[split] = images
        counts = []
        for image in images:
            boxes = boxes_for(args.data / split / "labels" / f"{image.stem}.txt")
            counts.append(len(boxes))
            if split == "train":
                areas.extend(w * h for _, _, w, h in boxes)
        counts_by_split[split] = counts

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(SPLITS, [len(images_by_split[s]) for s in SPLITS], color="#2675a8")
    ax.set(title="Images by split", ylabel="Image count")
    fig.tight_layout()
    fig.savefig(args.output / "split_counts.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    hist = Counter(counts_by_split["train"])
    ax.bar(sorted(hist), [hist[n] for n in sorted(hist)], color="#c76b2b")
    ax.set(title="Players per training image", xlabel="Player boxes", ylabel="Image count")
    fig.tight_layout()
    fig.savefig(args.output / "players_per_image.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist([100 * area for area in areas], bins=20, color="#5c9b5d")
    ax.set(title="Player box area in training images", xlabel="Percent of image area", ylabel="Box count")
    fig.tight_layout()
    fig.savefig(args.output / "player_box_area.png", dpi=160)
    plt.close(fig)

    # Sample spread across the sorted training list to show several source clips.
    train_images = images_by_split["train"]
    sample = [train_images[i] for i in range(0, len(train_images), max(1, len(train_images) // 6))][:6]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, path in zip(axes.flat, sample):
        with Image.open(path) as original:
            image = original.convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size
        boxes = boxes_for(args.data / "train" / "labels" / f"{path.stem}.txt")
        for x, y, w, h in boxes:
            draw.rectangle(
                ((x - w / 2) * width, (y - h / 2) * height,
                 (x + w / 2) * width, (y + h / 2) * height),
                outline="#f4c542", width=5,
            )
        ax.imshow(image)
        ax.set_title(f"{len(boxes)} player boxes")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.output / "annotated_examples.png", dpi=140)
    plt.close(fig)
    print(f"Figures saved to {args.output}")


if __name__ == "__main__":
    main()

