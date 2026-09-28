#!/usr/bin/env python3
"""Create a tiny zero-byte-image YOLO dataset for local testing."""

from pathlib import Path


def write_image(images: Path, name: str) -> None:
    (images / name).write_bytes(b"")


def write_label(labels: Path, name: str, text: str) -> None:
    (labels / f"{Path(name).stem}.txt").write_text(text, encoding="utf-8")


def create_dataset(root: Path) -> None:
    train_images = root / "train" / "images"
    train_labels = root / "train" / "labels"
    val_images = root / "val" / "images"
    val_labels = root / "val" / "labels"
    test_images = root / "test" / "images"
    test_labels = root / "test" / "labels"
    for directory in (train_images, train_labels, val_images, val_labels, test_images, test_labels):
        directory.mkdir(parents=True, exist_ok=True)

    write_image(train_images, "background.jpg")
    write_label(train_labels, "background.jpg", "")
    write_image(train_images, "fire.jpg")
    write_label(train_labels, "fire.jpg", "0 0.5 0.5 0.2 0.3\n")
    write_image(train_images, "smoke.jpg")
    write_label(train_labels, "smoke.jpg", "1 0.4 0.4 0.3 0.2\n")
    write_image(train_images, "both.jpg")
    write_label(train_labels, "both.jpg", "0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n")
    write_image(train_images, "bad.jpg")
    write_label(train_labels, "bad.jpg", "0 0.5 0.5\n0 0.5 0.5 1.5 0.2\n")
    write_image(train_images, "unknown.jpg")
    write_label(train_labels, "unknown.jpg", "9 0.5 0.5 0.2 0.2\n")
    write_image(train_images, "missing_label.jpg")
    write_image(val_images, "val_fire.jpg")
    write_label(val_labels, "val_fire.jpg", "0 0.5 0.5 0.2 0.2\n")
    write_image(test_images, "test_background.jpg")
    write_label(test_labels, "test_background.jpg", "")
    write_label(test_labels, "orphan.jpg", "1 0.5 0.5 0.2 0.2\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "generated_dataset")
    args = parser.parse_args()
    create_dataset(args.output)
    print(f"Created sample dataset at {args.output}")


if __name__ == "__main__":
    main()
