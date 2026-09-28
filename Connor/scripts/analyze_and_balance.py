#!/usr/bin/env python3
"""Analyze YOLO labels and build a training-only balancing manifest.

The implementation intentionally reads directory entries and label text only.
Image files are never opened and no third-party packages are required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import tomllib
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CATEGORIES = ("none", "fire_only", "smoke_only", "fire_and_smoke", "other")
SPLITS = ("train", "val", "test")


@dataclass
class ValidationIssues:
    images_missing_labels: int = 0
    labels_missing_images: int = 0
    empty_labels: int = 0
    malformed_rows: int = 0
    unknown_class_ids: int = 0
    invalid_coordinates: int = 0
    details: list[str] = field(default_factory=list)

    def total(self) -> int:
        return sum(
            (
                self.images_missing_labels,
                self.labels_missing_images,
                self.empty_labels,
                self.malformed_rows,
                self.unknown_class_ids,
                self.invalid_coordinates,
            )
        )


@dataclass
class SplitAnalysis:
    name: str
    split_root: str
    images: list[Path] = field(default_factory=list)
    labels: list[Path] = field(default_factory=list)
    categories: Counter = field(default_factory=Counter)
    boxes_by_class: Counter = field(default_factory=Counter)
    total_boxes: int = 0
    issues: ValidationIssues = field(default_factory=ValidationIssues)

    @property
    def total_images(self) -> int:
        return len(self.images)

    @property
    def total_labels(self) -> int:
        return len(self.labels)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    if not isinstance(config.get("classes"), dict) or not config["classes"]:
        raise ValueError("config must contain a non-empty [classes] table")
    classes: dict[int, str] = {}
    for raw_id, raw_name in config["classes"].items():
        try:
            class_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"class ID {raw_id!r} is not an integer") from exc
        classes[class_id] = str(raw_name)
    config["classes"] = classes
    balancing = config.setdefault("balancing", {})
    balancing.setdefault("strategy", "oversample_manifest")
    balancing.setdefault("seed", 42)
    strategy = balancing["strategy"]
    if strategy not in {"none", "downsample", "oversample_manifest"}:
        raise ValueError(f"unsupported balancing strategy: {strategy}")
    paths = config.setdefault("paths", {})
    paths.setdefault("images", "images")
    paths.setdefault("labels", "labels")
    for split in SPLITS:
        paths.setdefault(split, split)
    config.setdefault("name", "Unnamed dataset")
    return config


def _relative_key(path: Path, base: Path) -> str:
    return path.relative_to(base).with_suffix(".txt").as_posix()


def _record_detail(issues: ValidationIssues, detail: str) -> None:
    if len(issues.details) < 12:
        issues.details.append(detail)


def _classify(class_ids: set[int], *, fire_id: int | None, smoke_id: int | None, has_content: bool) -> str:
    if not has_content:
        return "none"
    has_fire = fire_id is not None and fire_id in class_ids
    has_smoke = smoke_id is not None and smoke_id in class_ids
    if has_fire and has_smoke:
        return "fire_and_smoke"
    if has_fire:
        return "fire_only"
    if has_smoke:
        return "smoke_only"
    return "other"


def _parse_label_file(
    label_path: Path,
    known_classes: set[int],
    issues: ValidationIssues,
    split_name: str,
) -> tuple[set[int], int, Counter, bool]:
    try:
        text = label_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        _record_detail(issues, f"{split_name}: could not read {label_path}: {exc}")
        issues.malformed_rows += 1
        return set(), 0, Counter(), True

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        issues.empty_labels += 1
        return set(), 0, Counter(), False

    class_ids: set[int] = set()
    box_counts: Counter = Counter()
    valid_rows = 0
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) != 5:
            issues.malformed_rows += 1
            _record_detail(issues, f"{label_path}:{line_number}: expected 5 fields")
            continue
        try:
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError:
            issues.malformed_rows += 1
            _record_detail(issues, f"{label_path}:{line_number}: non-numeric YOLO row")
            continue

        valid_rows += 1
        class_ids.add(class_id)
        box_counts[class_id] += 1
        if class_id not in known_classes:
            issues.unknown_class_ids += 1
            _record_detail(issues, f"{label_path}:{line_number}: unknown class {class_id}")
        x, y, width, height = coordinates
        valid_coordinates = (
            all(math.isfinite(value) for value in coordinates)
            and 0 <= x <= 1
            and 0 <= y <= 1
            and 0 < width <= 1
            and 0 < height <= 1
        )
        if not valid_coordinates:
            issues.invalid_coordinates += 1
            _record_detail(issues, f"{label_path}:{line_number}: invalid normalized coordinates")
    return class_ids, valid_rows, box_counts, True


def analyze_split(
    split_name: str,
    split_root: Path,
    images_dir_name: str,
    labels_dir_name: str,
    classes: dict[int, str],
    progress_every: int = 1000,
) -> SplitAnalysis:
    result = SplitAnalysis(split_name, str(split_root))
    images_root = split_root / images_dir_name
    labels_root = split_root / labels_dir_name
    if images_root.is_dir():
        result.images = sorted(
            (path for path in images_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
            key=lambda path: path.relative_to(images_root).as_posix().lower(),
        )
    if labels_root.is_dir():
        result.labels = sorted(
            (path for path in labels_root.rglob("*.txt") if path.is_file()),
            key=lambda path: path.relative_to(labels_root).as_posix().lower(),
        )

    image_keys = {_relative_key(path, images_root) for path in result.images}
    label_keys = {path.relative_to(labels_root).as_posix() for path in result.labels} if labels_root.is_dir() else set()
    result.issues.images_missing_labels = len(image_keys - label_keys)
    result.issues.labels_missing_images = len(label_keys - image_keys)
    if result.issues.images_missing_labels:
        _record_detail(result.issues, f"{split_name}: {result.issues.images_missing_labels} image(s) missing labels")
    if result.issues.labels_missing_images:
        _record_detail(result.issues, f"{split_name}: {result.issues.labels_missing_images} label(s) missing images")

    label_by_key = {
        path.relative_to(labels_root).as_posix(): path for path in result.labels
    } if labels_root.is_dir() else {}
    fire_id = next((class_id for class_id, name in classes.items() if name.strip().lower() == "fire"), None)
    smoke_id = next((class_id for class_id, name in classes.items() if name.strip().lower() == "smoke"), None)
    known_classes = set(classes)

    for index, image_path in enumerate(result.images, start=1):
        key = _relative_key(image_path, images_root)
        label_path = label_by_key.get(key)
        if label_path is None:
            category = "none"
        else:
            class_ids, row_count, box_counts, has_content = _parse_label_file(
                label_path, known_classes, result.issues, split_name
            )
            category = _classify(class_ids, fire_id=fire_id, smoke_id=smoke_id, has_content=has_content)
            result.total_boxes += row_count
            result.boxes_by_class.update(box_counts)
        result.categories[category] += 1
        if progress_every and index % progress_every == 0:
            print(f"Analyzed {index} / {len(result.images)} {split_name} images...", flush=True)
    return result


def classify_training_images(
    analysis: SplitAnalysis,
    images_dir_name: str,
    labels_dir_name: str,
    classes: dict[int, str],
) -> dict[str, list[Path]]:
    images_root = Path(analysis.split_root) / images_dir_name
    labels_root = Path(analysis.split_root) / labels_dir_name
    label_by_key = {
        path.relative_to(labels_root).as_posix(): path for path in analysis.labels
    } if labels_root.is_dir() else {}
    fire_id = next((class_id for class_id, name in classes.items() if name.strip().lower() == "fire"), None)
    smoke_id = next((class_id for class_id, name in classes.items() if name.strip().lower() == "smoke"), None)
    known_classes = set(classes)
    entries = {category: [] for category in CATEGORIES}
    for image_path in analysis.images:
        label_path = label_by_key.get(_relative_key(image_path, images_root))
        if label_path is None:
            category = "none"
        else:
            class_ids, _, _, has_content = _parse_label_file(
                label_path, known_classes, ValidationIssues(), analysis.name
            )
            category = _classify(class_ids, fire_id=fire_id, smoke_id=smoke_id, has_content=has_content)
        entries[category].append(image_path)
    return entries


def build_manifest(entries: dict[str, list[Path]], strategy: str, seed: int) -> list[Path]:
    rng = random.Random(seed)
    ordered = [entries[category] for category in CATEGORIES]
    if strategy == "none":
        return [path for category_entries in ordered for path in category_entries]
    populated = [paths for paths in ordered if paths]
    if not populated:
        return []
    if strategy == "downsample":
        target = min(len(paths) for paths in populated)
        return [path for paths in ordered if paths for path in rng.sample(paths, target)]
    target = max(len(paths) for paths in populated)
    manifest: list[Path] = []
    for paths in ordered:
        if not paths:
            continue
        manifest.extend(paths)
        for _ in range(target - len(paths)):
            manifest.append(rng.choice(paths))
    return manifest


def _percent(value: int, total: int) -> float:
    return round((value / total) * 100, 2) if total else 0.0


def _csv_write(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _status(split_results: dict[str, SplitAnalysis], train_root: Path, train: SplitAnalysis, paths: dict[str, Any]) -> str:
    if not train_root.is_dir() or not (train_root / paths["images"]).is_dir() or not (train_root / paths["labels"]).is_dir():
        return "FAIL"
    if not train.images:
        return "FAIL"
    if any(result.issues.total() for result in split_results.values()):
        return "PASS WITH WARNINGS"
    return "PASS"


def write_outputs(
    output_dir: Path,
    dataset_root: Path,
    config: dict[str, Any],
    split_results: dict[str, SplitAnalysis],
    manifest: list[Path],
    strategy: str,
    status: str,
    category_entries: dict[str, list[Path]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    classes: dict[int, str] = config["classes"]
    dataset_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        result = split_results[split]
        dataset_rows.append(
            {
                "split": split,
                "total_images": result.total_images,
                "total_labels": result.total_labels,
                "total_bounding_boxes": result.total_boxes,
                "empty_background_images": result.categories["none"],
                "malformed_rows": result.issues.malformed_rows,
                "unknown_class_ids": result.issues.unknown_class_ids,
                "invalid_coordinates": result.issues.invalid_coordinates,
                "images_missing_labels": result.issues.images_missing_labels,
                "labels_missing_images": result.issues.labels_missing_images,
            }
        )
        for category in CATEGORIES:
            count = result.categories[category]
            category_rows.append(
                {
                    "split": split,
                    "category": category,
                    "count": count,
                    "percentage": _percent(count, result.total_images),
                }
            )
        for class_id, name in sorted(classes.items()):
            class_rows.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": name,
                    "box_count": result.boxes_by_class[class_id],
                    "percentage_of_boxes": _percent(result.boxes_by_class[class_id], result.total_boxes),
                }
            )
        for unknown_id in sorted(class_id for class_id in result.boxes_by_class if class_id not in classes):
            class_rows.append(
                {
                    "split": split,
                    "class_id": unknown_id,
                    "class_name": f"unknown:{unknown_id}",
                    "box_count": result.boxes_by_class[unknown_id],
                    "percentage_of_boxes": _percent(result.boxes_by_class[unknown_id], result.total_boxes),
                }
            )

    _csv_write(output_dir / "dataset_summary.csv", list(dataset_rows[0]), dataset_rows)
    _csv_write(output_dir / "category_summary.csv", list(category_rows[0]), category_rows)
    _csv_write(output_dir / "class_summary.csv", list(class_rows[0]), class_rows)
    (output_dir / "balanced_train.txt").write_text(
        "".join(f"{path.resolve()}\n" for path in manifest), encoding="utf-8"
    )

    json_summary = {
        "dataset": config["name"],
        "dataset_root": str(dataset_root.resolve()),
        "status": status,
        "balancing": {
            "strategy": strategy,
            "seed": config["balancing"]["seed"],
            "original_samples": split_results["train"].total_images,
            "manifest_entries": len(manifest),
            "category_counts": {category: len(category_entries[category]) for category in CATEGORIES},
        },
        "splits": {
            split: {
                "total_images": result.total_images,
                "total_labels": result.total_labels,
                "total_bounding_boxes": result.total_boxes,
                "categories": {category: result.categories[category] for category in CATEGORIES},
                "boxes_by_class": {str(class_id): count for class_id, count in result.boxes_by_class.items()},
                "validation": asdict(result.issues),
            }
            for split, result in split_results.items()
        },
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(json_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "=" * 60,
        "DATASET ANALYSIS",
        "=" * 60,
        "",
        f"Dataset: {config['name']}",
        f"Root: {dataset_root}",
        f"Overall status: {status}",
        "",
    ]
    for split in SPLITS:
        result = split_results[split]
        lines.extend(
            [
                split.upper(),
                f"Images: {result.total_images}",
                f"Label files: {result.total_labels}",
                f"Bounding boxes: {result.total_boxes}",
                "",
                "Categories:",
            ]
        )
        for category in CATEGORIES:
            lines.append(f"{category + ':':18}{result.categories[category]}")
        lines.extend(
            [
                "",
                "Bounding Boxes:",
                *[f"{name + ':':18}{result.boxes_by_class[class_id]}" for class_id, name in sorted(classes.items())],
                "",
                "Validation:",
                f"Missing labels:   {result.issues.images_missing_labels}",
                f"Labels missing images: {result.issues.labels_missing_images}",
                f"Empty labels:     {result.issues.empty_labels}",
                f"Malformed rows:   {result.issues.malformed_rows}",
                f"Unknown classes:  {result.issues.unknown_class_ids}",
                f"Invalid coords:   {result.issues.invalid_coordinates}",
                "",
            ]
        )
    lines.extend(
        [
            "=" * 60,
            "BALANCING",
            "=" * 60,
            "",
            f"Strategy: {strategy}",
            f"Original samples: {split_results['train'].total_images}",
            f"Manifest entries: {len(manifest)}",
            "",
            "Training category counts used:",
            *[f"{category + ':':18}{len(category_entries[category])}" for category in CATEGORIES],
            "",
            "Original images and labels were not modified.",
            "Validation and test distributions were not balanced.",
            "",
        ]
    )
    (output_dir / "analysis_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    report_lines = ["VALIDATION REPORT", "=" * 60, f"Overall status: {status}", ""]
    for split in SPLITS:
        issues = split_results[split].issues
        report_lines.extend(
            [
                split.upper(),
                f"Images missing labels: {issues.images_missing_labels}",
                f"Labels missing images: {issues.labels_missing_images}",
                f"Empty labels: {issues.empty_labels}",
                f"Malformed YOLO rows: {issues.malformed_rows}",
                f"Unknown class IDs: {issues.unknown_class_ids}",
                f"Invalid normalized coordinates: {issues.invalid_coordinates}",
            ]
        )
        if issues.details:
            report_lines.append("Examples:")
            report_lines.extend(f"  - {detail}" for detail in issues.details)
        report_lines.append("")
    (output_dir / "validation_report.txt").write_text("\n".join(report_lines), encoding="utf-8")


def run(dataset_root: Path, config_path: Path, output_dir: Path, progress_every: int = 1000) -> int:
    config = load_config(config_path)
    paths = config["paths"]
    split_results: dict[str, SplitAnalysis] = {}
    for split in SPLITS:
        split_results[split] = analyze_split(
            split,
            dataset_root / paths[split],
            paths["images"],
            paths["labels"],
            config["classes"],
            progress_every=progress_every,
        )
    train = split_results["train"]
    category_entries = classify_training_images(train, paths["images"], paths["labels"], config["classes"])
    strategy = config["balancing"]["strategy"]
    manifest = build_manifest(category_entries, strategy, int(config["balancing"]["seed"]))
    status = _status(split_results, dataset_root / paths["train"], train, paths)
    write_outputs(output_dir, dataset_root, config, split_results, manifest, strategy, status, category_entries)
    print(f"Dataset: {config['name']}")
    print(f"Status: {status}")
    print(f"Training images: {train.total_images}; manifest entries: {len(manifest)}")
    print(f"Outputs written to: {output_dir}")
    return 0 if status != "FAIL" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args(argv)
    try:
        return run(args.dataset_root, args.config, args.output_dir, max(0, args.progress_every))
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
