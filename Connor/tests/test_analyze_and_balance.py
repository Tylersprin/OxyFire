import tempfile
import unittest
from pathlib import Path

from scripts.analyze_and_balance import _classify, analyze_split, build_manifest, run


class AnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.images = self.root / "train" / "images"
        self.labels = self.root / "train" / "labels"
        self.images.mkdir(parents=True)
        self.labels.mkdir(parents=True)
        self.classes = {0: "fire", 1: "smoke"}

    def tearDown(self):
        self.temp.cleanup()

    def add(self, name, label=None):
        image = self.images / f"{name}.jpg"
        image.write_bytes(f"image-{name}".encode())
        if label is not None:
            (self.labels / f"{name}.txt").write_text(label, encoding="utf-8")
        return image

    def test_categories(self):
        self.add("empty", "")
        self.add("fire", "0 0.5 0.5 0.2 0.2\n")
        self.add("smoke", "1 0.5 0.5 0.2 0.2\n")
        self.add("both", "0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n")
        result = analyze_split("train", self.root / "train", "images", "labels", self.classes, 0)
        self.assertEqual(result.categories["none"], 1)
        self.assertEqual(result.categories["fire_only"], 1)
        self.assertEqual(result.categories["smoke_only"], 1)
        self.assertEqual(result.categories["fire_and_smoke"], 1)

    def test_classify_helper(self):
        self.assertEqual(_classify(set(), fire_id=0, smoke_id=1, has_content=False), "none")
        self.assertEqual(_classify({0}, fire_id=0, smoke_id=1, has_content=True), "fire_only")
        self.assertEqual(_classify({1}, fire_id=0, smoke_id=1, has_content=True), "smoke_only")
        self.assertEqual(_classify({0, 1}, fire_id=0, smoke_id=1, has_content=True), "fire_and_smoke")

    def test_malformed_invalid_and_unknown(self):
        self.add("bad", "0 0.5 0.5\n0 0.5 0.5 1.2 0.2\n9 0.5 0.5 0.2 0.2\n")
        result = analyze_split("train", self.root / "train", "images", "labels", self.classes, 0)
        self.assertEqual(result.issues.malformed_rows, 1)
        self.assertEqual(result.issues.invalid_coordinates, 1)
        self.assertEqual(result.issues.unknown_class_ids, 1)

    def test_missing_label(self):
        self.add("missing")
        result = analyze_split("train", self.root / "train", "images", "labels", self.classes, 0)
        self.assertEqual(result.issues.images_missing_labels, 1)
        self.assertEqual(result.categories["none"], 1)

    def test_manifest_strategies_are_deterministic(self):
        entries = {
            "none": [Path("none.jpg")],
            "fire_only": [Path("fire1.jpg"), Path("fire2.jpg")],
            "smoke_only": [Path("smoke.jpg")],
            "fire_and_smoke": [],
            "other": [],
        }
        first = build_manifest(entries, "oversample_manifest", 42)
        second = build_manifest(entries, "oversample_manifest", 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        downsampled = build_manifest(entries, "downsample", 42)
        self.assertEqual(len(downsampled), 3)
        self.assertEqual(
            build_manifest(entries, "none", 42),
            [Path("none.jpg"), Path("fire1.jpg"), Path("fire2.jpg"), Path("smoke.jpg")],
        )

    def test_run_writes_outputs_and_preserves_sources(self):
        self.add("fire", "0 0.5 0.5 0.2 0.2\n")
        original_image = (self.images / "fire.jpg").read_bytes()
        original_label = (self.labels / "fire.txt").read_bytes()
        config = self.root / "config.toml"
        config.write_text(
            'name = "Test"\n[classes]\n0 = "fire"\n1 = "smoke"\n[balancing]\nstrategy = "oversample_manifest"\nseed = 7\n',
            encoding="utf-8",
        )
        output = self.root / "output"
        self.assertEqual(run(self.root, config, output, 0), 0)
        for filename in (
            "analysis_summary.txt",
            "dataset_summary.csv",
            "category_summary.csv",
            "class_summary.csv",
            "validation_report.txt",
            "balanced_train.txt",
            "dataset_summary.json",
        ):
            self.assertTrue((output / filename).is_file(), filename)
        self.assertEqual((self.images / "fire.jpg").read_bytes(), original_image)
        self.assertEqual((self.labels / "fire.txt").read_bytes(), original_label)


if __name__ == "__main__":
    unittest.main()
