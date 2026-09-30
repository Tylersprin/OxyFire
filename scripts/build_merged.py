#!/usr/bin/env python3
"""Build one YOLO-ready dataset from the manifest, with duplicate-aware splits.

Images are symlinks to the originals in raw/ (nothing copied or modified); labels
are new files with fire = 0, smoke = 1. Output (in --out):
  data.yaml                        Ultralytics dataset config
  {train,val,test}/images/*.jpg    symlinks
  {train,val,test}/labels/*.txt    labels (empty file = background image)
  merged_index.csv                 one row per image: source, old/new split, group
  merged_summary.md                counts per split, dataset and category

Selection (see reports/REPORT.md):
  dfire         all images
  aiformankind  all images (license CC BY-NC-SA: drop via SELECT if needed)
  azimjaan21    smoke images and cloud/glare negatives; fire-only images are
                left out because their smoke is unlabeled
Byte-identical files (same md5) become one image with the boxes of every copy:
azimjaan21 stores 714 images twice, once labeled fire and once smoke, so those
become fire-and-smoke images. Selection is applied after that. Splits are assigned per loose
(pHash <= 5) duplicate group, so copies never straddle train/val/test.

Usage:
  python scripts/build_merged.py --limit 50 --out $SCRATCH/oxyfire_data/outputs/test_merged
  python scripts/build_merged.py                       # full, inside a Slurm job
"""
import argparse
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

OUT_BASE = Path(os.environ["SCRATCH"]) / "oxyfire_data/outputs"
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42
# dataset -> image categories to include
SELECT = {
    "dfire": {"fire", "smoke", "fire_and_smoke", "neither"},
    "aiformankind": {"fire", "smoke", "fire_and_smoke", "neither"},
    # fire-only images are left out (smoke unlabeled); fire_and_smoke only arises after
    # identical copies are combined, see combine_copies()
    "azimjaan21": {"smoke", "neither", "fire_and_smoke"},
}
MAX_ASPECT = 50        # boxes thinner than 1:50 are labeling errors
MIN_SIDE_PX = 1        # boxes under 1 px after clipping are dropped


def clip_box(x, y, w, h):
    """Clip a normalized center box to the image; returns None if nothing is left."""
    x1, y1 = max(0.0, x - w / 2), max(0.0, y - h / 2)
    x2, y2 = min(1.0, x + w / 2), min(1.0, y + h / 2)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def combine_copies(idx, lab):
    """Byte-identical files (same md5) become one image carrying the boxes of every copy.
    azimjaan21 stores some images twice, once with fire boxes and once with smoke boxes."""
    idx = idx.sort_values("image_id")
    rep = idx.groupby("md5").image_id.transform("first")
    to_rep = dict(zip(idx.image_id, rep))
    lab = lab.assign(image_id=lab.image_id.map(to_rep))
    lab = lab.drop_duplicates(["image_id", "class_id", "x_center", "y_center", "width", "height"])
    n_copies = (idx.image_id != rep).sum()
    idx = idx[idx.image_id == rep].copy()
    counts = lab.groupby(["image_id", "class_name"]).size().unstack(fill_value=0)
    fire = idx.image_id.map(counts.get("fire", pd.Series(dtype=int))).fillna(0) > 0
    smoke = idx.image_id.map(counts.get("smoke", pd.Series(dtype=int))).fillna(0) > 0
    idx["category"] = np.select([fire & smoke, fire, smoke], ["fire_and_smoke", "fire", "smoke"], "neither")
    return idx, lab, int(n_copies)


def assign_splits(index):
    """Assign each duplicate group to a split, per dataset, to approach SPLITS ratios.
    Groups are visited largest first (hardest to place) then in random order."""
    rng = random.Random(SEED)
    target = {d: {s: r * n for s, r in SPLITS.items()}
              for d, n in index.source_dataset.value_counts().items()}
    filled = {d: Counter() for d in target}
    groups = [(g, sub.source_dataset.value_counts().idxmax(), len(sub))
              for g, sub in index.groupby("group_id")]
    rng.shuffle(groups)
    groups.sort(key=lambda t: -t[2])
    split_of = {}
    for g, d, n in groups:
        s = min(SPLITS, key=lambda s: (filled[d][s] + n) / target[d][s])
        split_of[g] = s
        filled[d][s] += n
    return index.group_id.map(split_of)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest-dir", type=Path, default=OUT_BASE)
    ap.add_argument("--report-dir", type=Path, default=OUT_BASE / "report")
    ap.add_argument("--out", type=Path, default=OUT_BASE / "merged")
    ap.add_argument("--limit", type=int, default=0, help="images per dataset (0 = all), for testing")
    ap.add_argument("--force", action="store_true", help="replace an existing --out folder")
    a = ap.parse_args()

    if a.out.exists() and not a.force:
        raise SystemExit(f"{a.out} exists; pass --force to rebuild it")

    man = pd.read_csv(a.manifest_dir / "manifest.csv")
    box = pd.read_csv(a.manifest_dir / "boxes.csv")
    stats = pd.read_csv(a.report_dir / "image_stats.csv", usecols=["image_id", "md5", "decode_ok"], dtype=str)
    groups = pd.read_csv(a.report_dir / "08_duplicate_groups_loose_le5.csv", usecols=["image_id", "group_id"])

    # --- select images ---
    idx = man[(man.status == "ok")].merge(stats, on="image_id", how="left")
    idx = idx[pd.to_numeric(idx.decode_ok, errors="coerce") == 1]
    lab = box[box.class_name.isin(["fire", "smoke"]) & box.image_id.isin(idx.image_id)]
    idx, lab, exact_dropped = combine_copies(idx, lab)
    keep = np.array([c in SELECT.get(d, set()) for d, c in zip(idx.source_dataset, idx.category)])
    dropped_by_rule = idx[~keep].groupby(["source_dataset", "category"]).size()
    idx = idx[keep]
    if a.limit:
        idx = idx.groupby("source_dataset", group_keys=False).apply(
            lambda g: g.iloc[np.linspace(0, len(g) - 1, min(a.limit, len(g))).astype(int)])

    # images with no duplicate get their own group
    idx = idx.merge(groups, on="image_id", how="left")
    solo = idx.group_id.isna()
    start = int(groups.group_id.max()) + 1 if len(groups) else 0
    idx.loc[solo, "group_id"] = np.arange(start, start + solo.sum())
    idx["group_id"] = idx.group_id.astype(int)
    idx["new_split"] = assign_splits(idx)

    # --- labels: fire/smoke boxes only, clipped, degenerate boxes dropped ---
    lab = lab[lab.image_id.isin(idx.image_id)]
    size = idx.set_index("image_id")[["width", "height"]]
    lines, fixes = defaultdict(list), Counter()
    for r in lab.itertuples(index=False):
        W, H = size.at[r.image_id, "width"], size.at[r.image_id, "height"]
        c = clip_box(r.x_center, r.y_center, r.width, r.height)
        if c is None:
            fixes["dropped_zero_or_outside"] += 1
            continue
        if c != (r.x_center, r.y_center, r.width, r.height) and r.out_of_bounds:
            fixes["clipped_to_image"] += 1
        x, y, w, h = c
        if w * W < MIN_SIDE_PX or h * H < MIN_SIDE_PX:
            fixes["dropped_under_1px"] += 1
            continue
        ar = (w * W) / (h * H)
        if ar > MAX_ASPECT or ar < 1 / MAX_ASPECT:
            fixes["dropped_extreme_aspect"] += 1
            continue
        lines[r.image_id].append(f"{int(r.class_id)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    # --- write the folder (to .partial first, renamed when complete) ---
    tmp = a.out.with_name(a.out.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    for s in SPLITS:
        (tmp / s / "images").mkdir(parents=True)
        (tmp / s / "labels").mkdir(parents=True)
    names = []
    for r in idx.itertuples(index=False):
        # unique, readable name: dataset + original relative path, flattened
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", r.image_id.replace("/", "__"))
        name = Path(stem).stem
        ext = Path(r.image_path).suffix.lower()
        os.symlink(r.image_path, tmp / r.new_split / "images" / f"{name}{ext}")
        (tmp / r.new_split / "labels" / f"{name}.txt").write_text(
            "\n".join(lines.get(r.image_id, [])) + ("\n" if lines.get(r.image_id) else ""))
        names.append(name)
    idx["merged_name"] = names
    idx["n_fire"] = [sum(l.startswith("0 ") for l in lines.get(i, [])) for i in idx.image_id]
    idx["n_smoke"] = [sum(l.startswith("1 ") for l in lines.get(i, [])) for i in idx.image_id]

    final = a.out.resolve()
    (tmp / "data.yaml").write_text(
        f"# Built by scripts/build_merged.py. Images are symlinks into raw/.\n"
        f"path: {final}\ntrain: train/images\nval: val/images\ntest: test/images\n\n"
        f"nc: 2\nnames:\n  0: fire\n  1: smoke\n")
    idx[["image_id", "merged_name", "source_dataset", "original_split", "new_split", "group_id",
         "category", "n_fire", "n_smoke"]].to_csv(tmp / "merged_index.csv", index=False)

    # --- summary ---
    split_ds = pd.crosstab(idx.source_dataset, idx.new_split).reindex(columns=list(SPLITS), fill_value=0)
    split_ds.loc["TOTAL"] = split_ds.sum()
    split_pct = split_ds.div(split_ds.sum(axis=1), axis=0).mul(100).round(1)
    split_cat = pd.crosstab(idx.category, idx.new_split).reindex(columns=list(SPLITS), fill_value=0)
    boxes_split = idx.groupby("new_split")[["n_fire", "n_smoke"]].sum().reindex(list(SPLITS))
    leaks = (idx.groupby("group_id").new_split.nunique() > 1).sum()
    with open(tmp / "merged_summary.md", "w") as f:
        f.write("# Merged dataset summary\n\n")
        f.write(f"Images: {len(idx):,}. Identical copies combined into one image "
                f"(boxes merged): {exact_dropped:,}. "
                f"Duplicate groups split across train/val/test: {leaks} (should be 0).\n\n")
        f.write("## Images left out by the selection rule\n\n```\n" + dropped_by_rule.to_string() + "\n```\n\n")
        f.write("## Images per split\n\n```\n" + split_ds.to_string() + "\n```\n\n")
        f.write("## Split % per dataset\n\n```\n" + split_pct.to_string() + "\n```\n\n")
        f.write("## Image categories per split\n\n```\n" + split_cat.to_string() + "\n```\n\n")
        f.write("## Boxes per split\n\n```\n" + boxes_split.to_string() + "\n```\n\n")
        f.write("## Box fixes applied\n\n```\n" + "\n".join(f"{k}: {v}" for k, v in sorted(fixes.items())) + "\n```\n")
    if a.out.exists():
        shutil.rmtree(a.out)
    tmp.rename(a.out)
    print(open(a.out / "merged_summary.md").read())
    print(f"Ready: {a.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
