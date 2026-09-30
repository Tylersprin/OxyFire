#!/usr/bin/env python3
"""Data-quality stats for the combined manifest, per dataset and combined.

Stage 1 (heavy, cached): decode every image once -> image_stats.csv with a
  decode check, mean brightness, pHash of the image and of its mirror, file md5.
Stage 2: tables (CSV), plots (PNG) and summary.md in --out.

Checks: counts/splits/% negative, class balance, boxes per image, COCO box sizes
and aspect ratios, image resolution/aspect, label problems, brightness, and
exact / near-duplicates (pHash Hamming <= 5, mirror-aware) within and across
datasets, flagging pairs that cross train/val/test.

Outputs contain file paths: keep them in $SCRATCH, never commit them.

Usage:
  python scripts/quality_stats.py --limit 50 --out $SCRATCH/oxyfire_data/outputs/test_report
  python scripts/quality_stats.py --workers 32        # full, inside a Slurm job
"""
import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import imagehash
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

OUT_BASE = Path(os.environ["SCRATCH"]) / "oxyfire_data/outputs"
DATASETS = ["dfire", "aiformankind", "azimjaan21"]
# Categorical slots 1-3 of the validated default palette, fixed order by dataset.
DS_COLOR = {"dfire": "#2a78d6", "aiformankind": "#eb6834", "azimjaan21": "#1baf7a"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COCO_SMALL, COCO_MEDIUM = 32 ** 2, 96 ** 2          # box area thresholds in pixels
DUP_THRESHOLD = 5                                    # pHash Hamming distance
DARK_BRIGHTNESS = 60                                 # mean gray (0-255) below = "dark"
BANDS = [(0, 11), (11, 22), (22, 33), (33, 44), (44, 54), (54, 64)]  # 6 bands -> pigeonhole


# ---------------- stage 1: per-image pixel pass ----------------

def image_pass(item):
    image_id, path = item
    row = {"image_id": image_id, "decode_ok": 0, "decode_error": "", "mean_brightness": "",
           "phash": "", "phash_mirror": "", "md5": ""}
    try:
        with open(path, "rb") as f:
            row["md5"] = hashlib.md5(f.read()).hexdigest()
        with Image.open(path) as im:
            im.load()                                   # full decode = corruption check
            im = im.convert("RGB")
        row["decode_ok"] = 1
        row["mean_brightness"] = round(float(np.asarray(im.convert("L")).mean()), 2)
        row["phash"] = str(imagehash.phash(im))
        row["phash_mirror"] = str(imagehash.phash(ImageOps.mirror(im)))
    except Exception as e:
        row["decode_error"] = f"{type(e).__name__}: {e}"[:200]
    return row


def run_image_pass(man, out, workers):
    cache = out / "image_stats.csv"
    done = (pd.read_csv(cache, dtype=str, keep_default_na=False) if cache.exists()
            else pd.DataFrame(columns=["image_id"], dtype=str))
    todo = man.loc[~man.image_id.isin(set(done.image_id)), ["image_id", "image_path"]]
    print(f"Pixel pass: {len(done):,} cached, {len(todo):,} to process", flush=True)
    rows = []
    if len(todo):
        with Pool(workers) as pool:
            for i, r in enumerate(pool.imap_unordered(image_pass, todo.itertuples(index=False, name=None),
                                                      chunksize=32), 1):
                rows.append(r)
                if i % 5000 == 0:
                    print(f"  {i:,}/{len(todo):,}", flush=True)
    new = pd.DataFrame(rows, dtype=str) if rows else pd.DataFrame(columns=done.columns, dtype=str)
    stats = pd.concat([done, new], ignore_index=True).fillna("")
    stats.to_csv(cache, index=False)
    stats["decode_ok"] = pd.to_numeric(stats.decode_ok, errors="coerce").fillna(0).astype(int)
    return stats


# ---------------- duplicates ----------------

def find_duplicates(df):
    """Pairs of images with pHash distance <= DUP_THRESHOLD, comparing each image
    with the other's original AND mirrored hash. Uses band bucketing: two 64-bit
    hashes within distance 5 must agree exactly on at least one of 6 bands."""
    ok = df[df.phash.str.len() == 16].reset_index(drop=True)
    h = [int(x, 16) for x in ok.phash]
    hm = [int(x, 16) for x in ok.phash_mirror]
    entries = [(i, v, 0) for i, v in enumerate(h)] + [(i, v, 1) for i, v in enumerate(hm)]
    best = {}
    for lo, hi in BANDS:
        mask = (1 << (hi - lo)) - 1
        buckets = defaultdict(list)
        for e in entries:
            buckets[(e[1] >> lo) & mask].append(e)
        for bucket in buckets.values():
            if len(bucket) < 2:
                continue
            for (i, vi, mi), (j, vj, mj) in combinations(bucket, 2):
                if i == j or (mi and mj):       # mirror-vs-mirror equals original-vs-original
                    continue
                d = (vi ^ vj).bit_count()
                if d <= DUP_THRESHOLD:
                    key = (min(i, j), max(i, j))
                    if key not in best or d < best[key][0]:
                        best[key] = (d, int(mi or mj))
    cols = ["source_dataset", "original_split", "category", "image_id", "md5"]
    pairs = []
    for (i, j), (d, mirrored) in best.items():
        a, b = ok.loc[i, cols], ok.loc[j, cols]
        kind = "exact_file" if a.md5 == b.md5 else "same_hash" if d == 0 else "near"
        pairs.append({"image_a": a.image_id, "image_b": b.image_id,
                      "dataset_a": a.source_dataset, "dataset_b": b.source_dataset,
                      "split_a": a.original_split, "split_b": b.original_split,
                      "category_a": a.category, "category_b": b.category,
                      "distance": d, "mirrored": mirrored, "kind": kind,
                      "cross_dataset": int(a.source_dataset != b.source_dataset),
                      "cross_split": int(a.source_dataset == b.source_dataset
                                         and a.original_split != b.original_split
                                         and "unsplit" not in (a.original_split, b.original_split)),
                      "category_mismatch": int(a.category != b.category)})
    return pd.DataFrame(pairs)


def clusters(pairs):
    """image_id -> group root, via union-find. Groups chain: A~B and B~C puts A, B, C together."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in zip(pairs.image_a, pairs.image_b):
        parent[find(a)] = find(b)
    return {x: find(x) for x in parent}


def duplicate_tier(pairs, df, name):
    """Per-dataset and cross-dataset tables plus group ids for one distance tier."""
    root = clusters(pairs)
    sizes = Counter(root.values())
    groups = pd.DataFrame({"image_id": list(root), "group": [root[x] for x in root]})
    groups["group_size"] = groups.group.map(sizes)
    groups = groups.merge(df[["image_id", "source_dataset", "original_split"]], on="image_id")
    # group ids are the root image id; renumber so the file holds no paths
    groups["group_id"] = groups.group.astype("category").cat.codes
    within = pairs[pairs.cross_dataset == 0]
    rows = []
    for d in DATASETS:
        g = groups[groups.source_dataset == d]
        w = within[within.dataset_a == d]
        split_mix = g.groupby("group").original_split.nunique()
        rows.append({"tier": name, "dataset": d, "images": int((df.source_dataset == d).sum()),
                     "images_with_a_duplicate": len(g),
                     "pct_images_with_a_duplicate": pct(len(g), (df.source_dataset == d).sum()),
                     "groups": g.group.nunique(), "pairs": len(w),
                     "pairs_crossing_splits": int(w.cross_split.sum()),
                     "groups_spanning_splits": int((split_mix > 1).sum()),
                     "mirror_pairs": int(w.mirrored.sum()),
                     "exact_file_pairs": int((w.kind == "exact_file").sum())})
    size_bins = pd.cut(pd.Series(list(sizes.values())), [1, 2, 5, 20, 100, 10 ** 9],
                       labels=["2", "3-5", "6-20", "21-100", "100+"])
    size_dist = size_bins.value_counts().reindex(["2", "3-5", "6-20", "21-100", "100+"]).rename(f"groups_{name}")
    return pd.DataFrame(rows), size_dist, groups.drop(columns="group")


# ---------------- plotting helpers ----------------

def style(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def grouped_bars(table, title, ylabel, path, pct=False):
    """table: index = x categories, columns = datasets (fixed color per dataset)."""
    cols = [d for d in DATASETS if d in table.columns]
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=SURFACE)
    width = 0.8 / max(1, len(cols))
    x = np.arange(len(table.index))
    for k, d in enumerate(cols):
        ax.bar(x + (k - (len(cols) - 1) / 2) * width, table[d].values, width * 0.92,
               color=DS_COLOR[d], label=d, edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(x, [str(i) for i in table.index])
    style(ax, title, ylabel=ylabel)
    if pct:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def hist_by_dataset(df, col, title, xlabel, path, bins, logx=False):
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=SURFACE)
    for d in DATASETS:
        v = pd.to_numeric(df.loc[df.source_dataset == d, col], errors="coerce").dropna()
        if len(v):
            w = np.full(len(v), 100 / len(v))
            ax.hist(v, bins=bins, weights=w, histtype="step", linewidth=2, color=DS_COLOR[d], label=d)
    if logx:
        ax.set_xscale("log")
    style(ax, title, xlabel=xlabel, ylabel="% of dataset")
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def review_sheet(paths_boxes, path, n=24, tile=300):
    """Grid of example images (for a person to review; never commit)."""
    items = list(paths_boxes.items())
    items = random.Random(0).sample(items, min(n, len(items)))
    if not items:
        return None
    cols = 6
    sheet = Image.new("RGB", (cols * tile, ((len(items) + cols - 1) // cols) * (tile + 16)), "white")
    ds = ImageDraw.Draw(sheet)
    for i, (p, bxs) in enumerate(items):
        with Image.open(p) as im:
            im = im.convert("RGB")
            dr = ImageDraw.Draw(im)
            W, H = im.size
            for x, y, w, h in bxs:
                dr.rectangle([(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H],
                             outline="yellow", width=max(2, W // 200))
            im.thumbnail((tile, tile))
        cx, cy = (i % cols) * tile, (i // cols) * (tile + 16)
        sheet.paste(im, (cx, cy))
        ds.text((cx + 2, cy + tile + 1), Path(p).name[:44], fill="black")
    sheet.save(path)
    return path


# ---------------- stage 2: analysis ----------------

def pct(a, b):
    return round(100 * a / b, 1) if b else 0.0


def md_table(t):
    """DataFrame -> GitHub markdown table (avoids needing the tabulate package)."""
    t = t.reset_index() if not isinstance(t.index, pd.RangeIndex) else t
    cols = [" / ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in t.itertuples(index=False)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest-dir", type=Path, default=OUT_BASE)
    ap.add_argument("--out", type=Path, default=OUT_BASE / "report")
    ap.add_argument("--limit", type=int, default=0, help="images per dataset (0 = all)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(a.manifest_dir / "manifest.csv", dtype={"category_id": str})
    box = pd.read_csv(a.manifest_dir / "boxes.csv")
    orphans = pd.read_csv(a.manifest_dir / "orphan_labels.csv")
    if a.limit:
        man = man.groupby("source_dataset", group_keys=False).apply(
            lambda g: g.iloc[np.linspace(0, len(g) - 1, min(a.limit, len(g))).astype(int)])
        box = box[box.image_id.isin(man.image_id)]

    stats = run_image_pass(man, a.out, a.workers)
    df = man.merge(stats, on="image_id", how="left")
    df["mean_brightness"] = pd.to_numeric(df.mean_brightness, errors="coerce")
    labeled = box[box.class_name.isin(["fire", "smoke"])].copy()
    groups = {"ALL": df, **{d: df[df.source_dataset == d] for d in DATASETS}}
    summary = {}

    # 1. counts, splits, negatives
    rows = []
    for g, sub in groups.items():
        for split, s in [("all", sub)] + list(sub.groupby("original_split")):
            rows.append({"dataset": g, "split": split, "images": len(s),
                         "negatives": int(s.is_negative.sum()), "pct_negative": pct(s.is_negative.sum(), len(s)),
                         "hard_negatives_cloud_glare": int(((s.is_negative == 1) & (s.num_ignored_boxes > 0)).sum())})
    t1 = pd.DataFrame(rows)
    t1.to_csv(a.out / "01_counts_splits.csv", index=False)

    # 2. class balance
    rows = []
    for g, sub in groups.items():
        b = labeled if g == "ALL" else labeled[labeled.source_dataset == g]
        cats = sub.category.value_counts()
        rows.append({"dataset": g, "images": len(sub),
                     **{f"img_{c}": int(cats.get(c, 0)) for c in ["fire", "smoke", "fire_and_smoke", "neither"]},
                     **{f"pct_{c}": pct(cats.get(c, 0), len(sub)) for c in ["fire", "smoke", "fire_and_smoke", "neither"]},
                     "fire_boxes": int((b.class_name == "fire").sum()),
                     "smoke_boxes": int((b.class_name == "smoke").sum()),
                     "ignored_boxes": int((box.class_name == "ignored").sum() if g == "ALL"
                                          else ((box.class_name == "ignored") & (box.source_dataset == g)).sum())})
    t2 = pd.DataFrame(rows)
    t2.to_csv(a.out / "02_class_balance.csv", index=False)
    cat_pct = t2.set_index("dataset").loc[[d for d in DATASETS if d in t2.dataset.values],
                                          ["pct_fire", "pct_smoke", "pct_fire_and_smoke", "pct_neither"]].T
    cat_pct.index = ["fire", "smoke", "fire + smoke", "neither"]
    grouped_bars(cat_pct, "Image categories by dataset", "% of images", a.out / "fig_categories.png", pct=True)

    # 3. boxes per image (fire + smoke boxes only)
    df["n_boxes"] = df.num_fire_boxes + df.num_smoke_boxes
    bins = [0, 1, 2, 3, 4, 5, 10, 10 ** 6]
    labels = ["0", "1", "2", "3", "4", "5-9", "10+"]
    df["n_boxes_bin"] = pd.cut(df.n_boxes, bins=bins, labels=labels, right=False)
    t3 = pd.crosstab(df.n_boxes_bin, df.source_dataset, normalize="columns").mul(100).round(1)
    t3.to_csv(a.out / "03_boxes_per_image_pct.csv")
    grouped_bars(t3, "Fire/smoke boxes per image", "% of images", a.out / "fig_boxes_per_image.png", pct=True)
    summary["boxes_per_image_max"] = {d: int(df.loc[df.source_dataset == d, "n_boxes"].max()) for d in DATASETS
                                      if (df.source_dataset == d).any()}

    # 4. box size (COCO, pixels) and aspect ratio
    labeled["coco_size"] = np.where(labeled.area_px < COCO_SMALL, "small",
                                    np.where(labeled.area_px < COCO_MEDIUM, "medium", "large"))
    labeled["aspect"] = labeled.width_px / labeled.height_px.replace(0, np.nan)
    t4 = (labeled.groupby(["source_dataset", "class_name", "coco_size"]).size()
          .unstack(fill_value=0).reindex(columns=["small", "medium", "large"], fill_value=0))
    t4_pct = t4.div(t4.sum(axis=1), axis=0).mul(100).round(1)
    t4.join(t4_pct, rsuffix="_pct").to_csv(a.out / "04_box_sizes_coco.csv")
    for cls in ["fire", "smoke"]:
        sub = t4_pct.xs(cls, level="class_name") if cls in t4_pct.index.get_level_values(1) else None
        if sub is not None:
            grouped_bars(sub.T, f"{cls.capitalize()} box size (COCO: small <32², medium <96² px)",
                         "% of boxes", a.out / f"fig_box_size_{cls}.png", pct=True)
    asp = labeled.groupby(["source_dataset", "class_name"]).aspect.describe(percentiles=[.05, .5, .95]).round(2)
    asp.to_csv(a.out / "04_box_aspect_ratio.csv")
    hist_by_dataset(labeled, "aspect", "Box aspect ratio (width / height)", "width / height (log scale)",
                    a.out / "fig_box_aspect.png", bins=np.logspace(-1.5, 1.5, 40), logx=True)

    # 5. image resolution and aspect
    df["megapixels"] = df.width * df.height / 1e6
    df["img_aspect"] = df.width / df.height
    res = (df.assign(res=df.width.astype("Int64").astype(str) + "x" + df.height.astype("Int64").astype(str))
             .groupby(["source_dataset", "res"]).size().rename("images").reset_index()
             .sort_values(["source_dataset", "images"], ascending=[True, False]))
    res.groupby("source_dataset").head(10).to_csv(a.out / "05_top_resolutions.csv", index=False)
    df.groupby("source_dataset")[["width", "height", "megapixels", "img_aspect"]].describe(
        percentiles=[.05, .5, .95]).round(2).T.to_csv(a.out / "05_resolution_stats.csv")
    hist_by_dataset(df, "megapixels", "Image size", "megapixels (log scale)", a.out / "fig_resolution.png",
                    bins=np.logspace(-2, 1.7, 40), logx=True)

    # 6. label problems
    rows = []
    for d in DATASETS:
        s, b = df[df.source_dataset == d], box[box.source_dataset == d]
        rows.append({"dataset": d, "images": len(s),
                     "missing_label": int((s.status == "missing_label").sum()),
                     "label_parse_error": int(s.status.str.startswith("label_parse_error").sum()),
                     "unreadable_header": int(s.status.str.startswith("unreadable_image").sum()),
                     "corrupt_decode": int((s.decode_ok != 1).sum()),
                     "orphan_labels": int((orphans.source_dataset == d).sum()),
                     "boxes": len(b), "out_of_bounds": int(b.out_of_bounds.sum()),
                     "zero_area": int(b.zero_area.sum()), "from_polygon": int(b.from_polygon.sum()),
                     "tiny_lt_4px": int(((b.width_px < 4) | (b.height_px < 4)).sum()),
                     "unmapped_boxes": int((b.class_name == "unmapped").sum())})
    t6 = pd.DataFrame(rows)
    t6.to_csv(a.out / "06_label_problems.csv", index=False)
    df.loc[df.decode_ok != 1, ["image_id", "decode_error"]].to_csv(a.out / "06_corrupt_images.csv", index=False)
    box[(box.out_of_bounds == 1) | (box.zero_area == 1)].to_csv(a.out / "06_bad_boxes.csv", index=False)

    # 7. brightness
    df["dark"] = df.mean_brightness < DARK_BRIGHTNESS
    t7 = df.groupby("source_dataset").agg(images=("image_id", "size"),
                                          median_brightness=("mean_brightness", lambda x: round(x.median(), 1)),
                                          pct_dark=("dark", lambda x: round(100 * x.mean(), 1)))
    t7.to_csv(a.out / "07_brightness.csv")
    hist_by_dataset(df, "mean_brightness", f"Mean image brightness (dark = below {DARK_BRIGHTNESS})",
                    "mean gray level (0 = black, 255 = white)", a.out / "fig_brightness.png",
                    bins=np.linspace(0, 255, 52))

    # 8. duplicates
    pairs = find_duplicates(df)
    pairs.to_csv(a.out / "08_duplicate_pairs.csv", index=False)
    # Two tiers: strict (distance <= 2 or identical file) = true copies; loose (<= 5)
    # also catches same-scene frames from fixed cameras and video.
    t8, t8x, t8d, t8s = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if len(pairs):
        tiers = {"strict_le2": pairs[pairs.distance <= 2], "loose_le5": pairs}
        tabs, dists = [], []
        for name, p in tiers.items():
            tab, size_dist, groups = duplicate_tier(p, df, name)
            tabs.append(tab)
            dists.append(size_dist)
            groups.to_csv(a.out / f"08_duplicate_groups_{name}.csv", index=False)
        t8 = pd.concat(tabs, ignore_index=True)
        t8.to_csv(a.out / "08_duplicates_within.csv", index=False)
        t8s = pd.concat(dists, axis=1).fillna(0).astype(int)
        t8s.index.name = "group_size"
        t8s.to_csv(a.out / "08_duplicate_group_sizes.csv")
        t8d = pd.crosstab(pairs.distance, pairs.cross_dataset.map({0: "within_dataset", 1: "across_datasets"}))
        t8d.to_csv(a.out / "08_distance_distribution.csv")
        cross = pairs[pairs.cross_dataset == 1]
        if len(cross):
            t8x = (cross.assign(pair=[" vs ".join(sorted(p)) for p in zip(cross.dataset_a, cross.dataset_b)],
                                strict=(cross.distance <= 2).astype(int))
                   .groupby("pair").agg(pairs_le5=("image_a", "size"), pairs_le2=("strict", "sum"),
                                        exact_file=("kind", lambda k: int((k == "exact_file").sum())),
                                        mirrored=("mirrored", "sum"), category_mismatch=("category_mismatch", "sum")))
            t8x.to_csv(a.out / "08_duplicates_across.csv")
            cm = cross.groupby(["dataset_a", "category_a", "dataset_b", "category_b"]).size().rename("pairs")
            cm.reset_index().to_csv(a.out / "08_across_category_pairs.csv", index=False)
        plot = t8.pivot(index="tier", columns="dataset", values="pct_images_with_a_duplicate")
        plot.index = ["strict (distance ≤2)" if i == "strict_le2" else "loose (distance ≤5)" for i in plot.index]
        grouped_bars(plot.iloc[::-1], "Images that have at least one duplicate", "% of images",
                     a.out / "fig_duplicates.png", pct=True)

    # Hard-negative review: cloud/glare-only images, and any that duplicate a fire/smoke image.
    hn = df[(df.is_negative == 1) & (df.num_ignored_boxes > 0)]
    ign = box[box.class_name == "ignored"]
    path_of = dict(zip(df.image_id, df.image_path))
    hn_boxes = defaultdict(list)
    for r in ign[ign.image_id.isin(hn.image_id)].itertuples():
        hn_boxes[path_of[r.image_id]].append((r.x_center, r.y_center, r.width, r.height))
    sheet = review_sheet(hn_boxes, a.out / "review_cloud_glare_negatives.png")
    if len(pairs):
        hn_ids = set(hn.image_id)
        flag = pairs[(pairs.image_a.isin(hn_ids) & ~pairs.category_b.isin(["neither"])) |
                     (pairs.image_b.isin(hn_ids) & ~pairs.category_a.isin(["neither"]))]
        flag.to_csv(a.out / "08_hard_negatives_duplicating_labeled_images.csv", index=False)
        summary["hard_negatives_duplicating_labeled_images"] = len(flag)
        summary["hard_negatives_duplicating_labeled_images_strict_le2"] = int((flag.distance <= 2).sum())

    # summary.md: every table in one readable file (numbers only, no paths)
    with open(a.out / "summary.md", "w") as f:
        f.write("# Data-quality summary (auto-generated)\n\n")
        for title, t in [("1. Counts, splits, negatives", t1), ("2. Class balance", t2),
                         ("3. Boxes per image (% of images)", t3), ("4. Box sizes, COCO (count and %)", t4.join(t4_pct, rsuffix="_pct")),
                         ("4b. Box aspect ratio", asp), ("6. Label problems", t6), ("7. Brightness", t7),
                         ("8a. Duplicates within datasets (strict and loose tiers)", t8),
                         ("8b. Duplicate group sizes", t8s), ("8c. pHash distance of duplicate pairs", t8d),
                         ("8d. Duplicates across datasets", t8x)]:
            f.write(f"## {title}\n\n{md_table(t) if len(t) else '_none_'}\n\n")
        f.write(f"## Other\n\n```\n{json.dumps(summary, indent=2)}\n```\n")
    print(open(a.out / "summary.md").read())
    if sheet:
        print("Review sheet (view in the portal, never commit):", sheet)


if __name__ == "__main__":
    main()
