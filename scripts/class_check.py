#!/usr/bin/env python3
"""Evidence for what azimjaan21's class ids mean, compared with D-Fire's known classes.

1. class_color_stats.csv: mean hue / saturation / brightness inside boxes, per class.
   Fire boxes tend to be bright, saturated and orange; smoke boxes gray and low in saturation.
2. sheet_<dataset>_<class>.png: a grid of example images with that class's boxes drawn,
   for a person to look at. Written to --out only; never commit these.

Reads manifest boxes.csv. Usage:
  python scripts/class_check.py --max-boxes 50            # quick test
  python scripts/class_check.py --workers 16              # full, inside a Slurm job
"""
import argparse
import csv
import os
import random
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

OUT_DEFAULT = Path(os.environ["SCRATCH"]) / "oxyfire_data/outputs"
# (dataset, source_class) groups to compare. D-Fire rows are the known reference.
# azimjaan21 source names come from build_manifest.py (ids 0/1/2 = cloud_glare/fire/smoke).
GROUPS = [("dfire", "fire"), ("dfire", "smoke"), ("aiformankind", "smoke"),
          ("azimjaan21", "cloud_glare"), ("azimjaan21", "fire"), ("azimjaan21", "smoke")]
COLORS = {"fire": "red", "smoke": "cyan", "cloud_glare": "yellow"}


def box_color(item):
    """Mean HSV (0-1 scale) of the pixels inside one box, plus % 'fire-like' pixels."""
    path, boxes = item
    try:
        with Image.open(path) as im:
            hsv = np.asarray(im.convert("RGB").convert("HSV"), dtype=np.float32) / 255.0
    except Exception:
        return []
    H, W = hsv.shape[:2]
    out = []
    for key, x, y, w, h in boxes:
        x1, x2 = int(max(0, (x - w / 2) * W)), int(min(W, (x + w / 2) * W))
        y1, y2 = int(max(0, (y - h / 2) * H)), int(min(H, (y + h / 2) * H))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        crop = hsv[y1:y2, x1:x2].reshape(-1, 3)
        hue, sat, val = crop[:, 0], crop[:, 1], crop[:, 2]
        # red-orange-yellow hue (0-60 degrees), saturated and bright
        firelike = ((hue < 60 / 360) | (hue > 345 / 360)) & (sat > 0.4) & (val > 0.5)
        out.append((key, float(sat.mean()), float(val.mean()), float(firelike.mean())))
    return out


def contact_sheet(dataset, cls, rows, out, n=16, tile=320):
    """Grid of n example images with this class's boxes drawn."""
    by_img = defaultdict(list)
    for r in rows:
        by_img[r["image_path"]].append(r)
    picks = random.Random(0).sample(sorted(by_img), min(n, len(by_img)))
    cols = 4
    sheet = Image.new("RGB", (cols * tile, ((len(picks) + cols - 1) // cols) * (tile + 18)), "white")
    draw_sheet = ImageDraw.Draw(sheet)
    for i, path in enumerate(picks):
        with Image.open(path) as im:
            im = im.convert("RGB")
            d = ImageDraw.Draw(im)
            W, H = im.size
            for b in by_img[path]:
                x, y, w, h = (float(b[k]) for k in ("x_center", "y_center", "width", "height"))
                d.rectangle([(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H],
                            outline=COLORS.get(cls, "red"), width=max(2, W // 200))
            im.thumbnail((tile, tile))
        cx, cy = (i % cols) * tile, (i // cols) * (tile + 18)
        sheet.paste(im, (cx, cy))
        draw_sheet.text((cx + 2, cy + tile + 2), Path(path).name[:48], fill="black")
    dest = out / f"sheet_{dataset}_{cls}.png"
    sheet.save(dest)
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT / "class_check")
    ap.add_argument("--max-boxes", type=int, default=3000, help="boxes sampled per class")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    paths = {}
    with open(a.manifest_dir / "manifest.csv", newline="") as f:
        for r in csv.DictReader(f):
            paths[r["image_id"]] = r["image_path"]
    groups = defaultdict(list)
    with open(a.manifest_dir / "boxes.csv", newline="") as f:
        for r in csv.DictReader(f):
            key = (r["source_dataset"], r["source_class"])
            if key in GROUPS:
                r["image_path"] = paths[r["image_id"]]
                groups[key].append(r)

    rng = random.Random(0)
    per_image = defaultdict(list)
    for key in GROUPS:
        rows = groups.get(key, [])
        for r in rng.sample(rows, min(a.max_boxes, len(rows))):
            per_image[r["image_path"]].append((key, *(float(r[k]) for k in ("x_center", "y_center", "width", "height"))))
        print(f"{key}: {len(rows):,} boxes total")

    stats = defaultdict(list)
    with Pool(a.workers) as pool:
        for res in pool.imap_unordered(box_color, per_image.items(), chunksize=16):
            for key, sat, val, fire in res:
                stats[key].append((sat, val, fire))

    out_rows = []
    for key in GROUPS:
        rows, s = groups.get(key, []), np.array(stats.get(key, []))
        if not len(s):
            continue
        areas = [float(r["area_norm"]) for r in rows]
        out_rows.append({"dataset": key[0], "source_class": key[1], "boxes_total": len(rows),
                         "boxes_measured": len(s), "mean_saturation": round(s[:, 0].mean(), 3),
                         "mean_brightness": round(s[:, 1].mean(), 3),
                         "pct_firelike_pixels": round(100 * s[:, 2].mean(), 1),
                         "median_box_area_pct": round(100 * float(np.median(areas)), 2)})
    with open(a.out / "class_color_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n{'group':28} {'boxes':>7} {'sat':>6} {'bright':>7} {'%firelike':>10} {'med area%':>10}")
    for r in out_rows:
        print(f"{r['dataset'] + '/' + r['source_class']:28} {r['boxes_total']:7,} {r['mean_saturation']:6} "
              f"{r['mean_brightness']:7} {r['pct_firelike_pixels']:10} {r['median_box_area_pct']:10}")

    for key in GROUPS:
        if groups.get(key):
            print("sheet:", contact_sheet(*key, groups[key], a.out))


if __name__ == "__main__":
    main()
