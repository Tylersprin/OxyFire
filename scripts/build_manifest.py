#!/usr/bin/env python3
"""Build a combined index of the public fire/smoke datasets. No images are copied.

Outputs (in --out):
  manifest.csv          one row per image
  boxes.csv             one row per bounding box, normalized to the image (0-1)
  unmapped_classes.csv  source class names that did not map to fire/smoke
  orphan_labels.csv     label files with no matching image
  class_maps.csv        how each dataset's classes were mapped

Target box classes: fire = 0, smoke = 1. Every image also gets a `category`
(fire / smoke / fire_and_smoke / neither) for reporting and balancing.
Look-alike classes (IGNORE_NAMES) are kept in boxes.csv as "ignored"; an image whose
only boxes are ignored counts as a negative (hard negative).

Usage:
  python scripts/build_manifest.py --limit 50 --out $SCRATCH/oxyfire_data/outputs/test_manifest
  python scripts/build_manifest.py --workers 16        # full run, inside a Slurm job
"""
import argparse
import csv
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_NAMES = {"train": "train", "training": "train", "val": "val", "valid": "val",
               "validation": "val", "test": "test", "testing": "test"}
TARGET = {0: "fire", 1: "smoke"}
# Image-level category ids (team convention). Not YOLO box classes.
CATEGORY_IDS = {"fire": 0, "smoke": 1, "fire_and_smoke": 2, "neither": 3}
# Normalized source class name -> target id. Anything else is reported as unmapped.
NAME_TO_TARGET = {"fire": 0, "flame": 0, "flames": 0, "smoke": 1}
# Look-alike classes (not fire or smoke). Their boxes are kept in boxes.csv as
# "ignored" but do not count as objects, so images with only these are negatives.
IGNORE_NAMES = {"cloudglare"}

# format: "yolo" or "voc". class_names: fallback id -> name for YOLO sets whose
# data.yaml is missing (D-Fire's official order is smoke, fire).
DATASETS = {
    "dfire":        {"format": "yolo", "class_names": {0: "smoke", 1: "fire"}},
    "aiformankind": {"format": "voc"},
    # No data.yaml ships with azimjaan21 and labels use ids 0/1/2 although the author
    # lists two classes. Meaning inferred 2026-09-29 from box color stats
    # (scripts/class_check.py) and a visual spot check of 16 images per id:
    # id 0 boxes are clouds / glare look-alikes, id 1 fire, id 2 smoke.
    "azimjaan21":   {"format": "yolo", "class_names": {0: "cloud_glare", 1: "fire", 2: "smoke"}},
}

MANIFEST_FIELDS = ["image_id", "source_dataset", "original_split", "image_path", "label_path",
                   "label_format", "width", "height", "num_fire_boxes", "num_smoke_boxes",
                   "num_ignored_boxes", "num_unmapped_boxes", "is_negative", "category",
                   "category_id", "status"]
BOX_FIELDS = ["image_id", "source_dataset", "original_split", "box_index", "class_id", "class_name",
              "source_class", "x_center", "y_center", "width", "height", "area_norm",
              "width_px", "height_px", "area_px", "out_of_bounds", "zero_area", "from_polygon"]


def norm_name(name):
    return re.sub(r"[^a-z]", "", str(name).lower())


def read_yaml_names(path):
    """Read `names:` from a YOLO data.yaml without needing PyYAML.
    Handles `names: [a, b]`, a block list (`- a`), and a block dict (`0: a`)."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^names\s*:\s*(.*)$", line.strip())
        if not m:
            continue
        rest = m.group(1).split("#")[0].strip()
        if rest.startswith("["):
            items = [s.strip().strip("'\"") for s in rest.strip("[]").split(",") if s.strip()]
            return dict(enumerate(items))
        names = {}
        for sub in lines[i + 1:]:
            s = sub.strip()
            if not s or s.startswith("#"):
                continue
            if not sub.startswith((" ", "\t", "-")):
                break
            if s.startswith("-"):
                names[len(names)] = s[1:].strip().strip("'\"")
            elif ":" in s:
                k, v = s.split(":", 1)
                names[int(k.strip())] = v.strip().strip("'\"")
        return names
    return {}


def find_split(parts):
    for p in reversed(parts):
        if p.lower() in SPLIT_NAMES:
            return SPLIT_NAMES[p.lower()]
    return "unsplit"


def yolo_label_for(img):
    """images/.../x.jpg -> labels/.../x.txt (last 'images' component replaced)."""
    parts = list(img.parts)
    idx = max(i for i, p in enumerate(parts) if p.lower() == "images")
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def discover(name, root, cfg, limit):
    """Return (tasks, class_map rows, orphan label paths) for one dataset."""
    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file())
    tasks, id_to_name, orphans = [], {}, []

    if cfg["format"] == "yolo":
        images = [p for p in images if any(q.lower() == "images" for q in p.relative_to(root).parts)]
        yamls = sorted(root.rglob("*.yaml")) + sorted(root.rglob("*.yml"))
        for y in yamls:
            id_to_name = read_yaml_names(y)
            if id_to_name:
                print(f"[{name}] class names from {y.relative_to(root)}: {id_to_name}")
                break
        if not id_to_name:
            if not cfg.get("class_names"):
                sys.exit(f"[{name}] ERROR: no data.yaml with names and no fallback class map")
            id_to_name = cfg["class_names"]
            print(f"[{name}] no data.yaml found; using built-in class names {id_to_name}")
        labels = {yolo_label_for(p) for p in images}
        orphans = [t for t in root.rglob("*.txt")
                   if any(q.lower() == "labels" for q in t.relative_to(root).parts) and t not in labels]
        pairs = [(p, yolo_label_for(p)) for p in images]
    else:  # VOC: XML next to the image or in an Annotations/ folder, matched by file stem
        xmls = {x.stem: x for x in root.rglob("*.xml")}
        stems = {p.stem for p in images}
        orphans = [x for s, x in xmls.items() if s not in stems]
        pairs = [(p, xmls.get(p.stem)) for p in images]

    if limit and len(pairs) > limit:  # evenly spaced, so the sample covers every split
        step = len(pairs) / limit
        pairs = [pairs[int(i * step)] for i in range(limit)]
    for img, lbl in pairs:
        rel = img.relative_to(root)
        tasks.append({"dataset": name, "format": cfg["format"], "image": str(img),
                      "label": str(lbl) if lbl else "", "split": find_split(rel.parts[:-1]),
                      "image_id": f"{name}/{rel.as_posix()}", "id_to_name": id_to_name})
    return tasks, id_to_name, orphans


def to_target(source_class):
    """0 / 1 for fire / smoke, "ignore" for look-alike classes, None if unmapped."""
    name = norm_name(source_class)
    return "ignore" if name in IGNORE_NAMES else NAME_TO_TARGET.get(name)


def parse_yolo(path, id_to_name):
    """Yield (source_class, x, y, w, h, from_polygon). Raises ValueError on bad lines."""
    for n, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        f = line.split()
        if not f:
            continue
        cid = int(float(f[0]))
        vals = [float(v) for v in f[1:]]
        src = id_to_name.get(cid, f"id{cid}")
        if len(vals) == 4:
            yield (src, *vals, False)
        elif len(vals) >= 6 and len(vals) % 2 == 0:  # segmentation polygon -> enclosing box
            xs, ys = vals[0::2], vals[1::2]
            yield src, (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys), True
        else:
            raise ValueError(f"line {n}: {len(f)} fields")


def parse_voc(path):
    """Yield (source_class, x, y, w, h in pixels, False) and return XML size via attribute."""
    root = ET.parse(path).getroot()
    for obj in root.iter("object"):
        bb = obj.find("bndbox")
        x1, y1, x2, y2 = (float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        yield obj.findtext("name", ""), x1, y1, x2, y2, False


def process(t):
    row = {"image_id": t["image_id"], "source_dataset": t["dataset"], "original_split": t["split"],
           "image_path": t["image"], "label_path": t["label"], "label_format": t["format"],
           "width": "", "height": "", "num_fire_boxes": 0, "num_smoke_boxes": 0,
           "num_ignored_boxes": 0, "num_unmapped_boxes": 0, "is_negative": "", "category": "",
           "status": "ok"}
    boxes, unmapped = [], Counter()
    try:
        with Image.open(t["image"]) as im:  # reads the header only, not the pixels
            W, H = im.size
        row.update(width=W, height=H)
    except Exception as e:
        row["status"] = f"unreadable_image: {type(e).__name__}"
        return row, boxes, unmapped
    if not t["label"] or not os.path.exists(t["label"]):
        row["status"] = "missing_label"
        return row, boxes, unmapped
    try:
        if t["format"] == "yolo":
            raw = list(parse_yolo(t["label"], t["id_to_name"]))
        else:  # convert VOC pixel corners to normalized center/size
            raw = [(c, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H, p)
                   for c, x1, y1, x2, y2, p in parse_voc(t["label"])]
    except Exception as e:
        row["status"] = f"label_parse_error: {type(e).__name__}: {e}"
        return row, boxes, unmapped

    for i, (src, x, y, w, h, poly) in enumerate(raw):
        cid = to_target(src)
        # Unmapped and ignored boxes stay in boxes.csv (blank class_id) so they can be remapped.
        if cid is None:
            unmapped[src] += 1
            row["num_unmapped_boxes"] += 1
        elif cid == "ignore":
            row["num_ignored_boxes"] += 1
        else:
            row["num_fire_boxes" if cid == 0 else "num_smoke_boxes"] += 1
        oob = x - w / 2 < -1e-3 or y - h / 2 < -1e-3 or x + w / 2 > 1.001 or y + h / 2 > 1.001
        boxes.append({"image_id": t["image_id"], "source_dataset": t["dataset"],
                      "original_split": t["split"], "box_index": i,
                      "class_id": cid if cid in TARGET else "",
                      "class_name": TARGET.get(cid, "ignored" if cid == "ignore" else "unmapped"),
                      "source_class": src,
                      "x_center": round(x, 6), "y_center": round(y, 6), "width": round(w, 6),
                      "height": round(h, 6), "area_norm": round(w * h, 8),
                      "width_px": round(w * W, 2), "height_px": round(h * H, 2),
                      "area_px": round(w * W * h * H, 1), "out_of_bounds": int(oob),
                      "zero_area": int(w <= 0 or h <= 0), "from_polygon": int(poly)})
    fire, smoke = row["num_fire_boxes"] > 0, row["num_smoke_boxes"] > 0
    row["category"] = ("fire_and_smoke" if fire and smoke else "fire" if fire
                       else "smoke" if smoke
                       else "unmapped_only" if row["num_unmapped_boxes"] else "neither")
    row["category_id"] = CATEGORY_IDS.get(row["category"], "")  # blank for unmapped_only
    row["is_negative"] = int(not fire and not smoke and row["num_unmapped_boxes"] == 0)
    return row, boxes, unmapped


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path(os.environ["SCRATCH"]) / "oxyfire_data/raw")
    ap.add_argument("--out", type=Path, default=Path(os.environ["SCRATCH"]) / "oxyfire_data/outputs")
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--limit", type=int, default=0, help="images per dataset (0 = all)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    tasks, class_rows, orphan_rows = [], [], []
    for name in a.datasets:
        root = a.raw / name
        if not root.is_dir():
            print(f"[{name}] not found at {root}, skipping")
            continue
        t, id_to_name, orphans = discover(name, root, DATASETS[name], a.limit)
        print(f"[{name}] {len(t)} images queued, {len(orphans)} orphan labels")
        tasks += t
        orphan_rows += [{"source_dataset": name, "label_path": str(o)} for o in orphans]
        for cid, cname in sorted(id_to_name.items()):
            tgt = to_target(cname)
            class_rows.append({"source_dataset": name, "source_id": cid, "source_name": cname,
                               "target_id": tgt if tgt in TARGET else "",
                               "target_name": TARGET.get(tgt, "IGNORED (negative)" if tgt == "ignore"
                                                         else "UNMAPPED")})

    rows, boxes, unmapped = [], [], Counter()
    with Pool(a.workers) as pool:
        for i, (r, b, u) in enumerate(pool.imap(process, tasks, chunksize=64), 1):
            rows.append(r)
            boxes += b
            unmapped.update({(r["source_dataset"], k): v for k, v in u.items()})
            if i % 5000 == 0:
                print(f"  processed {i:,}/{len(tasks):,}", flush=True)

    write_csv(a.out / "manifest.csv", rows, MANIFEST_FIELDS)
    write_csv(a.out / "boxes.csv", boxes, BOX_FIELDS)
    write_csv(a.out / "orphan_labels.csv", orphan_rows, ["source_dataset", "label_path"])
    write_csv(a.out / "class_maps.csv", class_rows,
              ["source_dataset", "source_id", "source_name", "target_id", "target_name"])
    write_csv(a.out / "unmapped_classes.csv",
              [{"source_dataset": d, "source_class": c, "boxes": n} for (d, c), n in sorted(unmapped.items())],
              ["source_dataset", "source_class", "boxes"])

    print(f"\nWrote {len(rows):,} images, {len(boxes):,} boxes to {a.out}")
    for name in a.datasets:
        sub = [r for r in rows if r["source_dataset"] == name]
        if sub:
            print(f"  {name:13} images={len(sub):6,}  status={dict(Counter(r['status'].split(':')[0] for r in sub))}"
                  f"  category={dict(Counter(r['category'] for r in sub))}")
    if unmapped:
        print("UNMAPPED classes (not dropped silently; see unmapped_classes.csv):", dict(unmapped))


if __name__ == "__main__":
    main()
