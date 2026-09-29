"""Local D-Fire audit: python analyze_dfire.py [dataset] [--output report]."""
import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from PIL import Image

CLASSES = {0: "smoke", 1: "fire"}
CATEGORIES = ["fire", "fire + smoke", "smoke", "none"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def size_bin(area, small, large):
    return "small" if area < small else "medium" if area < large else "large"


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(root, output, small=1.0, large=10.0):
    root, output = root.resolve(), output.resolve()
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTENSIONS
                   and "images" in p.relative_to(root).parts)
    if not paths:
        raise ValueError(f"No images found under an images/ directory in {root}")
    output.mkdir(parents=True, exist_ok=True)
    images, boxes, issues = [], [], []
    matched_labels = set()
    for index, path in enumerate(paths, 1):
        relative = path.relative_to(root)
        parts = list(relative.parts)
        position = parts.index("images")
        # Both train/images/file.jpg and images/train/file.jpg are supported.
        split_parts = parts[:position] + parts[position + 1:-1]
        split = next((s for s in reversed(split_parts)
                      if s.lower() in {"train", "test", "val", "valid", "validation"}), "unsplit")
        parts[position] = "labels"
        label = root.joinpath(*parts).with_suffix(".txt")
        matched_labels.add(label)
        row = {"image": relative.as_posix(), "split": split, "category": "invalid",
               "width_px": "", "height_px": "", "fire_boxes": 0, "smoke_boxes": 0,
               "largest_fire_area_pct": 0, "largest_smoke_area_pct": 0,
               "fire_size": "none", "smoke_size": "none", "status": "ok"}
        image_boxes = []
        try:
            with Image.open(path) as im:
                width, height = im.size
            row.update(width_px=width, height_px=height)
            if not label.is_file():
                raise ValueError("Missing label (not assumed to be a negative image)")
            for line_number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"Label line {line_number}: expected 5 YOLO fields")
                class_id = int(fields[0])
                x, y, w, h = map(float, fields[1:])
                if class_id not in CLASSES or not all(math.isfinite(v) for v in (x, y, w, h)):
                    raise ValueError(f"Label line {line_number}: invalid class or non-finite coordinate")
                if not (0 <= x <= 1 and 0 <= y <= 1 and w >= 0 and h >= 0):
                    raise ValueError(f"Label line {line_number}: invalid center or negative dimensions")
                if min(x - w / 2, y - h / 2) < -0.001 or max(x + w / 2, y + h / 2) > 1.001:
                    issues.append({"image": relative.as_posix(), "issue": f"Label line {line_number}: box extends beyond image; original dimensions retained"})
                if w == 0 or h == 0:
                    issues.append({"image": relative.as_posix(), "issue": f"Label line {line_number}: zero-area box; retained for class counts, size invalid"})
                area = w * h * 100
                image_boxes.append({"image": relative.as_posix(), "split": split,
                                    "class": CLASSES[class_id], "box_number": line_number,
                                    "x_center": x, "y_center": y, "width_norm": w, "height_norm": h,
                                    "width_px": w * width, "height_px": h * height,
                                    "area_px": w * width * h * height, "area_pct": area,
                                    "size": size_bin(area, small, large) if area > 0 else "invalid"})
            present = {b["class"] for b in image_boxes}
            row["category"] = "fire + smoke" if len(present) == 2 else next(iter(present), "none")
            for name in CLASSES.values():
                areas = [b["area_pct"] for b in image_boxes if b["class"] == name]
                row[f"{name}_boxes"] = len(areas)
                row[f"largest_{name}_area_pct"] = max(areas, default=0)
                row[f"{name}_size"] = (size_bin(max(areas), small, large) if max(areas) > 0 else "invalid") if areas else "none"
            boxes.extend(image_boxes)
        except (ValueError, OSError) as exc:
            row["status"] = str(exc)
            issues.append({"image": relative.as_posix(), "issue": str(exc)})
        images.append(row)
        if index % 2000 == 0:
            print(f"Processed {index:,}/{len(paths):,} images", flush=True)
    for label in root.rglob("*.txt"):
        if "labels" in label.relative_to(root).parts and label not in matched_labels:
            issues.append({"image": label.relative_to(root).as_posix(), "issue": "Orphan label: no matching image"})

    valid = [r for r in images if r["status"] == "ok"]
    splits = sorted({r["split"] for r in images})
    category_rows, size_rows, group_rows = [], [], []
    for split in ["ALL"] + splits:
        subset = [r for r in valid if split == "ALL" or r["split"] == split]
        counts = Counter(r["category"] for r in subset)
        for category in CATEGORIES:
            category_rows.append({"split": split, "category": category, "images": counts[category],
                                  "percent": round(100 * counts[category] / len(subset), 2) if subset else 0})
        for name in CLASSES.values():
            objects = [b for b in boxes if b["class"] == name and (split == "ALL" or b["split"] == split)]
            for size in ["small", "medium", "large", "invalid"]:
                areas = [b["area_pct"] for b in objects if b["size"] == size]
                size_rows.append({"split": split, "class": name, "size": size, "boxes": len(areas),
                                  "percent": round(100 * len(areas) / len(objects), 2) if objects else 0,
                                  "median_area_pct": median(areas) if areas else 0})
        groups = Counter((r["category"], r["fire_size"], r["smoke_size"]) for r in subset)
        for (category, fire, smoke), count in sorted(groups.items()):
            group_rows.append({"split": split, "category": category, "fire_size": fire,
                               "smoke_size": smoke, "images": count})
    summary = {"dataset": str(root), "total_images": len(images), "valid_images": len(valid),
               "invalid_images": len(images) - len(valid), "boxes": len(boxes), "issues": len(issues),
               "categories": dict(Counter(r["category"] for r in valid)),
               "boxes_by_class": dict(Counter(b["class"] for b in boxes)),
               "size_thresholds_area_pct": {"small_below": small, "large_at_least": large}}
    write_csv(output / "images.csv", images, list(images[0]))
    write_csv(output / "boxes.csv", boxes, ["image", "split", "class", "box_number", "x_center", "y_center",
              "width_norm", "height_norm", "width_px", "height_px", "area_px", "area_pct", "size"])
    for name, rows in [("categories", category_rows), ("sizes", size_rows), ("balance_groups", group_rows)]:
        write_csv(output / f"{name}.csv", rows, list(rows[0]) if rows else ["split", "category", "fire_size", "smoke_size", "images"])
    write_csv(output / "issues.csv", issues, ["image", "issue"])
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_report(output, root, summary, category_rows, size_rows, group_rows, valid, boxes, small, large)
    print(json.dumps(summary, indent=2))
    print(f"\nOpen: {output / 'report.html'}")
    return summary


def make_report(output, root, summary, categories, sizes, groups, images, boxes, small, large):
    esc = lambda value: html.escape(str(value), quote=True)

    def table(rows):
        if not rows:
            return '<p>No valid records.</p>'
        headers = list(rows[0])
        return '<table><thead><tr>' + ''.join(f'<th>{esc(h.replace("_", " "))}</th>' for h in headers) + \
            '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{esc(round(r[h], 3) if isinstance(r[h], float) else r[h])}</td>'
            for h in headers) + '</tr>' for r in rows) + '</tbody></table>'

    charts = ''.join(f'<div class="barrow"><span>{esc(r["category"])}</span><div class="track"><div class="bar" style="width:{r["percent"]}%"></div></div><b>{r["images"]:,} ({r["percent"]}%)</b></div>'
                     for r in categories if r["split"] == "ALL")
    by_image = defaultdict(list)
    for box in boxes:
        by_image[box["image"]].append(box)
    gallery = []
    shown = Counter()
    for row in images:
        key = (row["category"], row["fire_size"], row["smoke_size"])
        if shown[key] >= 1:
            continue
        shown[key] += 1
        source = root / row["image"]
        thumb_dir = output / "thumbnails"
        thumb_dir.mkdir(exist_ok=True)
        thumb_name = f"{len(gallery):03}.jpg"
        try:
            with Image.open(source) as im:
                im = im.convert("RGB")
                im.thumbnail((480, 320))
                im.save(thumb_dir / thumb_name, quality=80)
        except OSError:
            continue
        overlays = ''.join(f'<div class="box {b["class"]}" style="left:{(b["x_center"]-b["width_norm"]/2)*100}%;top:{(b["y_center"]-b["height_norm"]/2)*100}%;width:{b["width_norm"]*100}%;height:{b["height_norm"]*100}%" title="{b["class"]}: {b["area_pct"]:.2f}% ({b["size"]})"></div>' for b in by_image[row["image"]])
        gallery.append(f'<article data-category="{esc(row["category"])}"><a href="{esc(source.as_uri())}" target="_blank"><div class="photo"><img loading="lazy" src="thumbnails/{thumb_name}">{overlays}</div></a><b>{esc(row["category"])}</b><p>Fire: {row["fire_size"]} ({row["largest_fire_area_pct"]:.2f}%)<br>Smoke: {row["smoke_size"]} ({row["largest_smoke_area_pct"]:.2f}%)</p><small>{esc(row["image"])}</small></article>')
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>D-Fire dataset report</title>
<style>body{{font:16px system-ui;background:#f2f5f7;color:#182333;max-width:1150px;margin:40px auto;padding:0 24px}}h1{{font-size:38px}}h2{{margin-top:38px}}.cards,.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}}.card,article{{background:white;border:1px solid #dce3ea;border-radius:10px;padding:20px}}.card b{{display:block;font-size:30px}}table{{width:100%;border-collapse:collapse;background:white;margin:16px 0;font-size:14px}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #e0e6eb}}th{{background:#e6edf2}}.scroll{{overflow:auto}}.barrow{{display:flex;gap:16px;align-items:center;margin:14px 0}}.barrow span{{width:115px}}.barrow b{{width:175px}}.track{{flex:1;background:#dce5eb;height:25px;border-radius:4px;overflow:hidden}}.bar{{height:100%;background:#db662f}}.photo{{position:relative;line-height:0}}img{{width:100%;height:auto}}.box{{position:absolute;border:2px solid;box-sizing:border-box}}.fire{{border-color:#ff491d}}.smoke{{border-color:#00bfff}}small{{overflow-wrap:anywhere}}select{{padding:10px;font:inherit}}a{{color:#135da1}}</style>
<h1>D-Fire dataset report</h1><p>Image composition and object sizes from existing YOLO annotations (0 = smoke, 1 = fire).</p>
<div class="cards"><div class="card"><b>{summary['total_images']:,}</b>Images scanned</div><div class="card"><b>{summary['boxes']:,}</b>Annotated boxes</div><div class="card"><b>{summary['invalid_images']:,}</b>Images excluded for errors</div><div class="card"><b>{summary['issues']:,}</b>Annotation issues flagged</div></div>
<h2>Image categories</h2>{charts}<details><summary>Category counts by dataset split</summary><div class="scroll">{table(categories)}</div></details>
<h2>How big are the objects?</h2><p>Area = normalized box width × height × 100. Small: &lt; {small}%; medium: {small}% to &lt; {large}%; large: ≥ {large}% of the image. These adjustable prototype thresholds describe apparent box size, not physical fire size or severity.</p>
<div class="scroll">{table([r for r in sizes if r['split']=='ALL'])}</div><details><summary>Object sizes by dataset split</summary><div class="scroll">{table([r for r in sizes if r['split']!='ALL'])}</div></details>
<h2>Groups for balancing</h2><p>Each image belongs to one group using its category and largest box of each class. Multiple overlapping boxes are counted separately in object statistics; their areas are not summed. Use the training rows to plan sampling; preserve validation and test distributions.</p><details><summary>Show category × fire size × smoke size counts</summary><div class="scroll">{table(groups)}</div></details>
<h2>Examples with bounding boxes</h2><p>One example per observed size/category group. Fire = orange; smoke = blue. Hover a box for its area, or click an image for the original.</p><label>Category <select id="filter"><option>all</option>{''.join(f'<option>{esc(c)}</option>' for c in CATEGORIES)}</select></label><div class="gallery">{''.join(gallery)}</div>
<h2>Exported data</h2><p>{' · '.join(f'<a href="{name}.csv">{name}.csv</a>' for name in ['images','boxes','categories','sizes','balance_groups','issues'])} · <a href="summary.json">summary.json</a></p><p>Empty label files mean none. Missing or unparseable annotations are excluded. Zero-area boxes retain their class but have invalid size; boxes crossing image edges retain original dimensions. Both are flagged in issues.csv. Image headers are read for dimensions; the full dataset is not decoded for corruption checks.</p><p>Source: <a href="https://github.com/gaia-solutions-on-demand/DFireDataset">D-Fire repository</a>.</p>
<script>document.querySelector('#filter').onchange = function() {{document.querySelectorAll('article').forEach(el => el.hidden = this.value !== 'all' && el.dataset.category !== this.value);}};</script></html>'''
    (output / "report.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, nargs="?", default=Path(__file__).resolve().parent.parent / "data" / "DFire")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "report")
    parser.add_argument("--small", type=float, default=1, help="Small box area cutoff, percent (default 1)")
    parser.add_argument("--large", type=float, default=10, help="Large box area cutoff, percent (default 10)")
    args = parser.parse_args()
    if not (0 < args.small < args.large <= 100):
        parser.error("Require 0 < --small < --large <= 100")
    analyze(args.dataset, args.output, args.small, args.large)
