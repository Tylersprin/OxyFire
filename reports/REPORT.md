# Public fire/smoke datasets: data-quality report

Generated 2026-09-30 from the combined manifest (41,279 images, 50,775 boxes). Updated 2026-10-08
with datasets that are not merged yet (see the last sections).
Scripts: `scripts/build_manifest.py`, `scripts/class_check.py`, `scripts/quality_stats.py`
(Slurm: `jobs/manifest.slurm`, `jobs/quality.slurm`). Full tables and plots are in
`$SCRATCH/oxyfire_data/outputs/report/` on Grace. They are not in the repo because they contain file paths.

Box classes are unified to **fire = 0, smoke = 1**. Each image also gets a category:
0 fire, 1 smoke, 2 fire and smoke, 3 neither.

## Bottom line

| Dataset | Recommendation | Why |
|---|---|---|
| **D-Fire** | **Keep. Core training set.** Clean boxes and re-split. | Largest, realistic mix of fire, smoke, both and background, 18% dark images. Its own train/val/test splits leak near-duplicate frames. |
| **AI For Mankind v2** | **Keep, but don't overweight it.** In the merged set (team decision). | Only source of distant, small wildfire smoke. But it's about 300 distinct scenes rather than 2,191 images, daytime only, and has no negatives. |
| **azimjaan21** | **Clean before use.** Use its cloud/glare negatives and smoke images now. Hold its fire-only images until they're relabeled. | Undocumented class ids (resolved below). Fire images are missing smoke labels. It has 715 exact duplicate files and augmented copies that leak across its splits. Its 8,822 cloud/glare negatives are the most useful false-alarm data we have. |

None of the datasets contain flare stacks, which remain the main false-positive risk for industrial sites. That gap needs its own data.

## Summary per dataset

| | D-Fire | AI For Mankind | azimjaan21 | Combined |
|---|---|---|---|---|
| Images | 21,527 | 2,191 | 17,561 | 41,279 |
| Label format | YOLO | Pascal VOC | YOLO (some polygons) | – |
| Original splits (train / val / test) | 14,122 / 3,099 / 4,306 | none | 11,035 / 3,260 / 3,266 | – |
| Fire / smoke boxes | 14,692 / 11,865 | 0 / 2,317 | 5,562 / 5,745 | 20,254 / 19,927 |
| Images: fire / smoke / both / neither | 5% / 27% / 22% / 46% | 0% / 100% / 0% / 0% | 22% / 27% / **0%** / 50% | 12% / 31% / 11% / 45% |
| Small boxes, COCO <32² px (fire / smoke) | 31% / 4% | – / **61%** | 17% / 2% | – |
| Large boxes, COCO ≥96² px (fire / smoke) | 23% / 75% | – / 5% | 43% / 84% | – |
| Dark images (mean gray <60) | 18.4% | 0% | 10.4% | – |
| Images with a near-identical copy (pHash ≤2) | 43% | 93% | 43% | – |
| Distinct scenes after strict dedup (approx.) | ~13,400 | ~300 | ~11,300 | ~25,000 |
| Corrupt images / missing labels / orphan labels | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 |
| Boxes off the image edge / zero-area | 329 / 18 | 0 / 0 | 40 / 5 | 369 / 23 |

"Distinct scenes" = images with no near-copy + one per duplicate group. It's a rough
measure of real variety, not a count of unique files.

## Key problems found

1. **Duplicates leak across the published splits.** At the strict threshold (pHash distance ≤2):
   - D-Fire: 827 duplicate groups contain images from more than one of its train/val/test splits.
   - azimjaan21: 1,008 such groups.

   A model evaluated on these splits is partly tested on images it trained on, so scores come
   out too high. Most D-Fire matches are likely neighboring video frames. azimjaan21's include
   715 byte-identical files and 3,615 mirrored copies (its `Mirror…` and `Noise…` augmentations).
   The largest groups have 100+ images.
2. **AI For Mankind is far less varied than its size suggests.** 93% of images are near-copies of
   another image. Its 2,191 images collapse to 136 duplicate groups (~300 distinct scenes).
   The likely cause is fixed-camera sequences, where frames look the same apart from a small
   smoke plume. It also has
   no background images and no night images.
3. **azimjaan21 class ids were undocumented.** Labels use ids 0/1/2, but the author lists two
   classes and ships no `data.yaml`. Evidence used to settle it:
   - color inside boxes (id 1: saturation 0.49 and 50% fire-coloured pixels, vs D-Fire fire at 0.50 and 41%)
   - a visual spot check of 16 images per id
   - cross-dataset duplicates: all 44 matches between its id 1 / id 2 images and D-Fire are
     D-Fire images containing fire or smoke, and none are D-Fire background images

   Result: **0 = cloud/glare look-alike, 1 = fire, 2 = smoke.** Id 0 boxes are kept but ignored,
   so the 8,822 images with only id 0 boxes count as hard negatives.
4. **azimjaan21 fire images are missing smoke labels.** No image has both classes, against 22% in
   D-Fire. 20 duplicate pairs match a D-Fire *fire and smoke* image to an azimjaan21 image
   labeled fire only.
   Training on them as-is teaches the model that visible smoke is background.
   Part of the cause: **714 images are stored twice as byte-identical files**, once with only
   fire boxes and once with only smoke boxes. Combining the copies recovers 714 correct
   fire-and-smoke images. The other 3,202 fire-only images have no smoke copy.
5. **Some cloud/glare negatives may contain fire.** 12 cloud-only images (2 at the strict
   threshold) match D-Fire images that do have fire or smoke boxes. They need a manual look
   before we trust all 8,822 negatives.
6. **Cross-dataset overlap.**
   - AI For Mankind vs D-Fire: 252 pairs (110 strict). 185 of them are D-Fire *neither* images
     matching AI For Mankind *smoke* images, which suggests the same wildfire cameras at a
     different time or with unlabeled smoke.
   - azimjaan21 vs D-Fire: 56 pairs (36 strict, 15 mirrored).

   Any combined split has to keep these groups together.
7. **Minor box problems.**
   - 369 boxes extend past the image edge (329 of them in D-Fire)
   - 23 have zero area
   - 127 are under 4 px on a side (99 in AI For Mankind)
   - one azimjaan21 box has a 114:1 aspect ratio, which is almost certainly a labeling error
8. **Domain gaps.** No flare stacks. AI For Mankind is daytime only. Most azimjaan21 boxes are
   large, close-up objects (84% of smoke boxes). Small objects, which matter most for
   wide-view site cameras, come mainly from D-Fire fire and AI For Mankind smoke.

## Recommendations, in order

1. **Build new train/val/test splits by duplicate group, across all datasets.** Don't reuse the
   published splits. Every image already has a group id at both thresholds. Use the strict
   groups at minimum; the loose groups are safer for evaluation.
2. **Combine exact duplicates** (716 files) into one image each, merging their boxes, rather than
   just dropping copies. Done in the merged dataset.
3. **Fix boxes when writing training labels:**
   - clip the 369 off-image boxes
   - drop the 23 zero-area boxes and the 114:1 box
   - review boxes under 4 px
4. **azimjaan21:**
   - Use its smoke images and cloud/glare negatives now.
   - For fire-only images, either add smoke labels (best) or leave them out of training.
   - Review the 12 flagged cloud images before using the negatives.
5. **AI For Mankind:** group its sequences so one camera is never in both train and test.
6. **Close the flare-stack gap** with site or look-alike footage kept outside this public repo,
   plus more night-time data.

## Merged dataset (built 2026-09-30)

Built by `scripts/build_merged.py` (`jobs/merge.slurm`) as a YOLO folder of symlinked images and
new labels, with nothing copied from raw data. Ultralytics config: `outputs/merged/data.yaml` on Grace.

- **Included:**
  - all of D-Fire
  - all of AI For Mankind (team decision to train on it)
  - azimjaan21 smoke images, cloud/glare negatives, and the 714 recovered fire-and-smoke images
- **Left out:** 3,202 azimjaan21 fire-only images (smoke unlabeled).
- **Identical copies:** 716 combined into one image each, with boxes merged.
- **Box fixes:**
  - 329 boxes clipped to the image
  - 19 zero-area or fully-outside boxes dropped
  - 6 boxes under 1 px dropped
- **Splits:** new 70 / 15 / 15, assigned per loose duplicate group (pHash ≤5), balanced per
  dataset. 0 groups straddle splits.

| | Train | Val | Test | Total |
|---|---|---|---|---|
| D-Fire | 15,069 | 3,220 | 3,238 | 21,527 |
| azimjaan21 | 9,549 | 2,049 | 2,046 | 13,644 |
| AI For Mankind | 1,535 | 335 | 320 | 2,190 |
| **Total images** | **26,153** | **5,604** | **5,604** | **37,361** |
| Fire / smoke boxes | 10,641 / 13,654 | 2,134 / 3,166 | 2,924 / 3,090 | |

**Known weakness:** splits are balanced per dataset but not per category. Test holds 26% of the
fire-only images and 21% of the fire-and-smoke images, because D-Fire fire images come in large
duplicate groups. A category-aware pass would even this out before serious training runs.

## Method notes and caveats

- Duplicates use a 64-bit perceptual hash (`imagehash.phash`). Each image is compared with every
  other image's normal and mirrored hash. Distances come out even-only, so "≤5" means ≤4 in practice.
- Duplicate groups chain (if A matches B and B matches C, all three are grouped), so large groups
  can contain images that aren't direct copies of each other. The strict tier limits this.
- pHash captures overall layout. It flags frames from the same fixed camera as duplicates even
  when the smoke differs. For splitting that's the right call; for counting "copies" it
  overstates.
- Brightness is a rough day/night proxy (mean gray level); it doesn't detect night directly.
- The azimjaan21 class meaning comes from strong evidence (color stats, a visual spot check and
  cross-dataset agreement), not from the author. Id 0 in particular deserves a larger review.

## Fire coverage in the merged set

Fire is present, but **fire-only images are scarce**:

| | Images | Share of 37,361 |
|---|---|---|
| Any fire (fire only + fire and smoke) | 6,536 | 17% |
| Fire only | 1,164 | 3% |
| Fire and smoke | 5,372 | 14% |
| Smoke only | 12,166 | 33% |
| Neither (negatives) | 18,659 | 50% |

Fire boxes: 15,699. Smoke boxes: 19,910. Half the images are negatives, which helps against
false alarms but means fire recall should be watched closely in training. The datasets below are
the main way to add fire.

## Datasets not yet merged

| Dataset | Status | Images | Classes | Notes |
|---|---|---|---|---|
| **roscoekerby** (Kaggle `roscoekerby/firesmoke-detection-yolo-v9`) | **Next to add.** Downloaded; extraction being redone. | 42,842, but only **~7,551 distinct originals** | fire, smoke | Roboflow export with up to 148 augmented copies per original. We will keep **one image per original**. Includes industrial fire video frames. |
| **ironwolf437** (Kaggle `ironwolf437/fire-detection-dataset`) | **Candidate, not yet downloaded.** | 17,344 (no augmentation, per its readme) | fire, light, nonfire, smoke | Roboflow export from surveillance footage; its description mentions indoor kitchen scenes. `light` (lamps, reflections) would be kept as hard negatives, like azimjaan21's cloud/glare. Needs the same class, duplicate and quality checks before merging. |
| FiSmo (dsw2017) | Excluded | – | – | Image-level labels only, no boxes. |
| metinmekiabullrahman/fire-detection | Excluded | ~2,500 | fire | Smoke unlabeled; low-resolution video frames. |
| DataCluster Labs | Not evaluated | – | – | |

Image counts for roscoekerby and ironwolf437 come from their published metadata, not from our
own checks yet. Neither contains flare stacks, so that gap remains.
