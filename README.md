# D-Fire local prototype
# TO TEST/RUN must have Dfire dataset installed
# DFire: https://github.coem/gaia-solutions-on-demand/DFireDataset


Folder structure is
dfire
|_
| data/Dfire (dfire data source)
|_
    OxyFire (the github repo)


Once installed, double-click `run_report.cmd` to process the whole dataset and open the report.

Run from PowerShell:

```powershell
cd C:\dfire\OxyFire #(This may be different if you are starting in the OxyFire folder)
python -m pip install -r requirements.txt
python analyze_dfire.py
Start-Process report\report.html
```

The default dataset is `C:\dfire\data\DFire`. To analyze another location or change size thresholds:

```powershell
python analyze_dfire.py C:\dfire\data\DFire --small 1 --large 10 --output report
```

This is annotation analysis, not model inference. Each image is classified as **fire**, **fire + smoke**, **smoke**, or **none**, using YOLO class IDs 0 = smoke and 1 = fire. An empty label is a negative image. Missing labels, unparseable boxes, and unreadable image headers are reported as errors rather than silently classified as none. Zero-area boxes retain their class but get an `invalid` size. Boxes crossing image edges retain their original annotated dimensions. Both quirks are flagged in `issues.csv`; they do not exclude otherwise usable images.

Every box gets pixel width, height, area, percentage of image area, and a size bucket. Defaults: **small <1%**, **medium 1% to <10%**, **large >=10%**. These are configurable presentation thresholds, not physical fire sizes. Image balancing groups use the largest box of each class; all boxes remain available individually. No image files are modified and no resampling happens automatically.

Outputs in `report/`:

- `report.html`: presentation with category bars, split tables, size summaries, and annotated example images.
- `images.csv`: one row per image, including category, counts, largest sizes, and status.
- `boxes.csv`: one row per valid bounding box, with normalized and pixel measurements.
- `categories.csv`, `sizes.csv`: counts and percentages overall and by split.
- `balance_groups.csv`: category × largest fire size × largest smoke size counts by split, for planning balanced training samples.
- `issues.csv`, `summary.json`: validation issues and overall totals.

The report embeds local thumbnails; keep its `thumbnails/` directory alongside the HTML when sharing. Original-image links require the dataset to remain at its local path. The script reads every image header and label, but only decodes report examples; it is not a full image-corruption scan.

Dataset source: [official D-Fire repository](https://github.com/gaia-solutions-on-demand/DFireDataset). The download uses the [Kaggle mirror linked by the authors](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo). Original dataset authors: Pedro Vinicius Almeida Borges de Venancio, Adriano Chaves Lisboa, and Adriano Vilela Barbosa. See the official repository for citation and license information.

To download again, run from `C:\dfire`:

```powershell
curl.exe -fL --retry 2 https://www.kaggle.com/api/v1/datasets/download/sayedgamal99/smoke-fire-detection-yolo -o data\dfire.zip
python -m zipfile -e data\dfire.zip data\DFire
```
