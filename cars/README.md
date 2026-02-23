# Cars Detection Pipeline (WMTS -> YOLOv8 -> GeoJSON)

This folder contains a small, end-to-end pipeline for detecting cars in satellite imagery.

The workflow is:

1. Download imagery tiles from a WMTS service for one or more AOIs
2. (Optional) Convert annotations to YOLO labels and train a model
3. Run inference on new AOIs
4. Export detections to GeoJSON (EPSG:3857) for QA in a GIS

The pipeline entrypoint is `cars/scripts/pipeline.py`.

## Requirements

From the repository root:

```bash
pip install -r requirements.txt
```

Notes:

- `geopandas`/`rasterio` may require native dependencies depending on OS. If pip install fails, use a conda environment or your existing GIS Python environment.
- The pipeline shells out to the Ultralytics CLI (`yolo`). Installing `ultralytics` via `requirements.txt` should provide it.
- `export` uses OpenCV (`opencv-python`).

## Repository Layout (cars/)

- `cars/config.yaml`: WMTS settings, dataset definition, training and inference defaults
- `cars/aois/`: AOI GeoJSON inputs (checked into git)
- `cars/scripts/pipeline.py`: CLI pipeline runner
- `cars/data/aoi_runs/<run_id>/...`: generated imagery + per-run manifest (ignored by git)
- `cars/runs/`: Ultralytics outputs (ignored by git)
- `cars/exports/`: GeoJSON exports (GeoJSON files are committed; other artifacts are ignored)
- `cars/models/yolov8s_cars_best.pt`: final trained model checkpoint (committed)
- `cars/models/yolov8s_cars_best.meta.json`: model parameters and environment info (committed)

## Model Parameters

The final trained model checkpoint is committed as `cars/models/yolov8s_cars_best.pt`.

To make runs reproducible, this repo also checks in a small metadata file:

- `cars/models/yolov8s_cars_best.meta.json`

It includes:

- the checkpoint SHA256 and size
- Python / Torch / Ultralytics versions
- the `training:` and `inference:` defaults from `cars/config.yaml`

## AOI GeoJSON Schema

Your AOI file must be a GeoJSON `FeatureCollection` with `Polygon` or `MultiPolygon` features.

Recommended properties per feature:

- `site_id`: string identifier used in manifests/logs
- `split`: one of `train`, `val`, `test` (defaults to `train` if omitted)

CRS:

- The ingestion step expects AOI geometry in EPSG:3857 (Web Mercator meters).
- If the AOI has a CRS and it is not EPSG:3857, the pipeline will reproject.
- If the AOI has no CRS, the pipeline assumes EPSG:3857.

## Configure WMTS

Edit the `wmts:` section in `cars/config.yaml`:

- `url`, `layer`, `tile_matrix_set`, `tile_matrix`, `style`, `format`
- Optional: `resource_url_template` for REST-style WMTS access

If your WMTS requires auth, set an API key in the environment:

- `WMTS_API_KEY`: added as a Bearer token header

## Running The Pipeline

All commands are run from the repository root.

### 1) Ingest tiles for an AOI (required for new areas)

`ingest` is intentionally guarded to avoid accidental overwrites. You must pass both `--run-id` and `--aoi`.

Example:

```bash
python cars/scripts/pipeline.py ingest \
  --run-id oakland_02222026 \
  --aoi oakland_aio_02222026.geojson
```

AOI path resolution:

- Absolute paths work
- Repo-relative paths work (e.g. `./cars/aois/oakland_aio_02222026.geojson`)
- Bare filenames are resolved from `cars/aois/`

Outputs:

- Imagery tiles: `cars/data/aoi_runs/<run_id>/<split>/images/`
- Per-run manifest: `cars/data/aoi_runs/<run_id>/manifest.csv`

Implementation note: ingestion computes a tile range from each AOI feature's bounding box, so it may download tiles slightly outside the polygon.

### 2) Prepare labels (optional; only for training)

This project assumes you label imagery in CVAT and export to COCO detection format.

Typical flow:

1. Install/run CVAT (for example via Docker Desktop)
2. Create a task using the ingested tiles under:
   - `cars/data/aoi_runs/<run_id>/train/images/`
   - `cars/data/aoi_runs/<run_id>/val/images/`
3. Annotate cars (single class: `car`)
4. Export annotations as COCO 1.0 (instances)

Place the exported COCO JSON files here:

- `cars/annotations/train_coco.json`
- `cars/annotations/val_coco.json`

Then run:

```bash
python cars/scripts/pipeline.py prepare
```

This writes YOLO label files to:

- `cars/data/train/labels/`
- `cars/data/val/labels/`

### 3) Train (optional)

Configure `training:` in `cars/config.yaml`, then run:

```bash
python cars/scripts/pipeline.py train
```

On success, the best checkpoint is promoted into:

- `cars/models/<training.name>_best.pt`

This repo is set up to commit the final checkpoint at:

- `cars/models/yolov8s_cars_best.pt`

### 4) Validate (optional)

```bash
python cars/scripts/pipeline.py val
```

This uses the Ultralytics Python API and saves metrics to the validation run directory.

### 5) Inference for a run-id

Run inference on the tiles you ingested without editing `cars/config.yaml`:

```bash
python cars/scripts/pipeline.py infer --run-id oakland_02222026 --split test
```

Defaults:

- `--split` controls the source folder under the run (default: `test`)
- If `--run-id` is set, the default source becomes `data/aoi_runs/<run-id>/<split>/images`
- If `--run-id` is set, the default Ultralytics output is `project=aoi_tests` and `name=<run-id>`

You can override if needed:

- `--source`, `--project`, `--name`

### 6) Export predictions to GeoJSON

```bash
python cars/scripts/pipeline.py export --run-id oakland_02222026 --split test
```

Output:

- `cars/exports/oakland_02222026.geojson`

How georeferencing works:

- For WMTS-ingested tiles, filenames and label stems include `_z<zoom>_r<row>_c<col>`.
- `export` uses that tile id to compute EPSG:3857 bounds and convert YOLO pixel bboxes into world coordinates.

## Troubleshooting

- `AOI file not found`: pass `--aoi` as an absolute path, `./cars/aois/<file>`, or just the filename if it lives in `cars/aois/`.
- `AOI has no CRS; assuming EPSG:3857`: OK only if your AOI coordinates are Web Mercator meters.
- `'yolo' CLI not found`: `pip install ultralytics` and ensure your environment exposes the `yolo` command.
- `Model checkpoint not found`: confirm `inference.model` in `cars/config.yaml` exists under `cars/`.
- `[export] OpenCV not available`: `pip install opencv-python`.
