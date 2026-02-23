# Geospatial Object Detection

End-to-end pipelines for detecting objects in satellite and aerial imagery and exporting georeferenced results (GeoJSON) for QA in a GIS.

## Projects

- Cars (WMTS -> YOLOv8 -> GeoJSON): `cars/README.md`

Planned / future modules:

- Trees
- Buildings

## Quick Start (Cars)

From the repository root:

```bash
pip install -r requirements.txt
```

Run an AOI end-to-end (no need to edit config files):

```bash
# 1) Download WMTS tiles for an AOI into an isolated run folder
python cars/scripts/pipeline.py ingest --run-id <run_id> --aoi <aoi_file>

# 2) Run inference using the committed checkpoint under cars/models/
python cars/scripts/pipeline.py infer --run-id <run_id> --split test

# 3) Export detections to EPSG:3857 GeoJSON
python cars/scripts/pipeline.py export --run-id <run_id> --split test
```

See `cars/README.md` for full setup details, labeling/training, and troubleshooting.

## Results (Cars)

The screenshot below shows detections exported to GeoJSON and viewed in a GIS.

![Cars detections GIS overlay](docs/images/cars_gis_overlay.png)

If the image is missing locally, add a screenshot at `docs/images/cars_gis_overlay.png`.

## Repository Conventions

- AOI inputs live under `<project>/aois/` (committed)
- Large generated artifacts are kept out of git (tiles, intermediate runs)
- Small, shareable outputs can be committed (final model checkpoint and GeoJSON exports)

To add a new detector (trees/buildings/etc.), mirror the `cars/` module layout and documentation, and keep CLI flags consistent (`--run-id`, `--aoi`, `infer`, `export`, `val`).
