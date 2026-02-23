# Model Card: YOLOv8 Car Detector

This document captures the current model, how it was trained, and how it performs.

## Model Summary

- **Task**: object detection (single class: `car`)
- **Checkpoint**: `cars/models/yolov8s_cars_best.pt`
- **Checkpoint SHA256**: `e353d78be8b25686fc68e87ad97ca3451cad4eea8b2c35e9a36235c91ae1cef8`
- **Model metadata**: `cars/models/yolov8s_cars_best.meta.json`

## Intended Use

- Detect cars in WMTS-ingested imagery tiles (Web Mercator / EPSG:3857 tiling).
- Primary output artifact is a GeoJSON of detections (EPSG:3857) for QA in a GIS.

Not intended for:

- imagery without a compatible georeferencing / tile-id scheme (unless you add a georeferencing adapter)
- safety-critical decisions

## Data

### Training/Validation Data

- Imagery is ingested via WMTS using `cars/scripts/pipeline.py ingest`.
- AOIs are stored under `cars/aois/`:
  - `cars/aois/alameda_aio_02222026.geojson`
  - `cars/aois/oakland_aio_02222026.geojson`

### Labeling

- Labels are created in CVAT (running via Docker Desktop).
- Cars are labeled in both the train and validation datasets.
- CVAT exports are stored as COCO instances JSON:
  - `cars/annotations/train_coco.json`
  - `cars/annotations/val_coco.json`

## Training Configuration

Training defaults live in `cars/config.yaml` under `training:`:

- **Base model**: `models/yolov8s.pt`
- **Epochs**: 100
- **Image size**: 640
- **Batch**: 16
- **Device**: 0

## Evaluation

### Quantitative Metrics (Validation)

Run:

```bash
python cars/scripts/pipeline.py val --run-id run_alameda_02222026
```

Validation run output:

- Metrics: `cars/runs/detect/val_run_alameda_02222026/metrics.json`
- Summary line: `all 73 238 0.913 0.927 0.973 0.672`

Recorded metrics (Alameda validation set):

| Metric | Value | Notes |
|---|---:|---|
| mAP@0.5 | 0.9729 | Ultralytics `metrics/mAP50(B)` |
| mAP@0.5:0.95 | 0.6717 | Ultralytics `metrics/mAP50-95(B)` |
| Precision | 0.9131 | Ultralytics `metrics/precision(B)` |
| Recall | 0.9271 | Ultralytics `metrics/recall(B)` |
| Num images | 73 | `cars/data/aoi_runs/run_alameda_02222026/val/images` |
| Num instances | 238 | From Ultralytics val summary |

### Qualitative / AOI Generalization Tests

The repo commits example exports for quick GIS inspection:

- `cars/exports/run_alameda_02222026.geojson`
- `cars/exports/oakland_02222026.geojson`

Recommended review checklist:

- Verify alignment: bounding boxes land on cars (no CRS/flip issues)
- Note false positives: rooftops, bright objects, parking lot artifacts
- Note false negatives: small cars, shadows, occlusions
- Check stability across different AOIs / lighting / ground sample distance

## Inference Defaults

Inference defaults live in `cars/config.yaml` under `inference:`:

- **Confidence**: 0.25
- **IoU**: 0.45
- **Image size**: 640

Example run:

```bash
python cars/scripts/pipeline.py ingest --run-id <run_id> --aoi <aoi_file>
python cars/scripts/pipeline.py infer --run-id <run_id> --split test
python cars/scripts/pipeline.py export --run-id <run_id> --split test
```

## Environment

The environment used to build the committed checkpoint:

- **Python**: 3.12.10
- **Torch**: 2.10.0+cu130
- **Ultralytics**: 8.4.11

See `cars/models/yolov8s_cars_best.meta.json` for the authoritative values.

## Known Limitations / Risks

- AOI ingestion downloads tiles based on AOI bounding boxes, so edges may include out-of-AOI imagery.
- Performance may degrade under domain shift (seasonality, different sensors, different resolutions).
- Export assumes WMTS-style tile-id filenames (`_z<z>_r<row>_c<col>`). If you run inference on arbitrary images, GeoJSON export may not be able to georeference results.
