# Cars Detection Pipeline (YOLOv8 on Satellite Imagery)

This directory contains the high-level workflow to detect cars in satellite imagery using YOLOv8, with end-to-end steps from reading imagery to exporting detections as GeoJSON and visualizing on a map.

## Overview

- Objective: Detect cars in satellite imagery and export detections as GeoJSON for mapping/QA.
- Core workflow (5 stages): Data ingestion, Labeling & formatting, Model training, Inference, GeoJSON export & visualization.
- Deliverables: Trained YOLOv8 model, GeoJSON detections, interactive map visualizations.

## Phases and Steps

### 1. Data Ingestion

- Gather satellite imagery with georeferencing (footprints or affine transforms).
- Ensure data is organized with a stable naming convention and metadata.
- Split data into train/validation/test sets.

#### Data Ingestion (implementation details)

- Imagery is pulled from an internal WMTS service configured in `cars/data.yaml` (see the `wmts` section).
- AOIs are defined as polygons in EPSG:3857 in `cars/aoi.geojson` (or `cars/aoi.example.geojson` if you are just testing).
- Run `python cars/scripts/pipeline.py ingest` to download tiles for each AOI feature and split (`train`/`val`/`test`).
- Tiles are saved under `cars/data/<split>/images/`, and a manifest called `cars/data/manifest.csv` records per-tile metadata.
- `cars/data/` is intended for local datasets (imagery and labels) and should not be committed to git.

### 2. Labeling & YOLO Formatting

- Define classes (minimum: car; optional: truck, bus).
- Annotate a representative subset of images; export to COCO or YOLO format.
- Convert annotations to YOLO format (class_id, x_center, y_center, width, height) normalized to image size.
- Organize data under this module as:

```text
cars/
  data/
    train/images/
    train/labels/
    val/images/
    val/labels/
    test/images/
    test/labels/
```

### 3. Model Training (YOLOv8)

- Choose a model size: YOLOv8n/s/l/x based on compute and accuracy needs.
- Image size: 640×640 (adjust with tile sizing for large scenes).
- Typical hyperparameters to start:
  - epochs: 50–150
  - batch size: depends on GPU memory
  - learning rate: default/Yolov8 defaults; tune if needed
- Validate with mAP@0.5 and optionally mAP@0.5:0.95.
- Save best model checkpoints for inference.

### 4. Inference

- Run on test/holdout imagery or tiles for large scenes; apply overlap tiling if needed.
- Apply a confidence threshold (e.g., 0.25–0.5) and NMS to deduplicate detections.
- Output per-detection data (class, confidence, bbox) per image.

### 5. GeoJSON Export & Visualization

- Map detections to geographic coordinates using per-image georeferencing (affine transform or footprints).
- Produce a GeoJSON FeatureCollection where each feature is a polygon (car bbox) with properties: class, confidence, image_id, footprint.
- Visualize on a map with Folium or Kepler.gl; enable layer toggling and color by confidence.

## Data & Labeling Guidance

- Data sources: PlanetScope, Sentinel, or other providers with suitable resolution.
- Annotations: label car instances, keep guidelines for occlusions and shadows; agree on a consistent class schema.
- Metadata: maintain a manifest mapping image_id to footprint/transform to enable geo-export.

## Tools and Tips

- Use a tiling strategy for large scenes (e.g., 512–640 px tiles) to fit GPU memory and improve detection accuracy.
- Keep a simple experiment log: model type, hyperparameters, data version, and evaluation metrics.
- Pin versions in a requirements.txt or environment.yml to ensure reproducibility.

## Quick Start (illustrative commands)

- Create data layout (from repo root):

```bash
mkdir -p cars/data/train/images cars/data/train/labels cars/data/val/images cars/data/val/labels cars/data/test/images cars/data/test/labels
```

- Train (example; adapt to your environment and path):

```bash
# Assuming Ultralytics YOLOv8 CLI is installed
yolo train data=data.yaml model=yolov8s.pt epochs=100 imgsz=640
```

- Inference (example):

```bash
yolo predict model=best.pt source=cars/data/test/images/ --conf 0.25 --iou 0.45
```

- GeoJSON export (conceptual):

```bash
python tools/export_geojson.py --detections predictions.json --geo meta.json --out detections.geojson
```

- Visualize: open `detections.geojson` in a Folium map or Kepler.gl project.

## Notes

- Replace placeholder scripts and paths with your actual data and tooling.
- This plan is meant as a template; adjust according to data availability, compute, and project goals.
