"""Starter end-to-end pipeline for car detection in satellite imagery.

This module wires together the main steps of the workflow and provides a
minimal, working implementation of the data-ingestion step that pulls tiles
from a WMTS service using AOIs defined in GeoJSON (EPSG:3857).
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, cast

import geopandas as gpd
import requests
import yaml

# Web Mercator world extent used for simple EPSG:3857 tile math
WEB_MERCATOR_BOUNDS: Tuple[float, float, float, float] = (
    -20037508.342789244,
    -20037508.342789244,
    20037508.342789244,
    20037508.342789244,
)


def _project_root() -> Path:
    """Return the repository root based on this file location."""

    # cars/scripts/pipeline.py -> cars -> repo root
    return Path(__file__).resolve().parents[2]


def _cars_dir() -> Path:
    """Return the path to the cars/ module directory."""

    return Path(__file__).resolve().parents[1]


def _load_wmts_config() -> Dict:
    """Load WMTS configuration from cars/config.yaml.

    The YAML file is expected to contain a top-level ``wmts`` section.
    """

    cars_dir = _cars_dir()
    config_path = cars_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Expected WMTS config at {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    wmts_cfg = cfg.get("wmts") or {}
    required_keys = [
        "url",
        "layer",
        "tile_matrix_set",
        "tile_matrix",
        "style",
        "format",
    ]
    missing = [k for k in required_keys if k not in wmts_cfg]
    if missing:
        raise KeyError(f"Missing WMTS config keys in cars/config.yaml: {missing}")

    return wmts_cfg


def _zoom_from_tile_matrix(tile_matrix_value) -> int:
    """Interpret wmts.tile_matrix as a Web Mercator zoom level.

    For this first implementation we assume that ``wmts.tile_matrix`` is a
    standard XYZ-style zoom between 0 and 30. If this is not the case for
    your WMTS service, update ``cars/config.yaml`` accordingly.
    """

    try:
        z = int(tile_matrix_value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"wmts.tile_matrix must be an integer zoom level: {tile_matrix_value!r}"
        ) from exc

    if not (0 <= z <= 30):
        raise ValueError(
            "wmts.tile_matrix is expected to be a zoom level between 0 and 30; "
            f"got {tile_matrix_value!r}. Update cars/config.yaml to a reasonable zoom."
        )
    return z


def _bbox_to_tile_range(
    bbox: Tuple[float, float, float, float], zoom: int
) -> Tuple[int, int, int, int]:
    """Return (row_min, row_max, col_min, col_max) covering a bbox at a zoom.

    Uses a simple global Web Mercator grid with 2**zoom tiles in each axis.
    """

    minx, miny, maxx, maxy = bbox
    world_minx, world_miny, world_maxx, world_maxy = WEB_MERCATOR_BOUNDS
    n = 2**zoom

    tile_size_x = (world_maxx - world_minx) / n
    tile_size_y = (world_maxy - world_miny) / n

    def _col(x: float) -> int:
        return int((x - world_minx) / tile_size_x)

    def _row(y: float) -> int:
        # y origin at world_maxy, increasing downward
        return int((world_maxy - y) / tile_size_y)

    col_min = max(0, _col(minx))
    col_max = min(n - 1, _col(maxx))
    row_min = max(0, _row(maxy))
    row_max = min(n - 1, _row(miny))

    return row_min, row_max, col_min, col_max


def _tile_bounds(
    zoom: int, tile_row: int, tile_col: int
) -> Tuple[float, float, float, float]:
    """Return (xmin, ymin, xmax, ymax) in EPSG:3857 for a tile index."""

    world_minx, world_miny, world_maxx, world_maxy = WEB_MERCATOR_BOUNDS
    n = 2**zoom
    tile_size_x = (world_maxx - world_minx) / n
    tile_size_y = (world_maxy - world_miny) / n

    xmin = world_minx + tile_col * tile_size_x
    xmax = xmin + tile_size_x
    ymax = world_maxy - tile_row * tile_size_y
    ymin = ymax - tile_size_y
    return xmin, ymin, xmax, ymax


def _format_to_extension(fmt: str) -> str:
    """Map MIME format string to a file extension."""

    fmt = (fmt or "").lower()
    if "jpeg" in fmt or "jpg" in fmt:
        return ".jpg"
    if "png" in fmt:
        return ".png"
    if "tiff" in fmt or "tif" in fmt:
        return ".tif"
    return ".img"


def _build_wmts_tile_url(wmts_cfg: Dict, tile_row: int, tile_col: int) -> str:
    """Construct a WMTS tile URL from config and tile indices.

    - If ``resource_url_template`` is provided in the config, we treat it as a
      REST-style template like the one advertised in WMTS capabilities
      ``<ResourceURL>`` elements and fill in the placeholders.
    - Otherwise, we fall back to a KVP-style ``GetTile`` request.
    """

    template = wmts_cfg.get("resource_url_template")
    if template:
        return template.format(
            Style=wmts_cfg.get("style", "default"),
            TileMatrixSet=wmts_cfg["tile_matrix_set"],
            TileMatrix=wmts_cfg["tile_matrix"],
            TileCol=tile_col,
            TileRow=tile_row,
        )

    from urllib.parse import urlencode

    base_url = wmts_cfg["url"].rstrip("?")
    params = {
        "service": "WMTS",
        "request": "GetTile",
        "version": "1.0.0",
        "layer": wmts_cfg["layer"],
        "style": wmts_cfg.get("style", "default"),
        "tilematrixset": wmts_cfg["tile_matrix_set"],
        "tilematrix": wmts_cfg["tile_matrix"],
        "tilerow": tile_row,
        "tilecol": tile_col,
        "format": wmts_cfg.get("format", "image/png"),
    }
    return f"{base_url}?{urlencode(params)}"


def _create_wmts_session() -> requests.Session:
    """Create a requests session for WMTS calls.

    If an environment variable ``WMTS_API_KEY`` is set, it is added as a bearer
    token in the Authorization header. Adjust this to match your auth setup.
    """

    session = requests.Session()
    api_key = os.getenv("WMTS_API_KEY")
    if api_key:
        session.headers.update({"Authorization": f"Bearer {api_key}"})
    return session


def ingest_data():
    """Download imagery tiles from WMTS into cars/data and write a manifest.

    - Reads WMTS configuration from ``cars/config.yaml``.
    - Reads AOIs from ``cars/aoi.geojson``
    - For each AOI (Polygon/MultiPolygon in EPSG:3857) and split
      (train/val/test), computes tile row/col ranges that cover its bounding
      box at the configured zoom level.
    - Requests each tile via WMTS GetTile and saves to
      ``cars/data/<split>/images/``.
    - Appends metadata for each tile to ``cars/data/manifest.csv``.
    """

    cars_dir = _cars_dir()
    data_dir = cars_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        wmts_cfg = _load_wmts_config()
    except Exception as exc:  # pragma: no cover - simple CLI feedback
        print(f"[ingest] Failed to load WMTS config: {exc}")
        return

    zoom = _zoom_from_tile_matrix(wmts_cfg["tile_matrix"])

    # AOI file: prefer user-provided aoi.geojson, fall back to example.
    aoi_path = cars_dir / "aoi.geojson"
    if not aoi_path.exists():
        aoi_path = cars_dir / "aoi.geojson"
        print(f"[ingest] Using example AOI at {aoi_path}")
    else:
        print(f"[ingest] Using AOI at {aoi_path}")

    try:
        aoi_gdf = gpd.read_file(aoi_path)
    except Exception as exc:  # pragma: no cover - simple CLI feedback
        print(f"[ingest] Failed to read AOI GeoJSON: {exc}")
        return

    if aoi_gdf.empty:
        print("[ingest] AOI file has no features; nothing to do.")
        return

    if aoi_gdf.crs is None:
        print("[ingest] AOI has no CRS; assuming EPSG:3857.")
        aoi_gdf = aoi_gdf.set_crs(epsg=3857)
    elif str(aoi_gdf.crs.to_epsg()) != "3857":
        print(f"[ingest] Reprojecting AOI from {aoi_gdf.crs} to EPSG:3857.")
        aoi_gdf = aoi_gdf.to_crs(epsg=3857)

    manifest_path = data_dir / "manifest.csv"
    manifest_exists = manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        if not manifest_exists:
            writer.writerow(
                [
                    "image_id",
                    "split",
                    "rel_path",
                    "layer",
                    "tile_matrix_set",
                    "tile_matrix",
                    "tile_row",
                    "tile_col",
                    "crs",
                    "xmin",
                    "ymin",
                    "xmax",
                    "ymax",
                    "site_id",
                ]
            )

        session = _create_wmts_session()

        total_tiles = 0
        print(
            f"[ingest] Starting ingestion for {len(aoi_gdf)} AOI feature(s) at zoom {zoom} "
            f"from {wmts_cfg['url']}"
        )

        for idx, row in aoi_gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                print(f"[ingest] Skipping empty geometry at index {idx}.")
                continue

            props = row.to_dict()
            site_id = props.get("site_id") or props.get("SITE_ID") or f"site_{idx}"
            split = (props.get("split") or "train").lower()
            if split not in {"train", "val", "test"}:
                print(
                    f"[ingest] Unknown split '{split}' for site {site_id}; "
                    "defaulting to 'train'."
                )
                split = "train"

            minx, miny, maxx, maxy = geom.bounds
            row_min, row_max, col_min, col_max = _bbox_to_tile_range(
                (minx, miny, maxx, maxy), zoom
            )

            print(
                f"[ingest] AOI site_id={site_id}, split={split}, "
                f"rows {row_min}-{row_max}, cols {col_min}-{col_max}."
            )

            split_img_dir = data_dir / split / "images"
            split_img_dir.mkdir(parents=True, exist_ok=True)

            img_ext = _format_to_extension(wmts_cfg.get("format", "image/jpeg"))

            for tile_row in range(row_min, row_max + 1):
                for tile_col in range(col_min, col_max + 1):
                    xmin, ymin, xmax, ymax = _tile_bounds(zoom, tile_row, tile_col)

                    image_id = (
                        f"{wmts_cfg['layer']}_z{wmts_cfg['tile_matrix']}"
                        f"_r{tile_row}_c{tile_col}"
                    )
                    img_filename = image_id + img_ext
                    img_path = split_img_dir / img_filename

                    if img_path.exists():
                        print(f"[ingest] Skipping existing tile {image_id}.")
                    else:
                        url = _build_wmts_tile_url(wmts_cfg, tile_row, tile_col)
                        try:
                            resp = session.get(url, timeout=30)
                        except (
                            Exception
                        ) as exc:  # pragma: no cover - network error path
                            print(f"[ingest] Error fetching {image_id}: {exc}")
                            continue

                        if resp.status_code != 200:
                            print(
                                f"[ingest] Failed to fetch {image_id}: "
                                f"HTTP {resp.status_code}"
                            )
                            continue

                        img_path.write_bytes(resp.content)

                    rel_path = img_path.relative_to(cars_dir).as_posix()
                    writer.writerow(
                        [
                            image_id,
                            split,
                            rel_path,
                            wmts_cfg["layer"],
                            wmts_cfg["tile_matrix_set"],
                            wmts_cfg["tile_matrix"],
                            tile_row,
                            tile_col,
                            "EPSG:3857",
                            xmin,
                            ymin,
                            xmax,
                            ymax,
                            site_id,
                        ]
                    )
                    total_tiles += 1

        print(
            f"[ingest] Ingestion complete. "
            f"Manifest: {manifest_path} ({total_tiles} tile(s) recorded)."
        )


def convert_to_yolo_format():
    """Convert external annotations into YOLO bbox txt files for detection.

    This implementation assumes that annotations have been exported in COCO
    format (instance/object detection) to:

    - ``cars/annotations/train_coco.json``
    - ``cars/annotations/val_coco.json`` (optional, but recommended)

    For each split, YOLO-style label files are written to:

    - ``cars/data/<split>/labels/<image_stem>.txt``

    Each line in a label file has the form::

        class_id x_center y_center width height

    where the coordinates are normalized to [0, 1] by image width/height.
    Currently this pipeline assumes a single class ``car`` with class_id 0.
    """

    cars_dir = _cars_dir()
    annotations_dir = cars_dir / "annotations"
    if not annotations_dir.exists():
        print(f"[prepare] No annotations directory found at {annotations_dir}.")
        print("[prepare] Please export COCO annotations before running this step.")
        return

    def _process_split(split: str, coco_path: Path) -> None:
        if not coco_path.exists():
            print(f"[prepare] Skipping split '{split}': {coco_path} not found.")
            return

        print(f"[prepare] Processing split '{split}' from {coco_path}.")

        with coco_path.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        images = coco.get("images", [])
        annotations = coco.get("annotations", [])
        categories = coco.get("categories", [])

        # Build category_id -> class_id map. For now we collapse everything
        # into a single class_id 0 if the category name contains "car".
        cat_id_to_class_id: Dict[int, int] = {}
        for cat in categories:
            cid = int(cat["id"])
            name = str(cat.get("name", "")).lower()
            if "car" in name:
                cat_id_to_class_id[cid] = 0

        if not cat_id_to_class_id:
            print(
                f"[prepare] No categories mapped to 'car' for split '{split}'. "
                "Ensure your COCO categories include a 'car' class."
            )
            return

        # Group annotations by image_id
        anns_by_image: Dict[int, List[Dict]] = {}
        for ann in annotations:
            image_id = int(ann["image_id"])
            anns_by_image.setdefault(image_id, []).append(ann)

        img_dir = cars_dir / "data" / split / "images"
        labels_dir = cars_dir / "data" / split / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        num_images_with_labels = 0
        num_boxes = 0

        for img in images:
            image_id = int(img["id"])
            file_name = str(img["file_name"])
            width = float(img.get("width", 0))
            height = float(img.get("height", 0))

            if width <= 0 or height <= 0:
                print(
                    f"[prepare] Skipping image_id={image_id} with invalid size "
                    f"(width={width}, height={height})."
                )
                continue

            image_path = img_dir / file_name
            if not image_path.exists():
                # Some COCO exports include full paths; try matching by stem.
                fallback = img_dir / Path(file_name).name
                if fallback.exists():
                    image_path = fallback
                else:
                    print(
                        f"[prepare] Image file for COCO entry '{file_name}' "
                        f"not found under {img_dir}; skipping."
                    )
                    continue

            anns = anns_by_image.get(image_id, [])
            if not anns:
                # It's fine to have unlabeled images; create an empty label file.
                label_path = labels_dir / (image_path.stem + ".txt")
                label_path.write_text("", encoding="utf-8")
                continue

            yolo_lines: List[str] = []
            for ann in anns:
                cat_id_raw = ann.get("category_id")
                if cat_id_raw is None:
                    continue
                try:
                    cat_id = int(cat_id_raw)
                except (TypeError, ValueError):
                    continue

                if cat_id not in cat_id_to_class_id:
                    # Ignore non-car categories for now.
                    continue

                bbox = ann.get("bbox") or []
                if len(bbox) != 4:
                    continue

                x, y, w, h = map(float, bbox)
                if w <= 0 or h <= 0:
                    continue

                x_center = (x + w / 2.0) / width
                y_center = (y + h / 2.0) / height
                w_norm = w / width
                h_norm = h / height

                # Clamp to [0,1] just in case of small numeric drift.
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                w_norm = max(0.0, min(1.0, w_norm))
                h_norm = max(0.0, min(1.0, h_norm))

                class_id = cat_id_to_class_id[cat_id]
                yolo_lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
                )

            label_path = labels_dir / (image_path.stem + ".txt")
            label_path.write_text(
                "\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8"
            )

            if yolo_lines:
                num_images_with_labels += 1
                num_boxes += len(yolo_lines)

        print(
            f"[prepare] Finished split '{split}': "
            f"{num_images_with_labels} images with labels, {num_boxes} boxes."
        )

    _process_split("train", annotations_dir / "train_coco.json")
    _process_split("val", annotations_dir / "val_coco.json")


def train_yolov8():
    """Train a YOLOv8 model using cars/config.yaml.

    This function reads optional ``training`` settings from the same YAML used
    for the dataset definition and WMTS config and forwards them to the
    Ultralytics ``yolo`` CLI.
    """

    cars_dir = _cars_dir()
    config_path = cars_dir / "config.yaml"

    if not config_path.exists():
        print(f"[train] Dataset/config file not found at {config_path}.")
        return

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    train_cfg = cfg.get("training") or {}

    model = str(train_cfg.get("model", "yolov8s.pt"))
    epochs = str(train_cfg.get("epochs", 100))
    imgsz = str(train_cfg.get("imgsz", 640))
    batch = str(train_cfg.get("batch", 16))
    device = str(train_cfg.get("device", 0))
    workers = str(train_cfg.get("workers", 8))
    project = str(train_cfg.get("project", "runs/train"))
    name = str(train_cfg.get("name", "cars"))

    cmd = [
        "yolo",
        "train",
        f"data={config_path}",
        f"model={model}",
        f"epochs={epochs}",
        f"imgsz={imgsz}",
        f"batch={batch}",
        f"device={device}",
        f"workers={workers}",
        f"project={project}",
        f"name={name}",
    ]

    print("[train] Running:", " ".join(cmd))
    try:
        # Run YOLO from the cars/ directory so that all runs are written under
        # cars/runs/... instead of the repository root.
        subprocess.run(cmd, check=True, cwd=str(cars_dir))
    except FileNotFoundError:
        print(
            "[train] 'yolo' CLI not found. Install Ultralytics with 'pip install ultralytics' and ensure 'yolo' is on your PATH."
        )
        return
    except subprocess.CalledProcessError as exc:
        print(f"[train] Training failed with exit code {exc.returncode}.")
        return

    # On successful training, promote the best checkpoint from the most recent
    # matching run directory into cars/models/.
    runs_root = cars_dir / "runs" / "detect" / project
    candidate_dirs: List[Path] = []
    if runs_root.exists():
        for d in runs_root.iterdir():
            if d.is_dir() and (d.name == name or d.name.startswith(f"{name}")):
                candidate_dirs.append(d)

    if not candidate_dirs:
        # Fallback to the original expected layout in case the above lookup
        # fails for any reason.
        candidate_dirs = [cars_dir / "runs" / "detect" / project / name]

    # Pick the most recently modified candidate directory.
    run_dir = max(candidate_dirs, key=lambda p: p.stat().st_mtime)
    best_src = run_dir / "weights" / "best.pt"
    if not best_src.exists():
        print(f"[train] Expected best checkpoint not found at {best_src}.")
        return

    models_dir = cars_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_dst = models_dir / f"{name}_best.pt"

    try:
        shutil.copy2(best_src, best_dst)
        print(f"[train] Copied best checkpoint from {best_src} to {best_dst}.")
    except OSError as exc:
        print(f"[train] Failed to copy best checkpoint to {best_dst}: {exc}.")


def run_inference():
    """Run YOLOv8 inference using the trained model.

    This function reads an ``inference`` section from cars/config.yaml and
    forwards the settings to the Ultralytics ``yolo predict`` CLI. By default
    it uses the best checkpoint promoted to ``cars/models`` by ``train_yolov8``.
    """

    cars_dir = _cars_dir()
    config_path = cars_dir / "config.yaml"

    if not config_path.exists():
        print(f"[infer] Dataset/config file not found at {config_path}.")
        return

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    infer_cfg = cfg.get("inference") or {}
    train_cfg = cfg.get("training") or {}

    # Prefer an explicit inference model; fall back to the training name-based
    # best checkpoint, then finally to the training base model.
    model = infer_cfg.get("model")
    if not model:
        fallback_best = f"models/{train_cfg.get('name', 'cars')}_best.pt"
        model = infer_cfg.get("model", fallback_best)

    model = str(model)
    source = str(infer_cfg.get("source", "data/test/images"))
    imgsz = str(infer_cfg.get("imgsz", train_cfg.get("imgsz", 640)))
    conf = str(infer_cfg.get("conf", 0.25))
    iou = str(infer_cfg.get("iou", 0.45))
    device = str(infer_cfg.get("device", train_cfg.get("device", 0)))
    project = str(infer_cfg.get("project", "cars"))
    name = str(infer_cfg.get("name", "yolov8s_cars_infer"))

    model_path = cars_dir / model
    if not model_path.exists():
        print(f"[infer] Model checkpoint not found at {model_path}.")
        return

    cmd = [
        "yolo",
        "predict",
        f"model={model}",
        f"source={source}",
        f"imgsz={imgsz}",
        f"conf={conf}",
        f"iou={iou}",
        f"device={device}",
        f"project={project}",
        f"name={name}",
        "save_txt=True",
        "save_conf=True",
    ]

    print("[infer] Running:", " ".join(cmd))
    try:
        # Run YOLO from the cars/ directory so that all runs are written under
        # cars/runs/... instead of the repository root.
        subprocess.run(cmd, check=True, cwd=str(cars_dir))
    except FileNotFoundError:
        print(
            "[infer] 'yolo' CLI not found. Install Ultralytics with 'pip install ultralytics' and ensure 'yolo' is on your PATH."
        )
    except subprocess.CalledProcessError as exc:
        print(f"[infer] Inference failed with exit code {exc.returncode}.")


def run_validation():
    """Run YOLOv8 validation and persist metrics to disk.

    Uses the Ultralytics Python API so that metrics are saved alongside the
    standard validation plots.
    """

    try:
        import importlib

        ultralytics_module = importlib.import_module("ultralytics")
    except ImportError:
        print(
            "[val] Ultralytics is not installed. Install with 'pip install ultralytics'."
        )
        return

    YOLO = getattr(ultralytics_module, "YOLO", None)
    if YOLO is None:
        print("[val] Ultralytics YOLO API not found.")
        return

    cars_dir = _cars_dir()
    config_path = cars_dir / "config.yaml"

    if not config_path.exists():
        print(f"[val] Dataset/config file not found at {config_path}.")
        return

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    train_cfg = cfg.get("training") or {}
    val_cfg = cfg.get("validation") or {}

    model = val_cfg.get("model")
    if not model:
        model = f"models/{train_cfg.get('name', 'cars')}_best.pt"

    imgsz = int(val_cfg.get("imgsz", train_cfg.get("imgsz", 640)))
    batch = int(val_cfg.get("batch", train_cfg.get("batch", 16)))
    device = str(val_cfg.get("device", train_cfg.get("device", 0)))
    project = str(val_cfg.get("project", "runs/detect"))
    name = str(val_cfg.get("name", "val"))

    model_path = cars_dir / model
    if not model_path.exists():
        print(f"[val] Model checkpoint not found at {model_path}.")
        return

    project_path = Path(project)
    if not project_path.is_absolute():
        project_path = cars_dir / project_path

    print(
        "[val] Running validation with model",
        model,
        "->",
        project_path / name,
    )

    try:
        yolo_model = YOLO(str(model_path))
        results = yolo_model.val(
            data=str(config_path),
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=str(project_path),
            name=name,
        )
    except Exception as exc:
        print(f"[val] Validation failed: {exc}")
        return

    save_dir = Path(getattr(results, "save_dir", project_path / name))
    metrics = getattr(results, "results_dict", {}) or {}

    if not metrics:
        print("[val] No metrics dictionary returned to save.")
        return

    metrics_path = save_dir / "metrics.json"
    metrics_csv_path = save_dir / "metrics.csv"

    try:
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        with metrics_csv_path.open("w", newline="", encoding="utf-8") as mf:
            writer = csv.writer(mf)
            writer.writerow(["metric", "value"])
            for key, value in metrics.items():
                writer.writerow([key, value])
        print(f"[val] Saved metrics to {metrics_path} and {metrics_csv_path}.")
    except OSError as exc:
        print(f"[val] Failed to write metrics files: {exc}.")


def export_geojson():
    """Export YOLO detections to GeoJSON in EPSG:3857."""

    try:
        import cv2
    except ImportError:
        print("[export] OpenCV not available. Install opencv-python.")
        return

    cars_dir = _cars_dir()
    config_path = cars_dir / "config.yaml"

    if not config_path.exists():
        print(f"[export] Dataset/config file not found at {config_path}.")
        return

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    infer_cfg = cfg.get("inference") or {}
    export_cfg = cfg.get("export") or {}

    project = str(export_cfg.get("project", infer_cfg.get("project", "cars")))
    name = str(export_cfg.get("name", infer_cfg.get("name", "yolov8s_cars_infer")))
    source = str(export_cfg.get("source", infer_cfg.get("source", "data/test/images")))
    output = str(export_cfg.get("output", "exports/detections.geojson"))

    project_path = Path(project)
    if not project_path.is_absolute():
        run_dir = cars_dir / "runs" / "detect" / project_path / name
    else:
        run_dir = project_path / name

    labels_dir = run_dir / "labels"
    if not labels_dir.exists():
        print(f"[export] Labels directory not found at {labels_dir}.")
        return

    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = cars_dir / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_path = cars_dir / "data" / "manifest.csv"
    manifest_index: Dict[str, Dict[str, str]] = {}
    if manifest_path.exists():
        with manifest_path.open("r", newline="", encoding="utf-8") as mf:
            reader = csv.DictReader(mf)
            for row in reader:
                rel_path = row.get("rel_path") or ""
                name_key = Path(rel_path).name
                if not name_key:
                    continue
                manifest_index[name_key] = row

    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = cars_dir / source_path

    def _find_image_path(stem: str, fallback_name: str) -> Path:
        if fallback_name:
            candidate = source_path / fallback_name
            if candidate.exists():
                return candidate
        for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"):
            candidate = source_path / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        return Path()

    def _float_or_none(value):
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    features: List[Dict] = []
    missing_bounds = 0
    missing_images = 0

    for label_path in labels_dir.glob("*.txt"):
        stem = label_path.stem
        manifest_row = manifest_index.get(stem)
        image_file = ""
        image_id = stem
        bounds = None
        split = None

        if manifest_row:
            image_file = manifest_row.get("rel_path", "")
            image_id = manifest_row.get("image_id", stem)
            split = manifest_row.get("split")
            xmin_val = _float_or_none(manifest_row.get("xmin"))
            ymin_val = _float_or_none(manifest_row.get("ymin"))
            xmax_val = _float_or_none(manifest_row.get("xmax"))
            ymax_val = _float_or_none(manifest_row.get("ymax"))
            if None not in (xmin_val, ymin_val, xmax_val, ymax_val):
                bounds = cast(
                    Tuple[float, float, float, float],
                    (xmin_val, ymin_val, xmax_val, ymax_val),
                )

        if bounds is None:
            missing_bounds += 1
            continue

        image_path = _find_image_path(stem, Path(image_file).name)
        if not image_path.exists() and image_file:
            image_path = cars_dir / image_file

        if not image_path.exists():
            missing_images += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            missing_images += 1
            continue

        height, width = image.shape[:2]
        if width == 0 or height == 0:
            missing_images += 1
            continue

        xmin, ymin, xmax, ymax = bounds
        x_span = xmax - xmin
        y_span = ymax - ymin

        with label_path.open("r", encoding="utf-8") as lf:
            for line in lf:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) not in {5, 6}:
                    continue
                class_id = int(float(parts[0]))
                x_center = float(parts[1]) * width
                y_center = float(parts[2]) * height
                box_w = float(parts[3]) * width
                box_h = float(parts[4]) * height
                conf = float(parts[5]) if len(parts) == 6 else None

                xmin_px = x_center - box_w / 2.0
                xmax_px = x_center + box_w / 2.0
                ymin_px = y_center - box_h / 2.0
                ymax_px = y_center + box_h / 2.0

                xmin_world = xmin + (xmin_px / width) * x_span
                xmax_world = xmin + (xmax_px / width) * x_span
                ymax_world = ymax - (ymin_px / height) * y_span
                ymin_world = ymax - (ymax_px / height) * y_span

                coords = [
                    [xmin_world, ymin_world],
                    [xmax_world, ymin_world],
                    [xmax_world, ymax_world],
                    [xmin_world, ymax_world],
                    [xmin_world, ymin_world],
                ]

                props = {
                    "image_id": image_id,
                    "image_file": image_file or image_path.name,
                    "class_id": class_id,
                }
                if split:
                    props["split"] = split
                if conf is not None:
                    props["confidence"] = conf

                features.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "Polygon", "coordinates": [coords]},
                        "properties": props,
                    }
                )

    geojson = {
        "type": "FeatureCollection",
        "name": "yolo_detections",
        "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
        "features": features,
    }

    try:
        output_path.write_text(json.dumps(geojson, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"[export] Failed to write GeoJSON to {output_path}: {exc}.")
        return

    print(
        f"[export] Wrote {len(features)} detections to {output_path}. "
        f"Missing bounds: {missing_bounds}, missing images: {missing_images}."
    )


def visualize_map():
    print("Visualizing detections on a map... (placeholder)")


def main():
    ap = argparse.ArgumentParser(
        description="Starter end-to-end car-detection pipeline (YOLOv8)."
    )
    ap.add_argument(
        "step",
        choices=[
            "ingest",
            "prepare",
            "train",
            "val",
            "infer",
            "export",
            "visualize",
            "all",
        ],
        help="Step to execute",
    )
    args = ap.parse_args()
    if args.step == "ingest":
        ingest_data()
    elif args.step == "prepare":
        convert_to_yolo_format()
    elif args.step == "train":
        train_yolov8()
    elif args.step == "val":
        run_validation()
    elif args.step == "infer":
        run_inference()
    elif args.step == "export":
        export_geojson()
    elif args.step == "visualize":
        visualize_map()
    elif args.step == "all":
        ingest_data()
        convert_to_yolo_format()
        train_yolov8()
        run_inference()
        export_geojson()
        visualize_map()


if __name__ == "__main__":
    main()
