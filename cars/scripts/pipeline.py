"""Starter end-to-end pipeline for car detection in satellite imagery.

This module wires together the main steps of the workflow and provides a
minimal, working implementation of the data-ingestion step that pulls tiles
from a WMTS service using AOIs defined in GeoJSON (EPSG:3857).
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Tuple

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
    """Load WMTS configuration from cars/data.yaml.

    The YAML file is expected to contain a top-level ``wmts`` section.
    """

    cars_dir = _cars_dir()
    data_yaml_path = cars_dir / "data.yaml"
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Expected WMTS config at {data_yaml_path}")

    with data_yaml_path.open("r", encoding="utf-8") as f:
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
        raise KeyError(f"Missing WMTS config keys in cars/data.yaml: {missing}")

    return wmts_cfg


def _zoom_from_tile_matrix(tile_matrix_value) -> int:
    """Interpret wmts.tile_matrix as a Web Mercator zoom level.

    For this first implementation we assume that ``wmts.tile_matrix`` is a
    standard XYZ-style zoom between 0 and 30. If this is not the case for
    your WMTS service, update ``cars/data.yaml`` accordingly.
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
            f"got {tile_matrix_value!r}. Update cars/data.yaml to a reasonable zoom."
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

    - Reads WMTS configuration from ``cars/data.yaml``.
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
    print("Converting annotations to YOLO format... (placeholder)")


def train_yolov8():
    print("Training YOLOv8 model... (placeholder)")


def run_inference():
    print("Running inference... (placeholder)")


def export_geojson():
    print("Exporting detections to GeoJSON... (placeholder)")


def visualize_map():
    print("Visualizing detections on a map... (placeholder)")


def main():
    ap = argparse.ArgumentParser(
        description="Starter end-to-end car-detection pipeline (YOLOv8)."
    )
    ap.add_argument(
        "step",
        choices=["ingest", "prepare", "train", "infer", "export", "visualize", "all"],
        help="Step to execute",
    )
    args = ap.parse_args()
    if args.step == "ingest":
        ingest_data()
    elif args.step == "prepare":
        convert_to_yolo_format()
    elif args.step == "train":
        train_yolov8()
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
