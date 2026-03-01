"""WMTS ingestion pipeline for tree canopy segmentation imagery."""

import argparse
import csv
import os
import random
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import requests
import yaml

# Web Mercator world bounds (EPSG:3857) used for deriving tile math
WEB_MERCATOR_BOUNDS: Tuple[float, float, float, float] = (
    -20037508.342789244,
    -20037508.342789244,
    20037508.342789244,
    20037508.342789244,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _tree_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_tree_config() -> Dict:
    config_path = _tree_dir() / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _load_wmts_config() -> Dict[str, str]:
    cfg = _load_tree_config()
    wmts_cfg = cfg.get("wmts") or {}
    required_keys = [
        "url",
        "layer",
        "tile_matrix_set",
        "tile_matrix",
        "style",
        "format",
    ]

    missing = [key for key in required_keys if key not in wmts_cfg]
    if missing:
        raise KeyError(f"Missing WMTS keys in config.yaml: {missing}")

    return wmts_cfg


def _load_dataset_split_config() -> Optional[Dict[str, float]]:
    cfg = _load_tree_config()
    split_cfg = cfg.get("dataset_split")
    if not split_cfg:
        return None

    try:
        train_pct = float(split_cfg.get("train", 0))
        val_pct = float(split_cfg.get("val", 0))
        test_pct = float(split_cfg.get("test", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("dataset_split values must be numeric") from exc

    if any(pct < 0 for pct in (train_pct, val_pct, test_pct)):
        raise ValueError("dataset_split values must be >= 0")

    total = train_pct + val_pct + test_pct
    if total <= 0:
        raise ValueError("dataset_split must total more than 0")

    if abs(total - 100.0) > 0.001:
        raise ValueError("dataset_split values must total 100")

    return {"train": train_pct, "val": val_pct, "test": test_pct}


def _zoom_from_tile_matrix(tile_matrix_value) -> int:
    try:
        zoom = int(tile_matrix_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"tile_matrix must be an integer zoom level, got {tile_matrix_value!r}"
        ) from exc

    if not (0 <= zoom <= 30):
        raise ValueError("tile_matrix should be between 0 and 30")

    return zoom


def _bbox_to_tile_range(
    bbox: Tuple[float, float, float, float], zoom: int
) -> Tuple[int, int, int, int]:
    minx, miny, maxx, maxy = bbox
    world_minx, world_miny, world_maxx, world_maxy = WEB_MERCATOR_BOUNDS
    n = 2**zoom

    tile_size_x = (world_maxx - world_minx) / n
    tile_size_y = (world_maxy - world_miny) / n

    def _col(x: float) -> int:
        return int((x - world_minx) / tile_size_x)

    def _row(y: float) -> int:
        return int((world_maxy - y) / tile_size_y)

    col_min = max(0, _col(minx))
    col_max = min(n - 1, _col(maxx))
    row_min = max(0, _row(maxy))
    row_max = min(n - 1, _row(miny))

    return row_min, row_max, col_min, col_max


def _tile_bounds(zoom: int, row: int, col: int) -> Tuple[float, float, float, float]:
    world_minx, world_miny, world_maxx, world_maxy = WEB_MERCATOR_BOUNDS
    n = 2**zoom
    tile_size_x = (world_maxx - world_minx) / n
    tile_size_y = (world_maxy - world_miny) / n

    xmin = world_minx + col * tile_size_x
    xmax = xmin + tile_size_x
    ymax = world_maxy - row * tile_size_y
    ymin = ymax - tile_size_y
    return xmin, ymin, xmax, ymax


def _format_extension(fmt: str) -> str:
    fmt = (fmt or "").lower()
    if "jpeg" in fmt or "jpg" in fmt:
        return ".jpg"
    if "png" in fmt:
        return ".png"
    if "tiff" in fmt or "tif" in fmt:
        return ".tif"
    return ".img"


def _build_tile_url(wmts_cfg: Dict[str, str], tile_row: int, tile_col: int) -> str:
    template = wmts_cfg.get("resource_url_template")
    if template:
        return template.format(
            Style=wmts_cfg.get("style", "default"),
            TileMatrixSet=wmts_cfg["tile_matrix_set"],
            TileMatrix=wmts_cfg["tile_matrix"],
            TileCol=tile_col,
            TileRow=tile_row,
            layer=wmts_cfg.get("layer", ""),
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
    session = requests.Session()
    api_key = os.getenv("WMTS_API_KEY")
    if api_key:
        session.headers.update({"Authorization": f"Bearer {api_key}"})
    return session


def ingest_data(run_id: Optional[str], aoi_path: Optional[str]) -> None:
    if not run_id or not aoi_path:
        print("[ingest] --run-id and --aoi are required for ingest.")
        return

    tree_dir = _tree_dir()
    data_root = tree_dir / "data" / "aoi_runs" / run_id
    data_root.mkdir(parents=True, exist_ok=True)

    try:
        wmts_cfg = _load_wmts_config()
    except Exception as exc:
        print(f"[ingest] Failed to load WMTS config: {exc}")
        return

    try:
        split_cfg = _load_dataset_split_config()
    except Exception as exc:
        print(f"[ingest] Failed to load dataset_split config: {exc}")
        return

    zoom = _zoom_from_tile_matrix(wmts_cfg["tile_matrix"])
    overlap_percent = wmts_cfg.get("overlap_percent")
    overlap_tiles = wmts_cfg.get("overlap_tiles")
    overlap_ratio = 0.0
    overlap_tile_count = 0

    if overlap_percent is not None:
        try:
            overlap_ratio = float(overlap_percent) / 100.0
        except (TypeError, ValueError):
            print(
                f"[ingest] overlap_percent must be a number, got {overlap_percent!r}"
            )
            return

        if not (0.0 <= overlap_ratio <= 1.0):
            print("[ingest] overlap_percent must be between 0 and 100.")
            return
    elif overlap_tiles is not None:
        try:
            overlap_tile_count = int(overlap_tiles)
        except (TypeError, ValueError):
            print(
                f"[ingest] overlap_tiles must be an integer, got {overlap_tiles!r}"
            )
            return

        if overlap_tile_count < 0:
            print("[ingest] overlap_tiles must be >= 0.")
            return

    aoi_candidates: List[Path]
    candidate = Path(aoi_path)
    if candidate.is_absolute():
        aoi_candidates = [candidate]
    else:
        aoi_candidates = [
            Path.cwd() / candidate,
            tree_dir / candidate,
            tree_dir / "aois" / candidate,
            tree_dir / "aois" / candidate.name,
        ]

    resolved_aoi: Optional[Path] = None
    for path in aoi_candidates:
        if path.exists():
            resolved_aoi = path
            break

    if resolved_aoi is None:
        print("[ingest] AOI file not found. Checked:")
        for path in aoi_candidates:
            print("  ", path)
        return

    try:
        aoi_gdf = gpd.read_file(resolved_aoi)
    except Exception as exc:
        print(f"[ingest] Failed to read AOI file: {exc}")
        return

    if aoi_gdf.empty:
        print("[ingest] AOI has no features.")
        return

    if aoi_gdf.crs is None:
        print("[ingest] AOI has no CRS; assuming EPSG:3857.")
        aoi_gdf = aoi_gdf.set_crs(epsg=3857)
    elif str(aoi_gdf.crs.to_epsg()) != "3857":
        print(f"[ingest] Reprojecting AOI from {aoi_gdf.crs} to EPSG:3857.")
        aoi_gdf = aoi_gdf.to_crs(epsg=3857)

    manifest_path = data_root / "manifest.csv"
    manifest_exists = manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        if not manifest_exists:
            writer.writerow([
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
            ])

        session = _create_wmts_session()
        img_ext = _format_extension(wmts_cfg.get("format", "image/png"))
        total = 0

        for idx, row in aoi_gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            site_id = str(row.get("site_id") or row.get("SITE_ID") or f"site_{idx}")
            split = str(row.get("split") or "train").lower()
            if split not in {"train", "val", "test"}:
                split = "train"

            geom_bounds = geom.bounds
            if overlap_ratio:
                world_minx, world_miny, world_maxx, world_maxy = WEB_MERCATOR_BOUNDS
                tiles_across = 2**zoom
                tile_size_x = (world_maxx - world_minx) / tiles_across
                tile_size_y = (world_maxy - world_miny) / tiles_across
                pad_x = tile_size_x * overlap_ratio
                pad_y = tile_size_y * overlap_ratio
                minx, miny, maxx, maxy = geom_bounds
                geom_bounds = (
                    minx - pad_x,
                    miny - pad_y,
                    maxx + pad_x,
                    maxy + pad_y,
                )

            row_min, row_max, col_min, col_max = _bbox_to_tile_range(geom_bounds, zoom)
            if overlap_tile_count:
                max_index = (2**zoom) - 1
                row_min = max(0, row_min - overlap_tile_count)
                row_max = min(max_index, row_max + overlap_tile_count)
                col_min = max(0, col_min - overlap_tile_count)
                col_max = min(max_index, col_max + overlap_tile_count)
            split_images = data_root / split / "images"
            split_images.mkdir(parents=True, exist_ok=True)

            for tile_row in range(row_min, row_max + 1):
                for tile_col in range(col_min, col_max + 1):
                    image_id = (
                        f"{wmts_cfg['layer']}_z{wmts_cfg['tile_matrix']}_r{tile_row}_c{tile_col}"
                    )
                    img_name = image_id + img_ext
                    img_path = split_images / img_name

                    if not img_path.exists():
                        url = _build_tile_url(wmts_cfg, tile_row, tile_col)
                        try:
                            resp = session.get(url, timeout=30)
                        except Exception as exc:
                            print(f"[ingest] Error fetching {image_id}: {exc}")
                            continue

                        if resp.status_code != 200:
                            print(
                                f"[ingest] Failed {image_id}: HTTP {resp.status_code}"
                            )
                            continue

                        img_path.write_bytes(resp.content)

                    xmin, ymin, xmax, ymax = _tile_bounds(zoom, tile_row, tile_col)
                    rel_path = img_path.relative_to(_tree_dir()).as_posix()
                    writer.writerow([
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
                    ])
                    total += 1

        if split_cfg:
            _redistribute_tiles(data_root, manifest_path, run_id, split_cfg)

        print(f"[ingest] Completed {total} tiles; manifest at {manifest_path}.")


def _redistribute_tiles(
    data_root: Path, manifest_path: Path, run_id: str, split_cfg: Dict[str, float]
) -> None:
    if not manifest_path.exists():
        return

    with manifest_path.open("r", newline="", encoding="utf-8") as mf:
        reader = csv.DictReader(mf)
        rows = list(reader)

    if not rows:
        return

    rel_paths = sorted({row.get("rel_path", "") for row in rows if row.get("rel_path")})
    if not rel_paths:
        return

    total = len(rel_paths)
    val_count = int(round(total * (split_cfg["val"] / 100.0)))
    test_count = int(round(total * (split_cfg["test"] / 100.0)))
    train_count = max(0, total - val_count - test_count)

    rng = random.Random(42)
    rng.shuffle(rel_paths)

    split_map: Dict[str, str] = {}
    for rel_path in rel_paths[:train_count]:
        split_map[rel_path] = "train"
    for rel_path in rel_paths[train_count : train_count + val_count]:
        split_map[rel_path] = "val"
    for rel_path in rel_paths[train_count + val_count :]:
        split_map[rel_path] = "test"

    tree_dir = _tree_dir()
    for rel_path, split in split_map.items():
        src_path = tree_dir / rel_path
        if not src_path.exists():
            continue

        dest_dir = data_root / split / "images"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / src_path.name
        if dest_path.resolve() == src_path.resolve():
            continue
        if dest_path.exists():
            dest_path.unlink()
        shutil.move(str(src_path), str(dest_path))

    updated_rows: List[Dict[str, str]] = []
    for row in rows:
        rel_path = row.get("rel_path", "")
        split = split_map.get(rel_path, row.get("split", "train"))
        filename = Path(rel_path).name if rel_path else ""
        new_rel_path = Path("data") / "aoi_runs" / run_id / split / "images" / filename
        row["split"] = split
        row["rel_path"] = new_rel_path.as_posix()
        updated_rows.append(row)

    temp_path = manifest_path.with_suffix(".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as mf:
        writer = csv.DictWriter(mf, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(updated_rows)

    temp_path.replace(manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tree canopy WMTS ingestion.")
    parser.add_argument("step", choices=["ingest"], help="Pipeline step to run")
    parser.add_argument("--run-id", required=True, help="Identifier for this ingest run")
    parser.add_argument("--aoi", required=True, help="AOI GeoJSON path (relative to tree_canopy/aois or absolute)")

    args = parser.parse_args()
    if args.step == "ingest":
        ingest_data(run_id=args.run_id, aoi_path=args.aoi)


if __name__ == "__main__":
    main()
