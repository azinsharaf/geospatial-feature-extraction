# AOI files

Place Area Of Interest (AOI) GeoJSON files in this folder.

The pipeline supports passing an AOI file to ingest via `--aoi`:

```bash
python cars/scripts/pipeline.py ingest --run-id <run_id> --aoi <aoi_file>
```

`<aoi_file>` can be:

- an absolute path
- a repo-relative path (e.g. `./cars/aois/oakland_aoi_02222026.geojson`)
- a path relative to `cars/` (e.g. `aois/oakland_aoi_02222026.geojson`)
- a bare filename that exists in this folder (e.g. `oakland_aoi_02222026.geojson`)

Recommended naming convention:

- `<place>_aoi_<yyyymmdd>.geojson` for AOI inputs
- Use the same `<place>_<yyyymmdd>` as the `--run-id` so outputs are easy to locate

Each feature can include properties like:

- `split`: one of `train`, `val`, or `test` (defaults to `train` if omitted)
- `site_id`: optional identifier used in the manifest

CRS notes:

- Prefer EPSG:3857 coordinates (Web Mercator meters).
- If the AOI has no CRS metadata, the pipeline assumes EPSG:3857.
