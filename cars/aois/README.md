# AOI files

Place Area Of Interest (AOI) GeoJSON files in this folder.

The pipeline supports passing an AOI file to ingest via `--aoi`:

```bash
python cars/scripts/pipeline.py ingest --run-id <run_id> --aoi <aoi_file>
```

`<aoi_file>` can be:

- an absolute path
- a path relative to `cars/` (e.g. `aois/oakland.geojson`)
- a bare filename that exists in this folder (e.g. `oakland.geojson`)

Each feature can include properties like:

- `split`: one of `train`, `val`, or `test` (defaults to `train` if omitted)
- `site_id`: optional identifier used in the manifest
