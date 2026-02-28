# Tree Canopy Segmentation Pipeline

This directory hosts the independent workflow for extracting tree canopy from WMTS imagery.

## First step: WMTS ingestion

1. Populate `tree_canopy/config.yaml` with the WMTS endpoint, layer name, tile matrix, and format for your provider. Optionally provide a `resource_url_template` if your WMTS offers REST-style tile URLs.
2. Prepare an AOI GeoJSON (EPSG:3857) and place it under `tree_canopy/aois/` or pass an absolute path.
3. Run:

```
python tree_canopy/scripts/pipeline.py ingest --run-id <id> --aoi <aoi-file>
```

The command downloads tiles into `tree_canopy/data/aoi_runs/<run_id>/<split>/images/` and appends metadata to `tree_canopy/data/aoi_runs/<run_id>/manifest.csv`.

Next steps (prepare, train, infer, export) will be implemented separately inside this folder once the ingestion baseline is stable.
