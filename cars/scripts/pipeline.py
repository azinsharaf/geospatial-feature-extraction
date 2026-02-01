"""
Starter end-to-end pipeline for car detection in satellite imagery using YOLOv8.
This is a skeleton script with placeholders for each step of the workflow.
"""

import argparse


def ingest_data():
    print("Ingesting satellite imagery... (placeholder)")


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
