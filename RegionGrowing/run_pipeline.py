import argparse
from pathlib import Path

from RegionGrowing.config_loader import load_config
from RegionGrowing.pipeline.classify import run_classification
from RegionGrowing.pipeline.segment import run_segmentation
from RegionGrowing.pipeline.measure import run_measurement


def resolve_output_dir(user_path: str) -> Path:
    """
    If user gives a valid directory → use it.
    If user gives '.' or '' → use root /outputs/
    """
    if not user_path or user_path.strip() in {".", "./"}:
        out = Path("outputs")
    else:
        out = Path(user_path)

    out.mkdir(parents=True, exist_ok=True)
    return out


def run_pipeline(input_file, output_folder, cfg):
    output_folder = resolve_output_dir(output_folder)

    classified_file = output_folder / "classified.las"
    segmented_file = output_folder / "segmented.las"
    map_png = output_folder / "room_map.png"
    areas_csv = output_folder / "room_areas.csv"
    polygons_json = output_folder / "room_polygons.json"

    print("=" * 70)
    print("PIPELINE START")
    print("=" * 70)

    # STEP 1 — CLASSIFICATION
    print("\nRunning STEP 1: CLASSIFICATION")
    run_classification(
        input_file=input_file,
        output_file=str(classified_file),
        cfg=cfg["classification"]
    )

    # STEP 2 — SEGMENTATION
    print("\nRunning STEP 2: SEGMENTATION")
    seg_results = run_segmentation(
        input_file=str(classified_file),
        output_file=str(segmented_file),
        cfg=cfg["segmentation"]
    )

    if seg_results is None:
        print("\nSegmentation failed → stopping.")
        return

    # STEP 3 — MEASUREMENT
    print("\nRunning STEP 3: MEASUREMENT")
    run_measurement(
        segmented_file=str(segmented_file),
        png_out=str(map_png),
        csv_out=str(areas_csv),
        json_out=str(polygons_json),
        cfg=cfg["measurement"]
    )

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_folder}")


def main():
    parser = argparse.ArgumentParser(description="Run full 3D room analysis pipeline")

    parser.add_argument("input", help="Input LAS/LAZ file")
    parser.add_argument(
        "output",
        help="Output directory ('.' or empty → uses /outputs/ automatically)"
    )
    parser.add_argument("--config", help="Optional override YAML config", default=None)

    args = parser.parse_args()

    cfg = load_config(
        default_path="configs/default.yaml",
        override_path=args.config
    )

    run_pipeline(args.input, args.output, cfg)

if __name__ == "__main__":
    main()
