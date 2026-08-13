import argparse
from pathlib import Path

from src.pipeline import run_inspection


def parse_args():
    parser = argparse.ArgumentParser(
        description="AVIS — Automatic Vehicle Inspection System"
    )

    parser.add_argument(
        "--image",
        type=Path,
        default=Path("./run_check/ex.jpg"),
        help="Path to vehicle image",
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=Path("./reports/report.html"),
        help="Path to save HTML inspection report",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    image_path = args.image
    report_path = args.report

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("AVIS — Automatic Vehicle Inspection System")
    print("=" * 60)

    print(f"Image: {image_path}")
    print()

    inspection_result = run_inspection(
        img_path=image_path,
        report_save_folder=report_path.parent)

    print()
    print("=" * 60)
    print("Inspection completed")
    print("=" * 60)

    vehicle = inspection_result["vehicle"]
    def format_prediction(field):
        status = field.get("status")
    
        if status == "certain":
            return (
                f"{field['value']} "
                f"({field['confidence']:.2f})"
            )
    
        if status == "uncertain":
            return " / ".join(
                f"{candidate['value']} "
                f"({candidate['confidence']:.2f})"
                for candidate in field.get("candidates", [])[:2]
            )
    
        return "N/A"
    
    print(
    f"Brand: {format_prediction(vehicle['brand'])}")
    
    print(
        f"Type: {format_prediction(vehicle['type'])}")
    
    print(
        f"Color: {format_prediction(vehicle['color'])}")
    
    print(
        f"View: {format_prediction(vehicle['view'])}")
    
    plate = inspection_result.get(
        "license_plate",
        {})

    print(
        f"License plate: "
        f"{plate.get('text', 'N/A')}"
    )

    damage_summary = (
        inspection_result
        .get("damage_analysis", {})
        .get("summary", {})
    )

    print(
        f"Damaged parts: "
        f"{damage_summary.get('damaged_parts_count', 0)}"
    )

    print(
        f"Damage detections: "
        f"{damage_summary.get('damage_count', 0)}"
    )

    print()
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()