from pathlib import Path
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automatic Vehicle Inspection System"
    )

    parser.add_argument(
        "image",
        type=Path,
        nargs="?",
        help="Path to the vehicle image",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.image is None:
        print("Automatic Vehicle Inspection System")
        print()
        print("Usage:")
        print("  python main.py path/to/image.jpg")
        return

    if not args.image.is_file():
        raise FileNotFoundError(
            f"Image not found: {args.image}"
        )

    print(f"Input image: {args.image}")
    print("AVIS pipeline will be executed here.")


if __name__ == "__main__":
    main()