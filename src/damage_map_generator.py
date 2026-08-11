import csv

from PIL import Image, ImageDraw


def parse_points(points_string):
    """
    Convert:
        "100:200;150:220;..."
    into:
        [(100, 200), (150, 220), ...]
    """

    points = []

    for point in points_string.split(";"):

        if not point:
            continue

        x, y = point.split(":")

        points.append(
            (int(float(x)), int(float(y)))
        )

    return points


def generate_damage_map(
    base_image_path,
    polygons_csv_path,
    damaged_parts,
    damage_color=(255, 0, 0, 150),
):
    """
    Generate damage map using polygons stored in CSV.

    Parameters
    ----------
    base_image_path : str
        Path to car_map.png.

    polygons_csv_path : str
        Path to car_part_polygons.csv.

    damaged_parts : list[str]
        Parts that should be highlighted.

    damage_color : tuple
        RGBA color.

    Returns
    -------
    PIL.Image.Image
        Final damage map.
    """

    # --------------------------------------------------
    # Load base scheme
    # --------------------------------------------------

    base = Image.open(
        base_image_path
    ).convert("RGBA")

    overlay = Image.new(
        "RGBA",
        base.size,
        (0, 0, 0, 0)
    )

    draw = ImageDraw.Draw(overlay)

    # --------------------------------------------------
    # Read polygons
    # --------------------------------------------------

    with open(
        polygons_csv_path,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            part_name = row["part_name"]

            if part_name not in damaged_parts:
                continue

            # ------------------------------------------
            # Main polygon
            # ------------------------------------------

            points = parse_points(
                row["points_px"]
            )

            draw.polygon(
                points,
                fill=damage_color
            )

            # ------------------------------------------
            # Polygon holes
            # ------------------------------------------

            holes = row.get(
                "holes_px",
                ""
            )

            if holes:

                for hole_string in holes.split("|"):

                    if not hole_string:
                        continue

                    hole_points = parse_points(
                        hole_string
                    )

                    draw.polygon(
                        hole_points,
                        fill=(0, 0, 0, 0)
                    )

    # --------------------------------------------------
    # Put original scheme above/below overlay
    # --------------------------------------------------

    result = Image.alpha_composite(
        base,
        overlay
    )

    return result