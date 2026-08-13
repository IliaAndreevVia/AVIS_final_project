import base64
import html as html_lib

from pathlib import Path

import cv2
import numpy as np

from IPython.display import HTML, display

def classification_to_result(
    prediction,
    certain_threshold=0.75,
):
    """
    Convert raw classifier output to NLP-compatible format.

    If top-1 confidence >= certain_threshold:
        return one certain value.

    Otherwise:
        return two most probable candidates.
    """

    confidence = float(
        prediction["confidence"]
    )

    # -----------------------------------------------------
    # Certain
    # -----------------------------------------------------
    if confidence >= certain_threshold:

        return {
            "status": "certain",
            "value": prediction["value"],
            "confidence": confidence,
        }

    # -----------------------------------------------------
    # Uncertain
    # -----------------------------------------------------
    return {
        "status": "uncertain",
        "candidates": [
            {
                "value": candidate["value"],
                "confidence": float(
                    candidate["confidence"]
                ),
            }
            for candidate
            in prediction["top2"]
        ],
    }
    
def inspection_result_to_vehicle_data(
    inspection_result
):

    # =====================================================
    # View name converter
    # =====================================================

    def view_to_nlp_name(view):
        if view is None:
            return None

        return (
            str(view)
            .strip()
            .lower()
            .replace(" ", "_")
        )

    # =====================================================
    # Vehicle fields
    # =====================================================

    def make_vehicle_field(
        field,
        transform_value=None,
    ):
        status = field.get("status")

        # -------------------------------------------------
        # Certain
        # -------------------------------------------------
        if status == "certain":

            value = field.get("value")

            if transform_value is not None:
                value = transform_value(value)

            return {
                "status": "certain",
                "value": value,
                "confidence": float(
                    field.get(
                        "confidence",
                        0.0,
                    )
                ),
            }

        # -------------------------------------------------
        # Uncertain
        # -------------------------------------------------
        if status == "uncertain":

            candidates = []

            for candidate in field.get(
                "candidates",
                [],
            ):
                value = candidate["value"]

                if transform_value is not None:
                    value = transform_value(
                        value
                    )

                candidates.append({
                    "value": value,
                    "confidence": float(
                        candidate["confidence"]
                    ),
                })

            return {
                "status": "uncertain",
                "candidates": candidates,
            }

        # -------------------------------------------------
        # Unknown
        # -------------------------------------------------
        return {
            "status": "unknown",
            "value": None,
            "confidence": float(
                field.get(
                    "confidence",
                    0.0,
                )
            ),
        }

    # =====================================================
    # Vehicle information
    # =====================================================

    vehicle_data = {

        "brand": make_vehicle_field(
            inspection_result[
                "vehicle"
            ]["brand"]
        ),

        "vehicle_type": make_vehicle_field(
            inspection_result[
                "vehicle"
            ]["type"]
        ),

        "color": make_vehicle_field(
            inspection_result[
                "vehicle"
            ]["color"]
        ),

        "viewpoint": make_vehicle_field(
            inspection_result[
                "vehicle"
            ]["view"],
            transform_value=view_to_nlp_name,
        ),
    }

    # =====================================================
    # License plate
    # =====================================================

    plate_confidence = float(
        inspection_result[
            "license_plate"
        ].get(
            "confidence",
            0.0,
        )
    )

    plate_text = inspection_result[
        "license_plate"
    ].get(
        "text"
    )

    if plate_text is None:

        plate_status = "unknown"

    elif plate_confidence >= 0.82:

        plate_status = "certain"

    elif plate_confidence >= 0.42:

        plate_status = "uncertain"

    else:

        plate_status = "unknown"
        plate_text = None

    vehicle_data["license_plate"] = {
        "status": plate_status,
        "value": plate_text,
        "confidence": plate_confidence,
    }

    # =====================================================
    # Damages
    # =====================================================

    vehicle_data["damaged_parts"] = {

        part_name: [
            {
                "damage_type": damage[
                    "damage_type"
                ],
                "confidence": float(
                    damage.get(
                        "confidence",
                        0.0,
                    )
                ),
            }
            for damage
            in part_result["damages"]
        ]

        for part_name, part_result
        in inspection_result[
            "damage_analysis"
        ]["parts"].items()

        if len(
            part_result["damages"]
        ) > 0
    }

    return vehicle_data

def generate_vehicle_report(
    report_data,
    output_path="./reports/report.html",
):
    html = render_report_html(report_data)

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(html)

    return output_path

def generate_inspection_report(
    inspection_result,
    report_model="bart",
    save_path=None,
):
    """
    Generate HTML vehicle inspection report from inspection_result.

    Parameters
    ----------
    inspection_result : dict
        Combined result of the whole inspection pipeline.

    report_model : str
        NLP report to use:
        "bart", "flan_t5", "gpt2"

    save_path : str | Path | None
        If specified, saves report as HTML file.

    Returns
    -------
    str
        Generated HTML.
    """

    # =========================================================
    # Helper: convert image to base64
    # =========================================================
    def image_to_base64(image):

        if image is None:
            return None

        # Image path
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            if image is None:
                return None

        # PIL -> numpy
        if not isinstance(image, np.ndarray):
            try:
                image = np.array(image)
            except Exception:
                return None

        # Convert datatype if necessary
        if image.dtype != np.uint8:
            image = np.clip(
                image,
                0,
                255
            ).astype(np.uint8)

        # Encode
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        success, buffer = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, 90]
        )

        if not success:
            return None

        encoded = base64.b64encode(
            buffer
        ).decode("utf-8")

        return (
            f"data:image/jpeg;base64,"
            f"{encoded}"
        )


    # =========================================================
    # Helper
    # =========================================================
    def safe(value, default="N/A"):

        if value is None:
            return default

        return html_lib.escape(
            str(value)
        )


    # =========================================================
    # <<< ИСПРАВЛЕНИЕ 1
    # Helper for certain / uncertain vehicle fields
    # =========================================================
    def format_vehicle_field(field):
        """
        Format classification result.

        Certain:
            mazda (1.00)

        Uncertain:
            beige (0.55) / tan (0.31)
        """

        if not field:
            return (
                'N/A '
                '<span class="confidence">'
                '(0.00)'
                '</span>'
            )

        status = field.get("status")

        # -----------------------------------------------------
        # Uncertain -> show top-2 candidates
        # -----------------------------------------------------
        if status == "uncertain":

            candidates = field.get(
                "candidates",
                []
            )

            # Highest confidence first
            candidates = sorted(
                candidates,
                key=lambda candidate: candidate.get(
                    "confidence",
                    0.0
                ),
                reverse=True,
            )[:2]

            if not candidates:
                return (
                    'N/A '
                    '<span class="confidence">'
                    '(0.00)'
                    '</span>'
                )

            candidate_html = []

            for candidate in candidates:

                value = safe(
                    candidate.get("value")
                )

                confidence = float(
                    candidate.get(
                        "confidence",
                        0.0
                    )
                )

                candidate_html.append(
                    f'{value} '
                    f'<span class="confidence">'
                    f'({confidence:.2f})'
                    f'</span>'
                )

            return " / ".join(
                candidate_html
            )

        # -----------------------------------------------------
        # Certain / fallback
        # -----------------------------------------------------
        value = field.get("value")

        confidence = float(
            field.get(
                "confidence",
                0.0
            )
        )

        return (
            f'{safe(value)} '
            f'<span class="confidence">'
            f'({confidence:.2f})'
            f'</span>'
        )


    # =========================================================
    # Vehicle data
    # =========================================================
    vehicle = inspection_result.get(
        "vehicle",
        {}
    )

    color = vehicle.get(
        "color",
        {}
    )

    vehicle_type = vehicle.get(
        "type",
        {}
    )

    brand = vehicle.get(
        "brand",
        {}
    )

    view = vehicle.get(
        "view",
        {}
    )


    # =========================================================
    # License plate
    # =========================================================
    plate = inspection_result.get(
        "license_plate",
        {}
    )

    plate_text = plate.get(
        "text"
    )

    plate_conf = plate.get(
        "confidence"
    )

    plate_image = image_to_base64(
        plate.get("image")
    )


    # =========================================================
    # Original image
    # =========================================================
    original_image = image_to_base64(
        inspection_result.get(
            "original_image"
        )
    )


    # =========================================================
    # Damage map
    # =========================================================
    damage_map = image_to_base64(
        inspection_result.get(
            "damage_map"
        )
    )


    # =========================================================
    # Damage analysis
    # =========================================================
    damage_analysis = inspection_result.get(
        "damage_analysis",
        {}
    )

    parts = damage_analysis.get(
        "parts",
        {}
    )

    damage_rows = []
    damage_images = []

    total_damage_detections = 0
    damaged_part_names = []

    for part_name, part_result in parts.items():

        damages = part_result.get(
            "damages",
            []
        )

        if not damages:
            continue

        damaged_part_names.append(
            part_name
        )

        total_damage_detections += len(
            damages
        )

        # -----------------------------------------------------
        # Damage table
        # -----------------------------------------------------
        for damage in damages:

            damage_type = damage.get(
                "damage_type",
                "Unknown"
            )

            confidence = damage.get(
                "confidence"
            )

            area = damage.get(
                "area_percent"
            )

            damage_rows.append({
                "part": part_name,
                "damage": damage_type,
                "confidence": confidence,
                "area": area,
            })

        # -----------------------------------------------------
        # Part image
        # -----------------------------------------------------
        part_image = part_result.get(
            "image"
        )

        encoded_image = image_to_base64(
            part_image
        )

        if encoded_image is not None:

            damage_names = [
                damage.get(
                    "damage_type",
                    "Unknown"
                )
                for damage in damages
            ]

            damage_images.append({
                "part": part_name,
                "image": encoded_image,
                "damages": damage_names,
            })


    # =========================================================
    # Summary
    # =========================================================
    parts_detection = inspection_result.get(
        "parts_detection",
        {}
    )

    detected_parts = len(
        parts_detection.get(
            "parts",
            {}
        )
    )

    damaged_parts = len(
        damaged_part_names
    )


    # =========================================================
    # NLP report
    # =========================================================
    reports = inspection_result.get(
        "reports",
        {}
    )

    report_text = reports.get(
        report_model,
        ""
    )

    if report_text is None:
        report_text = ""

    report_text = html_lib.escape(
        str(report_text)
    )


    # =========================================================
    # Damage table HTML
    # =========================================================
    damage_table_html = ""

    for row in damage_rows:

        confidence = row["confidence"]
        area = row["area"]

        confidence_text = (
            f"{confidence:.2f}"
            if isinstance(
                confidence,
                (int, float)
            )
            else "N/A"
        )

        area_text = (
            f"{area:.2f}%"
            if isinstance(
                area,
                (int, float)
            )
            else "N/A"
        )

        damage_table_html += f"""
        <tr>
            <td>{safe(row["part"])}</td>
            <td>{safe(row["damage"])}</td>
            <td>{confidence_text}</td>
            <td>{area_text}</td>
        </tr>
        """


    # =========================================================
    # Damage photos HTML
    # =========================================================
    damage_photos_html = ""

    for item in damage_images:

        damages_text = ", ".join(
            item["damages"]
        )

        damage_photos_html += f"""
        <div class="damage-photo">
            <img src="{item['image']}">

            <div class="photo-title">
                {safe(item["part"])}
            </div>

            <div class="photo-subtitle">
                {safe(damages_text)}
            </div>
        </div>
        """


    # =========================================================
    # Plate image HTML
    # =========================================================
    plate_image_html = ""

    if plate_image is not None:

        plate_image_html = f"""
        <img
            src="{plate_image}"
            class="plate-image"
        >
        """


    # =========================================================
    # Original image HTML
    # =========================================================
    original_image_html = ""

    if original_image is not None:

        original_image_html = f"""
        <img
            src="{original_image}"
            class="vehicle-image"
        >
        """


    # =========================================================
    # Damage map HTML
    # =========================================================
    damage_map_html = ""

    if damage_map is not None:

        damage_map_html = f"""
        <img
            src="{damage_map}"
            class="damage-map"
        >
        """


    # =========================================================
    # HTML
    # =========================================================
    report_html = f"""
    <html>

    <head>

    <style>

    body {{
        font-family: Arial, sans-serif;
        background: #f4f6f8;
        margin: 0;
        padding: 30px;
        color: #20252b;
    }}

    .report {{
        max-width: 1400px;
        margin: auto;
        background: white;
        padding: 35px;
        border-radius: 10px;
    }}

    h1 {{
        margin-top: 0;
        border-bottom: 3px solid #252a30;
        padding-bottom: 15px;
    }}

    h2 {{
        font-size: 19px;
        margin-top: 0;
        margin-bottom: 18px;
    }}

    .grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
    }}

    .section {{
        border: 1px solid #d8dde3;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        background: white;
    }}

    .vehicle-layout {{
        display: grid;
        grid-template-columns: 1fr 1.2fr;
        gap: 20px;
        align-items: center;
    }}

    .vehicle-image {{
        width: 100%;
        max-height: 280px;
        object-fit: contain;
        border-radius: 8px;
    }}

    .plate-image {{
        max-width: 280px;
        max-height: 100px;
        object-fit: contain;
        margin-top: 15px;
    }}

    .damage-map {{
        width: 100%;
        max-height: 350px;
        object-fit: contain;
    }}

    .info-row {{
        display: flex;
        justify-content: space-between;
        border-bottom: 1px solid #eeeeee;
        padding: 8px 0;
    }}

    .label {{
        font-weight: bold;
    }}

    .confidence {{
        color: #656d76;
        font-size: 13px;
    }}

    table {{
        width: 100%;
        border-collapse: collapse;
    }}

    th {{
        background: #eef1f4;
        text-align: left;
        padding: 10px;
    }}

    td {{
        padding: 10px;
        border-bottom: 1px solid #e3e6e8;
    }}

    .photos {{
        display: grid;
        grid-template-columns:
            repeat(auto-fit, minmax(220px, 1fr));
        gap: 15px;
    }}

    .damage-photo {{
        border: 1px solid #dddddd;
        padding: 10px;
        border-radius: 7px;
    }}

    .damage-photo img {{
        width: 100%;
        height: 170px;
        object-fit: contain;
    }}

    .photo-title {{
        font-weight: bold;
        margin-top: 8px;
    }}

    .photo-subtitle {{
        font-size: 13px;
        color: #666666;
    }}

    .summary {{
        display: grid;
        grid-template-columns:
            repeat(3, 1fr);
        gap: 15px;
    }}

    .summary-item {{
        background: #eef1f4;
        padding: 18px;
        border-radius: 7px;
        text-align: center;
    }}

    .summary-number {{
        font-size: 28px;
        font-weight: bold;
    }}

    .report-text {{
        line-height: 1.7;
        white-space: pre-wrap;
    }}

    </style>

    </head>


    <body>

    <div class="report">

        <h1>
            Vehicle Inspection Report
        </h1>


        <!-- ============================================= -->
        <!-- Vehicle Information -->
        <!-- ============================================= -->

        <div class="section">

            <h2>
                1. Vehicle Information
            </h2>

            <div class="vehicle-layout">

                <div>

                    <!-- ================================= -->
                    <!-- <<< ИСПРАВЛЕНИЕ 2 -->
                    <!-- Используем format_vehicle_field -->
                    <!-- ================================= -->

                    <div class="info-row">

                        <span class="label">
                            Brand
                        </span>

                        <span>
                            {format_vehicle_field(brand)}
                        </span>

                    </div>


                    <div class="info-row">

                        <span class="label">
                            Type
                        </span>

                        <span>
                            {format_vehicle_field(vehicle_type)}
                        </span>

                    </div>


                    <div class="info-row">

                        <span class="label">
                            Color
                        </span>

                        <span>
                            {format_vehicle_field(color)}
                        </span>

                    </div>


                    <div class="info-row">

                        <span class="label">
                            View
                        </span>

                        <span>
                            {format_vehicle_field(view)}
                        </span>

                    </div>

                </div>


                <div>
                    {original_image_html}
                </div>

            </div>

        </div>


        <div class="grid">


            <!-- ========================================= -->
            <!-- License Plate -->
            <!-- ========================================= -->

            <div class="section">

                <h2>
                    2. License Plate
                </h2>

                <div class="info-row">

                    <span class="label">
                        Plate
                    </span>

                    <span>
                        {safe(plate_text)}
                    </span>

                </div>


                <div class="info-row">

                    <span class="label">
                        OCR Confidence
                    </span>

                    <span>
                        {
                            f"{plate_conf:.3f}"
                            if isinstance(
                                plate_conf,
                                (int, float)
                            )
                            else "N/A"
                        }
                    </span>

                </div>

                {plate_image_html}

            </div>


            <!-- ========================================= -->
            <!-- Damage Map -->
            <!-- ========================================= -->

            <div class="section">

                <h2>
                    3. Damage Map
                </h2>

                {damage_map_html}

            </div>

        </div>


        <!-- ============================================= -->
        <!-- Damage Details -->
        <!-- ============================================= -->

        <div class="section">

            <h2>
                4. Damage Details
            </h2>

            <table>

                <thead>

                    <tr>
                        <th>Part</th>
                        <th>Damage</th>
                        <th>Confidence</th>
                        <th>Area</th>
                    </tr>

                </thead>

                <tbody>

                    {damage_table_html}

                </tbody>

            </table>

        </div>


        <!-- ============================================= -->
        <!-- Damage Photos -->
        <!-- ============================================= -->

        <div class="section">

            <h2>
                5. Damage Photos
            </h2>

            <div class="photos">

                {damage_photos_html}

            </div>

        </div>


        <!-- ============================================= -->
        <!-- Summary -->
        <!-- ============================================= -->

        <div class="section">

            <h2>
                6. Summary
            </h2>

            <div class="summary">

                <div class="summary-item">

                    <div class="summary-number">
                        {detected_parts}
                    </div>

                    Detected Parts

                </div>


                <div class="summary-item">

                    <div class="summary-number">
                        {damaged_parts}
                    </div>

                    Damaged Parts

                </div>


                <div class="summary-item">

                    <div class="summary-number">
                        {total_damage_detections}
                    </div>

                    Damage Detections

                </div>

            </div>

        </div>


        <!-- ============================================= -->
        <!-- NLP Report -->
        <!-- ============================================= -->

        <div class="section">

            <h2>
                7. Report
            </h2>

            <div class="report-text">
                {report_text}
            </div>

        </div>

    </div>

    </body>

    </html>
    """


    # =========================================================
    # Save
    # =========================================================
    if save_path is not None:

        save_path = Path(
            save_path
        )

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            save_path,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                report_html
            )

        print(
            f"Report saved to: {save_path}"
        )


    # =========================================================
    # Display in Jupyter
    # =========================================================
    display(
        HTML(report_html)
    )