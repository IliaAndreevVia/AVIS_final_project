import math

from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

from src.train_test import create_efficientnet_b0

def efficientnet_loader(model_path, num_class_path, device):

    num_classes = torch.load(
        Path(num_class_path),
        weights_only=False
    )

    model = create_efficientnet_b0(
        num_classes=num_classes,
        pretrained=False
    )

    state_dict = torch.load(
        Path(model_path),
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model


def predict_classification(
    model,
    image,
    transform,
    classes,
    device,
):
    """
    Predict top-2 classes.

    Returns
    -------
    dict
        {
            "value": str,
            "confidence": float,
            "top2": [
                {
                    "value": str,
                    "confidence": float
                },
                {
                    "value": str,
                    "confidence": float
                }
            ]
        }
    """

    image_tensor = transform(image)

    image_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(device)
    )

    model.eval()

    outputs = model(image_tensor)

    probabilities = torch.softmax(
        outputs,
        dim=1,
    )

    k = min(
        2,
        probabilities.shape[1],
    )

    top_confidences, top_indices = torch.topk(
        probabilities,
        k=k,
        dim=1,
    )

    top_confidences = (
        top_confidences[0]
        .cpu()
        .tolist()
    )

    top_indices = (
        top_indices[0]
        .cpu()
        .tolist()
    )

    top2 = [
        {
            "value": classes[index],
            "confidence": float(confidence),
        }
        for index, confidence
        in zip(
            top_indices,
            top_confidences,
        )
    ]

    return {
        "value": top2[0]["value"],
        "confidence": top2[0]["confidence"],
        "top2": top2,
    }


def get_expected_parts(
    view_result,
    expected_parts_by_view,
):
    """
    Build expected parts list based on view prediction.

    Certain viewpoint:
        parts from one view.

    Uncertain viewpoint:
        union of parts from top-2 views.
    """

    # -----------------------------------------------------
    # Certain viewpoint
    # -----------------------------------------------------
    if view_result["status"] == "certain":

        views = [
            view_result["value"]
        ]

    # -----------------------------------------------------
    # Uncertain viewpoint
    # -----------------------------------------------------
    else:

        views = [
            candidate["value"]
            for candidate
            in view_result["candidates"]
        ]

    # -----------------------------------------------------
    # Union of expected parts
    # -----------------------------------------------------
    expected_parts = sorted(
        set().union(
            *[
                expected_parts_by_view[view]
                for view in views
            ]
        )
    )

    return expected_parts

def detect_parts(
    expected_parts,
    image,
    parts_detection_model,
):
    """
    Detect vehicle parts using different CONF and IoU values
    and return the result with the number of detected unique parts
    closest to the expected number of parts.

    Parameters
    ----------
    expected_parts : list[str]
        Expected parts list.
        Only its length is used as the expected number of parts.

    image : str | Path | PIL.Image.Image | np.ndarray
        Vehicle image.

    parts_detection_model : YOLO
        Loaded Ultralytics YOLO model.

    Returns
    -------
    dict
        {
            "parts": {
                "part_name": {
                    "image": cropped_image,
                    "confidence": float,
                    "bbox": [x1, y1, x2, y2]
                }
            },

            "detected_parts": [...],

            "expected_parts_count": int,
            "detected_parts_count": int,
            "count_difference": int,
            "count_ratio": float,

            "mean_detection_confidence": float,

            "conf": float,
            "iou": float
        }
    """

    # =========================================================
    # Prepare image
    # =========================================================

    if isinstance(image, (str, Path)):

        image_cv = cv2.imread(str(image))

        if image_cv is None:
            raise FileNotFoundError(
                f"Could not load image: {image}"
            )

        image_yolo = image_cv

    elif isinstance(image, Image.Image):

        image_rgb = np.array(
            image.convert("RGB")
        )

        image_cv = cv2.cvtColor(
            image_rgb,
            cv2.COLOR_RGB2BGR
        )

        image_yolo = image_rgb

    elif isinstance(image, np.ndarray):

        image_cv = image.copy()
        image_yolo = image

    else:

        raise TypeError(
            "image must be str, Path, "
            "PIL.Image.Image or numpy.ndarray"
        )

    # =========================================================
    # Normalize class names
    # =========================================================

    def normalize_part_name(name):

        return (
            str(name)
            .lower()
            .strip()
            .replace("-", "_")
            .replace(" ", "_")
        )

    # =========================================================
    # Expected number of parts
    # =========================================================

    expected_parts_count = len(expected_parts)

    # =========================================================
    # Search parameters
    # =========================================================

    conf_values = [
        0.25,
        0.20,
        0.15,
        0.10,
        0.075,
        0.05,
        0.03,
        0.02,
        0.01,
    ]

    iou_values = [
        0.70,
        0.60,
        0.50,
    ]

    parameter_grid = [
        (conf, iou)
        for iou in iou_values
        for conf in conf_values
    ]

    # =========================================================
    # Best result
    # =========================================================

    best_result = None

    # =========================================================
    # Try all CONF / IoU combinations
    # =========================================================

    for conf, iou in parameter_grid:

        results = parts_detection_model.predict(
            source=image_yolo,
            conf=conf,
            iou=iou,
            verbose=False,
        )

        result = results[0]

        detected_parts = {}

        # =====================================================
        # Process detected boxes
        # =====================================================

        for box in result.boxes:

            class_id = int(
                box.cls.item()
            )

            class_name = (
                parts_detection_model
                .names[class_id]
            )

            normalized_name = normalize_part_name(
                class_name
            )

            confidence = float(
                box.conf.item()
            )

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
                .cpu()
                .numpy()
            )

            # =================================================
            # Keep coordinates inside image
            # =================================================

            height, width = image_cv.shape[:2]

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(width, x2)
            y2 = min(height, y2)

            # Ignore invalid crops
            if x2 <= x1 or y2 <= y1:
                continue

            # =================================================
            # Same class detected several times:
            # keep highest-confidence detection
            # =================================================

            if normalized_name in detected_parts:

                old_confidence = (
                    detected_parts[
                        normalized_name
                    ]["confidence"]
                )

                if confidence <= old_confidence:
                    continue

            cropped_part = image_cv[
                y1:y2,
                x1:x2
            ].copy()

            detected_parts[
                normalized_name
            ] = {
                "name": class_name,
                "image": cropped_part,
                "confidence": confidence,
                "bbox": [
                    x1,
                    y1,
                    x2,
                    y2,
                ],
            }

        # =====================================================
        # Number of detected parts
        # =====================================================

        detected_parts_count = len(
            detected_parts
        )

        count_difference = abs(
            detected_parts_count
            - expected_parts_count
        )

        # =====================================================
        # Count ratio
        #
        # 1.0 = perfect number of detected parts
        # =====================================================

        if expected_parts_count == 0:

            count_ratio = (
                1.0
                if detected_parts_count == 0
                else 0.0
            )

        else:

            count_ratio = min(
                detected_parts_count,
                expected_parts_count,
            ) / max(
                detected_parts_count,
                expected_parts_count,
            )

        # =====================================================
        # Mean confidence
        # =====================================================

        if detected_parts_count > 0:

            mean_detection_confidence = float(
                np.mean(
                    [
                        part["confidence"]
                        for part
                        in detected_parts.values()
                    ]
                )
            )

        else:

            mean_detection_confidence = 0.0

        # =====================================================
        # Current result
        # =====================================================

        current_result = {

            "parts": {
                part["name"]: {
                    "image": part["image"],
                    "confidence": part["confidence"],
                    "bbox": part["bbox"],
                }
                for part
                in detected_parts.values()
            },

            "detected_parts": [
                part["name"]
                for part
                in detected_parts.values()
            ],

            "expected_parts_count":
                expected_parts_count,

            "detected_parts_count":
                detected_parts_count,

            "count_difference":
                count_difference,

            "count_ratio":
                count_ratio,

            "mean_detection_confidence":
                mean_detection_confidence,

            "conf":
                conf,

            "iou":
                iou,
        }

        # =====================================================
        # Debug output
        # =====================================================

        print(
            f"CONF={conf:.3f} | "
            f"IoU={iou:.2f} | "
            f"Parts="
            f"{detected_parts_count}/"
            f"{expected_parts_count} | "
            f"Difference="
            f"{count_difference} | "
            f"Ratio="
            f"{count_ratio:.2f} | "
            f"Mean conf="
            f"{mean_detection_confidence:.3f}"
        )

        # =====================================================
        # Select best result
        #
        # 1. Minimum difference in number of parts
        # 2. Higher mean confidence if tied
        # =====================================================

        if best_result is None:

            best_result = current_result

        elif (
            current_result["count_difference"]
            <
            best_result["count_difference"]
        ):

            best_result = current_result

        elif (
            current_result["count_difference"]
            ==
            best_result["count_difference"]
            and
            current_result[
                "mean_detection_confidence"
            ]
            >
            best_result[
                "mean_detection_confidence"
            ]
        ):

            best_result = current_result

    # =========================================================
    # Return best result
    # =========================================================

    return best_result


def show_detected_parts_grid(parts_result, cols=3):

    parts = list(parts_result["parts"].items())

    n = len(parts)

    if n == 0:
        print("No detected parts")
        return

    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(5 * cols, 4 * rows)
    )

    axes = np.array(axes).reshape(-1)

    for ax, (part_name, part_data) in zip(axes, parts):

        crop = part_data["image"]

        crop_rgb = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2RGB
        )

        ax.imshow(crop_rgb)

        ax.set_title(
            f"{part_name}\n"
            f"conf={part_data['confidence']:.2f}"
        )

        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def analyze_vehicle_damage(
    image,
    parts_result,
    damage_model,
    min_damage_confidence=0.05,
    glass_min_damage_confidence=0.30,
    whole_vehicle_min_confidence=0.001,
):
    """
    Analyze vehicle damages on:
        1. Whole vehicle image
        2. Every detected vehicle part

    Parameters
    ----------
    image : PIL.Image | np.ndarray | str | Path
        Original vehicle image.

    parts_result : dict
        Result returned by detect_parts().

    damage_model : YOLO
        Trained YOLO damage model.

    min_damage_confidence : float
        Minimum accepted damage confidence
        for normal vehicle parts.

    glass_min_damage_confidence : float
        Minimum accepted damage confidence
        for glass parts:
            - Back-window
            - Back-windshield
            - Front-window
            - Windshield

    whole_vehicle_min_confidence : float
        Minimum accepted confidence
        for whole vehicle analysis.

    Returns
    -------
    dict
    """

    # =========================================================
    # Glass parts
    # =========================================================

    GLASS_PARTS = {
        "Back-window",
        "Back-windshield",
        "Front-window",
        "Windshield",
    }

    def is_glass_part(part_name):

        return part_name in GLASS_PARTS

    # =========================================================
    # Prepare image
    # =========================================================

    def prepare_image(
        img,
        bgr_input=False,
    ):

        # Path
        if isinstance(img, (str, Path)):

            img_bgr = cv2.imread(
                str(img)
            )

            if img_bgr is None:
                raise FileNotFoundError(
                    f"Could not load image: {img}"
                )

            return cv2.cvtColor(
                img_bgr,
                cv2.COLOR_BGR2RGB
            )

        # PIL
        if isinstance(img, Image.Image):

            return np.array(
                img.convert("RGB")
            )

        # NumPy
        if isinstance(img, np.ndarray):

            if bgr_input:

                return cv2.cvtColor(
                    img,
                    cv2.COLOR_BGR2RGB
                )

            return img.copy()

        raise TypeError(
            "Image must be str, Path, "
            "PIL.Image or numpy.ndarray"
        )

    # =========================================================
    # Initial parameters from parts detection
    # =========================================================

    start_conf = float(
        parts_result.get(
            "conf",
            0.25
        )
    )

    start_iou = float(
        parts_result.get(
            "iou",
            0.70
        )
    )

    # =========================================================
    # CONF search values
    # =========================================================

    base_conf_values = [
        0.25,
        0.20,
        0.15,
        0.10,
        0.075,
        0.05,
        0.03,
        0.02,
        0.01,
    ]

    # ---------------------------------------------------------
    # Parts CONF range
    # ---------------------------------------------------------

    parts_conf_values = [
        value
        for value in base_conf_values
        if (
            value <= start_conf + 1e-9
            and
            value >= min_damage_confidence - 1e-9
        )
    ]

    # Add exact starting value
    if (
        start_conf >= min_damage_confidence
        and
        start_conf not in parts_conf_values
    ):
        parts_conf_values.append(
            start_conf
        )

    # Add exact minimum
    if (
        min_damage_confidence <= start_conf
        and
        min_damage_confidence
        not in parts_conf_values
    ):
        parts_conf_values.append(
            min_damage_confidence
        )

    parts_conf_values = sorted(
        set(parts_conf_values),
        reverse=True
    )

    # ---------------------------------------------------------
    # Whole vehicle CONF range
    # ---------------------------------------------------------

    whole_conf_values = [
        value
        for value in base_conf_values
        if (
            value <= start_conf + 1e-9
            and
            value >= whole_vehicle_min_confidence - 1e-9
        )
    ]

    whole_low_conf_values = [
        0.005,
        0.003,
        0.002,
        0.001,
    ]

    for value in whole_low_conf_values:

        if (
            value <= start_conf
            and
            value >= whole_vehicle_min_confidence
        ):
            whole_conf_values.append(
                value
            )

    if (
        whole_vehicle_min_confidence
        <= start_conf
    ):
        whole_conf_values.append(
            whole_vehicle_min_confidence
        )

    whole_conf_values = sorted(
        set(whole_conf_values),
        reverse=True
    )

    # =========================================================
    # IoU search values
    # =========================================================

    base_iou_values = [
        0.70,
        0.60,
        0.50,
    ]

    iou_values = [
        value
        for value in base_iou_values
        if value <= start_iou + 1e-9
    ]

    if start_iou not in iou_values:

        iou_values.append(
            start_iou
        )

    iou_values = sorted(
        set(iou_values),
        reverse=True
    )

    # =========================================================
    # Analyze one image
    # =========================================================

    def analyze_single_image(
        img,
        conf_values,
        min_confidence,
        bgr_input=False,
    ):

        img_rgb = prepare_image(
            img,
            bgr_input=bgr_input
        )

        height, width = (
            img_rgb.shape[:2]
        )

        image_area = (
            height * width
        )

        best_result = None

        # =====================================================
        # CONF / IoU search
        # =====================================================

        for iou in iou_values:

            for conf in conf_values:

                results = damage_model.predict(
                    source=img_rgb,
                    conf=conf,
                    iou=iou,
                    verbose=False,
                )

                result = results[0]

                damages = []

                # -------------------------------------------------
                # Segmentation available?
                # -------------------------------------------------

                masks_available = (
                    result.masks is not None
                    and
                    result.masks.data is not None
                )

                # =================================================
                # Process detections
                # =================================================

                for index, box in enumerate(
                    result.boxes
                ):

                    confidence = float(
                        box.conf.item()
                    )

                    # ---------------------------------------------
                    # Part-specific confidence filter
                    # ---------------------------------------------

                    if confidence < min_confidence:
                        continue

                    class_id = int(
                        box.cls.item()
                    )

                    damage_type = (
                        damage_model.names[
                            class_id
                        ]
                    )

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                        .cpu()
                        .numpy()
                    )

                    # ---------------------------------------------
                    # Keep bbox inside image
                    # ---------------------------------------------

                    x1 = max(
                        0,
                        min(x1, width)
                    )

                    x2 = max(
                        0,
                        min(x2, width)
                    )

                    y1 = max(
                        0,
                        min(y1, height)
                    )

                    y2 = max(
                        0,
                        min(y2, height)
                    )

                    if (
                        x2 <= x1
                        or
                        y2 <= y1
                    ):
                        continue

                    # =============================================
                    # Damage area
                    # =============================================

                    if masks_available:

                        mask = (
                            result
                            .masks
                            .data[index]
                            .cpu()
                            .numpy()
                        )

                        mask = cv2.resize(
                            mask,
                            (
                                width,
                                height
                            ),
                            interpolation=(
                                cv2.INTER_NEAREST
                            ),
                        )

                        damage_area_pixels = int(
                            np.sum(
                                mask > 0.5
                            )
                        )

                        area_method = (
                            "segmentation_mask"
                        )

                    else:

                        damage_area_pixels = (
                            (x2 - x1)
                            *
                            (y2 - y1)
                        )

                        area_method = (
                            "bounding_box"
                        )

                    damage_area_percent = (
                        damage_area_pixels
                        /
                        image_area
                        *
                        100
                    )

                    # =============================================
                    # Damage crop
                    # =============================================

                    damage_crop = img_rgb[
                        y1:y2,
                        x1:x2
                    ].copy()

                    damages.append(
                        {
                            "damage_type":
                                damage_type,

                            "confidence":
                                confidence,

                            "bbox": [
                                x1,
                                y1,
                                x2,
                                y2,
                            ],

                            "area_pixels":
                                damage_area_pixels,

                            "area_percent":
                                damage_area_percent,

                            "area_method":
                                area_method,

                            "image":
                                damage_crop,
                        }
                    )

                # =================================================
                # Statistics
                # =================================================

                damage_count = len(
                    damages
                )

                if damage_count > 0:

                    mean_confidence = float(
                        np.mean(
                            [
                                damage[
                                    "confidence"
                                ]
                                for damage
                                in damages
                            ]
                        )
                    )

                else:

                    mean_confidence = 0.0

                current_result = {

                    "conf":
                        conf,

                    "iou":
                        iou,

                    "damage_count":
                        damage_count,

                    "mean_confidence":
                        mean_confidence,

                    "damages":
                        damages,
                }

                # =================================================
                # Select best result
                #
                # Priority:
                # 1. More detections
                # 2. Higher mean confidence
                # 3. Higher CONF threshold
                # =================================================

                if best_result is None:

                    best_result = (
                        current_result
                    )

                elif (
                    current_result[
                        "damage_count"
                    ]
                    >
                    best_result[
                        "damage_count"
                    ]
                ):

                    best_result = (
                        current_result
                    )

                elif (
                    current_result[
                        "damage_count"
                    ]
                    ==
                    best_result[
                        "damage_count"
                    ]
                    and
                    current_result[
                        "mean_confidence"
                    ]
                    >
                    best_result[
                        "mean_confidence"
                    ]
                ):

                    best_result = (
                        current_result
                    )

                elif (
                    current_result[
                        "damage_count"
                    ]
                    ==
                    best_result[
                        "damage_count"
                    ]
                    and
                    np.isclose(
                        current_result[
                            "mean_confidence"
                        ],
                        best_result[
                            "mean_confidence"
                        ],
                    )
                    and
                    current_result[
                        "conf"
                    ]
                    >
                    best_result[
                        "conf"
                    ]
                ):

                    best_result = (
                        current_result
                    )

        # =====================================================
        # Result summary
        # =====================================================

        damages = best_result[
            "damages"
        ]

        damage_types = Counter(
            damage["damage_type"]
            for damage in damages
        )

        # -----------------------------------------------------
        # Total area
        # -----------------------------------------------------

        total_damage_area_pixels = sum(
            damage["area_pixels"]
            for damage in damages
        )

        total_damage_area_percent = (
            total_damage_area_pixels
            /
            image_area
            *
            100
        )

        # -----------------------------------------------------
        # Annotated image
        # -----------------------------------------------------

        annotated_image = (
            img_rgb.copy()
        )

        for damage in damages:

            x1, y1, x2, y2 = (
                damage["bbox"]
            )

            cv2.rectangle(
                annotated_image,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2,
            )

            label = (
                f"{damage['damage_type']} "
                f"{damage['confidence']:.2f}"
            )

            cv2.putText(
                annotated_image,
                label,
                (
                    x1,
                    max(
                        20,
                        y1 - 5
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

        return {

            **best_result,

            "damage_types":
                dict(damage_types),

            "total_damage_area_pixels":
                total_damage_area_pixels,

            "total_damage_area_percent":
                total_damage_area_percent,

            "image":
                img_rgb,

            "annotated_image":
                annotated_image,
        }

    # =========================================================
    # 1. Analyze whole vehicle
    # =========================================================

    whole_vehicle = (
        analyze_single_image(
            img=image,

            conf_values=(
                whole_conf_values
            ),

            min_confidence=(
                whole_vehicle_min_confidence
            ),

            bgr_input=False,
        )
    )

    # =========================================================
    # 2. Analyze every detected part
    # =========================================================

    parts_damage = {}

    for (
        part_name,
        part_data
    ) in parts_result[
        "parts"
    ].items():

        part_image = (
            part_data["image"]
        )

        # =====================================================
        # Different minimum confidence for glass
        # =====================================================

        if is_glass_part(
            part_name
        ):

            current_min_confidence = (
                glass_min_damage_confidence
            )

        else:

            current_min_confidence = (
                min_damage_confidence
            )

        # =====================================================
        # Analyze part
        # =====================================================

        analysis = (
            analyze_single_image(
                img=part_image,

                conf_values=(
                    parts_conf_values
                ),

                min_confidence=(
                    current_min_confidence
                ),

                bgr_input=True,
            )
        )

        # -----------------------------------------------------
        # Information from parts detector
        # -----------------------------------------------------

        analysis[
            "part_detection_confidence"
        ] = part_data.get(
            "confidence"
        )

        analysis[
            "part_bbox"
        ] = part_data.get(
            "bbox"
        )

        # Useful for debugging
        analysis[
            "min_damage_confidence"
        ] = current_min_confidence

        analysis[
            "is_glass"
        ] = is_glass_part(
            part_name
        )

        parts_damage[
            part_name
        ] = analysis

    # =========================================================
    # 3. Damaged parts only
    # =========================================================

    damaged_parts = {

        part_name:
            result

        for (
            part_name,
            result
        )
        in parts_damage.items()

        if result[
            "damage_count"
        ] > 0
    }

    # =========================================================
    # 4. Flat damage list
    # =========================================================

    flat_damages = []

    for (
        part_name,
        part_result
    ) in damaged_parts.items():

        for damage in (
            part_result[
                "damages"
            ]
        ):

            flat_damages.append(
                {
                    "part":
                        part_name,

                    "damage_type":
                        damage[
                            "damage_type"
                        ],

                    "confidence":
                        damage[
                            "confidence"
                        ],

                    "bbox":
                        damage[
                            "bbox"
                        ],

                    "area_pixels":
                        damage[
                            "area_pixels"
                        ],

                    "area_percent":
                        damage[
                            "area_percent"
                        ],

                    "area_method":
                        damage[
                            "area_method"
                        ],

                    "image":
                        damage[
                            "image"
                        ],

                    "conf":
                        part_result[
                            "conf"
                        ],

                    "iou":
                        part_result[
                            "iou"
                        ],
                }
            )

    # =========================================================
    # 5. Damage type statistics
    # =========================================================

    all_damage_types = Counter(
        damage["damage_type"]
        for damage in flat_damages
    )

    # =========================================================
    # Final output
    # =========================================================

    return {

        "initial_conf":
            start_conf,

        "initial_iou":
            start_iou,

        "whole_vehicle":
            whole_vehicle,

        "parts":
            parts_damage,

        "damaged_parts":
            damaged_parts,

        "damages":
            flat_damages,

        "summary": {

            "damaged_parts_count":
                len(
                    damaged_parts
                ),

            "damaged_parts":
                list(
                    damaged_parts.keys()
                ),

            "damage_count":
                len(
                    flat_damages
                ),

            "damage_types":
                dict(
                    all_damage_types
                ),
        },
    }

def show_damaged_parts(
    damage_result,
    cols=3,
):
    """
    Show damaged vehicle parts in a subplot grid.

    Parameters
    ----------
    damage_result : dict
        Result returned by analyze_vehicle_damage().

    cols : int
        Number of columns in subplot grid.
    """

    damaged_parts = list(
        damage_result["damaged_parts"].items()
    )

    n = len(damaged_parts)

    if n == 0:
        print("No damaged parts detected")
        return

    rows = math.ceil(
        n / cols
    )

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(
            5 * cols,
            4 * rows
        )
    )

    axes = np.array(
        axes
    ).reshape(-1)

    for ax, (
        part_name,
        part_result
    ) in zip(
        axes,
        damaged_parts
    ):

        image_rgb = part_result["image"]

        ax.imshow(
            image_rgb
        )

        damages_text = "\n".join(
            [
                (
                    f"{damage['damage_type']} "
                    f"({damage['confidence']:.2f})"
                )
                for damage
                in part_result["damages"]
            ]
        )

        ax.set_title(
            f"{part_name}\n"
            f"{damages_text}\n"
            f"Damaged area: "
            f"{part_result['total_damage_area_percent']:.1f}%"
        )

        ax.axis("off")

    # Hide empty subplots
    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def show_damaged_parts_with_boxes(
    damage_result,
    cols=3,
    merge_iou_threshold=0.7,
):
    """
    Show damaged parts with merged overlapping damage boxes.

    Parameters
    ----------
    damage_result : dict
        Result returned by analyze_vehicle_damage().

    cols : int
        Number of subplot columns.

    merge_iou_threshold : float
        If IoU between two damage boxes is greater than
        this value, they are treated as the same damage area.
    """

    # Calculate IoU between two boxes
    def bbox_iou(box1, box2):

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])

        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = (
            max(0, x2 - x1)
            *
            max(0, y2 - y1)
        )

        area1 = (
            max(0, box1[2] - box1[0])
            *
            max(0, box1[3] - box1[1])
        )

        area2 = (
            max(0, box2[2] - box2[0])
            *
            max(0, box2[3] - box2[1])
        )

        union = (
            area1
            +
            area2
            -
            intersection
        )

        if union <= 0:
            return 0.0

        return (
            intersection
            /
            union
        )

    # Merge overlapping damage boxes
    def merge_damage_boxes(damages):

        merged = []

        # Highest confidence first
        damages = sorted(
            damages,
            key=lambda damage: damage["confidence"],
            reverse=True,
        )

        for damage in damages:

            matched = False

            for group in merged:

                iou = bbox_iou(
                    damage["bbox"],
                    group["bbox"],
                )

                if iou >= merge_iou_threshold:

                    group["damages"].append(
                        damage
                    )

                    # Union bbox
                    group["bbox"] = [
                        min(
                            group["bbox"][0],
                            damage["bbox"][0]
                        ),

                        min(
                            group["bbox"][1],
                            damage["bbox"][1]
                        ),

                        max(
                            group["bbox"][2],
                            damage["bbox"][2]
                        ),

                        max(
                            group["bbox"][3],
                            damage["bbox"][3]
                        ),
                    ]

                    matched = True
                    break

            if not matched:

                merged.append(
                    {
                        "bbox":
                            damage["bbox"].copy(),

                        "damages":
                            [damage],
                    }
                )

        return merged

    # Get damaged parts
    damaged_parts = list(
        damage_result[
            "damaged_parts"
        ].items()
    )

    n = len(
        damaged_parts
    )

    if n == 0:

        print(
            "No damaged parts detected"
        )

        return

    # Create subplot grid
    rows = math.ceil(
        n / cols
    )

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(
            5 * cols,
            4 * rows
        )
    )

    axes = np.array(
        axes
    ).reshape(-1)

    # Draw each damaged part
    for (
        ax,
        (
            part_name,
            part_result
        )
    ) in zip(
        axes,
        damaged_parts
    ):

        image_rgb = (
            part_result[
                "image"
            ].copy()
        )

        height, width = (
            image_rgb.shape[:2]
        )

        # Merge overlapping boxes
        damage_groups = (
            merge_damage_boxes(
                part_result[
                    "damages"
                ]
            )
        )

        # Union mask for total damaged area
        damage_union_mask = np.zeros(
            (
                height,
                width
            ),
            dtype=np.uint8
        )

        # Draw merged damage groups
        for group in damage_groups:

            x1, y1, x2, y2 = (
                group["bbox"]
            )

            # Keep bbox inside image
            x1 = max(
                0,
                min(x1, width - 1)
            )

            x2 = max(
                0,
                min(x2, width)
            )

            y1 = max(
                0,
                min(y1, height - 1)
            )

            y2 = max(
                0,
                min(y2, height)
            )

            if (
                x2 <= x1
                or
                y2 <= y1
            ):
                continue

            # Add area to union mask
            damage_union_mask[
                y1:y2,
                x1:x2
            ] = 1
 

            # Draw merged bounding box
            cv2.rectangle(
                image_rgb,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            # Create labels
            labels = [
                (
                    f"{damage['damage_type']} "
                    f"{damage['confidence']:.2f}"
                )
                for damage
                in group["damages"]
            ]

            # Draw labels one under another
            # INSIDE the bbox
            line_height = 18

            for i, label in enumerate(
                labels
            ):

                text_x = (
                    x1 + 5
                )

                text_y = (
                    y1
                    +
                    line_height
                    +
                    i * line_height
                )

                # If labels do not fit inside bbox,
                # stop before they leave the image
                if text_y >= height:
                    break

                cv2.putText(
                    image_rgb,
                    label,
                    (
                        text_x,
                        text_y
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA
                )

        # Calculate UNIQUE damaged area
        union_area_pixels = int(
            damage_union_mask.sum()
        )

        total_pixels = (
            height
            *
            width
        )

        union_area_percent = (
            union_area_pixels
            /
            total_pixels
            *
            100
        )

        # Show subplot
        ax.imshow(
            image_rgb
        )

        ax.set_title(
            f"{part_name}\n"
            f"Damage area: "
            f"{union_area_percent:.1f}%"
        )

        ax.axis(
            "off"
        )

    # Hide unused axes
    for ax in axes[n:]:

        ax.axis(
            "off"
        )

    plt.tight_layout()
    plt.show()
    

def parts_sep(model, conf, iou, image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )
                                
    results = model.predict(image, conf = conf, iou = iou)
    output = {}
    for result in results:
        for i, box in enumerate(result.boxes):

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
    
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]
            
            output[i] = {"class_name":class_name, "image":image[y1:y2, x1:x2], "confidence": confidence}
            
    return output