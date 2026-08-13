import re

from pathlib import Path

import cv2
import torch

from PIL import Image

from torchvision import transforms
from ultralytics import YOLO
from paddleocr import PaddleOCR

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)

from src.run_func import (
    efficientnet_loader,
    predict_classification,
    get_expected_parts,
    detect_parts,
    analyze_vehicle_damage,
    parts_sep,
)

from src.damage_map_generator import generate_damage_map

from src.nlp_report import generate_report

from src.report import (
    classification_to_result,
    inspection_result_to_vehicle_data,
    generate_inspection_report,
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

def run_inspection(img_path = "./run_check/ex.jpg",
                   report_save_folder = "./run_check/report/",
                   text_generation_model = "bart-base"
                  ):

    image = Image.open(img_path).convert("RGB")
    
    color_detection_model = efficientnet_loader("./trained_pytorch_models/color/EfficientNet_B0.pth", 
                                            "./trained_pytorch_models/color/num_classes.pth",
                                            device)
    orientation_detection_model = efficientnet_loader("./trained_pytorch_models/orientation/EfficientNet_B0.pth", 
                                                      "./trained_pytorch_models/orientation/num_classes.pth",
                                                      device)
    vehicle_type_detection_model = efficientnet_loader("./trained_pytorch_models/Vehicle_type/EfficientNet_B0.pth", 
                                                   "./trained_pytorch_models/Vehicle_type/num_classes.pth",
                                                   device)
    brand_detection_model = efficientnet_loader("./trained_pytorch_models/Vehicle_brand/EfficientNet_B0.pth",
                                            "./trained_pytorch_models/Vehicle_brand/num_classes.pth",
                                            device)

    plate_detection_model = YOLO(Path("./runs/detect/license_plate/train_best/weights/best.pt"))

    damage_detection_model = YOLO(Path("./runs/detect/damage_type/train_best/weights/best.pt"))
    
    parts_detection_model = YOLO(Path("./runs/detect/parts/train_best/weights/best.pt"))
    
    if text_generation_model.lower() == "bart-base":
    
        print("Text generation model is BART-base")
    
        nlp_path = "./trained_models/bart-base/best"
    
        nlp_tokenizer = AutoTokenizer.from_pretrained(
            nlp_path
        )
    
        nlp_model = AutoModelForSeq2SeqLM.from_pretrained(
            nlp_path
        )
    
    elif text_generation_model.lower() == "flan-t5":
    
        print("Text generation model is FLAN-T5")
    
        nlp_path = "./trained_models/flan-t5-small/best"
    
        nlp_tokenizer = AutoTokenizer.from_pretrained(
            nlp_path
        )
    
        nlp_model = AutoModelForSeq2SeqLM.from_pretrained(
            nlp_path
        )
    
    elif text_generation_model.lower() == "gpt2":
    
        print("Text generation model is GPT-2")
    
        nlp_path = "./trained_models/gpt2/best"
    
        nlp_tokenizer = AutoTokenizer.from_pretrained(
            nlp_path
        )
    
        nlp_model = AutoModelForCausalLM.from_pretrained(
            nlp_path
        )
    
    else:
    
        print(
            f"Unknown text generation model: "
            f"{text_generation_model}"
        )
    
        print("Using default model: BART-base")
    
        text_generation_model = "bart-base"
    
        nlp_path = "./trained_models/bart-base/best"
    
        nlp_tokenizer = AutoTokenizer.from_pretrained(
            nlp_path
        )
    
        nlp_model = AutoModelForSeq2SeqLM.from_pretrained(
            nlp_path
        )
    
    nlp_model = nlp_model.to(device)
    nlp_model.eval()

    orientation_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    brand_transform = transforms.Compose(
        [transforms.Resize((256, 256)),
         transforms.ToTensor(),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    
    sides_classes = torch.load(
        Path("./trained_pytorch_models/orientation/classes.pth"),
        weights_only=False
    )
    
    color_classes = torch.load(
        Path("./trained_pytorch_models/color/classes.pth"),
        weights_only=False
    )
    
    type_classes = torch.load(
        Path("./trained_pytorch_models/Vehicle_type/classes.pth"),
        weights_only=False
    )
    
    brand_classes = torch.load(
        Path("./trained_pytorch_models/Vehicle_brand/classes.pth"),
        weights_only=False
    )
    
    color = predict_classification(color_detection_model,
                                   image,
                                   transform,
                                   color_classes,
                                   device)
    
    vehicle_type = predict_classification(vehicle_type_detection_model,
                                   image,
                                   transform,
                                   type_classes,
                                   device)
    
    vehicle_brand = predict_classification(brand_detection_model,
                                   image,
                                   brand_transform,
                                   brand_classes,
                                   device)
    
    orientation = predict_classification(orientation_detection_model,
                               image,
                               orientation_transform,
                               sides_classes,
                               device)
    
    brand_result = classification_to_result(
        vehicle_brand
    )
    
    type_result = classification_to_result(
        vehicle_type
    )
    
    color_result = classification_to_result(
        color
    )
    
    view_result = classification_to_result(
        orientation
    )

    EXPECTED_PARTS_BY_VIEW = {

    "Front": [
        "Hood",
        "Front-bumper",
        "Windshield",
        "Headlight",
        "Grille",
        "License-plate",
    ],

    "Front Right": [
        "Hood",
        "Front-bumper",
        "Windshield",
        "Headlight",
        "Fender",
        "Front-door",
        "Mirror",
        "Front-wheel",
        "Front-window",
        "License-plate",
    ],

    "Right": [
        "Fender",
        "Front-door",
        "Back-door",
        "Mirror",
        "Front-wheel",
        "Back-wheel",
        "Front-window",
        "Back-window",
        "Quarter-panel",
        "Rocker-panel",
    ],

    "Rear Right": [
        "Trunk",
        "Back-bumper",
        "Back-windshield",
        "Tail-light",
        "Quarter-panel",
        "Back-door",
        "Back-wheel",
        "Back-window",
        "License-plate",
    ],

    "Rear": [
        "Trunk",
        "Back-bumper",
        "Back-windshield",
        "Tail-light",
        "License-plate",
    ],

    "Rear Left": [
        "Trunk",
        "Back-bumper",
        "Back-windshield",
        "Tail-light",
        "Quarter-panel",
        "Back-door",
        "Back-wheel",
        "Back-window",
        "License-plate",
    ],

    "Left": [
        "Fender",
        "Front-door",
        "Back-door",
        "Mirror",
        "Front-wheel",
        "Back-wheel",
        "Front-window",
        "Back-window",
        "Quarter-panel",
        "Rocker-panel",
    ],

    "Front Left": [
        "Hood",
        "Front-bumper",
        "Windshield",
        "Headlight",
        "Fender",
        "Front-door",
        "Mirror",
        "Front-wheel",
        "Front-window",
        "License-plate",
    ],
    }

    expected_parts = get_expected_parts(view_result,
                                        EXPECTED_PARTS_BY_VIEW
                                       )
                                
    detection_result = detect_parts(
        expected_parts,
        img_path,
        parts_detection_model
    )
    
    damage_result = analyze_vehicle_damage(
        image=img_path,
        parts_result=detection_result,
        damage_model=damage_detection_model,
        min_damage_confidence=0.05,
        glass_min_damage_confidence=0.8,
        whole_vehicle_min_confidence=0.05,
    )
    
    total_damage_detections = sum(
        len(part_result["damages"])
        for part_result in damage_result["parts"].values()
    )
    
    def read_plate(plate_crop, ocr, fxfy=3, min_conf=0.6):
    
        plate_crop = cv2.resize(
            plate_crop,
            None,
            fx=fxfy,
            fy=fxfy,
            interpolation=cv2.INTER_LANCZOS4
        )
    
        gray = cv2.cvtColor(
            plate_crop,
            cv2.COLOR_BGR2GRAY
        )
    
        gray = cv2.equalizeHist(gray)
    
        results = ocr.ocr(
            gray,
            cls=False
        )
    
        predictions = []
    
        if not results or results[0] is None:
            return predictions
    
        for result in results[0]:
    
            text = result[1][0]
            confidence = float(result[1][1])
    
            if confidence >= min_conf:
    
                cleaned_text = re.sub(
                    r"[^A-Z0-9]",
                    "",
                    text.upper()
                )
    
                if cleaned_text:
                    predictions.append({
                        "text": cleaned_text,
                        "confidence": confidence
                    })
    
        return predictions
    
    ocr = PaddleOCR(use_angle_cls=False,
                    lang="en",
                    show_log=False,
                    use_gpu=(device.type == "cuda"))
    
    out = parts_sep(plate_detection_model,
                    0.5,
                    0.7,
                    img_path)

    if len(out) > 0:
    
        first_plate = next(iter(out.values()))
    
        plate_predictions = read_plate(
            first_plate["image"],
            ocr,
            fxfy=3,
        )
    
        plate_image = first_plate["image"]
    
    else:
    
        plate_predictions = []
        plate_image = None
    
    
    plate_result = (
        plate_predictions[0]
        if len(plate_predictions) > 0
        else None)

    damaged_parts = []
    for detail, ch, in damage_result['parts'].items():
        if len(ch['damages'])>0:
            
            damaged_parts.append(detail)
            
    damage_map = generate_damage_map(
        base_image_path="./damage_map/car_map.png",
        polygons_csv_path="./damage_map/car_part_polygons.csv",
        damaged_parts=damaged_parts,
        )

    inspection_result = {
                        # Original image
                        "original_image": image,
                        "image_path": img_path,
                    
                        # Vehicle classification
                        "vehicle": {
                                "brand": brand_result,
                                "type": type_result,
                                "color": color_result,
                                "view": view_result,
                            },
                    
                        # Parts detection
                        "parts_detection": detection_result,
                    
                        # Damage detection
                        "damage_analysis": damage_result,
                    
                        # License plate
                        "license_plate": {
                            "text": (
                                plate_result["text"]
                                if plate_result is not None
                                else None
                            ),
                            "confidence": (
                                plate_result["confidence"]
                                if plate_result is not None
                                else 0.0
                            ),
                            "image": plate_image,
                        },
                    
                        # Damage map
                        "damage_map": damage_map,
                    
                        # Summary
                        "summary": {
                            "detected_parts": len(
                                detection_result["parts"]
                            ),
                            "damaged_parts": len(
                                damaged_parts
                            ),
                            "damage_detections": total_damage_detections,
                            "damaged_part_names": damaged_parts,
                        },
                    
                        # NLP reports
                        "reports": {},
                        }
        
    vehicle_data = inspection_result_to_vehicle_data(inspection_result)
    

    # Generate NLP report
    report = generate_report(
        vehicle_data,
        nlp_model,
        nlp_tokenizer,
        max_new_tokens=200,
    )
    
    inspection_result["reports"] = {
        text_generation_model: report
    }
    

    # Prepare report filename
    plate_text = (
        plate_result["text"]
        if plate_result is not None
        else "unknown"
    )
    
    def result_to_filename(field):
        if field["status"] == "certain":
            return str(field["value"])
    
        if field["status"] == "uncertain":
            return "-".join(
                str(candidate["value"])
                for candidate in field["candidates"][:2]
            )
    
        return "unknown"
        
    report_path = (
        Path(report_save_folder)
        / (
            f"report_"
            f"{text_generation_model}_"
            f"{result_to_filename(color_result)}_"
            f"{result_to_filename(brand_result)}_"
            f"{plate_text}.html"
        )
    )
    
    
    # Generate HTML report
    generate_inspection_report(
        inspection_result,
        report_model=text_generation_model,
        save_path=report_path,
    )
    
    
    # Return full inspection result
    return inspection_result