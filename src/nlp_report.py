def vehicle_data_to_structured_input(vehicle_data):
    lines = []

    # =====================================================
    # Vehicle classification
    # =====================================================

    for field_name in [
        "brand",
        "vehicle_type",
        "color",
        "viewpoint",
    ]:
        field = vehicle_data[field_name]

        lines.append(f"{field_name}:")
        lines.append(
            f"  status: {field['status']}"
        )

        # Certain
        if field["status"] == "certain":
            lines.append(
                f"  value: {field['value']}"
            )

            lines.append(
                f"  confidence: "
                f"{field['confidence'] * 100:.0f}%"
            )

        # Uncertain
        elif field["status"] == "uncertain":
            lines.append("  candidates:")

            for candidate in field.get(
                "candidates",
                [],
            ):
                lines.append(
                    f"    - {candidate['value']}: "
                    f"{candidate['confidence'] * 100:.0f}%"
                )

        # Unknown
        else:
            lines.append(
                "  value: unknown"
            )

            lines.append(
                f"  confidence: "
                f"{field.get('confidence', 0) * 100:.0f}%"
            )

        lines.append("")

    # =====================================================
    # License plate
    # =====================================================

    plate = vehicle_data["license_plate"]

    lines.append("license_plate:")
    lines.append(
        f"  status: {plate['status']}"
    )

    plate_value = plate.get("value")

    lines.append(
        f"  value: "
        f"{plate_value if plate_value else 'unknown'}"
    )

    lines.append(
        f"  confidence: "
        f"{plate.get('confidence', 0) * 100:.0f}%"
    )

    lines.append("")

    # =====================================================
    # Damages
    # =====================================================

    damaged_parts = vehicle_data.get(
        "damaged_parts",
        {},
    )

    lines.append("damages:")

    if not damaged_parts:
        lines.append(
            "  none detected"
        )

    else:
        damage_id = 1

        for part_name, damages in damaged_parts.items():

            # ---------------------------------------------
            # Merge duplicate damage types on same part
            # ---------------------------------------------

            unique_damages = {}

            for damage in damages:
                damage_type = damage["damage_type"]
                confidence = damage.get(
                    "confidence",
                    0.0,
                )

                if (
                    damage_type not in unique_damages
                    or confidence
                    > unique_damages[damage_type]
                ):
                    unique_damages[
                        damage_type
                    ] = confidence

            # ---------------------------------------------
            # Add each logical damage separately
            # ---------------------------------------------

            for (
                damage_type,
                confidence,
            ) in unique_damages.items():

                lines.append(
                    f"  - damage_id: {damage_id}"
                )

                lines.append(
                    f"    part: {part_name}"
                )

                lines.append(
                    f"    damage: {damage_type}"
                )

                lines.append(
                    f"    confidence: "
                    f"{confidence * 100:.0f}%"
                )

                damage_id += 1

    return "\n".join(lines)

def generate_report(
    vehicle_data,
    model,
    tokenizer,
    max_new_tokens=200,
):
    structured_input = vehicle_data_to_structured_input(
        vehicle_data
    )

    if model.config.is_encoder_decoder:
        prompt = (
            "Generate vehicle damage report:\n\n"
            + structured_input
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        report = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

    else:
        prompt = (
            "Generate vehicle damage report:\n\n"
            + structured_input
            + "\n\nREPORT:\n"
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        generated_ids = outputs[0][
            inputs["input_ids"].shape[1]:
        ]

        report = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

    return report.strip()