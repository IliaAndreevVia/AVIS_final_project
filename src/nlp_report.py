def vehicle_data_to_structured_input(vehicle_data):
    lines = []

    for key, value in vehicle_data.items():

        # Dictionary
        if isinstance(value, dict):
            lines.append(f"{key}:")

            for sub_key, sub_value in value.items():

                if isinstance(sub_value, list):
                    sub_value = ", ".join(map(str, sub_value))

                lines.append(
                    f"  {sub_key}: {sub_value}"
                )

        # List
        elif isinstance(value, list):
            value = ", ".join(map(str, value))
            lines.append(f"{key}: {value}")

        # String, number, etc.
        else:
            lines.append(f"{key}: {value}")

    return "\n".join(lines)

def generate_report(
    vehicle_data,
    model,
    tokenizer,
    max_new_tokens=200
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
            truncation=True
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        report = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

    else:

        prompt = (
            "Generate vehicle damage report:\n\n"
            + structured_input
            + "\n\nREPORT:\n"
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt"
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
            skip_special_tokens=True
        )

    return report.strip()