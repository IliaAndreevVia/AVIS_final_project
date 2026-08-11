import json

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models

from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    task="classification",
):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0
    total_samples = 0

    pbar = tqdm(loader, desc="Training")

    for X_batch, y_batch in pbar:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        if task == "segmentation":
            y_batch = y_batch.long()

        optimizer.zero_grad()

        outputs = model(X_batch)

        if isinstance(outputs, dict):
            outputs = outputs["out"]

        loss = criterion(outputs, y_batch)

        loss.backward()
        optimizer.step()

        batch_size = X_batch.size(0)

        running_loss += loss.item() * batch_size
        total_samples += batch_size

        predicted = outputs.argmax(dim=1)

        correct += (predicted == y_batch).sum().item()

        if task == "segmentation":
            total += y_batch.numel()
        else:
            total += batch_size

        pbar.set_postfix(
            loss=loss.item(),
            accuracy=correct / max(total, 1),
        )

    average_loss = running_loss / max(total_samples, 1)
    average_accuracy = correct / max(total, 1)

    return average_loss, average_accuracy

def evaluate_epoch(
    model,
    loader,
    criterion,
    device,
    task="classification",
):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    total_samples = 0

    pbar = tqdm(loader, desc="Evaluation")

    with torch.no_grad():
        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            if task == "segmentation":
                y_batch = y_batch.long()

            outputs = model(X_batch)

            if isinstance(outputs, dict):
                outputs = outputs["out"]

            loss = criterion(outputs, y_batch)

            batch_size = X_batch.size(0)

            running_loss += loss.item() * batch_size
            total_samples += batch_size

            predicted = outputs.argmax(dim=1)

            correct += (predicted == y_batch).sum().item()

            if task == "segmentation":
                total += y_batch.numel()
            else:
                total += batch_size

            pbar.set_postfix(
                loss=loss.item(),
                accuracy=correct / max(total, 1),
            )

    average_loss = running_loss / max(total_samples, 1)
    average_accuracy = correct / max(total, 1)

    return average_loss, average_accuracy

def train_evol_model(model_name, model, device, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS, task="classification"):

    model.to(device)
    
    best_test_loss = float('inf')
    patience = 5
    patience_counter = 0
    all_results = {}

    checkpoint_path = f"best_checkpoint_{model_name}.pth"
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, task=task)
        test_loss, test_acc = evaluate_epoch(model, val_loader, criterion, device, task=task)
    
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
    
        scheduler.step(test_loss)
    
        if test_loss < best_test_loss:
            best_test_loss = test_loss
    
            patience_counter = 0
    
            checkpoint = {'model_state_dict' : model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                          'epoch': epoch,
                          'best_test_loss': best_test_loss,
                          'scheduler_state_dict': scheduler.state_dict()}
    
            torch.save(checkpoint, checkpoint_path)
            
        else:
            patience_counter += 1
    
        if patience_counter >= patience:
            print("Early stopping triggered")
            break


    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    history_df = pd.DataFrame(history)
    results = {'best_test_loss': best_test_loss,
               'best_test_acc': max(history['test_acc']),
               'history_df': history_df}
        
        
            
    print(f"--- Best results ---")
    print(f'best_test_loss {results["best_test_loss"]}, best_test_acc {results["best_test_acc"]}')
    
    
    return model, results

def model_test(model,test_dataset, test_loader, path_for_saving, device="cpu"):
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            labels = labels.to(device)
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    
    torch.save({"accuracy":accuracy, "all_preds":all_preds, "all_labels":all_labels, "classes":test_dataset.classes},
               path_for_saving)
    
    model.to(device)
    
    print(f"Model test results saved to {path_for_saving}")
    print(f"Test Accuracy: {accuracy:.2f}%")

def segmentation_model_test(
    model,
    test_dataset,
    test_loader,
    path_for_saving,
    device
):
    model.eval()
    model = model.to(device)

    total_correct_pixels = 0
    total_pixels = 0

    all_preds = []
    all_masks = []

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            masks = masks.to(device).long()

            outputs = model(images)
            outputs = outputs["out"]

            predictions = outputs.argmax(dim=1)

            total_correct_pixels += (
                predictions == masks
            ).sum().item()

            total_pixels += masks.numel()

            all_preds.append(predictions.cpu())
            all_masks.append(masks.cpu())

    pixel_accuracy = total_correct_pixels / total_pixels


    results = {
        "pixel_accuracy": pixel_accuracy,
        "all_preds": torch.cat(all_preds),
        "all_masks": torch.cat(all_masks),
    }

    if hasattr(test_dataset, "classes"):
        results["classes"] = test_dataset.classes

    torch.save(results, path_for_saving)

    print(f"Pixel accuracy: {pixel_accuracy:.4f}")
    print(f"Test results saved to {path_for_saving}")

    return results
    
def create_efficientnet_b0(num_classes: int,
                           pretrained: bool) -> nn.Module:
    weights = (
        models.EfficientNet_B0_Weights.DEFAULT
        if pretrained
        else None
    )

    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features=in_features,
        out_features=num_classes,
    )

    return model

def preprocess_function(example, tokenizer, max_input_target_length):
    
    if tokenizer.name_or_path == "gpt2":
        prompt = (
        "Generate vehicle damage report:\n\n"
        + example["structured_input"]
        + "\n\nREPORT:\n")

        target = example["target_report"]
    
        full_text = (prompt
                     + target
                     + tokenizer.eos_token)
    
        return tokenizer(full_text,
                         truncation=True,
                         max_length=max_input_target_length)
        
    else:
        inputs = [
            "Generate vehicle damage report:\n" + text
            for text in example["structured_input"]
        ]
    
        model_inputs = tokenizer(inputs,
                                 max_length=max_input_target_length,
                                 truncation=True)
    
        labels = tokenizer(text_target=example["target_report"],
                           max_length=max_input_target_length,
                           truncation=True)
    
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

def train_save_flan_bart(model_name, dataset, num_train_epochs=10, per_device_train_batch_size=4, per_device_eval_batch_size=4, MAX_INPUT_LENGTH=1024):

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


    tokenized_dataset = dataset.map(
        lambda example: preprocess_function(
            example,
            tokenizer,
            MAX_INPUT_LENGTH,
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./trained_models/{model_name.split('/')[-1]}",

        num_train_epochs=num_train_epochs,

        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,

        gradient_accumulation_steps=2,

        learning_rate=5e-5,
        weight_decay=0.01,

        eval_strategy="epoch",
        save_strategy="epoch",

        predict_with_generate=True,

        logging_steps=50,

        load_best_model_at_end=True,

        metric_for_best_model="eval_loss",
        greater_is_better=False,

        fp16=False,
        bf16=True,

        save_total_limit=2,

        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,

        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],

        data_collator=data_collator,
        processing_class=tokenizer,
    )

    trainer.train()

    SAVE_DIR = Path(
        f"./trained_models/{model_name.split('/')[-1]}/best"
    )
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Model
    trainer.save_model(SAVE_DIR)
    print(f"Model saved to {SAVE_DIR}")

    # 2. Tokenizer
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"Tokinizer saved to {SAVE_DIR}")

    # 3. Full training history
    with open(SAVE_DIR / "log_history.json", "w") as f:
        json.dump(
            trainer.state.log_history,
            f,
            indent=4,
        )
    print(f"History saved to {SAVE_DIR}/log_history.json")

    # 4. Trainer state
    trainer.state.save_to_json(
        SAVE_DIR / "trainer_state.json"
    )
    print(f"Trainer state saved to {SAVE_DIR}/trainer_state.json")

    # 5. Training arguments
    torch.save(
        trainer.args,
        SAVE_DIR / "training_args.pt",
    )
    print(f"Training arguments saved to {SAVE_DIR}/trainer_state.json")

    # 6. Final validation metrics
    eval_metrics = trainer.evaluate()

    with open(SAVE_DIR / "eval_metrics.json", "w") as f:
        json.dump(
            eval_metrics,
            f,
            indent=4,
        )
    print(f"Final validation metrics saved to {SAVE_DIR}/eval_metrics.json")

    return trainer