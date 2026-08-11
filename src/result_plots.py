import json

from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from sklearn.metrics import confusion_matrix
from ultralytics import YOLO


from sklearn.metrics import confusion_matrix

def train_val_acc_loss_plot(train_results, model_purpouse, plot_name, plot_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']):
    
    df = train_results['history_df']
    fig, ax = plt.subplots(1, 2, figsize = (15, 5))
    
    
    ax[0].plot(df.index, 
             df.train_acc, 
             label="Train", 
             color=plot_colors[1])
    
    ax[0].plot(df.index, 
             df.test_acc, 
             label="Validation", 
             color=plot_colors[0],
             linestyle = '--')
    
    best_epoch = list(df[df.test_acc == train_results['best_test_acc']].index)[0]
    
    ax[0].axvline(best_epoch, 
                color=plot_colors[5])
    
    ax[0].text(best_epoch*0.95, 
             df.train_acc.min(), 
             "Best Epoch", 
             rotation="vertical",
             fontsize = 9) 
    
    ax[0].text(best_epoch*0.65,
             df.test_acc[best_epoch],
             f"Best validation accuracy:\n{df.test_acc[best_epoch]:.2f}",
             fontsize = 9)
    
    ax[0].text(best_epoch*0.7, 
             df.train_acc[best_epoch]*0.99, 
             f"Best train accuracy:\n{df.train_acc[best_epoch]:.2f}",
             fontsize = 9)
    
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].set_title(f"{model_purpouse} model accuracy vs Epoch")
    ax[0].legend()
    
    ax[1].plot(df.index, 
             df.train_loss, 
             label="Train", 
             color=plot_colors[1])
    
    ax[1].plot(df.index, 
             df.test_loss, 
             label="Validation", 
             color=plot_colors[0],
             linestyle = '--')
    
    ax[1].axvline(best_epoch, 
                color=plot_colors[5])
    
    ax[1].text(best_epoch*0.95, 
             0.6, 
             "Best Epoch", 
             rotation="vertical",
             fontsize = 9) 
    
    ax[1].text(best_epoch*0.7,
             df.test_loss[best_epoch]*1.1,
             f"Best validation loss:\n{df.test_loss[best_epoch]:.2f}",
             fontsize = 9)
    
    ax[1].text(best_epoch*0.7, 
             df.train_loss[best_epoch]*1.1, 
             f"Best train loss:\n{df.train_loss[best_epoch]:.2f}",
             fontsize = 9)
    
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_title(f"{model_purpouse} model loss vs Epoch")
    ax[1].legend()
    
    plt.savefig(f"./plots/{plot_name}.png",
                dpi=300,
                bbox_inches="tight")
    plt.show()

    test_color_detection_results = torch.load('./trained_pytorch_models/color/test_results.pth', 
                                          weights_only=False)


def conf_matrix_heatmap(test_results, model_purpouse, plot_name):
    conf_mat = confusion_matrix(test_results['all_labels'], 
                                test_results['all_preds'], 
                                normalize="true")
    
    plt.figure(figsize=(15,15))
    sns.heatmap(conf_mat, 
                annot=True, 
                fmt=".2f", 
                cmap='Blues', 
                xticklabels=test_results['classes'], 
                yticklabels=test_results['classes'])
    
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f"{model_purpouse}\n Actual vs Predicted\n (normalozed by rows)")
    plt.savefig(f"./plots/{plot_name}.png",
                dpi=300,
                bbox_inches="tight")
    plt.show()

def dataset_class_distribution(data, plot_name):

    # Path with class folders
    if isinstance(data, (str, Path)):

        dataset_path = Path(data)

        image_extensions = {
            ".jpg", ".jpeg", ".png",
            ".bmp", ".webp"
        }

        class_counts = {}

        for class_dir in sorted(dataset_path.iterdir()):

            if not class_dir.is_dir():
                continue

            count = sum(
                1
                for file in class_dir.rglob("*")
                if file.is_file()
                and file.suffix.lower() in image_extensions
            )

            class_counts[class_dir.name] = count

    # PyTorch Dataset
    else:

        labels = []

        for i in range(len(data)):
            _, label = data[i]
            labels.append(label)

        counts = Counter(labels)

        class_counts = {
            class_name: counts.get(class_idx, 0)
            for class_idx, class_name in enumerate(data.classes)
        }

    df = pd.DataFrame({
        "class": class_counts.keys(),
        "count": class_counts.values()
    })

    total = df["count"].sum()

    print(f"Total images: {total}")

    display(df)

    plt.figure(figsize=(12, 6))

    bars = plt.bar(
        df["class"],
        df["count"]
    )

    for bar in bars:

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(bar.get_height())}",
            ha="center",
            va="bottom"
        )

    plt.xlabel("Class")
    plt.ylabel("Number of images")

    plt.title(
        f"Class Distribution — Total: {total}"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()
    
    plt.savefig(f"./plots/{plot_name}.png",
                dpi=300,
                bbox_inches="tight")
    plt.show()

    return df

def segmentation_class_distribution(dataset, classes_csv_path, plot_name):

    classes_df = pd.read_csv(classes_csv_path)
    classes = classes_df["class"].tolist()

    counts = Counter()

    for sample in dataset.samples:

        mask_path = sample[1]

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            raise ValueError(f"Cannot read mask: {mask_path}")

        values, frequencies = np.unique(
            mask,
            return_counts=True
        )

        for value, frequency in zip(values, frequencies):
            counts[int(value)] += int(frequency)

    data = []

    for class_idx, class_name in enumerate(classes):
        data.append({
            "class": class_name,
            "pixels": counts.get(class_idx, 0)
        })

    df = pd.DataFrame(data)

    total_pixels = df["pixels"].sum()

    df["percent"] = (
        df["pixels"] / total_pixels * 100
    )

    print(f"Total pixels: {total_pixels:,}")
    display(df)

    plt.figure(figsize=(14, 6))

    bars = plt.bar(
        df["class"],
        df["percent"]
    )

    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.1f}%",
            ha="center",
            va="bottom"
        )

    plt.xlabel("Class")
    plt.ylabel("Pixels (%)")
    plt.title("Segmentation Class Distribution")

    plt.xticks(rotation=90)

    plt.tight_layout()
    
    plt.savefig(f"./plots/{plot_name}.png",
                dpi=300,
                bbox_inches="tight")
    plt.show()

    return df

def plot_yolo_training_results(results_csv_path, plot_name, color_1, color_2):
    results_csv_path = Path(results_csv_path)

    df = pd.read_csv(results_csv_path)

    df.columns = df.columns.str.strip()

    plots = [
        ("train/box_loss", "train/box_loss"),
        ("train/cls_loss", "train/cls_loss"),
        ("train/dfl_loss", "train/dfl_loss"),
        ("metrics/precision(B)", "metrics/precision(B)"),
        ("metrics/recall(B)", "metrics/recall(B)"),

        ("val/box_loss", "val/box_loss"),
        ("val/cls_loss", "val/cls_loss"),
        ("val/dfl_loss", "val/dfl_loss"),
        ("metrics/mAP50(B)", "metrics/mAP50(B)"),
        ("metrics/mAP50-95(B)", "metrics/mAP50-95(B)"),
    ]

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(18, 9)
    )

    axes = axes.flatten()

    for ax, (column, title) in zip(axes, plots):

        if column not in df.columns:
            ax.set_visible(False)
            continue

        y = df[column]

        ax.plot(
            df["epoch"],
            y,
            label="results",
            color=color_1
        )

        smooth = y.rolling(
            window=5,
            center=True,
            min_periods=1
        ).mean()

        ax.plot(
            df["epoch"],
            smooth,
            linestyle="--",
            label="smooth",
            color=color_2
        )

        ax.set_title(title)

        if column == "train/cls_loss":
            ax.legend()

    plt.tight_layout()
    
    plt.savefig(f"./plots/{plot_name}.png",
                dpi=300,
                bbox_inches="tight")
    plt.show()

def plot_conf_matrix_yolo(
    best_weights_path,
    data_yaml_path,
    plot_name,
    conf=0.25,
    iou=0.7
):
    model = YOLO(best_weights_path)

    metrics = model.val(
        data=data_yaml_path,
        split="val",
        conf=conf,
        iou=iou,
        verbose=False,
        plots=True
    )

    # Confusion matrix
    cm = metrics.confusion_matrix.matrix

    # Class names
    names = metrics.confusion_matrix.names

    if isinstance(names, dict):
        class_names = list(names.values())
    else:
        class_names = list(names)

    labels = class_names + ["background"]

    # Normalize by True class
    cm_normalized = cm / (cm.sum(axis=0, keepdims=True) + 1e-9)

    # Plot
    plt.figure(figsize=(18, 14))

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=1
    )

    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("YOLO Confusion Matrix (Normalized)")

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(f"./plots/{plot_name}.png",
                dpi=300,
                bbox_inches="tight")
    plt.show()

def plot_nlp_training_results(model_path, plot_name):
    model_path = Path(model_path)
    log_path = model_path / "log_history.json"

    if not log_path.exists():
        raise FileNotFoundError(
            f"log_history.json not found: {log_path}"
        )

    # Load training history
    with open(log_path, "r") as f:
        history = json.load(f)

    df = pd.DataFrame(history)

    # Train / Validation Loss
    train_df = (
        df[df["loss"].notna()].copy()
        if "loss" in df.columns
        else pd.DataFrame()
    )

    eval_df = (
        df[df["eval_loss"].notna()].copy()
        if "eval_loss" in df.columns
        else pd.DataFrame()
    )

    if not train_df.empty or not eval_df.empty:
        plt.figure(figsize=(10, 6))

        if not train_df.empty:
            plt.plot(
                train_df["epoch"],
                train_df["loss"],
                label="Train Loss"
            )

        if not eval_df.empty:
            plt.plot(
                eval_df["epoch"],
                eval_df["eval_loss"],
                marker="o",
                label="Validation Loss"
            )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")

        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"./plots/{plot_name}_val_loss.png",
                    dpi=300,
                    bbox_inches="tight")
        plt.show()

    # Gradient Norm
    if (
        not train_df.empty
        and "grad_norm" in train_df.columns
        and train_df["grad_norm"].notna().any()
    ):
        plt.figure(figsize=(10, 6))

        plt.plot(
            train_df["epoch"],
            train_df["grad_norm"]
        )

        plt.xlabel("Epoch")
        plt.ylabel("Gradient Norm")
        plt.title("Gradient Norm")

        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"./plots/{plot_name}_gradient_norm.png",
                dpi=300,
                bbox_inches="tight")
        plt.show()

    return df