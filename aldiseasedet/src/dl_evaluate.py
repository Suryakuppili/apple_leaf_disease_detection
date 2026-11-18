#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Deep‑learning model evaluation script for the Apple Leaf Disease dataset.

* Loads the two saved checkpoints (custom CNN & MobileNet‑V2) using the project’s
  ``get_model`` helpers.
* Evaluates them on the untouched test split (AppleLeafDataset(split='test')).
* Computes accuracy, macro precision/recall/F1, per‑class scores, a 4 × 4
  confusion matrix, average inference time per image, and model size.
* Saves:
    - results/dl_evaluation_metrics.csv (scalar metrics per model)
    - results/dl_classification_report.csv (combined per‑class report)
    - results/confusion_matrix_custom_cnn.png
    - results/confusion_matrix_mobilenet_v2.png
    - results/model_comparison.png (accuracy & macro scores)
    - results/model_inference_time.png (average inference time)
* Works on CPU/GPU automatically and includes robust validation.
"""

import pathlib
import sys
import time
import warnings
from typing import List, Dict

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    classification_report,
)

# ---------------------------------------------------------------------
# Project imports (the script lives under ``aldiseasedet/src``)
# ---------------------------------------------------------------------
from src.dl_dataset import AppleLeafDataset
from src.dl_models.custom_cnn import get_model as get_custom_cnn
from src.dl_models.mobilenet_v2 import get_model as get_mobilenet_v2

# ---------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------
def load_checkpoint(model: torch.nn.Module, ckpt_path: pathlib.Path) -> None:
    """Load a checkpoint saved with the key ``model_state_dict``.
    Raises a clear exception if the file is missing or malformed.
    """
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "model_state_dict" not in ckpt:
            raise KeyError("'model_state_dict' key missing in checkpoint")
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint {ckpt_path}: {e}") from e


def get_device() -> torch.device:
    """Return the best available device (GPU if present, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync_and_time(fn, *args, device=None, **kwargs):
    if device is None:
        device = get_device()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.time()
    result = fn(*args, **kwargs)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.time() - start
    return result, elapsed


def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_classes: int,
) -> Dict:
    """Run inference on the whole test loader and return metrics and data.
    Returns a dict containing ``metrics`` (scalar values), ``confusion`` (matrix),
    ``y_true`` and ``y_pred``.
    """
    model = model.to(device)
    model.eval()

    all_true, all_pred = [], []
    total_time, total_samples = 0.0, 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # timed forward pass – sync handled inside ``sync_and_time``
            (outputs, ), elapsed = sync_and_time(
                lambda x: (model(x),), images, device=device
            )
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            _, preds = torch.max(outputs, dim=1)

            total_time += elapsed
            total_samples += images.size(0)

            all_true.append(labels.cpu().numpy())
            all_pred.append(preds.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    # -----------------------------------------------------------------
    # Validation of predictions
    # -----------------------------------------------------------------
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred have different lengths")
    if not np.all((y_pred >= 0) & (y_pred < n_classes)):
        raise ValueError("Predictions contain invalid class indices")

    # -----------------------------------------------------------------
    # Metric calculation
    # -----------------------------------------------------------------
    acc = accuracy_score(y_true, y_pred)
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(n_classes), average="macro"
    )
    per_prec, per_rec, per_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(n_classes), average=None
    )
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))

    if cm.shape != (n_classes, n_classes):
        raise ValueError(f"Confusion matrix shape mismatch: expected {(n_classes, n_classes)}, got {cm.shape}")

    avg_inf_time = total_time / total_samples if total_samples else float("nan")

    metrics = {
        "accuracy": acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "avg_inference_time_s": avg_inf_time,
    }
    for i in range(n_classes):
        metrics[f"class_{i}_precision"] = per_prec[i]
        metrics[f"class_{i}_recall"] = per_rec[i]
        metrics[f"class_{i}_f1"] = per_f1[i]

    return {
        "metrics": metrics,
        "confusion": cm,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def plot_confusion(cm: np.ndarray, class_names: List[str], title: str, out_path: pathlib.Path):
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_comparison(summary: pd.DataFrame, out_path: pathlib.Path):
    """Bar chart for accuracy and the three macro scores (higher is better)."""
    metrics = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    df = summary.set_index("model")[metrics].astype(float)
    ax = df.plot(kind="bar", figsize=(8, 5))
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison (Accuracy & Macro Scores)")
    plt.xticks(rotation=0)
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_inference_times(summary: pd.DataFrame, out_path: pathlib.Path):
    df = summary.set_index("model")[["avg_inference_time_s"]].astype(float)
    ax = df.plot(kind="bar", color="orange", figsize=(6, 4))
    ax.set_ylabel("Seconds per image")
    ax.set_title("Average Inference Time")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ---------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------
def main() -> None:
    # -----------------------------------------------------------------
    # Paths & constants – project root is two levels up from this file
    # -----------------------------------------------------------------
    ROOT = pathlib.Path(__file__).resolve().parents[1]
    RESULT_DIR = ROOT / "results"
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    CHECKPOINTS = {
        "custom_cnn": ROOT / "models" / "dl" / "custom_cnn.pt",
        "mobilenet_v2": ROOT / "models" / "dl" / "mobilenet_v2.pt",
    }

    # -----------------------------------------------------------------
    # Device selection
    # -----------------------------------------------------------------
    device = get_device()
    print(f"🚀 Using device: {device}")

    # -----------------------------------------------------------------
    # Test dataset & loader (single source – untouched test split)
    # -----------------------------------------------------------------
    try:
        test_set = AppleLeafDataset(split="test")
    except Exception as e:
        print(f"❌ Failed to initialise test dataset: {e}")
        sys.exit(1)

    class_to_idx = test_set.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(class_to_idx))]
    n_classes = len(class_names)

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # -----------------------------------------------------------------
    # Containers for results
    # -----------------------------------------------------------------
    summary_rows = []
    report_frames = []  # per‑model classification reports

    for model_name, ckpt_path in CHECKPOINTS.items():
        print(f"\n🔎 Evaluating {model_name} …")

        # -------------------------------------------------------------
        # Build model via project's ``get_model`` helper
        # -------------------------------------------------------------
        if model_name == "custom_cnn":
            model = get_custom_cnn(num_classes=n_classes)
        elif model_name == "mobilenet_v2":
            model = get_mobilenet_v2(num_classes=n_classes)
        else:
            raise ValueError(f"Unexpected model name: {model_name}")

        # -------------------------------------------------------------
        # Load checkpoint (key = model_state_dict)
        # -------------------------------------------------------------
        try:
            load_checkpoint(model, ckpt_path)
        except Exception as e:
            print(f"⚠️  Skipping {model_name} – {e}")
            continue

        model_mb = ckpt_path.stat().st_size / (1024 * 1024)

        # -------------------------------------------------------------
        # Evaluation
        # -------------------------------------------------------------
        try:
            out = evaluate(model, test_loader, device, n_classes)
        except Exception as e:
            print(f"⚠️  Evaluation failed for {model_name}: {e}")
            continue

        metrics = out["metrics"]
        metrics["model"] = model_name
        metrics["model_size_mb"] = model_mb
        summary_rows.append(metrics)

        # -------------------------------------------------------------
        # Save confusion matrix figure
        # -------------------------------------------------------------
        cm_path = RESULT_DIR / f"confusion_matrix_{model_name}.png"
        plot_confusion(
            out["confusion"], class_names, f"Confusion Matrix – {model_name}", cm_path
        )
        print(f"✅ Confusion matrix saved → {cm_path}")

        # -------------------------------------------------------------
        # Classification report – convert to a tidy DataFrame
        # -------------------------------------------------------------
        rep_dict = classification_report(
            out["y_true"], out["y_pred"], target_names=class_names, digits=4, output_dict=True
        )
        rep_df = pd.DataFrame(rep_dict).transpose().reset_index().rename(columns={"index": "class"})
        rep_df["model"] = model_name
        report_frames.append(rep_df)

    # -----------------------------------------------------------------
    # No successful evaluations?
    # -----------------------------------------------------------------
    if not summary_rows:
        print("❌ No model could be evaluated. Exiting.")
        sys.exit(1)

    # -----------------------------------------------------------------
    # Save aggregated scalar metrics
    # -----------------------------------------------------------------
    summary_df = pd.DataFrame(summary_rows)
    metrics_path = RESULT_DIR / "dl_evaluation_metrics.csv"
    summary_df.to_csv(metrics_path, index=False)
    print(f"\n✅ Summary metrics saved → {metrics_path}")

    # -----------------------------------------------------------------
    # Combined classification report CSV
    # -----------------------------------------------------------------
    combined_report = pd.concat(report_frames, ignore_index=True)
    report_path = RESULT_DIR / "dl_classification_report.csv"
    combined_report.to_csv(report_path, index=False)
    print(f"✅ Combined classification report saved → {report_path}")

    # -----------------------------------------------------------------
    # Plot comparison charts
    # -----------------------------------------------------------------
    comp_path = RESULT_DIR / "model_comparison.png"
    plot_comparison(summary_df, comp_path)
    print(f"✅ Model comparison chart saved → {comp_path}")

    inf_path = RESULT_DIR / "model_inference_time.png"
    plot_inference_times(summary_df, inf_path)
    print(f"✅ Inference‑time chart saved → {inf_path}")

    # -----------------------------------------------------------------
    # Terminal table
    # -----------------------------------------------------------------
    display_cols = [
        "model",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "avg_inference_time_s",
        "model_size_mb",
    ]
    print("\n=== Model Comparison Table ===")
    print(summary_df[display_cols].to_string(index=False, float_format="{:.4f}".format))

    print("\n🎉 Evaluation finished successfully!")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
