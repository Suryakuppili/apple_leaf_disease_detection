#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single‑image prediction script for the Apple Leaf Disease classification models.

Usage:
    python -m src.dl_predict --image "path/to/image.jpg"

The script:
* Loads the MobileNetV2 model (default) from ``models/dl/mobilenet_v2.pt``.
* Re‑uses the exact preprocessing pipeline defined in ``AppleLeafDataset``
  (resize → tensor → ImageNet normalisation).
* Runs on GPU when available, otherwise falls back to CPU.
* Prints the predicted disease, confidence, and the probability for each of the four
  classes.
"""

import argparse
import pathlib
import sys
import warnings
from typing import List

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Project imports (relative to the package root ``aldiseasedet``)
from src.dl_dataset import AppleLeafDataset
from src.dl_models.mobilenet_v2 import get_model as get_mobilenet_v2


def get_device() -> torch.device:
    """Return CUDA if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(model: torch.nn.Module, ckpt_path: pathlib.Path) -> None:
    """Load a checkpoint saved with the key ``model_state_dict``.
    Raises a clear error if the file is missing or malformed.
    """
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" not in ckpt:
        raise KeyError("'model_state_dict' key missing in checkpoint")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)


def build_transform() -> transforms.Compose:
    """Return the same transform used during evaluation (no augmentations)."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_image(image_path: pathlib.Path, transform: transforms.Compose) -> torch.Tensor:
    """Open an image, apply the transform, and return a 4‑D tensor ready for the model.
    Handles missing files and unsupported extensions gracefully.
    """
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Verify extension – mirrors SUPPORTED_EXTS in the dataset module
    if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError(
            f"Unsupported image extension '{image_path.suffix}'. Supported: .jpg, .jpeg, .png"
        )

    try:
        pil_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise IOError(f"Failed to open image '{image_path}': {e}") from e

    tensor = transform(pil_img)  # shape: C×H×W
    return tensor.unsqueeze(0)  # add batch dimension → 1×C×H×W


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Predict Apple leaf disease from a single image.")
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the leaf image (jpg/jpeg/png).",
    )
    args = parser.parse_args(argv)

    # -----------------------------------------------------------------
    # Resolve paths
    # -----------------------------------------------------------------
    root_dir = pathlib.Path(__file__).resolve().parents[1]  # project root (aldiseasedet)
    checkpoint_path = root_dir / "models" / "dl" / "mobilenet_v2.pt"
    image_path = pathlib.Path(args.image).expanduser().resolve()

    device = get_device()
    print(f"🚀 Using device: {device}")

    # -----------------------------------------------------------------
    # Load class mapping from the dataset (test split is fine – mapping is static)
    # -----------------------------------------------------------------
    dummy_dataset = AppleLeafDataset(split="test")
    class_to_idx = dummy_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    n_classes = len(class_to_idx)

    # -----------------------------------------------------------------
    # Model creation & checkpoint loading
    # -----------------------------------------------------------------
    model = get_mobilenet_v2(num_classes=n_classes)
    try:
        load_checkpoint(model, checkpoint_path)
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        sys.exit(1)

    model = model.to(device)
    model.eval()

    # -----------------------------------------------------------------
    # Image preprocessing
    # -----------------------------------------------------------------
    transform = build_transform()
    try:
        input_tensor = load_image(image_path, transform).to(device)
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)

    # -----------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).cpu().squeeze(0).numpy()

    pred_idx = int(probs.argmax())
    pred_class = idx_to_class[pred_idx]
    confidence = probs[pred_idx] * 100

    # -----------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------
    print("\n=== Prediction Result ===")
    print(f"Image:   {image_path}")
    print(f"Class:   {pred_class} ({confidence:.2f}% confidence)")
    print("\nProbabilities:")
    for i, prob in enumerate(probs):
        print(f"  {idx_to_class[i]:<25}: {prob * 100:>6.2f}%")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
