import os
import torch
import json
import hashlib
from pathlib import Path
from typing import List, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms

# Constants
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
MANIFEST_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "manifest.json"))
SPLITS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "splits.npz"))
RANDOM_STATE = 42

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

def _hash_entry(path: str, label: str) -> str:
    """Create a SHA256 hash of the file contents, its absolute path, and class label.
    This detects any modification of the image data, renaming, or relocation.
    """
    h = hashlib.sha256()
    # Include the absolute path and label
    h.update((path + "|" + label).encode("utf-8"))
    # Include file contents (read in binary chunks to avoid loading whole file into memory)
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                h.update(chunk)
    except Exception as e:
        raise IOError(f"Failed to read image file for hashing: {path}. Error: {e}")
    return h.hexdigest()

def _build_manifest() -> List[dict]:
    """Iterate over the raw data directory and build a manifest list.
    Each entry is a dict with absolute path, class label, and a hash for integrity.
    """
    manifest = []
    if not os.path.isdir(DATA_ROOT):
        raise FileNotFoundError(f"Data root not found: {DATA_ROOT}")
    for class_dir in sorted(Path(DATA_ROOT).iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name
        for file_path in sorted(class_dir.iterdir()):
            if file_path.suffix not in SUPPORTED_EXTS:
                continue
            abs_path = str(file_path.resolve())
            entry_hash = _hash_entry(abs_path, label)
            manifest.append({"path": abs_path, "label": label, "hash": entry_hash})
    return manifest

def _save_manifest(manifest: List[dict]):
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

def _load_manifest() -> List[dict]:
    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(f"Manifest file not found at {MANIFEST_PATH}")
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _verify_manifest(manifest: List[dict]):
    """Re‑hash each entry and ensure the stored hash matches.
    If any mismatch is found, raise an exception.
    """
    for entry in manifest:
        expected = entry["hash"]
        recomputed = _hash_entry(entry["path"], entry["label"])
        if expected != recomputed:
            raise ValueError(f"Manifest verification failed for {entry['path']}." )
    # also ensure file still exists
    missing = [e["path"] for e in manifest if not os.path.exists(e["path"])]
    if missing:
        raise FileNotFoundError(f"Files listed in manifest missing: {missing[:5]} (and {len(missing)-5} more)")

def _create_splits(manifest: List[dict]):
    """Create stratified train/val/test splits (80/10/10) and save to splits.npz.
    The splits are index arrays referencing the manifest list.
    """
    paths = [e["path"] for e in manifest]
    labels = [e["label"] for e in manifest]
    # First split: train 80%, temp 20%
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        np.arange(len(paths)), labels,
        test_size=0.20,
        stratify=labels,
        random_state=RANDOM_STATE,
    )
    # Second split: val 10%, test 10% (i.e., split temp 50/50)
    val_idx, test_idx, y_val, y_test = train_test_split(
        temp_idx, y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )
    np.savez_compressed(SPLITS_PATH, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    return train_idx, val_idx, test_idx

def _load_splits():
    if not os.path.exists(SPLITS_PATH):
        raise FileNotFoundError(f"Splits file not found at {SPLITS_PATH}")
    data = np.load(SPLITS_PATH)
    return data["train_idx"], data["val_idx"], data["test_idx"]

class AppleLeafDataset(torch.utils.data.Dataset):
    """PyTorch dataset that returns (image_tensor, label_index).
    Supports 'train', 'val', and 'test' splits.
    """
    def __init__(self, split: str = "train"):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split '{split}'. Must be one of train/val/test.")
        # Load or create manifest & splits
        if not os.path.exists(MANIFEST_PATH) or not os.path.exists(SPLITS_PATH):
            # Build from scratch
            manifest = _build_manifest()
            _save_manifest(manifest)
            _create_splits(manifest)
        else:
            manifest = _load_manifest()
            _verify_manifest(manifest)
        self.manifest = manifest
        train_idx, val_idx, test_idx = _load_splits()
        if split == "train":
            self.indices = train_idx
        elif split == "val":
            self.indices = val_idx
        else:
            self.indices = test_idx
        # Build label <-> index mapping
        self.classes = sorted({e["label"] for e in self.manifest})
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        # Transforms – training augmentations added later by user if needed
        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # Placeholder for optional training augmentations (set externally)
        self.train_augment = None

    def set_train_augment(self, augment_transform):
        """Attach a training‑only augmentation transform (torchvision.transforms)."""
        self.train_augment = augment_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        manifest_idx = self.indices[idx]
        entry = self.manifest[manifest_idx]
        img_path = entry["path"]
        label = entry["label"]
        # Load image using PIL
        from PIL import Image
        image = Image.open(img_path).convert("RGB")
        if self.train_augment is not None:
            # Apply augmentation before base transform
            image = self.train_augment(image)
        tensor = self.base_transform(image)
        label_idx = self.class_to_idx[label]
        return tensor, label_idx

if __name__ == "__main__":
    # Simple sanity check – build manifest and splits, then print summary
    manifest = _build_manifest()
    _save_manifest(manifest)
    train_idx, val_idx, test_idx = _create_splits(manifest)
    print(f"Dataset manifest saved to {MANIFEST_PATH} ({len(manifest)} images).")
    print(f"Splits saved to {SPLITS_PATH}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
