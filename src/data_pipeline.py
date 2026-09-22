"""
EN3150 Assignment 03 - Dataset & Data Preparation (Member 1)

Common data pipeline for the whole group.

Dataset : EuroSAT (RGB satellite patches, 10 land-use classes, native 64x64)
Split   : 70% train / 15% validation / 15% test, stratified, fixed seed
Leakage : the split is made ONCE on image indices and saved to disk.
          Normalisation statistics (mean/std) come from the TRAIN set only.
          Augmentation is applied to the TRAIN set only.

Everyone in the team uses:
    from data_pipeline import get_dataloaders
    train_loader, val_loader, test_loader, info = get_dataloaders()

Run this file directly to produce the dataset analysis (counts, sizes,
class distribution, split distribution) used in the report.

Requirements: pip install torch torchvision scikit-learn numpy matplotlib pillow
"""

import json
import hashlib
import os
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import EuroSAT
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------
# Settings (shared by the whole team - do not change without telling everyone)
# ----------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
DATA_DIR = os.path.join(ROOT, "data")
SPLIT_FILE = os.path.join(ROOT, "data", "split_indices.json")
IMG_SIZE = 64                  # maximum resolution required by the assignment
SEED = 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15
BATCH_SIZE = 64
NUM_WORKERS = 2


# ----------------------------------------------------------------------------
# 1. Load raw dataset (no transforms yet, PIL images)
# ----------------------------------------------------------------------------
def load_raw_dataset():
    """Download (first run only) and return the raw EuroSAT dataset."""
    return EuroSAT(root=DATA_DIR, download=True)


# ----------------------------------------------------------------------------
# 2. Dataset analysis (for the report)
# ----------------------------------------------------------------------------
def analyse_dataset(raw):
    """Print and return number of images, classes, image sizes, class counts."""
    labels = np.array([y for _, y in raw.samples])
    class_names = raw.classes
    counts = Counter(labels.tolist())

    # Check image sizes on every image (fast enough for 27k small files)
    sizes = Counter()
    modes = Counter()
    from PIL import Image
    for path, _ in raw.samples:
        with Image.open(path) as im:
            sizes[im.size] += 1
            modes[im.mode] += 1

    print("=" * 60)
    print("DATASET ANALYSIS: EuroSAT (RGB)")
    print("=" * 60)
    print(f"Total images     : {len(raw)}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Image sizes      : {dict(sizes)}")
    print(f"Colour modes     : {dict(modes)}")
    print("\nClass distribution:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name:<22} {counts[i]:>5}  ({100*counts[i]/len(raw):.1f}%)")
    return {"total": len(raw), "classes": class_names,
            "sizes": {str(k): v for k, v in sizes.items()},
            "counts": {class_names[i]: counts[i] for i in range(len(class_names))}}


# ----------------------------------------------------------------------------
# 3. Stratified 70/15/15 split (done once, saved to disk)
# ----------------------------------------------------------------------------
def make_split(labels):
    """Stratified split on indices so each class keeps the same proportion."""
    idx = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(
        idx, test_size=(VAL_FRAC + TEST_FRAC), stratify=labels, random_state=SEED)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
        stratify=labels[temp_idx], random_state=SEED)
    return train_idx, val_idx, test_idx


def load_or_create_split(raw):
    """Reuse the saved split if it exists so the whole team shares one split."""
    labels = np.array([y for _, y in raw.samples])
    if os.path.exists(SPLIT_FILE):
        with open(SPLIT_FILE) as f:
            s = json.load(f)
        return np.array(s["train"]), np.array(s["val"]), np.array(s["test"])
    train_idx, val_idx, test_idx = make_split(labels)
    with open(SPLIT_FILE, "w") as f:
        json.dump({"seed": SEED, "train": train_idx.tolist(),
                   "val": val_idx.tolist(), "test": test_idx.tolist()}, f)
    return train_idx, val_idx, test_idx


# ----------------------------------------------------------------------------
# 4. Leakage checks
# ----------------------------------------------------------------------------
def check_no_leakage(raw, train_idx, val_idx, test_idx, check_duplicates=True):
    """Confirm the three sets are disjoint and (optionally) contain no
    identical image files across sets."""
    tr, va, te = set(train_idx.tolist()), set(val_idx.tolist()), set(test_idx.tolist())
    assert not (tr & va), "Leakage: train and val overlap"
    assert not (tr & te), "Leakage: train and test overlap"
    assert not (va & te), "Leakage: val and test overlap"
    assert len(tr) + len(va) + len(te) == len(raw), "Some images are missing from the split"
    print("[OK] Index sets are disjoint and cover all images.")

    if check_duplicates:
        def file_hash(p):
            with open(p, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        hashes = {name: {file_hash(raw.samples[i][0]) for i in idxs}
                  for name, idxs in [("train", train_idx), ("val", val_idx), ("test", test_idx)]}
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            common = hashes[a] & hashes[b]
            print(f"[{'OK' if not common else 'WARN'}] Identical images shared by {a} and {b}: {len(common)}")


# ----------------------------------------------------------------------------
# 5. Transforms and normalisation (statistics from TRAIN set only)
# ----------------------------------------------------------------------------
class SubsetWithTransform(Dataset):
    """Wraps a list of raw sample indices and applies a transform."""
    def __init__(self, raw, indices, transform):
        self.raw, self.indices, self.transform = raw, indices, transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        from PIL import Image
        path, label = self.raw.samples[self.indices[i]]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def compute_train_mean_std(raw, train_idx):
    """Per-channel mean/std computed on the training images only."""
    tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])
    ds = SubsetWithTransform(raw, train_idx, tf)
    loader = DataLoader(ds, batch_size=256, num_workers=NUM_WORKERS)
    n, s, s2 = 0, torch.zeros(3), torch.zeros(3)
    for x, _ in loader:
        n += x.size(0) * x.size(2) * x.size(3)
        s += x.sum(dim=(0, 2, 3))
        s2 += (x ** 2).sum(dim=(0, 2, 3))
    mean = s / n
    std = (s2 / n - mean ** 2).sqrt()
    return mean.tolist(), std.tolist()


def get_dataloaders(batch_size=BATCH_SIZE, augment=True):
    """Main function everyone calls. Returns train/val/test loaders + info."""
    raw = load_raw_dataset()
    train_idx, val_idx, test_idx = load_or_create_split(raw)
    mean, std = compute_train_mean_std(raw, train_idx)

    base = [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
            transforms.Normalize(mean, std)]
    eval_tf = transforms.Compose(base)
    train_tf = transforms.Compose(
        [transforms.Resize((IMG_SIZE, IMG_SIZE)),
         transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
         transforms.ToTensor(), transforms.Normalize(mean, std)]
    ) if augment else eval_tf   # augmentation on TRAIN only

    train_ds = SubsetWithTransform(raw, train_idx, train_tf)
    val_ds = SubsetWithTransform(raw, val_idx, eval_tf)
    test_ds = SubsetWithTransform(raw, test_idx, eval_tf)

    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, generator=g)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False, num_workers=NUM_WORKERS)

    info = {"classes": raw.classes, "num_classes": len(raw.classes),
            "mean": mean, "std": std, "img_size": IMG_SIZE,
            "sizes": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)}}
    return train_loader, val_loader, test_loader, info


# ----------------------------------------------------------------------------
# 6. Report figures: class distribution overall and per split
# ----------------------------------------------------------------------------
def plot_distributions(raw, train_idx, val_idx, test_idx, out="class_distribution.png"):
    import matplotlib.pyplot as plt
    labels = np.array([y for _, y in raw.samples])
    names = raw.classes
    x = np.arange(len(names))
    w = 0.27
    plt.figure(figsize=(11, 4.5))
    for k, (nm, idxs) in enumerate([("Train", train_idx), ("Validation", val_idx), ("Test", test_idx)]):
        c = np.bincount(labels[idxs], minlength=len(names))
        plt.bar(x + (k - 1) * w, c, w, label=nm)
    plt.xticks(x, names, rotation=35, ha="right")
    plt.ylabel("Number of images")
    plt.title("Class distribution across train / validation / test splits")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    print(f"Saved {out}")


def plot_samples(raw, out="sample_images.png"):
    import matplotlib.pyplot as plt
    from PIL import Image
    labels = np.array([y for _, y in raw.samples])
    fig, axes = plt.subplots(len(raw.classes), 5, figsize=(8, 16))
    rng = np.random.default_rng(SEED)
    for c, name in enumerate(raw.classes):
        picks = rng.choice(np.where(labels == c)[0], 5, replace=False)
        for j, p in enumerate(picks):
            axes[c, j].imshow(Image.open(raw.samples[p][0]).convert("RGB"))
            axes[c, j].axis("off")
        axes[c, 0].set_title(name, fontsize=8, loc="left")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


# ----------------------------------------------------------------------------
# Run this file directly to produce everything needed for the report
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    raw = load_raw_dataset()
    analyse_dataset(raw)

    train_idx, val_idx, test_idx = load_or_create_split(raw)
    print(f"\nSplit sizes -> train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")
    print(f"Fractions   -> {len(train_idx)/len(raw):.3f} / {len(val_idx)/len(raw):.3f} / {len(test_idx)/len(raw):.3f}")

    check_no_leakage(raw, train_idx, val_idx, test_idx)
    plot_distributions(raw, train_idx, val_idx, test_idx)
    plot_samples(raw)

    train_loader, val_loader, test_loader, info = get_dataloaders()
    print(f"\nTrain-set mean: {[round(m, 4) for m in info['mean']]}")
    print(f"Train-set std : {[round(s, 4) for s in info['std']]}")
    xb, yb = next(iter(train_loader))
    print(f"One batch -> images {tuple(xb.shape)}, labels {tuple(yb.shape)}")
