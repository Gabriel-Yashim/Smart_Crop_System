"""
Shared settings and helpers for the crop disease pipeline.
Used by 03_train_compare.py and 04_evaluate_export.py.
Keep this file in the same folder as the scripts.
"""

import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

# ==== SETTINGS =====================================
DATA_ROOT = Path("data_split")   # output folder of 01_pool_and_split.py
ARTIFACTS = Path("artifacts")    # models, reports and plots go here
IMG_SIZE = 224                   # use 160 for faster experiments
MEAN = (0.485, 0.456, 0.406)     # ImageNet statistics, matches the pretrained weights
STD = (0.229, 0.224, 0.225)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42
# "robust" adds stronger color, grayscale, crop and erasing augmentation to reduce reliance on
# background and lighting cues. Use "standard" to train a baseline for comparison.
AUG_MODE = "standard"
# ====


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_classes():
    """Class names are the folder names under train/. Folders starting with _ are ignored."""
    train_dir = DATA_ROOT / "train"
    return sorted(p.name for p in train_dir.iterdir() if p.is_dir() and not p.name.startswith("_"))


def scan_split(split, classes):
    """Return a list of (image_path, class_index) for one split."""
    items = []
    for idx, cls in enumerate(classes):
        folder = DATA_ROOT / split / cls
        if not folder.is_dir():
            continue
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                items.append((str(p), idx))
    return items


class ImageListDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform
        self.labels = [y for _, y in items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        with Image.open(path) as im:
            img = im.convert("RGB")
        return self.transform(img), y


def train_transform():
    if AUG_MODE == "robust":
        # Wider crops change what the border contains, stronger brightness, contrast and saturation
        # jitter weakens lighting and white balance cues, occasional grayscale forces texture and shape
        # cues, and random erasing stops the model from relying on one patch. Hue stays small because
        # real disease color (yellowing, rust, browning) is a genuine signal.
        return T.Compose([
            T.RandomResizedCrop(IMG_SIZE, scale=(0.4, 1.0), ratio=(0.8, 1.25)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(25),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.04),
            T.RandomGrayscale(p=0.15),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
            T.RandomErasing(p=0.25, scale=(0.02, 0.15), value=0),
        ])
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0), ratio=(0.85, 1.18)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(20),
        T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.03),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def eval_transform():
    # Whole image resized to a square, no cropping, so lesions near the edge are kept.
    # The Streamlit app must apply exactly this preprocessing.
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def build_model(name, num_classes, pretrained):
    import timm
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


@torch.no_grad()
def predict_logits(model, loader, device):
    """Return (logits, labels) as CPU tensors."""
    model.eval()
    all_logits, all_y = [], []
    for x, y in loader:
        out = model(x.to(device))
        all_logits.append(out.float().cpu())
        all_y.append(y)
    return torch.cat(all_logits), torch.cat(all_y)


@torch.no_grad()
def cpu_latency_ms(model, runs=30):
    """Median single-image CPU latency in milliseconds."""
    m = copy.deepcopy(model).cpu().eval()
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    for _ in range(5):
        m(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        m(x)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))
