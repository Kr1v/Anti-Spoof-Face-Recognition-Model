"""
=============================================================================
DenseNet121 Anti-Spoofing — v4.0
   (Transfer Learning + Freeze/Unfreeze Schedule + Warm-Restart LR Boosts
    + Group-Level Split with Val/Test Swap + EMA + Focal Loss + Hard Mining)
=============================================================================

Key changes vs the CDCN-from-scratch v3.2 script, based on the v3 training
log you shared:

  1. BACKBONE: ImageNet-pretrained DenseNet121 instead of a from-scratch
     CDCN. Your v3 run had Val ACER=0.0424 but Test ACER=0.2255 — a ~5x
     generalization gap. That gap is the signature of a from-scratch model
     with ~8M params memorizing a training set of only 49 identity groups.
     Pretrained ImageNet features generalize far better on datasets this
     size.

  2. LOWER LR + WARM RESTARTS EVERY 5 EPOCHS: the backbone uses a much
     smaller LR (5e-5) than the fresh classification head (5e-4), and both
     follow CosineAnnealingWarmRestarts with T_0=5 — LR decays smoothly
     within each 5-epoch block, then "boosts" back up at the start of the
     next block. This is aligned with hard-example-mining (BOOST_EVERY=5)
     so the LR kick and the harder sampling distribution land together.

  3. FREEZE -> UNFREEZE SCHEDULE: for the first FREEZE_EPOCHS epochs, only
     the new classifier head trains (backbone frozen) so the head doesn't
     send destructive gradients into the pretrained features while it's
     still random. After that, the backbone unfreezes and fine-tunes at
     its low LR. The earliest DenseNet stem + first two dense blocks stay
     frozen for the whole run (they encode generic low-level texture/edge
     features that don't need retraining and are the most overfitting-prone
     to unfreeze on a small dataset).

  4. VAL/TEST SWAP: your val split had fewer images (1296) than test
     (2417). As requested, the two sets are swapped post-split so
     validation now uses the larger set and the held-out test set is the
     smaller one — grouping by identity folder is preserved exactly as
     before, so there is still zero frame/identity leakage between them.

  5. Simplified output head: DenseNet121 doesn't naturally produce the
     multi-scale depth maps CDCN's central-difference convolutions did, so
     this version uses a single binary logit (real/fake) with focal loss +
     label smoothing + class-weighted BCE, keeping every other part of the
     regularization stack (mixup, EMA, hard-example mining, ACER-based
     early stopping) identical in spirit to v3.
"""

# ───────────────────────────── IMPORTS ───────────────────────────────────────
import csv
import argparse
import logging
import random
import sys
import copy
import os
from pathlib import Path

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import densenet121, DenseNet121_Weights
from tqdm import tqdm

from sklearn.metrics import roc_auc_score

# ───────────────────────────── REPRODUCIBILITY ───────────────────────────────
SEED = 42

def seed_everything(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

seed_everything()

# ───────────────────────────── CONFIG ────────────────────────────────────────
class Config:
    IMG_SIZE       = 224            # matches DenseNet121 ImageNet pretraining resolution
    BATCH_SIZE     = 32
    EPOCHS         = 60

    LR_BACKBONE    = 5e-5           # small — protects pretrained features from overfitting
    LR_HEAD        = 5e-4           # larger — head is randomly initialized, needs to move faster
    WEIGHT_DECAY   = 1e-2           # stronger reg than v3 (2e-3) since backbone is much larger

    FREEZE_EPOCHS  = 5              # epochs 1..5: head-only training, backbone frozen
    BOOST_EVERY    = 5              # hard-example remining cadence
    LR_T0          = 5              # warm-restart period — LR "boost" lands every 5 epochs

    VAL_SPLIT      = 0.15
    TEST_SPLIT     = 0.15
    THRESHOLD      = 0.5
    PATIENCE       = 15
    EMA_DECAY      = 0.999
    MIXUP_ALPHA    = 0.1
    LABEL_SMOOTH   = 0.05
    FOCAL_GAMMA    = 2.0
    DROPOUT        = 0.5

    NUM_WORKERS    = min(4, os.cpu_count() or 0)

    MODEL_DIR      = Path("/kaggle/working/models")
    TRAIN_DIR      = Path("/kaggle/working/training")
    MODEL_SAVE     = MODEL_DIR / "densenet121_best_v4.pt"
    CKPT_LAST      = MODEL_DIR / "checkpoint_last_v4.pt"
    DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────────────────────── LOGGING ───────────────────────────────────────
Config.TRAIN_DIR.mkdir(parents=True, exist_ok=True)
Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(Config.TRAIN_DIR / "train_v4.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ───────────────────────────── ARGUMENT PARSING ──────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--real_dir",      default="dataset/real")
    p.add_argument("--fake_dir",      default="dataset/fake")
    p.add_argument("--epochs",        type=int,   default=Config.EPOCHS)
    p.add_argument("--batch_size",    type=int,   default=Config.BATCH_SIZE)
    p.add_argument("--lr_backbone",   type=float, default=Config.LR_BACKBONE)
    p.add_argument("--lr_head",       type=float, default=Config.LR_HEAD)
    p.add_argument("--freeze_epochs", type=int,   default=Config.FREEZE_EPOCHS)
    p.add_argument("--patience",      type=int,   default=Config.PATIENCE)
    p.add_argument("--boost_every",   type=int,   default=Config.BOOST_EVERY)
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint if it exists")
    return p.parse_args()

# ───────────────────────────── DATA COLLECTION ───────────────────────────────
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def collect_grouped_paths(real_dir: Path, fake_dir: Path):
    paths, labels, groups = [], [], []
    _solo_counter = 0

    for class_dir, label in [(real_dir, 1), (fake_dir, 0)]:
        class_dir = Path(class_dir)
        class_tag = class_dir.name

        if not class_dir.exists():
            raise FileNotFoundError(f"Directory not found: {class_dir}")

        for sub in sorted(class_dir.iterdir()):
            if not sub.is_dir():
                continue
            imgs = [p for p in sub.rglob("*") if p.suffix.lower() in EXTS]
            group_id = f"{class_tag}__{sub.name}"
            for p in imgs:
                paths.append(str(p))
                labels.append(label)
                groups.append(group_id)

        loose = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTS]
        for p in loose:
            _solo_counter += 1
            paths.append(str(p))
            labels.append(label)
            groups.append(f"{class_tag}__LOOSE__{_solo_counter:06d}")

    if not paths:
        raise FileNotFoundError("No images found. Check --real_dir / --fake_dir.")

    n_real = sum(labels)
    n_fake = len(labels) - n_real
    logger.info(f"Total images  → real: {n_real} | fake: {n_fake}")
    logger.info(f"Total groups  → {len(set(groups))} (person folders + loose images)")
    return paths, labels, groups


def group_split(paths, labels, groups, val_frac, test_frac, seed):
    """Identity-folder-level split — guarantees zero frame leakage across
    train/val/test since every image from a given person folder stays in
    exactly one split."""
    paths   = np.array(paths)
    labels  = np.array(labels)
    groups  = np.array(groups)
    idx_all = np.arange(len(paths))

    train_idx_parts, val_idx_parts, test_idx_parts = [], [], []

    for cls in (0, 1):
        cls_idx = idx_all[labels == cls]
        cls_groups = groups[cls_idx]

        unique_groups = np.array(list(set(cls_groups)))
        folder_groups = sorted([g for g in unique_groups if "__LOOSE__" not in g])
        loose_groups = sorted([g for g in unique_groups if "__LOOSE__" in g])

        local_rng = random.Random(seed)
        local_rng.shuffle(folder_groups)
        local_rng.shuffle(loose_groups)

        n_folders = len(folder_groups)
        if n_folders >= 3:
            n_val = max(1, int(round(n_folders * val_frac)))
            n_test = max(1, int(round(n_folders * test_frac)))
            if n_val + n_test >= n_folders:
                n_val, n_test = 1, 1
            val_f = folder_groups[:n_val]
            test_f = folder_groups[n_val:n_val+n_test]
            train_f = folder_groups[n_val+n_test:]
        elif n_folders == 2:
            train_f, val_f, test_f = [folder_groups[0]], [folder_groups[1]], []
        elif n_folders == 1:
            train_f, val_f, test_f = [folder_groups[0]], [], []
        else:
            train_f, val_f, test_f = [], [], []

        n_loose = len(loose_groups)
        n_val_l = int(round(n_loose * val_frac))
        n_test_l = int(round(n_loose * test_frac))

        if len(val_f) == 0 and n_loose >= 1:
            n_val_l = max(1, n_val_l)
        if len(test_f) == 0 and n_loose >= 1:
            n_test_l = max(1, n_test_l)

        val_l = loose_groups[:n_val_l]
        test_l = loose_groups[n_val_l:n_val_l+n_test_l]
        train_l = loose_groups[n_val_l+n_test_l:]

        cls_train_g = set(train_f + train_l)
        cls_val_g = set(val_f + val_l)
        cls_test_g = set(test_f + test_l)

        for idx in cls_idx:
            g = groups[idx]
            if g in cls_train_g:
                train_idx_parts.append(idx)
            elif g in cls_val_g:
                val_idx_parts.append(idx)
            elif g in cls_test_g:
                test_idx_parts.append(idx)

    train_idx = np.array(train_idx_parts, dtype=int)
    val_idx = np.array(val_idx_parts, dtype=int)
    test_idx = np.array(test_idx_parts, dtype=int)

    train_s = [(paths[i], int(labels[i])) for i in train_idx]
    val_s   = [(paths[i], int(labels[i])) for i in val_idx]
    test_s  = [(paths[i], int(labels[i])) for i in test_idx]

    train_g, val_g, test_g = set(groups[train_idx]), set(groups[val_idx]), set(groups[test_idx])
    if (train_g & val_g) or (train_g & test_g) or (val_g & test_g):
        raise RuntimeError("Data split group leakage detected.")

    # ── VAL/TEST SWAP (requested) ─────────────────────────────────────────
    # Your v3 log showed Val (1296 imgs) smaller than Test (2417 imgs).
    # Swap the two sets so validation — the split that steers early
    # stopping and checkpoint selection — gets the larger, more
    # representative pool, and the held-out test set becomes the smaller
    # one. Group membership (and therefore zero-leakage) is untouched;
    # only which named split each group of images belongs to changes.
    val_s, test_s = test_s, val_s
    val_idx, test_idx = test_idx, val_idx
    val_g, test_g = test_g, val_g
    logger.info("↔ Val/Test swapped per request — validation now uses the larger split.")

    logger.info(f"Train split   → real: {sum(l for _, l in train_s)} | fake: {len(train_s)-sum(l for _, l in train_s)} | groups: {len(train_g)}")
    logger.info(f"Val   split   → real: {sum(l for _, l in val_s)} | fake: {len(val_s)-sum(l for _, l in val_s)} | groups: {len(val_g)}")
    logger.info(f"Test  split   → real: {sum(l for _, l in test_s)} | fake: {len(test_s)-sum(l for _, l in test_s)} | groups: {len(test_g)}")
    logger.info("✓ Zero data leakage across Train, Val, and Test splits confirmed.")

    return train_s, val_s, test_s

# ───────────────────────────── SAMPLER ───────────────────────────────────────
def make_sampler(samples, sample_weights=None):
    if sample_weights is None:
        labels       = [l for _, l in samples]
        class_counts = np.bincount(labels, minlength=2).clip(1)
        cw           = 1.0 / class_counts
        sample_weights = [float(cw[l]) for l in labels]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

# ───────────────────────────── TRANSFORMS ────────────────────────────────────
def get_transforms(img_size):
    train_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08),
        transforms.RandomRotation(15),
        transforms.RandomGrayscale(0.05),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])
    val_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf

# ───────────────────────────── DATASET ───────────────────────────────────────
class FaceDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((128, 128, 3), dtype=np.uint8)
        img    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img)
        return tensor, torch.tensor(label, dtype=torch.float32), idx

# ───────────────────────────── MIXUP ─────────────────────────────────────────
def mixup_data(x, y, alpha=0.1):
    if alpha <= 0:
        return x, y, x, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(x.size(0), device=x.device)
    return x, y, x[index], y[index], lam

# ───────────────────────────── MODEL ─────────────────────────────────────────
class DenseNet121AntiSpoof(nn.Module):
    """DenseNet121 backbone (ImageNet-pretrained) + a lightweight binary
    classification head. Early layers (stem + first two dense blocks) are
    frozen for the entire run — they encode generic edge/texture features
    that are the most prone to overfitting if fine-tuned on a small
    dataset. The rest of the backbone starts frozen too and is released
    after Config.FREEZE_EPOCHS via unfreeze_backbone()."""

    ALWAYS_FROZEN = {"conv0", "norm0", "relu0", "pool0", "denseblock1", "transition1", "denseblock2", "transition2"}

    def __init__(self, dropout=0.5, pretrained=True):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = densenet121(weights=weights)
        self.features = backbone.features
        in_features = backbone.classifier.in_features  # 1024

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 1),
        )

        self._freeze_backbone()          # everything off by default
        self._unfreeze_always_frozen_exempt = False

    def _freeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self):
        """Release all backbone layers EXCEPT the permanently frozen
        stem + first two dense blocks."""
        for name, module in self.features.named_children():
            if name in self.ALWAYS_FROZEN:
                continue
            for p in module.parameters():
                p.requires_grad_(True)

    def forward(self, x):
        feat = self.features(x)
        feat = F.relu(feat, inplace=True)
        pooled = self.pool(feat)
        logit = self.head(pooled)
        return logit.squeeze(1)   # raw logit, apply sigmoid outside for scoring

# ───────────────────────────── EMA ───────────────────────────────────────────
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay  = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.copy_(self.decay * s + (1 - self.decay) * m)
        for s_buf, m_buf in zip(self.shadow.buffers(), model.buffers()):
            s_buf.copy_(m_buf)

    def get_model(self):
        return self.shadow

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)

# ───────────────────────────── LOSS ──────────────────────────────────────────
class FocalBCEWithLogits(nn.Module):
    def __init__(self, pos_weight=1.0, gamma=2.0, smooth=0.05):
        super().__init__()
        self.gamma  = gamma
        self.smooth = smooth
        self.register_buffer("pw", torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, logits, labels):
        device_type = logits.device.type
        with torch.amp.autocast(device_type, enabled=False):
            logits = logits.float()
            labels = labels.float()
            probs  = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
            y_soft = labels * (1 - self.smooth) + 0.5 * self.smooth
            p_t    = torch.where(labels >= 0.5, probs, 1 - probs)
            focal  = (1 - p_t) ** self.gamma
            weight = torch.where(labels >= 0.5, self.pw.expand_as(labels), torch.ones_like(labels))
            bce    = F.binary_cross_entropy_with_logits(logits, y_soft, weight=weight, reduction="none")
            loss   = (focal * bce).mean()
        return loss


def per_sample_loss(logits, labels):
    device_type = logits.device.type
    with torch.amp.autocast(device_type, enabled=False):
        bce = F.binary_cross_entropy_with_logits(logits.float(), labels.float(), reduction="none")
    return bce.detach().cpu().numpy()

# ───────────────────────────── METRICS ───────────────────────────────────────
def compute_metrics(labels, scores, threshold):
    preds    = np.array([1 if s >= threshold else 0 for s in scores])
    labels_a = np.array(labels)
    accuracy = float((preds == labels_a).mean())
    rm = labels_a == 1; fm = ~rm
    bpcer = float((preds[rm] == 0).mean()) if rm.any() else 0.0
    apcer = float((preds[fm] == 1).mean()) if fm.any() else 0.0
    acer  = (apcer + bpcer) / 2.0
    try:
        auc = float(roc_auc_score(labels_a, scores))
    except ValueError:
        auc = float("nan")
    return {"accuracy": accuracy, "APCER": apcer, "BPCER": bpcer, "ACER": acer, "AUC": auc}

# ───────────────────────────── EARLY STOPPING ────────────────────────────────
class EarlyStopping:
    def __init__(self, patience, path):
        self.patience  = patience
        self.path      = path
        self.best_acer = float("inf")
        self.counter   = 0
        self.triggered = False

    def step(self, acer, model):
        if acer < self.best_acer - 1e-5:
            self.best_acer = acer
            self.counter   = 0
            self.triggered = False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.path)
            logger.info(f"  [SAVED] Best ACER={acer:.4f} → {self.path}")
        else:
            self.counter += 1
            logger.info(f"  No improvement — patience {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered

    def state_dict(self):
        return {"best_acer": self.best_acer, "counter": self.counter, "triggered": self.triggered}

    def load_state_dict(self, sd):
        self.best_acer = sd["best_acer"]
        self.counter   = sd["counter"]
        self.triggered = sd["triggered"]

# ───────────────────────────── HARD-EXAMPLE MINING ───────────────────────────
@torch.no_grad()
def compute_sample_weights_hard(model, samples, cfg_obj):
    logger.info("  [BOOST] Recomputing hard-example sample weights …")
    _, val_tf = get_transforms(cfg_obj.IMG_SIZE)
    ds     = FaceDataset(samples, val_tf)
    loader = DataLoader(
        ds, batch_size=cfg_obj.BATCH_SIZE * 2, shuffle=False,
        num_workers=cfg_obj.NUM_WORKERS, pin_memory=(cfg_obj.DEVICE.type == "cuda"),
        persistent_workers=(cfg_obj.NUM_WORKERS > 0)
    )

    model.eval()
    losses_by_idx = np.zeros(len(samples), dtype=np.float32)

    with torch.amp.autocast("cuda", enabled=(cfg_obj.DEVICE.type == "cuda")):
        for imgs, labels, idxs in loader:
            imgs   = imgs.to(cfg_obj.DEVICE, non_blocking=True)
            labels = labels.to(cfg_obj.DEVICE, non_blocking=True)
            logits = model(imgs)
            sl     = per_sample_loss(logits, labels)
            for i, idx in enumerate(idxs.numpy()):
                losses_by_idx[idx] = sl[i]

    class_labels = np.array([l for _, l in samples])
    class_counts = np.bincount(class_labels, minlength=2).clip(1)
    base_w       = 1.0 / class_counts[class_labels]
    hard_w       = losses_by_idx / (losses_by_idx.max() + 1e-8)
    combined     = base_w * (1.0 + hard_w)
    return combined.tolist()

# ───────────────────────────── CHECKPOINT I/O ─────────────────────────────────
def save_checkpoint(path, epoch, model, ema, optimizer, scheduler, stopper, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "ema_state": ema.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "stopper_state": stopper.state_dict(),
        "history": history,
        "rng_state_torch": torch.get_rng_state(),
        "rng_state_numpy": np.random.get_state(),
    }, path)


def load_checkpoint(path, model, ema, optimizer, scheduler, stopper, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    ema.load_state_dict(ckpt["ema_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    stopper.load_state_dict(ckpt["stopper_state"])
    try:
        torch.set_rng_state(ckpt["rng_state_torch"].cpu())
        np.random.set_state(ckpt["rng_state_numpy"])
    except Exception:
        logger.warning("Could not fully restore RNG state.")
    logger.info(f"Resumed from checkpoint at epoch {ckpt['epoch']}.")
    return ckpt["epoch"], ckpt["history"]

# ───────────────────────────── TRAINING LOOP ─────────────────────────────────
def train(cfg_obj: Config, real_dir: Path, fake_dir: Path, resume: bool):

    paths, labels, groups = collect_grouped_paths(real_dir, fake_dir)
    train_s, val_s, test_s = group_split(paths, labels, groups, cfg_obj.VAL_SPLIT, cfg_obj.TEST_SPLIT, SEED)

    train_tf, val_tf = get_transforms(cfg_obj.IMG_SIZE)

    train_ds = FaceDataset(train_s, train_tf)
    val_ds   = FaceDataset(val_s,   val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=cfg_obj.BATCH_SIZE, sampler=make_sampler(train_s),
        num_workers=cfg_obj.NUM_WORKERS, pin_memory=(cfg_obj.DEVICE.type == "cuda"),
        persistent_workers=(cfg_obj.NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg_obj.BATCH_SIZE, shuffle=False,
        num_workers=cfg_obj.NUM_WORKERS, pin_memory=(cfg_obj.DEVICE.type == "cuda"),
        persistent_workers=(cfg_obj.NUM_WORKERS > 0),
    )

    model  = DenseNet121AntiSpoof(dropout=cfg_obj.DROPOUT, pretrained=True).to(cfg_obj.DEVICE)
    ema    = EMA(model, decay=cfg_obj.EMA_DECAY)

    n_real = sum(l for _, l in train_s)
    n_fake = len(train_s) - n_real
    pos_w  = n_fake / max(n_real, 1)

    criterion = FocalBCEWithLogits(pos_weight=pos_w, gamma=cfg_obj.FOCAL_GAMMA, smooth=cfg_obj.LABEL_SMOOTH).to(cfg_obj.DEVICE)

    # Differential LR: backbone params (frozen initially, unfrozen later)
    # get a low LR; the fresh head gets a higher LR. Params with
    # requires_grad=False simply produce no gradient, so AdamW silently
    # skips them until unfreeze_backbone() is called — no need to rebuild
    # the optimizer mid-training.
    optimizer = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": cfg_obj.LR_BACKBONE, "name": "backbone"},
        {"params": model.head.parameters(),     "lr": cfg_obj.LR_HEAD,     "name": "head"},
    ], weight_decay=cfg_obj.WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg_obj.LR_T0, T_mult=1, eta_min=1e-7)
    stopper   = EarlyStopping(cfg_obj.PATIENCE, cfg_obj.MODEL_SAVE)

    scaler = torch.amp.GradScaler("cuda", enabled=(cfg_obj.DEVICE.type == "cuda"))

    start_epoch = 1
    history     = []
    if resume and cfg_obj.CKPT_LAST.exists():
        last_epoch, history = load_checkpoint(cfg_obj.CKPT_LAST, model, ema, optimizer, scheduler, stopper, cfg_obj.DEVICE)
        start_epoch = last_epoch + 1
        if start_epoch > cfg_obj.FREEZE_EPOCHS:
            model.unfreeze_backbone()   # keep unfreeze state consistent across resume

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params : {n_total:,} total | {n_trainable:,} trainable at start")
    logger.info(f"Device       : {cfg_obj.DEVICE}")
    logger.info(f"Freeze phase : epochs 1-{cfg_obj.FREEZE_EPOCHS} (head only), then backbone unfreezes")
    logger.info(f"LR schedule  : warm restarts every {cfg_obj.LR_T0} epochs (backbone={cfg_obj.LR_BACKBONE:.1e}, head={cfg_obj.LR_HEAD:.1e})")
    logger.info(f"Starting at epoch {start_epoch}")

    for epoch in range(start_epoch, cfg_obj.EPOCHS + 1):

        if epoch == cfg_obj.FREEZE_EPOCHS + 1:
            model.unfreeze_backbone()
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"  [UNFREEZE] Backbone released — {n_trainable:,} trainable params now.")

        if epoch > 1 and (epoch - 1) % cfg_obj.BOOST_EVERY == 0:
            hard_weights = compute_sample_weights_hard(model, train_s, cfg_obj)
            train_loader = DataLoader(
                train_ds, batch_size=cfg_obj.BATCH_SIZE,
                sampler=make_sampler(train_s, sample_weights=hard_weights),
                num_workers=cfg_obj.NUM_WORKERS, pin_memory=(cfg_obj.DEVICE.type == "cuda"),
                persistent_workers=(cfg_obj.NUM_WORKERS > 0),
            )

        model.train()
        run_loss, correct, total = 0.0, 0, 0

        for imgs, lbls, _ in tqdm(train_loader, desc=f"Epoch {epoch:03d} train", leave=False):
            imgs = imgs.to(cfg_obj.DEVICE, non_blocking=True)
            lbls = lbls.to(cfg_obj.DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(cfg_obj.DEVICE.type == "cuda")):
                imgs1, lbls1, imgs2, lbls2, lam = mixup_data(imgs, lbls, alpha=cfg_obj.MIXUP_ALPHA)
                mixed_imgs = lam * imgs1 + (1.0 - lam) * imgs2
                logits = model(mixed_imgs)
                loss = lam * criterion(logits, lbls1) + (1.0 - lam) * criterion(logits, lbls2)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)

            scores = torch.sigmoid(logits.detach())
            preds  = (scores >= cfg_obj.THRESHOLD).float()

            corr1 = (preds == (lbls1 >= 0.5).float()).sum().item()
            corr2 = (preds == (lbls2 >= 0.5).float()).sum().item()
            correct  += lam * corr1 + (1.0 - lam) * corr2
            total    += lbls.size(0)
            run_loss += loss.item()

        scheduler.step()
        avg_loss  = run_loss / len(train_loader)
        train_acc = correct / total

        ema_model = ema.get_model()
        ema_model.eval()
        all_scores, all_labels = [], []

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(cfg_obj.DEVICE.type == "cuda")):
                for imgs, lbls, _ in tqdm(val_loader, desc=f"Epoch {epoch:03d} val  ", leave=False):
                    imgs   = imgs.to(cfg_obj.DEVICE, non_blocking=True)
                    logits = ema_model(imgs)
                    sc     = torch.sigmoid(logits).cpu().numpy()
                    all_scores.extend(sc.tolist())
                    all_labels.extend(lbls.numpy().tolist())

        m = compute_metrics(all_labels, all_scores, cfg_obj.THRESHOLD)
        lr_backbone_now = optimizer.param_groups[0]["lr"]
        lr_head_now     = optimizer.param_groups[1]["lr"]

        logger.info(
            f"Epoch {epoch:03d}/{cfg_obj.EPOCHS} | Loss: {avg_loss:.4f} | TrainAcc: {train_acc:.4f} | "
            f"ValAcc: {m['accuracy']:.4f} | APCER: {m['APCER']:.4f} | BPCER: {m['BPCER']:.4f} | "
            f"ACER: {m['ACER']:.4f} | AUC: {m['AUC']:.4f} | LR(bb): {lr_backbone_now:.2e} | LR(head): {lr_head_now:.2e}"
        )

        history.append({"epoch": epoch, "loss": avg_loss, "train_acc": train_acc, **m,
                         "lr_backbone": lr_backbone_now, "lr_head": lr_head_now})
        save_checkpoint(cfg_obj.CKPT_LAST, epoch, model, ema, optimizer, scheduler, stopper, history)

        if stopper.step(m["ACER"], ema_model):
            logger.info(f"Early stopping triggered at epoch {epoch}.")
            break

    hist_path = cfg_obj.TRAIN_DIR / "training_history_v4.csv"
    with open(hist_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    logger.info(f"History saved → {hist_path}")

    if len(test_s) > 0 and cfg_obj.MODEL_SAVE.exists():
        ema_model.load_state_dict(torch.load(cfg_obj.MODEL_SAVE, map_location=cfg_obj.DEVICE))
        logger.info("Reloaded best EMA checkpoint for holdout testing...")

        test_loader = DataLoader(
            FaceDataset(test_s, val_tf), batch_size=cfg_obj.BATCH_SIZE, shuffle=False,
            num_workers=cfg_obj.NUM_WORKERS, pin_memory=(cfg_obj.DEVICE.type == "cuda"),
            persistent_workers=(cfg_obj.NUM_WORKERS > 0)
        )
        ema_model.eval()
        test_scores, test_labels = [], []

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(cfg_obj.DEVICE.type == "cuda")):
                for imgs, lbls, _ in tqdm(test_loader, desc="Testing set evaluation", leave=False):
                    imgs   = imgs.to(cfg_obj.DEVICE, non_blocking=True)
                    logits = ema_model(imgs)
                    sc     = torch.sigmoid(logits).cpu().numpy()
                    test_scores.extend(sc.tolist())
                    test_labels.extend(lbls.numpy().tolist())

        test_metrics = compute_metrics(test_labels, test_scores, cfg_obj.THRESHOLD)
        logger.info(
            f"[TEST HIGHLIGHTS] Acc: {test_metrics['accuracy']:.4f} | APCER: {test_metrics['APCER']:.4f} | "
            f"BPCER: {test_metrics['BPCER']:.4f} | ACER: {test_metrics['ACER']:.4f} | AUC: {test_metrics['AUC']:.4f}"
        )

        test_hist_path = cfg_obj.TRAIN_DIR / "test_metrics_v4.csv"
        with open(test_hist_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=test_metrics.keys())
            writer.writeheader()
            writer.writerow(test_metrics)
        logger.info(f"Test metrics saved successfully → {test_hist_path}")

    return ema_model

# ───────────────────────────── ENTRY POINT ───────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    cfg  = Config()
    cfg.EPOCHS        = args.epochs
    cfg.BATCH_SIZE    = args.batch_size
    cfg.LR_BACKBONE   = args.lr_backbone
    cfg.LR_HEAD       = args.lr_head
    cfg.FREEZE_EPOCHS = args.freeze_epochs
    cfg.PATIENCE      = args.patience
    cfg.BOOST_EVERY   = args.boost_every

    train(cfg, Path(args.real_dir), Path(args.fake_dir), resume=args.resume)