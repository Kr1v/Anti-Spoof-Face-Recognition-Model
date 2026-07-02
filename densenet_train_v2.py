"""
=============================================================================
DenseNet121 Anti-Spoofing — v5.0
   (v4 backbone/training recipe + Group-Level K-Fold Cross-Validation
    with a Locked, Never-Touched Held-Out Test Set)
=============================================================================

What changed vs v4:

  1. TEST SET IS CARVED OFF FIRST, ONCE, AT THE IDENTITY-FOLDER LEVEL.
     Before any cross-validation happens, a fraction of identity groups
     (TEST_SPLIT) is set aside as `test_samples` and never appears in any
     fold's train or val set. It is only touched exactly once, at the very
     end, for final reporting. This guarantees your test number can't be
     inflated by having "seen" that data during model selection.

  2. THE REMAINING POOL IS K-FOLDED AT THE GROUP LEVEL.
     StratifiedGroupKFold (sklearn >= 1.1) keeps every image from a given
     identity folder in exactly one fold, and tries to balance real/fake
     ratio across folds. If your sklearn is older and doesn't have
     StratifiedGroupKFold, this falls back to plain GroupKFold (still
     leakage-safe, just without the class-balance guarantee per fold).

  3. EACH FOLD TRAINS ITS OWN MODEL, INDEPENDENTLY.
     Same DenseNet121 + freeze/unfreeze + warm-restart LR + EMA + focal
     loss + hard-example mining recipe as v4, just run K times with a
     shorter per-fold epoch budget (FOLD_EPOCHS) and its own early
     stopping (FOLD_PATIENCE), since you're now training K models instead
     of 1.

  4. FINAL TEST METRIC = ENSEMBLE OF THE K FOLD MODELS.
     After CV, all K best-fold checkpoints are loaded and their sigmoid
     outputs on the held-out test set are averaged. Averaging independently
     trained models is one of the cheapest, most reliable generalization
     boosts available — it cancels out fold-specific overfitting. Solo
     per-fold test numbers are also logged for comparison.

  5. DEPLOYMENT NOTE (see logged warning at the end): a 5-model DenseNet121
     ensemble is too heavy for CPU-only rural Android inference. Use the
     ensemble to get an honest generalization estimate and as a teacher for
     distillation into your single deployable model — see the "how to
     generalize better" notes at the bottom of this file.
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
from sklearn.model_selection import GroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_STRAT_GROUP_KFOLD = True
except ImportError:
    HAS_STRAT_GROUP_KFOLD = False

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
    IMG_SIZE       = 224

    BATCH_SIZE     = 32

    LR_BACKBONE    = 5e-5
    LR_HEAD        = 5e-4
    WEIGHT_DECAY   = 1e-2

    FREEZE_EPOCHS  = 5
    BOOST_EVERY    = 5
    LR_T0          = 5

    # ---- CV-specific ----
    N_FOLDS        = 5              # number of cross-validation folds
    TEST_SPLIT     = 0.15           # fraction of GROUPS locked away as test, before CV
    FOLD_EPOCHS    = 25             # epoch budget PER FOLD (you're training N_FOLDS models)
    FOLD_PATIENCE  = 8              # early-stopping patience PER FOLD

    THRESHOLD      = 0.5
    EMA_DECAY      = 0.999
    MIXUP_ALPHA    = 0.1
    LABEL_SMOOTH   = 0.05
    FOCAL_GAMMA    = 2.0
    DROPOUT        = 0.5

    NUM_WORKERS    = min(4, os.cpu_count() or 0)

    MODEL_DIR      = Path("/kaggle/working/models")
    TRAIN_DIR      = Path("/kaggle/working/training")
    DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────────────────────── LOGGING ───────────────────────────────────────
Config.TRAIN_DIR.mkdir(parents=True, exist_ok=True)
Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(Config.TRAIN_DIR / "train_v5_kfold.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

if not HAS_STRAT_GROUP_KFOLD:
    logger.warning(
        "StratifiedGroupKFold not available in this sklearn version — "
        "falling back to plain GroupKFold. Leakage safety is unaffected; "
        "you just lose the per-fold class-balance guarantee. "
        "`pip install -U scikit-learn` to get it."
    )

# ───────────────────────────── ARGUMENT PARSING ──────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--real_dir",      default="dataset/real")
    p.add_argument("--fake_dir",      default="dataset/fake")
    p.add_argument("--n_folds",       type=int,   default=Config.N_FOLDS)
    p.add_argument("--fold_epochs",   type=int,   default=Config.FOLD_EPOCHS)
    p.add_argument("--fold_patience", type=int,   default=Config.FOLD_PATIENCE)
    p.add_argument("--test_split",    type=float, default=Config.TEST_SPLIT)
    p.add_argument("--batch_size",    type=int,   default=Config.BATCH_SIZE)
    p.add_argument("--lr_backbone",   type=float, default=Config.LR_BACKBONE)
    p.add_argument("--lr_head",       type=float, default=Config.LR_HEAD)
    p.add_argument("--freeze_epochs", type=int,   default=Config.FREEZE_EPOCHS)
    p.add_argument("--boost_every",   type=int,   default=Config.BOOST_EVERY)
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

# ───────────────────────────── HOLDOUT TEST SPLIT ─────────────────────────────
def holdout_test_split(paths, labels, groups, test_frac, seed):
    """Carve out a held-out TEST set at the identity/group level, ONCE,
    before any cross-validation. This set is never passed to any fold's
    train() or val() and is only evaluated a single time at the very end,
    so it can't leak into model selection or hyperparameter choices."""
    paths  = np.array(paths)
    labels = np.array(labels)
    groups = np.array(groups)
    idx_all = np.arange(len(paths))

    pool_idx_parts, test_idx_parts = [], []

    for cls in (0, 1):
        cls_idx = idx_all[labels == cls]
        cls_groups = groups[cls_idx]
        unique_groups = sorted(set(cls_groups))
        folder_groups = [g for g in unique_groups if "__LOOSE__" not in g]
        loose_groups  = [g for g in unique_groups if "__LOOSE__" in g]

        local_rng = random.Random(seed)
        local_rng.shuffle(folder_groups)
        local_rng.shuffle(loose_groups)

        n_folders = len(folder_groups)
        n_test = max(1, int(round(n_folders * test_frac))) if n_folders >= 2 else 0
        test_f = folder_groups[:n_test]
        pool_f = folder_groups[n_test:]

        n_loose = len(loose_groups)
        n_test_l = int(round(n_loose * test_frac))
        if len(test_f) == 0 and n_loose >= 1:
            n_test_l = max(1, n_test_l)
        test_l = loose_groups[:n_test_l]
        pool_l = loose_groups[n_test_l:]

        cls_test_g = set(test_f + test_l)
        cls_pool_g = set(pool_f + pool_l)

        for idx in cls_idx:
            g = groups[idx]
            if g in cls_test_g:
                test_idx_parts.append(idx)
            elif g in cls_pool_g:
                pool_idx_parts.append(idx)

    pool_idx = np.array(pool_idx_parts, dtype=int)
    test_idx = np.array(test_idx_parts, dtype=int)

    pool_groups_arr = groups[pool_idx]
    test_groups_arr = groups[test_idx]

    if set(pool_groups_arr) & set(test_groups_arr):
        raise RuntimeError("Test holdout leakage detected — a group ended up in both pool and test.")

    pool_samples = [(paths[i], int(labels[i])) for i in pool_idx]
    test_samples = [(paths[i], int(labels[i])) for i in test_idx]

    logger.info(
        f"LOCKED TEST set (never touched during CV) → {len(test_samples)} images, "
        f"{len(set(test_groups_arr))} groups"
    )
    logger.info(
        f"CV pool (will be k-folded)                → {len(pool_samples)} images, "
        f"{len(set(pool_groups_arr))} groups"
    )
    logger.info("✓ Zero leakage between CV pool and locked test set confirmed.")

    return pool_samples, pool_groups_arr, test_samples, test_groups_arr

# ───────────────────────────── K-FOLD SPLIT ───────────────────────────────────
def make_kfold_splits(pool_samples, pool_groups_arr, n_folds, seed):
    """Group-level k-fold split of the CV pool. Every image from a given
    identity folder lands in exactly one fold — so within CV, a fold's
    validation set never shares an identity with that fold's training set."""
    labels_arr = np.array([l for _, l in pool_samples])
    groups_arr = np.array(pool_groups_arr)
    X_dummy = np.zeros(len(pool_samples))

    if HAS_STRAT_GROUP_KFOLD:
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    else:
        splitter = GroupKFold(n_splits=n_folds)

    folds = []
    for fold_i, (train_idx, val_idx) in enumerate(splitter.split(X_dummy, labels_arr, groups_arr)):
        train_groups = set(groups_arr[train_idx])
        val_groups   = set(groups_arr[val_idx])
        if train_groups & val_groups:
            raise RuntimeError(f"Fold {fold_i}: group leakage between train and val.")

        train_s = [pool_samples[i] for i in train_idx]
        val_s   = [pool_samples[i] for i in val_idx]
        folds.append((train_s, val_s))

        n_real_t = sum(l for _, l in train_s); n_fake_t = len(train_s) - n_real_t
        n_real_v = sum(l for _, l in val_s);   n_fake_v = len(val_s) - n_real_v
        logger.info(
            f"Fold {fold_i + 1}/{n_folds} → "
            f"train: {len(train_s)} (real {n_real_t}/fake {n_fake_t}, groups {len(train_groups)}) | "
            f"val: {len(val_s)} (real {n_real_v}/fake {n_fake_v}, groups {len(val_groups)})"
        )

    return folds

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
    ALWAYS_FROZEN = {"conv0", "norm0", "relu0", "pool0", "denseblock1", "transition1", "denseblock2", "transition2"}

    def __init__(self, dropout=0.5, pretrained=True):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = densenet121(weights=weights)
        self.features = backbone.features
        in_features = backbone.classifier.in_features

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

        self._freeze_backbone()

    def _freeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self):
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
        return logit.squeeze(1)

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
            logger.info(f"    [SAVED] Best ACER={acer:.4f} → {self.path}")
        else:
            self.counter += 1
            logger.info(f"    No improvement — patience {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered

# ───────────────────────────── HARD-EXAMPLE MINING ───────────────────────────
@torch.no_grad()
def compute_sample_weights_hard(model, samples, cfg_obj):
    logger.info("    [BOOST] Recomputing hard-example sample weights …")
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

# ───────────────────────────── PER-FOLD TRAINING ──────────────────────────────
def run_training_fold(cfg_obj: Config, train_s, val_s, fold_id: int):
    """Trains one fold's model end to end using the v4 recipe (freeze/
    unfreeze, warm-restart LR, EMA, focal loss, hard-example mining,
    ACER-based early stopping). Returns the best EMA model for this fold,
    its history, best val ACER, and the checkpoint path."""

    model_path = cfg_obj.MODEL_DIR / f"densenet121_fold{fold_id}_best.pt"

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

    model = DenseNet121AntiSpoof(dropout=cfg_obj.DROPOUT, pretrained=True).to(cfg_obj.DEVICE)
    ema   = EMA(model, decay=cfg_obj.EMA_DECAY)

    n_real = sum(l for _, l in train_s)
    n_fake = len(train_s) - n_real
    pos_w  = n_fake / max(n_real, 1)

    criterion = FocalBCEWithLogits(pos_weight=pos_w, gamma=cfg_obj.FOCAL_GAMMA, smooth=cfg_obj.LABEL_SMOOTH).to(cfg_obj.DEVICE)

    optimizer = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": cfg_obj.LR_BACKBONE, "name": "backbone"},
        {"params": model.head.parameters(),     "lr": cfg_obj.LR_HEAD,     "name": "head"},
    ], weight_decay=cfg_obj.WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg_obj.LR_T0, T_mult=1, eta_min=1e-7)
    stopper   = EarlyStopping(cfg_obj.FOLD_PATIENCE, model_path)
    scaler    = torch.amp.GradScaler("cuda", enabled=(cfg_obj.DEVICE.type == "cuda"))

    history = []
    ema_model = ema.get_model()

    for epoch in range(1, cfg_obj.FOLD_EPOCHS + 1):

        if epoch == cfg_obj.FREEZE_EPOCHS + 1:
            model.unfreeze_backbone()
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"    [UNFREEZE] Backbone released — {n_trainable:,} trainable params now.")

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

        for imgs, lbls, _ in tqdm(train_loader, desc=f"Fold {fold_id} | Epoch {epoch:03d} train", leave=False):
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
                for imgs, lbls, _ in tqdm(val_loader, desc=f"Fold {fold_id} | Epoch {epoch:03d} val  ", leave=False):
                    imgs   = imgs.to(cfg_obj.DEVICE, non_blocking=True)
                    logits = ema_model(imgs)
                    sc     = torch.sigmoid(logits).cpu().numpy()
                    all_scores.extend(sc.tolist())
                    all_labels.extend(lbls.numpy().tolist())

        m = compute_metrics(all_labels, all_scores, cfg_obj.THRESHOLD)
        lr_backbone_now = optimizer.param_groups[0]["lr"]
        lr_head_now     = optimizer.param_groups[1]["lr"]

        logger.info(
            f"  Fold {fold_id} | Epoch {epoch:03d}/{cfg_obj.FOLD_EPOCHS} | Loss: {avg_loss:.4f} | "
            f"TrainAcc: {train_acc:.4f} | ValAcc: {m['accuracy']:.4f} | APCER: {m['APCER']:.4f} | "
            f"BPCER: {m['BPCER']:.4f} | ACER: {m['ACER']:.4f} | AUC: {m['AUC']:.4f} | "
            f"LR(bb): {lr_backbone_now:.2e} | LR(head): {lr_head_now:.2e}"
        )

        history.append({"fold": fold_id, "epoch": epoch, "loss": avg_loss, "train_acc": train_acc, **m,
                         "lr_backbone": lr_backbone_now, "lr_head": lr_head_now})

        if stopper.step(m["ACER"], ema_model):
            logger.info(f"  Fold {fold_id}: early stopping triggered at epoch {epoch}.")
            break

    # reload best checkpoint for this fold before returning
    best_model = DenseNet121AntiSpoof(dropout=cfg_obj.DROPOUT, pretrained=False).to(cfg_obj.DEVICE)
    best_model.load_state_dict(torch.load(model_path, map_location=cfg_obj.DEVICE))
    best_model.eval()

    return best_model, history, stopper.best_acer, model_path

# ───────────────────────────── FINAL TEST: ENSEMBLE ────────────────────────────
def evaluate_ensemble_on_test(cfg_obj: Config, fold_model_paths, test_samples):
    """The ONLY point in the whole run where the locked test set is used.
    Averages the sigmoid outputs of all K fold models (a cheap, reliable
    generalization boost) and reports both the ensemble metric and each
    fold's solo metric for comparison."""
    if not test_samples:
        logger.warning("No test samples available — skipping final evaluation.")
        return None

    _, val_tf = get_transforms(cfg_obj.IMG_SIZE)
    test_loader = DataLoader(
        FaceDataset(test_samples, val_tf), batch_size=cfg_obj.BATCH_SIZE, shuffle=False,
        num_workers=cfg_obj.NUM_WORKERS, pin_memory=(cfg_obj.DEVICE.type == "cuda"),
    )

    all_fold_scores = []
    test_labels = None

    for path in fold_model_paths:
        model = DenseNet121AntiSpoof(dropout=cfg_obj.DROPOUT, pretrained=False).to(cfg_obj.DEVICE)
        model.load_state_dict(torch.load(path, map_location=cfg_obj.DEVICE))
        model.eval()

        scores, labels = [], []
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(cfg_obj.DEVICE.type == "cuda")):
                for imgs, lbls, _ in tqdm(test_loader, desc=f"Testing {path.name}", leave=False):
                    imgs   = imgs.to(cfg_obj.DEVICE, non_blocking=True)
                    logits = model(imgs)
                    sc     = torch.sigmoid(logits).cpu().numpy()
                    scores.extend(sc.tolist())
                    labels.extend(lbls.numpy().tolist())

        all_fold_scores.append(scores)
        test_labels = labels

    ensemble_scores = np.mean(np.array(all_fold_scores), axis=0)
    m = compute_metrics(test_labels, ensemble_scores, cfg_obj.THRESHOLD)

    logger.info(f"\n{'='*70}\nFINAL TEST — {len(fold_model_paths)}-model ensemble, LOCKED set, seen for the first time\n{'='*70}")
    logger.info(
        f"  Acc: {m['accuracy']:.4f} | APCER: {m['APCER']:.4f} | BPCER: {m['BPCER']:.4f} | "
        f"ACER: {m['ACER']:.4f} | AUC: {m['AUC']:.4f}"
    )

    for i, path in enumerate(fold_model_paths, start=1):
        m_i = compute_metrics(test_labels, all_fold_scores[i - 1], cfg_obj.THRESHOLD)
        logger.info(f"  Fold {i} solo on test → ACER: {m_i['ACER']:.4f} | AUC: {m_i['AUC']:.4f} | Acc: {m_i['accuracy']:.4f}")

    test_csv = cfg_obj.TRAIN_DIR / "final_test_ensemble_metrics.csv"
    with open(test_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=m.keys())
        writer.writeheader()
        writer.writerow(m)
    logger.info(f"Final test metrics saved → {test_csv}")

    logger.info(
        "\n[DEPLOYMENT NOTE] A 5-model DenseNet121 ensemble (~5x the compute/memory of one model) "
        "is not something you want running on a low-end rural Android device. Use this ensemble "
        "result as your honest generalization estimate, and as a teacher to distill into your "
        "single deployable model (see notes at bottom of this script)."
    )

    return m

# ───────────────────────────── CROSS-VALIDATION ORCHESTRATION ─────────────────
def cross_validate(cfg_obj: Config, real_dir: Path, fake_dir: Path):
    paths, labels, groups = collect_grouped_paths(real_dir, fake_dir)

    pool_samples, pool_groups, test_samples, test_groups = holdout_test_split(
        paths, labels, groups, cfg_obj.TEST_SPLIT, SEED
    )

    folds = make_kfold_splits(pool_samples, pool_groups, cfg_obj.N_FOLDS, SEED)

    fold_results, fold_model_paths = [], []

    for fold_id, (train_s, val_s) in enumerate(folds, start=1):
        logger.info(f"\n{'='*70}\nFOLD {fold_id}/{cfg_obj.N_FOLDS}\n{'='*70}")
        _, history, best_acer, model_path = run_training_fold(cfg_obj, train_s, val_s, fold_id)
        fold_results.append({"fold": fold_id, "best_val_acer": best_acer, **history[-1]})
        fold_model_paths.append(model_path)

    acers = [r["best_val_acer"] for r in fold_results]
    logger.info(f"\n{'='*70}\nCROSS-VALIDATION SUMMARY ({cfg_obj.N_FOLDS} folds)\n{'='*70}")
    logger.info(f"  Mean val ACER: {np.mean(acers):.4f} ± {np.std(acers):.4f}")
    logger.info(f"  Per-fold ACER: {[round(a, 4) for a in acers]}")

    cv_csv = cfg_obj.TRAIN_DIR / "cv_fold_results.csv"
    with open(cv_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fold_results[0].keys())
        writer.writeheader()
        writer.writerows(fold_results)
    logger.info(f"Per-fold summary saved → {cv_csv}")

    evaluate_ensemble_on_test(cfg_obj, fold_model_paths, test_samples)

    return fold_model_paths, fold_results

# ───────────────────────────── ENTRY POINT ───────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    cfg  = Config()
    cfg.N_FOLDS        = args.n_folds
    cfg.FOLD_EPOCHS    = args.fold_epochs
    cfg.FOLD_PATIENCE  = args.fold_patience
    cfg.TEST_SPLIT     = args.test_split
    cfg.BATCH_SIZE     = args.batch_size
    cfg.LR_BACKBONE    = args.lr_backbone
    cfg.LR_HEAD        = args.lr_head
    cfg.FREEZE_EPOCHS  = args.freeze_epochs
    cfg.BOOST_EVERY    = args.boost_every

    cross_validate(cfg, Path(args.real_dir), Path(args.fake_dir))

# =============================================================================
# GENERALIZATION NOTES — see the chat response for the full writeup.
# Highest-leverage items for your specific setup (small identity count,
# rural low-end Android deployment, high-res replay attacks):
#   1. Ensembling (done above) — biggest free win from CV.
#   2. More identity groups >> more images from the same identities.
#   3. Distill the ensemble into a lightweight student (MobileNetV3 /
#      your CDCN) for actual on-device deployment.
#   4. Replay-specific augmentation: synthesize moiré, screen-glare,
#      recapture JPEG artifacts, rather than generic color jitter alone.
#   5. Test-time augmentation (average over hflip / slight crops) at
#      inference — cheap, no retraining needed.
#   6. Consider an auxiliary task (e.g. predicting a pseudo-depth map or
#      rPPG signal) to discourage the model from keying on texture shortcuts.
# =============================================================================