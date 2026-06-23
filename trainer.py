"""
=============================================================================
CDCN Anti-Spoofing -- Training & Validation Script (VS Code / Local)
=============================================================================
Run preprocessing first:
    python preprocess_faces.py --raw_dir raw_dataset --out_dir dataset

Then train:
    python cdcn_train.py
    python cdcn_train.py --real_dir dataset/real --fake_dir dataset/fake
    python cdcn_train.py --epochs 80 --patience 15

Expected dataset layout:
    dataset/
        real/    <- face-cropped images (from preprocess_faces.py)
        fake/    <- face-cropped images (from preprocess_faces.py)

Key design decisions:
  * Strict train/val split BEFORE any augmentation or transform -- no leakage
  * Stratified split preserves real/fake ratio in both folds
  * WeightedRandomSampler balances imbalanced classes during training
  * Early stopping on ACER -- best checkpoint reloaded into model after training
  * APCER / BPCER / ACER / AUC reported every epoch (ISO 30107-3 metrics)
  * Dropout in depth-head for regularisation
  * Cosine LR schedule + gradient clipping
=============================================================================
"""

# ─────────────────────── 1. IMPORTS ──────────────────────────────────────────
import csv
import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# ─────────────────────── 2. REPRODUCIBILITY ──────────────────────────────────
SEED = 42

def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

seed_everything()

# ─────────────────────── 3. LOGGING ──────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/train.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ─────────────────────── 4. CONFIG ───────────────────────────────────────────
class Config:
    IMG_SIZE     = 256
    BATCH_SIZE   = 16
    EPOCHS       = 60
    LR           = 1e-3
    WEIGHT_DECAY = 5e-4
    VAL_SPLIT    = 0.2       # 80/20 stratified split
    THRESHOLD    = 0.5       # depth-score cutoff: >= threshold -> LIVE
    PATIENCE     = 10        # early stopping patience in epochs
    MODEL_SAVE = Path("/kaggle/working/models/cdcn_best.pt")
    DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────── 5. ARGUMENT PARSING ─────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="CDCN Anti-Spoofing Trainer")
    parser.add_argument("--real_dir",   type=str,   default="dataset/real",
                        help="Path to real (live) face images folder")
    parser.add_argument("--fake_dir",   type=str,   default="dataset/fake",
                        help="Path to fake (spoof) face images folder")
    parser.add_argument("--epochs",     type=int,   default=Config.EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=Config.BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=Config.LR)
    parser.add_argument("--patience",   type=int,   default=Config.PATIENCE)
    return parser.parse_args()

# ─────────────────────── 6. DATA COLLECTION ──────────────────────────────────
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def collect_paths(real_dir: Path, fake_dir: Path):
    """
    Walks real/ and fake/ and returns two SEPARATE lists of (path, label) --
    train and val. The split happens HERE on raw paths before any Dataset or
    transform is created, so there is zero leakage between folds.
    """
    data = []
    for p in real_dir.rglob("*"):
        if p.suffix.lower() in EXTS:
            data.append((str(p), 1))    # 1 = live

    for p in fake_dir.rglob("*"):
        if p.suffix.lower() in EXTS:
            data.append((str(p), 0))    # 0 = spoof

    if not data:
        raise FileNotFoundError(
            f"No images found in {real_dir} or {fake_dir}.\n"
            "Run preprocess_faces.py first, or check your folder paths."
        )

    labels = [lbl for _, lbl in data]
    n_real = int(sum(labels))
    n_fake = len(labels) - n_real
    logger.info(f"Dataset total  -> real: {n_real} | fake: {n_fake}")

    if n_real == 0 or n_fake == 0:
        raise ValueError(
            f"Need both real and fake images (got real={n_real}, fake={n_fake})."
        )

    train_s, val_s = train_test_split(
        data,
        test_size=Config.VAL_SPLIT,
        random_state=SEED,
        stratify=labels,
    )

    tr_labels = [l for _, l in train_s]
    va_labels = [l for _, l in val_s]
    logger.info(f"Train split    -> real: {sum(tr_labels)} | fake: {len(tr_labels) - sum(tr_labels)}")
    logger.info(f"Val   split    -> real: {sum(va_labels)} | fake: {len(va_labels) - sum(va_labels)}")

    return train_s, val_s, {"real": n_real, "fake": n_fake}


def make_sampler(samples):
    """WeightedRandomSampler for class balance -- train loader only."""
    labels        = [lbl for _, lbl in samples]
    class_counts  = np.bincount(labels, minlength=2)
    class_counts  = np.where(class_counts == 0, 1, class_counts)
    class_weights = 1.0 / class_counts
    sample_weights = [float(class_weights[lbl]) for lbl in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

# ─────────────────────── 7. TRANSFORMS ───────────────────────────────────────
TRAIN_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomRotation(degrees=10),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ─────────────────────── 8. DATASET ──────────────────────────────────────────
class FaceDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples   = samples
        self.transform = transform
        self.map_size  = Config.IMG_SIZE // 4

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            logger.warning(f"Corrupt/missing: {path} -- using blank image.")
            img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        img      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor   = self.transform(img)
        depth_gt = torch.full((1, self.map_size, self.map_size), float(label))
        label_t  = torch.tensor(label, dtype=torch.float32)
        return tensor, depth_gt, label_t

# ─────────────────────── 9. MODEL ────────────────────────────────────────────
class CentralDifferenceConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, theta=0.7):
        super().__init__()
        self.theta = theta
        self.conv  = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)

    def forward(self, x):
        out_vanilla = self.conv(x)
        if self.theta == 0:
            return out_vanilla
        kernel      = self.conv.weight
        kernel_diff = kernel.sum(dim=[2, 3], keepdim=True)
        stride      = self.conv.stride[0]   # plain int -- required by F.conv2d
        out_cd      = F.conv2d(x, kernel_diff, stride=stride, padding=0)
        return out_vanilla - self.theta * out_cd


class ConvBNPReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, theta=0.7):
        super().__init__(
            CentralDifferenceConv2d(in_ch, out_ch, k, s, p, theta),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
        )


class CDCN(nn.Module):
    def __init__(self, base_ch=64, theta=0.7, dropout=0.3):
        super().__init__()

        self.stem = nn.Sequential(
            ConvBNPReLU(3,           base_ch,     k=3, s=1, p=1, theta=theta),
            ConvBNPReLU(base_ch,     base_ch * 2, k=3, s=2, p=1, theta=theta),
            ConvBNPReLU(base_ch * 2, base_ch * 2, k=3, s=1, p=1, theta=theta),
        )
        self.layer1 = nn.Sequential(
            ConvBNPReLU(base_ch * 2, base_ch * 2, theta=theta),
            ConvBNPReLU(base_ch * 2, base_ch * 2, theta=theta),
            ConvBNPReLU(base_ch * 2, base_ch * 2, theta=theta),
            nn.MaxPool2d(3, 2, 1),
        )
        self.layer2 = nn.Sequential(
            ConvBNPReLU(base_ch * 2, base_ch * 4, theta=theta),
            ConvBNPReLU(base_ch * 4, base_ch * 4, theta=theta),
            ConvBNPReLU(base_ch * 4, base_ch * 4, theta=theta),
            nn.MaxPool2d(3, 2, 1),
        )
        self.layer3 = nn.Sequential(
            ConvBNPReLU(base_ch * 4, base_ch * 8, theta=theta),
            ConvBNPReLU(base_ch * 8, base_ch * 8, theta=theta),
            ConvBNPReLU(base_ch * 8, base_ch * 8, theta=theta),
        )

        self.conv_s1   = nn.Conv2d(base_ch * 2, 1, 1)
        self.conv_s2   = nn.Conv2d(base_ch * 4, 1, 1)
        self.conv_s3   = nn.Conv2d(base_ch * 8, 1, 1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.depth_head = nn.Sequential(
            nn.Dropout2d(p=dropout),
            nn.Conv2d(3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.layer1(s0)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        m1 = self.conv_s1(s1)
        m2 = self.upsample2(self.conv_s2(s2))
        m3 = self.upsample2(self.conv_s3(s3))
        return self.depth_head(torch.cat([m1, m2, m3], dim=1))

# ─────────────────────── 10. LOSS ────────────────────────────────────────────
class CDCNLoss(nn.Module):
    def __init__(self, lam: float = 0.5, pos_weight: float = 1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lam = lam
        self.register_buffer("pw", torch.tensor([pos_weight]))

    def forward(self, depth_pred, depth_gt, label):
        loss_depth = self.mse(depth_pred, depth_gt)
        score      = depth_pred.mean(dim=[1, 2, 3]).clamp(1e-6, 1 - 1e-6)
        weight     = torch.where(label == 1,
                                 self.pw.expand_as(label),
                                 torch.ones_like(label))
        loss_cls   = F.binary_cross_entropy(score, label, weight=weight)
        return loss_depth + self.lam * loss_cls

# ─────────────────────── 11. METRICS ─────────────────────────────────────────
def compute_metrics(labels: list, scores: list, threshold: float) -> dict:
    preds    = np.array([1 if s >= threshold else 0 for s in scores])
    labels_a = np.array(labels)
    accuracy = float((preds == labels_a).mean())
    real_mask = labels_a == 1
    fake_mask = ~real_mask
    bpcer = float((preds[real_mask] == 0).mean()) if real_mask.any() else 0.0
    apcer = float((preds[fake_mask] == 1).mean()) if fake_mask.any() else 0.0
    acer  = (apcer + bpcer) / 2.0
    try:
        auc = float(roc_auc_score(labels, scores))
    except ValueError:
     return {"accuracy": accuracy, "APCER": apcer, "BPCER": bpcer, "ACER": acer, "AUC": auc}

# ─────────────────────── 12. EARLY STOPPING ──────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int, model_path: Path):
        self.patience   = patience
        self.model_path = model_path
        self.best_acer  = float("inf")
        self.counter    = 0
        self.triggered  = False

    def step(self, acer: float, model: nn.Module) -> bool:
        if acer < self.best_acer:
            self.best_acer = acer
            self.counter   = 0
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.model_path)
            logger.info(f"  [SAVED] Best model (ACER={acer:.4f}) -> {self.model_path}")
        else:
            self.counter += 1
            logger.info(f"  No improvement -- patience {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered

# ─────────────────────── 13. TRAINING LOOP ───────────────────────────────────
def train(cfg: Config, real_dir: Path, fake_dir: Path) -> CDCN:

    # Step 1: split paths
    train_samples, val_samples, counts = collect_paths(real_dir, fake_dir)

    # Step 2: datasets
    train_ds = FaceDataset(train_samples, transform=TRAIN_TRANSFORM)
    val_ds   = FaceDataset(val_samples,   transform=VAL_TRANSFORM)

    # Step 3: loaders
    # num_workers=0 on Windows avoids multiprocessing pickle errors
    nw = 0 if sys.platform == "win32" else 4

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        sampler=make_sampler(train_samples),
        num_workers=0,
        pin_memory=(cfg.DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(cfg.DEVICE.type == "cuda"),
    )

    # Step 4: model / loss / optimiser
    model      = CDCN(dropout=0.3).to(cfg.DEVICE)
    pos_weight = counts["fake"] / max(counts["real"], 1)
    criterion  = CDCNLoss(lam=0.5, pos_weight=pos_weight).to(cfg.DEVICE)
    optimizer  = torch.optim.Adam(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS, eta_min=1e-6
    )
    stopper    = EarlyStopping(patience=cfg.PATIENCE, model_path=cfg.MODEL_SAVE)

    logger.info(f"Model params : {sum(p.numel() for p in model.parameters()):,}")

    # Step 5: epoch loop
    history   = []
    epoch_bar = tqdm(range(1, cfg.EPOCHS + 1), desc="Training",
                     unit="epoch", file=sys.stdout)
    for imgs, depth_gt, labels in train_loader:
        print("Batch loaded")
        print(imgs.shape)
        break
    for epoch in epoch_bar:

        # Train
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for imgs, depth_gt, labels in train_loader:
            imgs     = imgs.to(cfg.DEVICE)
            depth_gt = depth_gt.to(cfg.DEVICE)
            labels   = labels.to(cfg.DEVICE)
            optimizer.zero_grad()
            depth_pred = model(imgs)
            scores = depth_pred.mean(dim=[1,2,3])
            preds = (scores >= cfg.THRESHOLD).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            loss = criterion(depth_pred, depth_gt, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        train_acc = train_correct / train_total
        # Validation
        model.eval()
        all_scores: list = []
        all_labels: list = []
        with torch.no_grad():
            for imgs, _, labels in val_loader:
                imgs       = imgs.to(cfg.DEVICE)
                depth_pred = model(imgs)
                scores     = depth_pred.mean(dim=[1, 2, 3]).cpu().numpy()
                all_scores.extend(scores.tolist())
                all_labels.extend(labels.numpy().tolist())

        m = compute_metrics(all_labels, all_scores, cfg.THRESHOLD)

        logger.info(
            f"Epoch {epoch:03d}/{cfg.EPOCHS} | "
            f"Loss: {avg_loss:.4f} | "
            f"TrainAcc: {train_acc:.4f} | "
            f"ValAcc: {m['accuracy']:.4f} | "
            f"APCER: {m['APCER']:.4f} | "
            f"BPCER: {m['BPCER']:.4f} | "
            f"ACER: {m['ACER']:.4f} | "
            f"AUC: {m['AUC']:.4f}")
        epoch_bar.set_postfix(
            loss=f"{avg_loss:.4f}",
            ACER=f"{m['ACER']:.4f}",
            AUC=f"{m['AUC']:.4f}",
        )

        history.append({"epoch": epoch, "loss": avg_loss, **m})

        if stopper.step(m["ACER"], model):
            logger.info(f"Early stopping triggered at epoch {epoch}.")
            break

    logger.info(f"Training complete. Best ACER: {stopper.best_acer:.4f}")

    # Step 6: reload best weights (not the final epoch's weights)
    if cfg.MODEL_SAVE.exists():
        model.load_state_dict(torch.load(cfg.MODEL_SAVE, map_location=cfg.DEVICE))
        logger.info(f"Best checkpoint reloaded (ACER={stopper.best_acer:.4f}).")

    # Step 7: save history CSV
    if history:
        hist_path = Path("/kaggle/working/models/training_history.csv")
        with open(hist_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        logger.info(f"History saved -> {hist_path}")

    return model

# ─────────────────────── 14. ENTRY POINT ─────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    cfg            = Config()
    cfg.EPOCHS     = args.epochs
    cfg.BATCH_SIZE = args.batch_size
    cfg.LR         = args.lr
    cfg.PATIENCE   = args.patience

    real_dir = Path(args.real_dir)
    fake_dir = Path(args.fake_dir)

    logger.info(f"Device     : {cfg.DEVICE}")
    if cfg.DEVICE.type == "cpu":
        logger.warning(
            "No GPU detected -- training will be slow. "
            "Consider enabling a GPU in your environment."
        )
    logger.info(
        f"Config     : IMG={cfg.IMG_SIZE} | BS={cfg.BATCH_SIZE} | "
        f"LR={cfg.LR} | Patience={cfg.PATIENCE} | Epochs={cfg.EPOCHS}"
    )
    logger.info(f"real_dir   : {real_dir.resolve()}")
    logger.info(f"fake_dir   : {fake_dir.resolve()}")

    train(cfg, real_dir, fake_dir)
