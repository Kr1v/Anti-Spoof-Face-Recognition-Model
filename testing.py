import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, confusion_matrix
import json

# ─────────────────────────────────────────────
# CDCN Model Definition (must match training)
# ─────────────────────────────────────────────

class CDC(nn.Module):
    """Central Difference Convolution"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, theta=0.7):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)
        if abs(self.theta) < 1e-8:
            return out_normal
        kernel = self.conv.weight
        kernel_diff = kernel.sum(2).sum(2)
        kernel_diff = kernel_diff[:, :, None, None]
        out_diff = nn.functional.conv2d(x, kernel_diff, stride=self.conv.stride, padding=0)
        return out_normal - self.theta * out_diff


class CDCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, theta=0.7):
        super().__init__()
        self.cdc = CDC(in_ch, out_ch, theta=theta)
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.cdc(x)))


class CDCN(nn.Module):
    def __init__(self, theta=0.7):
        super().__init__()
        self.layer1 = CDCNBlock(3,   64,  theta)
        self.layer2 = CDCNBlock(64,  128, theta)
        self.pool   = nn.MaxPool2d(2, 2)
        self.layer3 = CDCNBlock(128, 256, theta)
        self.layer4 = CDCNBlock(256, 128, theta)
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(self.layer1(x))
        x = self.pool(self.layer2(x))
        x = self.pool(self.layer3(x))
        x = self.layer4(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)


# ─────────────────────────────────────────────
# Dataset  (test split – no augmentation)
# ─────────────────────────────────────────────

class TestDataset(Dataset):
    """
    Expects folder layout:
        test/
            real/   *.jpg / *.png
            fake/   *.jpg / *.png
    """
    EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

    def __init__(self, root: str, img_size: int = 128):
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        self.samples = []   # (path, label)  label: 1=real, 0=fake

        for label, cls in [(1, 'real'), (0, 'fake')]:
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                raise FileNotFoundError(f"Expected folder: {cls_dir}")
            for fname in sorted(os.listdir(cls_dir)):
                if os.path.splitext(fname)[1].lower() in self.EXTS:
                    self.samples.append((os.path.join(cls_dir, fname), label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), torch.tensor(label, dtype=torch.float32)


# ─────────────────────────────────────────────
# ISO 30107-3 metrics
# ─────────────────────────────────────────────

def compute_iso_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    y_true = np.array(y_true)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # APCER: attack samples classified as real  →  FP / (FP + TN)
    apcer = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    # BPCER: real samples classified as attack  →  FN / (FN + TP)
    bpcer = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    # ACER
    acer = (apcer + bpcer) / 2.0
    # Standard accuracy
    acc  = (tp + tn) / len(y_true) * 100

    try:
        auc = roc_auc_score(y_true, y_prob) * 100
    except ValueError:
        auc = float('nan')

    return dict(accuracy=acc, apcer=apcer*100, bpcer=bpcer*100,
                acer=acer*100, auc_roc=auc,
                tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))


# ─────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Device] {device}")

    # ── Load model ──
    model = CDCN(theta=args.theta).to(device)

    ckpt = torch.load(args.model, map_location=device)
    # Support raw state_dict or checkpoint dict
    state = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"[Model]  Loaded  ← {args.model}")

    # ── Dataset ──
    dataset = TestDataset(args.test_dir, img_size=args.img_size)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=False, num_workers=args.workers,
                         pin_memory=(device.type == 'cuda'))

    real_count = sum(1 for _, l in dataset.samples if l == 1)
    fake_count = len(dataset) - real_count
    print(f"[Data]   {len(dataset)} images  |  real={real_count}  fake={fake_count}")
    print(f"         Source: {args.test_dir}\n")

    # ── Inference ──
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs).squeeze(1)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    # ── Metrics ──
    m = compute_iso_metrics(all_labels, all_probs, threshold=args.threshold)

    bar = "─" * 42
    print(bar)
    print(f"  Test Accuracy  : {m['accuracy']:.2f}%")
    print(f"  AUC-ROC        : {m['auc_roc']:.2f}%")
    print(bar)
    print(f"  APCER          : {m['apcer']:.4f}%")
    print(f"  BPCER          : {m['bpcer']:.4f}%")
    print(f"  ACER           : {m['acer']:.4f}%")
    print(bar)
    print(f"  Confusion Matrix  (threshold={args.threshold})")
    print(f"    TP={m['tp']}  FP={m['fp']}")
    print(f"    FN={m['fn']}  TN={m['tn']}")
    print(bar)

    if args.save_results:
        out_path = args.save_results
        with open(out_path, 'w') as f:
            json.dump({**m, 'threshold': args.threshold,
                       'model': args.model, 'test_dir': args.test_dir}, f, indent=2)
        print(f"\n[Saved]  Results → {out_path}")

    return m


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CDCN Face Anti-Spoofing – Test Evaluation')

    parser.add_argument('--model',       required=True,
                        help='Path to .pt model file  (e.g. best_model.pt)')
    parser.add_argument('--test_dir',    required=True,
                        help='Root of test dataset  (must contain real/ and fake/ subdirs)')
    parser.add_argument('--img_size',    type=int,   default=128,
                        help='Image resize dimension (default: 128)')
    parser.add_argument('--batch_size',  type=int,   default=32,
                        help='Inference batch size (default: 32)')
    parser.add_argument('--workers',     type=int,   default=2,
                        help='DataLoader num_workers (default: 2)')
    parser.add_argument('--theta',       type=float, default=0.7,
                        help='CDC theta value – must match training (default: 0.7)')
    parser.add_argument('--threshold',   type=float, default=0.5,
                        help='Classification threshold (default: 0.5)')
    parser.add_argument('--save_results', default=None,
                        help='Optional path to save JSON results  (e.g. results.json)')

    args = parser.parse_args()
    evaluate(args)