import os
import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, confusion_matrix
import json

# ─────────────────────────────────────────────
# CDCN Model Definition — must match cdcn_train.py EXACTLY
# ─────────────────────────────────────────────

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
        stride      = self.conv.stride[0]
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


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class TestDataset(Dataset):
    EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

    def __init__(self, root: str, img_size: int = 128):
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        self.samples = []   # (path, label, true_class_name)

        for label, cls in [(1, 'real'), (0, 'fake')]:
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                raise FileNotFoundError(f"Expected folder: {cls_dir}")
            for fname in sorted(os.listdir(cls_dir)):
                if os.path.splitext(fname)[1].lower() in self.EXTS:
                    self.samples.append((os.path.join(cls_dir, fname), label, cls))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, cls = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), torch.tensor(label, dtype=torch.float32), idx


# ─────────────────────────────────────────────
# ISO 30107-3 metrics
# ─────────────────────────────────────────────

def compute_iso_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    y_true = np.array(y_true)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    apcer = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    bpcer = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    acer  = (apcer + bpcer) / 2.0
    acc   = (tp + tn) / len(y_true) * 100

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
    model = CDCN(base_ch=64, theta=args.theta, dropout=0.3).to(device)

    ckpt  = torch.load(args.model, map_location=device)
    state = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"[Model]  Loaded  ← {args.model}")

    # ── Dataset ──
    dataset = TestDataset(args.test_dir, img_size=args.img_size)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=False, num_workers=args.workers,
                         pin_memory=(device.type == 'cuda'))

    real_count = sum(1 for _, l, _ in dataset.samples if l == 1)
    fake_count = len(dataset) - real_count
    print(f"[Data]   {len(dataset)} images  |  real={real_count}  fake={fake_count}")
    print(f"         Source: {args.test_dir}\n")

    # ── Inference ──
    all_probs, all_labels, all_indices = [], [], []

    with torch.no_grad():
        for imgs, labels, indices in loader:
            imgs       = imgs.to(device)
            depth_pred = model(imgs)
            # score = mean of depth map (same as training inference)
            probs      = depth_pred.mean(dim=[1, 2, 3]).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_indices.extend(indices.numpy().tolist())

    # ── Aggregate metrics ──
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

    # ── Per-image CSV ──
    csv_path = args.save_results.replace('.json', '_per_image.csv') \
               if args.save_results else '/kaggle/working/predictions.csv'

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'filename', 'true_label', 'true_class',
            'predicted_class', 'score', 'correct'
        ])
        for idx, prob in zip(all_indices, all_probs):
            path, label, true_cls = dataset.samples[idx]
            pred_cls   = 'real' if prob >= args.threshold else 'fake'
            correct    = (pred_cls == true_cls)
            writer.writerow([
                os.path.basename(path),
                int(label),
                true_cls,
                pred_cls,
                f"{prob:.6f}",
                correct,
            ])

    print(f"\n[Saved]  Per-image predictions → {csv_path}")

    # ── Summary JSON ──
    if args.save_results:
        with open(args.save_results, 'w') as f:
            json.dump({**m, 'threshold': args.threshold,
                       'model': args.model, 'test_dir': args.test_dir}, f, indent=2)
        print(f"[Saved]  Summary metrics     → {args.save_results}")

    return m


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CDCN Face Anti-Spoofing – Test Evaluation')

    parser.add_argument('--model',        required=True,
                        help='Path to .pt model file')
    parser.add_argument('--test_dir',     required=True,
                        help='Root of test dataset (must contain real/ and fake/ subdirs)')
    parser.add_argument('--img_size',     type=int,   default=128)
    parser.add_argument('--batch_size',   type=int,   default=32)
    parser.add_argument('--workers',      type=int,   default=2)
    parser.add_argument('--theta',        type=float, default=0.7)
    parser.add_argument('--threshold',    type=float, default=0.5)
    parser.add_argument('--save_results', default='/kaggle/working/results.json',
                        help='Path to save JSON summary (CSV is saved alongside it)')

    args = parser.parse_args()
    evaluate(args)