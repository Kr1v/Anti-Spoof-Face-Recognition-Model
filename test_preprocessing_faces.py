"""
=============================================================================
Face Crop Preprocessor — CelebA-Spoof Edition (Kaggle)
=============================================================================
Reads the CelebA-Spoof dataset layout:

    /kaggle/input/<dataset>/CelebA_Spoof/Data/
        train/
            <id_folder>/
                live/    ← real faces
                spoof/   ← fake faces
        test/
            <id_folder>/
                live/
                spoof/

Only the FIRST N numbered subfolders (sorted) are processed (default: 5).

Output layout (ready for cdcn_train.py):
    /kaggle/working/dataset/
        real/   ← face-cropped 256×256 images
        fake/   ← face-cropped 256×256 images

Usage (Kaggle notebook cell):
    !python preprocess_faces.py
    # or override defaults:
    !python preprocess_faces.py \
        --data_root /kaggle/input/<your-dataset-name>/CelebA_Spoof/Data \
        --split     test \
        --n_folders 5 \
        --out_dir   /kaggle/working/dataset \
        --margin    0.25

Each image's BB.txt companion file (if present) is used first for the face
box — skipping detection entirely, which is faster and perfectly accurate.
Falls back to MediaPipe → Haar if the BB file is missing or invalid.
=============================================================================
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

# ─────────────────────── LOGGING ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("/kaggle/working/preprocess.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ─────────────────────── CONFIG ───────────────────────────────────────────────
EXTS        = {".jpg", ".jpeg", ".png", ".bmp"}
OUTPUT_SIZE = 256
MARGIN      = 0.25
MIN_FACE_PX = 40

# ─────────────────────── BB.TXT READER ───────────────────────────────────────
def read_bb_file(img_path: Path):
    """
    CelebA-Spoof ships a <stem>_BB.txt next to each image.
    Format (first line): x y w h [confidence] [landmarks...]
    Returns (x, y, w, h) as ints, or None if file is absent / malformed.
    """
    bb_path = img_path.parent / (img_path.stem + "_BB.txt")
    if not bb_path.exists():
        return None
    try:
        with open(bb_path) as f:
            parts = f.readline().split()
        x, y, w, h = int(float(parts[0])), int(float(parts[1])), \
                     int(float(parts[2])), int(float(parts[3]))
        if w >= MIN_FACE_PX and h >= MIN_FACE_PX:
            return (x, y, w, h)
    except Exception:
        pass
    return None

# ─────────────────────── DETECTOR LOADER ─────────────────────────────────────
def load_mediapipe():
    try:
        import mediapipe as mp
        detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        logger.info("Fallback detector: MediaPipe")
        return ("mediapipe", detector)
    except ImportError:
        return None


def load_haar():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    logger.info("Fallback detector: Haar Cascade")
    return ("haar", cascade)


def get_detector():
    det = load_mediapipe()
    return det if det else load_haar()

# ─────────────────────── DETECTION ───────────────────────────────────────────
def detect_face_mediapipe(detector, img_rgb):
    results = detector.process(img_rgb)
    if not results.detections:
        return None
    h, w = img_rgb.shape[:2]
    best, best_area = None, 0
    for det in results.detections:
        bb  = det.location_data.relative_bounding_box
        fx, fy = int(bb.xmin * w), int(bb.ymin * h)
        fw, fh = int(bb.width * w), int(bb.height * h)
        area = fw * fh
        if area > best_area and fw >= MIN_FACE_PX and fh >= MIN_FACE_PX:
            best_area = area
            best = (fx, fy, fw, fh)
    return best


def detect_face_haar(cascade, img_bgr):
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5,
        minSize=(MIN_FACE_PX, MIN_FACE_PX)
    )
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def detect_face(detector_type, detector, img_bgr):
    if detector_type == "mediapipe":
        return detect_face_mediapipe(detector, cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return detect_face_haar(detector, img_bgr)

# ─────────────────────── CROP ────────────────────────────────────────────────
def crop_face(img_bgr, box, margin, output_size):
    x, y, w, h = box
    ih, iw = img_bgr.shape[:2]
    pad_x, pad_y = int(w * margin), int(h * margin)
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(iw, x + w + pad_x), min(ih, y + h + pad_y)
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)

# ─────────────────────── COLLECT IMAGE PATHS ─────────────────────────────────
def collect_images(data_root: Path, split: str, n_folders: int):
    """
    Returns list of (img_path, label) where label is 'real' or 'fake'.
    Scans the first n_folders sorted numeric subdirectories under
    data_root/<split>/.
    """
    split_dir = data_root / split
    if not split_dir.exists():
        logger.error(f"Split directory not found: {split_dir}")
        sys.exit(1)

    # Get all numeric subdirectories, sorted
    all_id_dirs = sorted(
        [d for d in split_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name)
    )

    if not all_id_dirs:
        logger.error(f"No numeric subdirectories found in {split_dir}")
        sys.exit(1)

    selected = all_id_dirs[:n_folders]
    logger.info(f"Using {len(selected)} ID folders: {[d.name for d in selected]}")

    entries = []   # (Path, 'real'|'fake')

    label_map = {
        "live":  "real",
        "spoof": "fake",
    }

    for id_dir in selected:
        for subfolder, label in label_map.items():
            sub_path = id_dir / subfolder
            if not sub_path.exists():
                logger.warning(f"  Missing: {sub_path}")
                continue
            imgs = [p for p in sub_path.rglob("*") if p.suffix.lower() in EXTS]
            entries.extend((p, label) for p in imgs)
            logger.info(f"  {id_dir.name}/{subfolder}: {len(imgs)} images → label={label}")

    logger.info(f"Total images collected: {len(entries)}")
    return entries

# ─────────────────────── MAIN PROCESSING ─────────────────────────────────────
def run(data_root: Path, split: str, n_folders: int,
        out_dir: Path, margin: float, output_size: int):

    # Build output dirs
    (out_dir / "real").mkdir(parents=True, exist_ok=True)
    (out_dir / "fake").mkdir(parents=True, exist_ok=True)

    # Load fallback detector (used only when BB.txt is absent)
    detector_type, detector = get_detector()

    # Collect all image paths
    entries = collect_images(data_root, split, n_folders)

    saved   = {"real": 0, "fake": 0}
    skipped = 0
    bb_hits = 0   # how many images used the BB.txt fast path

    skipped_path = Path("/kaggle/working/skipped.txt")
    with open(skipped_path, "w", encoding="utf-8") as skipped_log:
        skipped_log.write("REASON\tLABEL\tPATH\n")

        for idx, (img_path, label) in enumerate(entries, 1):

            img = cv2.imread(str(img_path))
            if img is None:
                skipped_log.write(f"UNREADABLE\t{label}\t{img_path}\n")
                skipped += 1
                continue

            # ── Fast path: use pre-computed BB.txt ──────────────────────────
            box = read_bb_file(img_path)
            if box:
                bb_hits += 1
            else:
                # ── Fallback: run detector ───────────────────────────────────
                box = detect_face(detector_type, detector, img)

            if box is None:
                skipped_log.write(f"NO_FACE\t{label}\t{img_path}\n")
                skipped += 1
                continue

            crop = crop_face(img, box, margin, output_size)
            if crop is None:
                skipped_log.write(f"BAD_CROP\t{label}\t{img_path}\n")
                skipped += 1
                continue

            # ── Unique output filename: <id_folder>_<original_stem>_crop.ext
            id_folder = img_path.parents[1].name   # e.g. "10001"
            out_name  = f"{id_folder}_{img_path.stem}_crop{img_path.suffix}"
            out_path  = out_dir / label / out_name
            cv2.imwrite(str(out_path), crop)
            saved[label] += 1

            if idx % 200 == 0 or idx == len(entries):
                logger.info(
                    f"[{idx}/{len(entries)}] "
                    f"real={saved['real']} fake={saved['fake']} "
                    f"skipped={skipped} bb_hits={bb_hits}"
                )

    logger.info("=" * 60)
    logger.info(f"Done.")
    logger.info(f"  real saved : {saved['real']}")
    logger.info(f"  fake saved : {saved['fake']}")
    logger.info(f"  skipped    : {skipped}")
    logger.info(f"  BB.txt used: {bb_hits} / {len(entries)}")
    logger.info(f"  Output     : {out_dir}")
    logger.info(f"  Skipped log: {skipped_path}")
    logger.info("Next step: python cdcn_train.py --data_dir /kaggle/working/dataset")

    if saved["real"] + saved["fake"] == 0:
        logger.error(
            "No images saved! Check:\n"
            "  1. --data_root points to the CelebA_Spoof/Data folder\n"
            "  2. --split matches an existing subfolder (train or test)\n"
            "  3. ID folders contain live/ and spoof/ subfolders\n"
        )

# ─────────────────────── ENTRY POINT ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess CelebA-Spoof for CDCN training."
    )
    parser.add_argument(
        "--data_root", type=str,
        default="/kaggle/input/celeba-spoof/CelebA_Spoof/Data",
        help="Path to CelebA_Spoof/Data/ (contains train/ and/or test/)"
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["train", "test"],
        help="Which split to process (default: test)"
    )
    parser.add_argument(
        "--n_folders", type=int, default=5,
        help="Number of first numeric ID folders to use (default: 5)"
    )
    parser.add_argument(
        "--out_dir", type=str, default="/kaggle/working/dataset",
        help="Output directory for cropped images"
    )
    parser.add_argument(
        "--margin", type=float, default=MARGIN,
        help="Fractional padding around face box (default 0.25)"
    )
    parser.add_argument(
        "--size", type=int, default=OUTPUT_SIZE,
        help="Output crop size in pixels (default 256)"
    )
    args = parser.parse_args()

    run(
        data_root  = Path(args.data_root),
        split      = args.split,
        n_folders  = args.n_folders,
        out_dir    = Path(args.out_dir),
        margin     = args.margin,
        output_size= args.size,
    )