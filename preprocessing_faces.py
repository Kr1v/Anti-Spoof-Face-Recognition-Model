"""
=============================================================================
Face Crop Preprocessor
=============================================================================
Run this ONCE before training. It reads raw images from your dataset,
detects the face region, crops + resizes to 256x256, and saves the result
into a new folder structure that cdcn_train.py can directly consume.

Input layout (your raw dataset):
    raw_dataset/
        real/   ← original full-frame images
        fake/   ← original full-frame images

Output layout (what training will use):
    dataset/
        real/   ← face-cropped 256x256 images
        fake/   ← face-cropped 256x256 images

Usage:
    python preprocess_faces.py
    python preprocess_faces.py --raw_dir raw_dataset --out_dir dataset
    python preprocess_faces.py --raw_dir raw_dataset --out_dir dataset --margin 0.3

Detection strategy (falls back automatically):
    1. MediaPipe Face Detection  ← most accurate, handles varied lighting
    2. OpenCV DNN (res10 model)  ← good backup, ships with opencv-contrib
    3. Haar Cascade              ← always available, used as last resort
    Images where NO face is detected are skipped and logged to skipped.txt
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
        logging.FileHandler("preprocess.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ─────────────────────── CONFIG ───────────────────────────────────────────────
EXTS       = {".jpg", ".jpeg", ".png", ".bmp"}
OUTPUT_SIZE = 256      # final crop size fed to CDCN
MARGIN      = 0.25     # fraction of face box to expand on each side
                       # 0.25 = 25% extra context around the face
MIN_FACE_PX = 40       # ignore detections smaller than this (pixels)

# ─────────────────────── DETECTOR LOADER ─────────────────────────────────────
def load_mediapipe():
    """Try loading MediaPipe face detector. Returns detector or None."""
    try:
        import mediapipe as mp
        mp_face = mp.solutions.face_detection
        detector = mp_face.FaceDetection(
            model_selection=1,          # 1 = full-range model (better for varied distances)
            min_detection_confidence=0.5,
        )
        logger.info("Detector: MediaPipe Face Detection (primary)")
        return ("mediapipe", detector)
    except ImportError:
        return None


def load_dnn():
    """
    Try loading OpenCV DNN face detector (res10_300x300_ssd).
    Model files must be in the same folder as this script, or on PATH.
    Download from:
      https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector
    """
    proto  = Path("deploy.prototxt")
    weights = Path("res10_300x300_ssd_iter_140000.caffemodel")
    if proto.exists() and weights.exists():
        net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
        logger.info("Detector: OpenCV DNN res10 (secondary)")
        return ("dnn", net)
    return None


def load_haar():
    """Haar cascade — always available as last resort."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    logger.info("Detector: Haar Cascade (fallback)")
    return ("haar", cascade)


def get_detector():
    """Returns (detector_type_str, detector_object) using best available."""
    det = load_mediapipe()
    if det:
        return det
    det = load_dnn()
    if det:
        return det
    return load_haar()


# ─────────────────────── DETECTION ───────────────────────────────────────────
def detect_face_mediapipe(detector, img_rgb: np.ndarray):
    """
    Returns (x, y, w, h) of the largest detected face, or None.
    img_rgb: uint8 RGB image.
    """
    results = detector.process(img_rgb)
    if not results.detections:
        return None

    h, w = img_rgb.shape[:2]
    best = None
    best_area = 0

    for det in results.detections:
        bb = det.location_data.relative_bounding_box
        x = int(bb.xmin * w)
        y = int(bb.ymin * h)
        fw = int(bb.width  * w)
        fh = int(bb.height * h)
        area = fw * fh
        if area > best_area and fw >= MIN_FACE_PX and fh >= MIN_FACE_PX:
            best_area = area
            best = (x, y, fw, fh)

    return best


def detect_face_dnn(net, img_bgr: np.ndarray):
    """Returns (x, y, w, h) of highest-confidence face, or None."""
    h, w = img_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img_bgr, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()

    best = None
    best_conf = 0.5   # confidence threshold

    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf > best_conf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            fw, fh = x2 - x1, y2 - y1
            if fw >= MIN_FACE_PX and fh >= MIN_FACE_PX:
                best_conf = conf
                best = (x1, y1, fw, fh)

    return best


def detect_face_haar(cascade, img_bgr: np.ndarray):
    """Returns (x, y, w, h) of largest face, or None."""
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(MIN_FACE_PX, MIN_FACE_PX),
    )
    if len(faces) == 0:
        return None
    # pick the largest face
    return max(faces, key=lambda f: f[2] * f[3])


def detect_face(detector_type: str, detector, img_bgr: np.ndarray):
    """Unified detection call. Returns (x, y, w, h) or None."""
    if detector_type == "mediapipe":
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return detect_face_mediapipe(detector, img_rgb)
    elif detector_type == "dnn":
        return detect_face_dnn(detector, img_bgr)
    else:
        return detect_face_haar(detector, img_bgr)


# ─────────────────────── CROP ────────────────────────────────────────────────
def crop_face(img_bgr: np.ndarray, box, margin: float, output_size: int):
    """
    Expands the detected box by `margin` fraction on all sides,
    clamps to image boundaries, crops, and resizes to output_size x output_size.

    margin=0.25 means add 25% of face width/height as padding on each side.
    This keeps ears, forehead, chin in frame — important texture cues for CDCN.
    """
    x, y, w, h = box
    ih, iw = img_bgr.shape[:2]

    pad_x = int(w * margin)
    pad_y = int(h * margin)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(iw, x + w + pad_x)
    y2 = min(ih, y + h + pad_y)

    crop = img_bgr[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    resized = cv2.resize(crop, (output_size, output_size),
                         interpolation=cv2.INTER_AREA)
    return resized


# ─────────────────────── MAIN PROCESSING ─────────────────────────────────────
def process_split(
    src_dir: Path,
    dst_dir: Path,
    detector_type: str,
    detector,
    margin: float,
    output_size: int,
    skipped_log,
):
    """Process all images in src_dir → dst_dir. Returns (saved, skipped) counts."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    image_paths = [p for p in src_dir.rglob("*") if p.suffix.lower() in EXTS]

    if not image_paths:
        logger.warning(f"No images found in {src_dir}")
        return 0, 0

    saved   = 0
    skipped = 0

    for idx, src_path in enumerate(image_paths, 1):
        img = cv2.imread(str(src_path))
        if img is None:
            logger.warning(f"  Cannot read: {src_path}")
            skipped_log.write(f"UNREADABLE\t{src_path}\n")
            skipped += 1
            continue

        box = detect_face(detector_type, detector, img)

        if box is None:
            logger.debug(f"  No face: {src_path.name}")
            skipped_log.write(f"NO_FACE\t{src_path}\n")
            skipped += 1
            continue

        crop = crop_face(img, box, margin, output_size)
        if crop is None:
            skipped_log.write(f"BAD_CROP\t{src_path}\n")
            skipped += 1
            continue

        # Preserve original filename; add _crop suffix to avoid collisions
        out_name = src_path.stem + "_crop" + src_path.suffix
        out_path = dst_dir / out_name
        cv2.imwrite(str(out_path), crop)
        saved += 1

        if idx % 100 == 0 or idx == len(image_paths):
            logger.info(f"  [{idx}/{len(image_paths)}] saved={saved} skipped={skipped}")

    return saved, skipped


def run(raw_dir: Path, out_dir: Path, margin: float, output_size: int):
    detector_type, detector = get_detector()

    splits = [("real", 1), ("fake", 0)]
    total_saved   = 0
    total_skipped = 0

    skipped_path = Path("skipped.txt")
    with open(skipped_path, "w", encoding="utf-8") as skipped_log:
        skipped_log.write("REASON\tPATH\n")

        for split_name, _ in splits:
            src = raw_dir / split_name
            dst = out_dir / split_name

            if not src.exists():
                logger.warning(f"Source folder not found, skipping: {src}")
                continue

            logger.info(f"Processing {split_name}/ ...")
            s, sk = process_split(
                src, dst, detector_type, detector,
                margin, output_size, skipped_log
            )
            logger.info(f"  {split_name}: saved={s} | skipped={sk}")
            total_saved   += s
            total_skipped += sk

    logger.info("=" * 60)
    logger.info(f"Done. Total saved: {total_saved} | Total skipped: {total_skipped}")
    logger.info(f"Skipped image list → {skipped_path}")
    logger.info(f"Cropped dataset    → {out_dir}")
    logger.info("Next step: python cdcn_train.py")

    if total_saved == 0:
        logger.error(
            "No images were saved. Possible reasons:\n"
            "  1. Wrong --raw_dir path\n"
            "  2. Face detector not finding faces (try --margin 0.1 or check lighting)\n"
            "  3. Images are already cropped faces — in that case skip this script"
        )


# ─────────────────────── ENTRY POINT ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crop faces from raw dataset images before CDCN training."
    )
    parser.add_argument(
        "--raw_dir", type=str, default="raw_dataset",
        help="Root folder containing real/ and fake/ subfolders of raw images"
    )
    parser.add_argument(
        "--out_dir", type=str, default="dataset",
        help="Output folder — will contain real/ and fake/ of cropped faces"
    )
    parser.add_argument(
        "--margin", type=float, default=MARGIN,
        help="Fractional padding around detected face box (default 0.25)"
    )
    parser.add_argument(
        "--size", type=int, default=OUTPUT_SIZE,
        help="Output crop size in pixels (default 256, matches CDCN input)"
    )
    args = parser.parse_args()

    run(
        raw_dir     = Path(args.raw_dir),
        out_dir     = Path(args.out_dir),
        margin      = args.margin,
        output_size = args.size,
    )