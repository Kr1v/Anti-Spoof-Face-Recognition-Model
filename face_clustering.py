"""
=============================================================================
Face Clustering — Group Same Faces Into Person Folders
=============================================================================
Problem:  Raw dataset has images named arbitrarily (img001.jpg, etc.)
          Same person may appear in both train and val → data leakage.

Solution: 
  1. Detect face in every image
  2. Embed each face into a 128-d vector (FaceNet/ArcFace via face_recognition)
  3. Cluster embeddings with DBSCAN (no need to specify number of people)
  4. Move images into  clustered/person_001/, person_002/, ... person_N/
  5. Split AT IDENTITY LEVEL → person_001 goes entirely to train OR val, never both

Output structure:
  clustered/
    real/
      person_001/   ← all images of this person
      person_002/
      noise/        ← faces DBSCAN couldn't assign (blurry, partial)
    fake/
      person_001/
      ...

  split/
    train/
      real/
        person_001/
        person_003/
      fake/
        person_001/
    val/
      real/
        person_002/
      fake/
        person_002/

=============================================================================
Install:
    pip install face_recognition scikit-learn opencv-python numpy tqdm
    (face_recognition requires dlib — on RPi: pip install dlib --no-cache-dir)
=============================================================================
"""

# ─────────────────────────── 1. IMPORTS ──────────────────────────────────────
import os
import shutil
import logging
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import cv2
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

try:
    import face_recognition          # wraps dlib's FaceNet-like 128-d embedder
    FACE_REC_AVAILABLE = True
except ImportError:
    FACE_REC_AVAILABLE = False
    logging.warning("face_recognition not installed. Falling back to OpenCV HOG embedder.")

# ─────────────────────────── 2. LOGGING ──────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("face_clustering.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────── 3. CONFIG ───────────────────────────────────────
class Config:
    # ── Input ──────────────────────────────────────────────────────────────
    # Your raw dataset with class subfolders (real/, fake/)
    RAW_DIR       = Path("dataset")

    # ── Output ─────────────────────────────────────────────────────────────
    CLUSTERED_DIR = Path("clustered")   # grouped by identity
    SPLIT_DIR     = Path("split")       # train/val split by identity

    # ── Face detection ──────────────────────────────────────────────────────
    # "hog"  → CPU-friendly, less accurate
    # "cnn"  → GPU-accelerated, more accurate (needs dlib compiled with CUDA)
    DETECTION_MODEL = "hog"

    # ── Embedding ───────────────────────────────────────────────────────────
    # face_recognition uses a ResNet model trained on 3M faces (FaceNet-style)
    # Each face → 128-dimensional L2-normalized vector
    EMBED_DIM     = 128

    # ── DBSCAN clustering ───────────────────────────────────────────────────
    # eps: max distance between two embeddings to be in the same cluster
    #   Lower  → stricter matching (fewer false merges, more clusters)
    #   Higher → looser matching (may merge different people)
    #   0.5–0.6 is standard for 128-d face embeddings
    DBSCAN_EPS     = 0.5
    DBSCAN_MIN_SAMPLES = 2   # min images to form a cluster (noise otherwise)

    # ── Train / Val split ───────────────────────────────────────────────────
    VAL_SPLIT      = 0.2     # 20% of IDENTITIES go to val (not images)
    SEED           = 42

    # ── Image formats ───────────────────────────────────────────────────────
    IMG_EXTS       = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

cfg = Config()

# ─────────────────────────── 4. FACE DETECTION + EMBEDDING ───────────────────
def get_face_embedding_face_recognition(img_path: Path,
                                         model: str = "hog") -> np.ndarray | None:
    """
    Uses the face_recognition library (dlib under the hood).
    Detects the largest face, returns its 128-d embedding.
    Returns None if no face found.
    """
    img = face_recognition.load_image_file(str(img_path))   # RGB numpy array

    # Step 1: locate face bounding boxes
    locations = face_recognition.face_locations(img, model=model)
    if not locations:
        return None

    # Take the largest face (by bounding box area) if multiple detected
    largest = max(locations, key=lambda loc: (loc[2]-loc[0]) * (loc[1]-loc[3]))

    # Step 2: compute 128-d embedding for that face location
    encodings = face_recognition.face_encodings(img, known_face_locations=[largest])
    if not encodings:
        return None

    return np.array(encodings[0])   # shape: (128,)


def get_face_embedding_opencv(img_path: Path) -> np.ndarray | None:
    """
    Fallback when face_recognition is not available.
    Uses OpenCV Haar Cascade for detection and HOG descriptor as embedding.
    NOTE: HOG is NOT identity-discriminative — clustering quality will be lower.
          Install face_recognition for production use.
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)

    img = cv2.imread(str(img_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return None

    # Largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_crop = gray[y:y+h, x:x+w]
    face_crop = cv2.resize(face_crop, (64, 64))

    # HOG descriptor as a crude embedding
    hog = cv2.HOGDescriptor((64,64), (16,16), (8,8), (8,8), 9)
    descriptor = hog.compute(face_crop).flatten()

    # L2-normalize so DBSCAN distances are comparable
    norm = np.linalg.norm(descriptor)
    return descriptor / norm if norm > 0 else descriptor


def get_embedding(img_path: Path, detection_model: str = "hog") -> np.ndarray | None:
    """Router: use face_recognition if available, else OpenCV HOG."""
    if FACE_REC_AVAILABLE:
        return get_face_embedding_face_recognition(img_path, detection_model)
    else:
        logger.warning("Using OpenCV HOG fallback — install face_recognition for better clustering")
        return get_face_embedding_opencv(img_path)


# ─────────────────────────── 5. COLLECT EMBEDDINGS ───────────────────────────
def collect_embeddings(class_dir: Path, detection_model: str):
    """
    Walks a class folder (e.g. dataset/real/), embeds every face image.
    Returns:
        embeddings : np.ndarray shape (N, 128)
        img_paths  : list of Path, length N
        failed     : list of Path that had no detectable face
    """
    img_paths  = [p for p in class_dir.rglob("*")
                  if p.suffix.lower() in cfg.IMG_EXTS]

    logger.info(f"  Found {len(img_paths)} images in {class_dir.name}/")

    embeddings = []
    valid_paths = []
    failed      = []

    for img_path in tqdm(img_paths, desc=f"Embedding {class_dir.name}", unit="img"):
        emb = get_embedding(img_path, detection_model)
        if emb is not None:
            embeddings.append(emb)
            valid_paths.append(img_path)
        else:
            failed.append(img_path)

    logger.info(f"  Embedded: {len(embeddings)} | No face detected: {len(failed)}")

    if not embeddings:
        return np.array([]), [], failed

    return np.array(embeddings), valid_paths, failed


# ─────────────────────────── 6. CLUSTER WITH DBSCAN ─────────────────────────
def cluster_embeddings(embeddings: np.ndarray,
                        eps: float, min_samples: int) -> np.ndarray:
    """
    DBSCAN clustering on face embeddings.

    Why DBSCAN (not K-Means)?
      - You don't know how many people are in the dataset (no K to specify)
      - DBSCAN finds arbitrary-shaped clusters
      - Points too far from any cluster → label -1 (noise) → goes to noise/
      - Metric: euclidean distance on L2-normalized vectors
        = equivalent to cosine distance (standard for face embeddings)

    Returns:
        labels: array of length N, each value is cluster ID (or -1 for noise)
    """
    # L2 normalize embeddings (makes euclidean ≈ cosine distance)
    emb_normalized = normalize(embeddings, norm="l2")

    db = DBSCAN(
        eps          = eps,
        min_samples  = min_samples,
        metric       = "euclidean",
        n_jobs       = -1            # use all CPU cores
    )
    labels = db.fit_predict(emb_normalized)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = list(labels).count(-1)
    logger.info(f"  Clusters (identities): {n_clusters} | Noise images: {n_noise}")

    return labels


# ─────────────────────────── 7. ORGANIZE INTO FOLDERS ────────────────────────
def organize_into_folders(img_paths: list, labels: np.ndarray,
                           failed: list, out_dir: Path):
    """
    Copies images into:
        out_dir/person_001/img.jpg
        out_dir/person_002/img.jpg
        out_dir/noise/img.jpg        ← DBSCAN label -1
        out_dir/no_face/img.jpg      ← no face detected at all
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Map cluster label → zero-padded folder name
    label_set = sorted(set(labels))
    label_to_folder = {}
    person_counter = 1
    for lbl in label_set:
        if lbl == -1:
            label_to_folder[lbl] = out_dir / "noise"
        else:
            label_to_folder[lbl] = out_dir / f"person_{person_counter:03d}"
            person_counter += 1

    # Create all folders
    for folder in label_to_folder.values():
        folder.mkdir(parents=True, exist_ok=True)
    (out_dir / "no_face").mkdir(parents=True, exist_ok=True)

    # Copy images
    for img_path, lbl in zip(img_paths, labels):
        dest_folder = label_to_folder[lbl]
        dest = dest_folder / img_path.name
        # Handle duplicate filenames across subdirs
        if dest.exists():
            dest = dest_folder / f"{img_path.stem}_{img_path.parent.name}{img_path.suffix}"
        shutil.copy2(str(img_path), str(dest))

    # Copy failed (no face detected)
    no_face_dir = out_dir / "no_face"
    for img_path in failed:
        shutil.copy2(str(img_path), str(no_face_dir / img_path.name))

    logger.info(f"  Organized into {out_dir}")
    return label_to_folder


# ─────────────────────────── 8. IDENTITY-LEVEL TRAIN/VAL SPLIT ───────────────
def split_by_identity(clustered_class_dir: Path, split_dir: Path,
                       class_name: str, val_split: float, seed: int):
    """
    Splits at the IDENTITY (person folder) level, not the image level.

    This guarantees:
        - person_001 is ENTIRELY in train OR val, never both
        - No face leakage across splits
        - Stratification is meaningless here (we have 1 class label per identity cluster)

    Algorithm:
        1. List all person_XXX/ folders (exclude noise/ and no_face/)
        2. Shuffle with fixed seed
        3. First 80% → train,  last 20% → val
        4. Copy entire folder to split/train/class/ or split/val/class/
    """
    rng = np.random.default_rng(seed)

    person_dirs = sorted([
        d for d in clustered_class_dir.iterdir()
        if d.is_dir() and d.name.startswith("person_")
    ])

    if not person_dirs:
        logger.warning(f"No person folders found in {clustered_class_dir}")
        return

    # Shuffle identities
    indices = np.arange(len(person_dirs))
    rng.shuffle(indices)
    person_dirs = [person_dirs[i] for i in indices]

    # Split point
    n_val   = max(1, int(len(person_dirs) * val_split))
    n_train = len(person_dirs) - n_val

    train_persons = person_dirs[:n_train]
    val_persons   = person_dirs[n_train:]

    logger.info(f"  {class_name}: {n_train} identities → train | "
                f"{n_val} identities → val")

    # Copy to split directories
    for person_dir in train_persons:
        dest = split_dir / "train" / class_name / person_dir.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(str(person_dir), str(dest))

    for person_dir in val_persons:
        dest = split_dir / "val" / class_name / person_dir.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(str(person_dir), str(dest))


# ─────────────────────────── 9. MAIN PIPELINE ────────────────────────────────
def run_pipeline(cfg: Config):
    """
    Full pipeline:
      RAW dataset → embed → cluster → organize → identity split
    """
    logger.info("=" * 60)
    logger.info("FACE CLUSTERING PIPELINE")
    logger.info("=" * 60)

    # Discover class folders (real/, fake/, or any subfolders)
    class_dirs = [d for d in cfg.RAW_DIR.iterdir() if d.is_dir()]
    if not class_dirs:
        logger.error(f"No subfolders found in {cfg.RAW_DIR}. "
                     f"Expected: dataset/real/, dataset/fake/")
        return

    for class_dir in class_dirs:
        class_name = class_dir.name
        logger.info(f"\n{'─'*40}")
        logger.info(f"Processing class: {class_name.upper()}")
        logger.info(f"{'─'*40}")

        # ── Step 1: Embed all faces ───────────────────────────────────────
        embeddings, img_paths, failed = collect_embeddings(
            class_dir, cfg.DETECTION_MODEL
        )

        if len(embeddings) == 0:
            logger.warning(f"  No embeddings for {class_name} — skipping.")
            continue

        # ── Step 2: Cluster embeddings ────────────────────────────────────
        labels = cluster_embeddings(
            embeddings, cfg.DBSCAN_EPS, cfg.DBSCAN_MIN_SAMPLES
        )

        # ── Step 3: Organize into person folders ──────────────────────────
        out_class_dir = cfg.CLUSTERED_DIR / class_name
        organize_into_folders(img_paths, labels, failed, out_class_dir)

        # ── Step 4: Print cluster summary ─────────────────────────────────
        from collections import Counter
        counts = Counter(labels)
        logger.info(f"\n  Cluster sizes (top 10):")
        for lbl, count in counts.most_common(10):
            name = f"person_{list(counts.keys()).index(lbl)+1:03d}" if lbl != -1 else "noise"
            logger.info(f"    {name:15s}: {count} images")

        # ── Step 5: Identity-level train/val split ────────────────────────
        logger.info(f"\n  Splitting by identity...")
        split_by_identity(
            out_class_dir, cfg.SPLIT_DIR, class_name,
            cfg.VAL_SPLIT, cfg.SEED
        )

    logger.info("\n" + "=" * 60)
    logger.info("DONE")
    logger.info(f"Clustered dataset → {cfg.CLUSTERED_DIR}")
    logger.info(f"Train/Val split   → {cfg.SPLIT_DIR}")
    logger.info("=" * 60)

    # Print final split summary
    print_split_summary(cfg.SPLIT_DIR)


def print_split_summary(split_dir: Path):
    """Prints image counts for each class in train and val."""
    logger.info("\nFINAL SPLIT SUMMARY:")
    for subset in ["train", "val"]:
        subset_dir = split_dir / subset
        if not subset_dir.exists():
            continue
        for class_dir in sorted(subset_dir.iterdir()):
            imgs = list(class_dir.rglob("*"))
            imgs = [p for p in imgs if p.suffix.lower() in cfg.IMG_EXTS]
            n_persons = len([d for d in class_dir.iterdir() if d.is_dir()])
            logger.info(f"  {subset}/{class_dir.name}: "
                        f"{n_persons} identities | {len(imgs)} images")


# ─────────────────────────── 10. INSPECT CLUSTERS ────────────────────────────
def inspect_clusters(clustered_dir: Path, class_name: str = "real",
                      max_persons: int = 5, imgs_per_person: int = 4):
    """
    Quick visual sanity check — prints a grid of images per cluster.
    Run this after clustering to verify correctness before training.
    """
    import random

    class_dir = clustered_dir / class_name
    person_dirs = sorted([d for d in class_dir.iterdir()
                          if d.is_dir() and d.name.startswith("person_")])

    logger.info(f"\nSANITY CHECK — {class_name} — showing {max_persons} clusters:")

    for person_dir in person_dirs[:max_persons]:
        imgs = list(person_dir.glob("*"))
        imgs = [p for p in imgs if p.suffix.lower() in cfg.IMG_EXTS]
        sample = random.sample(imgs, min(imgs_per_person, len(imgs)))
        logger.info(f"  {person_dir.name}: {len(imgs)} total images | "
                    f"sample: {[p.name for p in sample]}")

        # Display using OpenCV if available
        panels = []
        for p in sample:
            img = cv2.imread(str(p))
            if img is not None:
                img = cv2.resize(img, (100, 100))
                cv2.putText(img, person_dir.name, (2, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)
                panels.append(img)
        if panels:
            row = np.hstack(panels)
            cv2.imshow(f"Cluster: {person_dir.name}", row)

    if cv2.getWindowProperty("Cluster: person_001", cv2.WND_PROP_VISIBLE) >= 0:
        logger.info("  Press any key to close windows.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ─────────────────────────── 11. ENTRY POINT ─────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Clustering — Group by Identity")
    parser.add_argument("--mode",
                        choices=["cluster", "inspect"],
                        default="cluster",
                        help="cluster: run full pipeline | inspect: view cluster samples")
    parser.add_argument("--class_name", type=str, default="real",
                        help="Which class folder to inspect (for --mode inspect)")
    parser.add_argument("--eps", type=float, default=cfg.DBSCAN_EPS,
                        help=f"DBSCAN eps (default {cfg.DBSCAN_EPS}). "
                             "Lower = stricter identity matching.")
    parser.add_argument("--val_split", type=float, default=cfg.VAL_SPLIT,
                        help="Fraction of identities for validation (default 0.2)")
    args = parser.parse_args()

    cfg.DBSCAN_EPS = args.eps
    cfg.VAL_SPLIT  = args.val_split

    if args.mode == "cluster":
        run_pipeline(cfg)
    elif args.mode == "inspect":
        inspect_clusters(cfg.CLUSTERED_DIR, args.class_name)