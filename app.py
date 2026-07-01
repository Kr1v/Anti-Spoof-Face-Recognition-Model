"""
=============================================================================
Face Liveness + Anti-Spoofing Check — Streamlit App
=============================================================================
Flow:
  1. Visitor opens the site, browser asks for camera permission.
  2. MediaPipe FaceMesh tracks eye landmarks and waits for a natural BLINK
     (Eye Aspect Ratio dips below threshold then recovers) — this is the
     liveness challenge. A printed photo or frozen replay frame can't blink.
  3. Once a blink is confirmed, the next VERIFY_FRAMES cropped face frames
     are each passed through the DenseNet121 anti-spoofing model.
  4. Majority vote across those frames decides REAL vs FAKE.

Run locally:
    streamlit run app.py

See DEPLOYMENT NOTES at the bottom of this file for hosting instructions.
=============================================================================
"""

import threading
from pathlib import Path
from collections import deque

import av
import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import densenet121
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration

# ───────────────────────────── CONFIG ────────────────────────────────────────
MODEL_PATH          = Path(r"https://github.com/Kr1v/Anti-Spoof-Face-Recognition-Model/blob/main/models/densenet121_best_v4.pt")   # your local checkpoint
IMG_SIZE            = 224
THRESHOLD            = 0.5          # sigmoid score >= THRESHOLD -> predicted "real"
BLINK_EAR_THRESHOLD  = 0.21         # eye-aspect-ratio below this = eyes closed
BLINK_CONSEC_FRAMES  = 2            # consecutive closed-eye frames to count as a real blink
VERIFY_FRAMES        = 15           # frames used for the majority vote (odd -> no ties)
FACE_MARGIN          = 0.35         # crop margin (fraction of box size) around detected face
DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Standard MediaPipe FaceMesh eye landmark indices for EAR calculation
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# ───────────────────────────── MODEL ─────────────────────────────────────────
# Must match the architecture in train_densenet121_v4.py exactly so the
# checkpoint's state_dict keys line up.
class DenseNet121AntiSpoof(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        backbone = densenet121(weights=None)   # weights are loaded from your checkpoint below
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

    def forward(self, x):
        feat = self.features(x)
        feat = F.relu(feat, inplace=True)
        pooled = self.pool(feat)
        logit = self.head(pooled)
        return logit.squeeze(1)


@st.cache_resource
def load_model(model_path: str):
    model = DenseNet121AntiSpoof(dropout=0.5)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


PREPROCESS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ───────────────────────────── EAR (blink) HELPER ────────────────────────────
def eye_aspect_ratio(landmarks, eye_idx, w, h):
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in eye_idx])
    vert1 = np.linalg.norm(pts[1] - pts[5])
    vert2 = np.linalg.norm(pts[2] - pts[4])
    horiz = np.linalg.norm(pts[0] - pts[3])
    return (vert1 + vert2) / (2.0 * horiz + 1e-6)

# ───────────────────────────── VIDEO PROCESSOR ───────────────────────────────
class LivenessProcessor(VideoProcessorBase):
    """Runs once per WebRTC session. `recv()` is called for every incoming
    browser frame on a background thread, so all shared state is guarded
    by a lock."""

    def __init__(self, model):
        self.model = model
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with self.lock:
            self.phase = "waiting_blink"     # waiting_blink -> collecting -> done
            self.ear_below_count = 0
            self.frame_preds = deque(maxlen=VERIFY_FRAMES)
            self.result = None               # "REAL" / "FAKE" once done

    def _crop_face(self, frame_rgb, box, w, h):
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        mx, my = int(bw * FACE_MARGIN), int(bh * FACE_MARGIN)
        x1, y1 = max(0, x1 - mx), max(0, y1 - my)
        x2, y2 = min(w, x2 + mx), min(h, y2 + my)
        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

    @torch.no_grad()
    def _predict(self, face_rgb):
        tensor = PREPROCESS(face_rgb).unsqueeze(0).to(DEVICE)
        logit = self.model(tensor)
        score = torch.sigmoid(logit).item()       # closer to 1 -> real, closer to 0 -> fake
        return 1 if score >= THRESHOLD else 0      # 1 = real, 0 = fake (matches training labels)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)
        status_text = ""

        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0].landmark
            xs = [int(p.x * w) for p in lms]
            ys = [int(p.y * h) for p in lms]
            box = (min(xs), min(ys), max(xs), max(ys))
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

            with self.lock:
                phase = self.phase

            if phase == "waiting_blink":
                left_ear = eye_aspect_ratio(lms, LEFT_EYE, w, h)
                right_ear = eye_aspect_ratio(lms, RIGHT_EYE, w, h)
                ear = (left_ear + right_ear) / 2.0

                with self.lock:
                    if ear < BLINK_EAR_THRESHOLD:
                        self.ear_below_count += 1
                    else:
                        if self.ear_below_count >= BLINK_CONSEC_FRAMES:
                            self.phase = "collecting"
                        self.ear_below_count = 0
                status_text = "Please blink naturally to verify liveness..."

            elif phase == "collecting":
                face_crop = self._crop_face(rgb, box, w, h)
                if face_crop is not None:
                    pred = self._predict(face_crop)
                    with self.lock:
                        self.frame_preds.append(pred)
                        n_collected = len(self.frame_preds)
                        if n_collected >= VERIFY_FRAMES:
                            fake_votes = sum(1 for p in self.frame_preds if p == 0)
                            real_votes = len(self.frame_preds) - fake_votes
                            # Majority vote per spec: fake only if fake votes are the majority
                            self.result = "FAKE" if fake_votes > real_votes else "REAL"
                            self.phase = "done"
                    status_text = f"Verifying... {n_collected}/{VERIFY_FRAMES}"

            elif phase == "done":
                with self.lock:
                    result = self.result
                status_text = f"Result: {result}"
                color = (0, 200, 0) if result == "REAL" else (0, 0, 220)
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 3)
        else:
            status_text = "No face detected -- center your face in frame."

        # White text with dark outline so it stays readable on any background
        cv2.putText(img, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(img, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 1, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ───────────────────────────── STREAMLIT UI ───────────────────────────────────
st.set_page_config(page_title="Face Liveness Attendance Check", layout="centered")
st.title("Face Anti-Spoofing Attendance Check")
st.write(
    "1. Allow camera access when prompted.\n"
    "2. Look at the camera and **blink naturally** to prove liveness.\n"
    "3. Hold still for about a second while your face is verified."
)

if not MODEL_PATH.exists():
    st.error(f"Model checkpoint not found at `{MODEL_PATH}`.")
    uploaded = st.file_uploader("Upload a .pt checkpoint instead", type=["pt"])
    if uploaded is not None:
        tmp_path = Path("uploaded_model.pt")
        tmp_path.write_bytes(uploaded.read())
        model = load_model(str(tmp_path))
    else:
        st.stop()
else:
    model = load_model(str(MODEL_PATH))

ctx = webrtc_streamer(
    key="liveness-check",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=lambda: LivenessProcessor(model),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if st.button("Reset Check"):
    if ctx.video_processor:
        ctx.video_processor.reset()

st.caption("The verification status and final REAL/FAKE result are drawn directly on the video feed.")


# =============================================================================
# DEPLOYMENT NOTES
# =============================================================================
#
# requirements.txt (put these in your repo root):
#   streamlit
#   streamlit-webrtc
#   torch
#   torchvision
#   opencv-python-headless
#   mediapipe
#   av
#   numpy
#
# packages.txt (system libs Streamlit Cloud needs for opencv/av — put in repo root):
#   libgl1
#   libglib2.0-0
#   ffmpeg
#
# ── OPTION A: Streamlit Community Cloud (free, easiest) ────────────────────
#   1. Push this repo to GitHub, including cdcn/models/densenet121_best_v4.pt
#      - GitHub blocks files over 100MB by default. If your checkpoint is
#        bigger, use Git LFS (`git lfs track "*.pt"`) or host the file
#        externally (Hugging Face Hub / S3 / Google Drive) and download it
#        at app startup instead of committing it.
#   2. Go to https://share.streamlit.io, sign in with GitHub, click
#      "New app", pick this repo/branch and app.py as the entry point.
#   3. Streamlit Cloud auto-installs requirements.txt and packages.txt and
#      serves the app over HTTPS (required for browser camera access).
#   4. Done — you get a public https://<app-name>.streamlit.app URL.
#
# ── OPTION B: Your own server (Docker + VM) ─────────────────────────────────
#   1. Dockerfile:
#        FROM python:3.11-slim
#        RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 ffmpeg
#        WORKDIR /app
#        COPY . .
#        RUN pip install -r requirements.txt
#        EXPOSE 8501
#        CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
#   2. Build & run:
#        docker build -t liveness-app .
#        docker run -p 8501:8501 liveness-app
#      Then put a reverse proxy (nginx) in front of port 8501 to handle
#      HTTPS termination for the public-facing domain.
#   3. Put it behind nginx + Let's Encrypt (certbot) for HTTPS — browsers
#      refuse camera access on plain HTTP for any host other than localhost.
#   4. CPU inference is fine here: DenseNet121 (~7M params) runs a single
#      224x224 frame in well under 100ms on a normal cloud CPU core.
#
# ── WebRTC connectivity note (applies to both options) ──────────────────────
#   The STUN server above (Google's public one) works for most home/office
#   networks. Some users behind strict corporate NAT/firewalls will fail to
#   connect without a TURN server as a relay fallback. If you see users
#   stuck on "connecting", add a TURN server to RTC_CONFIGURATION — Twilio's
#   Network Traversal Service has a free tier, or self-host coturn.
#
# ── Note on your rural/low-end Android target ───────────────────────────────
#   This Streamlit+WebRTC app is a good demo/admin dashboard, but WebRTC in
#   a mobile browser on a low-end device / poor connectivity can be choppy.
#   For the actual field-deployed attendance app, plan on-device inference
#   (TorchScript/ONNX export + a native camera capture flow) rather than
#   routing every frame through this browser-streaming pipeline.
# =============================================================================
