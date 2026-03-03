from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent


def _load_face_cascade():
    """Load OpenCV's frontal face Haar cascade if available."""
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if cascade_path.exists():
        return cv2.CascadeClassifier(str(cascade_path))
    return None


_FACE_CASCADE = _load_face_cascade()


def enhance_lighting_bgr(bgr: np.ndarray) -> np.ndarray:
    """
    Make a BGR image more robust to dark / overexposed conditions using CLAHE + gamma.
    """
    # CLAHE on L channel in LAB color space
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    clahe_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Automatic gamma correction based on mean intensity
    gray = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2GRAY)
    mean_intensity = gray.mean()
    # Map mean in [30, 220] to gamma ~ [1.8, 0.6]
    mean_clipped = np.clip(mean_intensity, 30, 220)
    alpha = (mean_clipped - 30) / (220 - 30 + 1e-6)
    gamma = 1.8 - alpha * (1.8 - 0.6)

    inv_gamma = 1.0 / gamma
    table = np.array(
        [(i / 255.0) ** inv_gamma * 255 for i in range(256)],
        dtype="uint8",
    )
    corrected = cv2.LUT(clahe_bgr, table)

    return corrected


def detect_face_region(bgr: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Detect a face bounding box (x, y, w, h). If detection fails, return full image.
    """
    h, w = bgr.shape[:2]
    if _FACE_CASCADE is None:
        return 0, 0, w, h

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )
    if len(faces) == 0:
        return 0, 0, w, h

    # Take the largest detected face
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])

    # Expand box slightly to include chin/forehead
    pad_x = int(0.1 * fw)
    pad_y = int(0.15 * fh)
    x0 = max(x - pad_x, 0)
    y0 = max(y - pad_y, 0)
    x1 = min(x + fw + pad_x, w)
    y1 = min(y + fh + pad_y, h)

    return x0, y0, x1 - x0, y1 - y0


def enhance_face_image(pil_img: Image.Image, target_size=(128, 128)) -> Image.Image:
    """
    Full preprocessing pipeline to make faces robust to lighting:
      - Convert to OpenCV BGR
      - Enhance lighting (CLAHE + gamma correction)
      - Detect and crop the face region (eyes, mouth, full face)
      - Resize to target size and return as RGB PIL Image
    """
    rgb = pil_img.convert("RGB")
    bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

    bgr = enhance_lighting_bgr(bgr)

    x, y, w, h = detect_face_region(bgr)
    face_bgr = bgr[y : y + h, x : x + w]

    face_bgr = cv2.resize(face_bgr, target_size, interpolation=cv2.INTER_AREA)
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(face_rgb)

