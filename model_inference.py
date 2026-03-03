from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from tensorflow import keras

from preprocess_utils import enhance_face_image


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"
# Default to the best checkpoint saved during training (.keras format)
DEFAULT_MODEL_PATH = MODEL_DIR / "drowsiness_cnn.keras"

IMG_SIZE = (128, 128)

_MODEL = None


def load_drowsiness_model(model_path: Optional[Path] = None):
    """Load the trained drowsiness CNN model (singleton-style)."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at {path}. "
            f"Train the model first by running train_model.py."
        )

    _MODEL = keras.models.load_model(path)
    return _MODEL


def preprocess_image(img: Image.Image) -> np.ndarray:
    """
    Enhance and normalize an input PIL image to feed into the CNN.

    Steps:
      - Enhance lighting (dark / bright) and crop to face region
      - Resize to model input size
      - Keep pixels in [0, 255] so the model's internal Rescaling(1/255)
        layer sees the same range as during training.
    """
    enhanced = enhance_face_image(img, target_size=IMG_SIZE)
    enhanced = enhanced.convert("RGB")
    # IMPORTANT: do NOT divide by 255 here; the model itself has a Rescaling(1/255) layer.
    arr = np.array(enhanced).astype("float32")
    arr = np.expand_dims(arr, axis=0)  # add batch dimension
    return arr


def predict_drowsiness(img: Image.Image, model=None) -> float:
    """
    Predict drowsiness probability for a single frame.

    Returns:
        float between 0 and 1: probability person is drowsy/fatigued.
    """
    if model is None:
        model = load_drowsiness_model()

    x = preprocess_image(img)
    prob = model.predict(x, verbose=0)[0][0]
    return float(prob)

