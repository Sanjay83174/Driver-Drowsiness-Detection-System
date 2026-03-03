import io
import os
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from flask import Flask, jsonify, render_template, request

from model_inference import load_drowsiness_model, predict_drowsiness


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

model = None


def get_model():
    global model
    if model is None:
        model = load_drowsiness_model()
    return model


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict_image", methods=["POST"])
def api_predict_image():
    """
    Predict drowsiness from a single image.

    Accepts:
      - multipart/form-data with field `image`
      - or JSON with base64-encoded PNG/JPEG in `image_base64`
    """
    from base64 import b64decode

    pil_image = None

    if "image" in request.files:
        file = request.files["image"]
        pil_image = Image.open(file.stream)
    elif request.is_json:
        data = request.get_json(silent=True) or {}
        img_b64 = data.get("image_base64")
        if img_b64:
            # Remove data URL prefix if present
            if "," in img_b64:
                img_b64 = img_b64.split(",", 1)[1]
            try:
                raw = b64decode(img_b64)
                pil_image = Image.open(io.BytesIO(raw))
            except Exception:
                return jsonify({"error": "Invalid base64 image"}), 400

    if pil_image is None:
        return jsonify({"error": "No image provided"}), 400

    prob = predict_drowsiness(pil_image, model=get_model())
    # Use "awake" instead of "alert" for a more natural wording.
    state = "drowsy" if prob >= 0.6 else "awake"

    return jsonify(
        {
            "drowsy_probability": prob,
            "state": state,
        }
    )


@app.route("/api/predict_video", methods=["POST"])
def api_predict_video():
    """
    Predict drowsiness over an uploaded video.

    Strategy:
      - Sample frames at a fixed interval
      - Run the CNN on each sampled frame
      - Apply simple temporal logic to determine if there are
        segments where eyes are closed / yawning for > 3-4 seconds.
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    video_path = UPLOAD_DIR / file.filename
    file.save(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return jsonify({"error": "Unable to open uploaded video"}), 400

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = int(max(fps // 3, 1))  # ~3 samples per second

    model_instance = get_model()

    frame_index = 0
    timestamps = []
    probs = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            # Convert BGR (OpenCV) to RGB (PIL)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            prob = predict_drowsiness(pil_img, model=model_instance)

            t = frame_index / fps
            timestamps.append(t)
            probs.append(prob)

        frame_index += 1

    cap.release()

    # Temporal logic: mark intervals where prob >= 0.6 for >= 3 seconds
    threshold = 0.6
    min_duration = 3.0

    drowsy_segments = []
    segment_start = None

    for t, p in zip(timestamps, probs):
        if p >= threshold:
            if segment_start is None:
                segment_start = t
        else:
            if segment_start is not None:
                duration = t - segment_start
                if duration >= min_duration:
                    drowsy_segments.append({"start": segment_start, "end": t})
                segment_start = None

    # Close segment if video ends while drowsy
    if segment_start is not None and timestamps:
        last_t = timestamps[-1]
        duration = last_t - segment_start
        if duration >= min_duration:
            drowsy_segments.append({"start": segment_start, "end": last_t})

    overall_drowsy = bool(drowsy_segments)
    avg_prob = float(np.mean(probs)) if probs else 0.0

    return jsonify(
        {
            "overall_state": "drowsy" if overall_drowsy else "awake",
            "average_drowsy_probability": avg_prob,
            "drowsy_segments": drowsy_segments,
        }
    )


if __name__ == "__main__":
    # Example: FLASK_ENV=development python app.py
    app.run(host="0.0.0.0", port=5000, debug=True)

