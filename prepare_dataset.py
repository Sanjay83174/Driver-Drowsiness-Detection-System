from pathlib import Path

from PIL import Image

from preprocess_utils import enhance_face_image


BASE_DIR = Path(__file__).resolve().parent
RAW_DATASET_DIR = BASE_DIR / "0 FaceImages"
PROCESSED_DATASET_DIR = BASE_DIR / "0 FaceImages_enhanced"


def preprocess_dataset():
    """
    Create a lighting-normalized, face-focused version of the dataset.

    - Reads images from `0 FaceImages/Active Subjects` and `0 FaceImages/Fatigue Subjects`
    - Enhances lighting (dark / bright) and crops to face region
    - Saves the processed images into `0 FaceImages_enhanced/...`
    """
    if not RAW_DATASET_DIR.exists():
        raise FileNotFoundError(f"Raw dataset not found at {RAW_DATASET_DIR}")

    PROCESSED_DATASET_DIR.mkdir(exist_ok=True)

    count = 0
    for class_dir in sorted(p for p in RAW_DATASET_DIR.iterdir() if p.is_dir()):
        rel = class_dir.name
        out_class_dir = PROCESSED_DATASET_DIR / rel
        out_class_dir.mkdir(parents=True, exist_ok=True)

        for img_path in class_dir.glob("*.jpg"):
            out_path = out_class_dir / img_path.name

            # If already processed, skip (allows safe restart)
            if out_path.exists():
                continue

            try:
                with Image.open(img_path) as img:
                    enhanced = enhance_face_image(img)
                    enhanced.save(out_path, format="JPEG", quality=95)
                count += 1

                # Lightweight progress indicator every 100 images
                if count % 100 == 0:
                    print(f"Processed {count} images so far... (last: {out_path})")

            except Exception as e:
                print(f"Skipping {img_path} due to error: {e}")

    print(
        f"Finished preprocessing. Wrote/updated images in {PROCESSED_DATASET_DIR}. "
        "You can now point training to this folder for improved robustness to dark/bright faces."
    )


if __name__ == "__main__":
    preprocess_dataset()

