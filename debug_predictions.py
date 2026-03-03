from pathlib import Path
import random

from PIL import Image

from model_inference import load_drowsiness_model, predict_drowsiness


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "0 FaceImages_enhanced"


def collect_samples(class_name: str, n: int = 20):
    class_dir = DATASET_DIR / class_name
    if not class_dir.exists():
        raise FileNotFoundError(f"Class folder not found: {class_dir}")

    images = list(class_dir.glob("*.jpg"))
    if not images:
        raise RuntimeError(f"No images found in {class_dir}")

    random.shuffle(images)
    return images[: min(n, len(images))]


def main():
    model = load_drowsiness_model()

    active_name = "Active Subjects"
    fatigue_name = "Fatigue Subjects"

    active_paths = collect_samples(active_name, n=20)
    fatigue_paths = collect_samples(fatigue_name, n=20)

    print(f"Loaded model. Testing on {len(active_paths)} active and {len(fatigue_paths)} fatigue images.")

    active_probs = []
    fatigue_probs = []

    print("\n=== Active (should be ALERT / low drowsy prob) ===")
    for p in active_paths:
        img = Image.open(p)
        prob = predict_drowsiness(img, model=model)
        active_probs.append(prob)
        print(f"{p.name:30s} -> drowsy_prob={prob:.3f}")

    print("\n=== Fatigue (should be DROWSY / high drowsy prob) ===")
    for p in fatigue_paths:
        img = Image.open(p)
        prob = predict_drowsiness(img, model=model)
        fatigue_probs.append(prob)
        print(f"{p.name:30s} -> drowsy_prob={prob:.3f}")

    if active_probs:
        print(f"\nAverage drowsy prob for ACTIVE:  {sum(active_probs)/len(active_probs):.3f}")
    if fatigue_probs:
        print(f"Average drowsy prob for FATIGUE: {sum(fatigue_probs)/len(fatigue_probs):.3f}")


if __name__ == "__main__":
    main()

