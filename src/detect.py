"""Real-time face mask detection.

Usage:
    python -m src.detect                  # live webcam (press q to quit)
    python -m src.detect --image a.jpg    # one image -> results/annotated.jpg
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from . import config

GREEN, RED = (0, 200, 0), (0, 0, 255)  # BGR


def load_face_detector():
    """Load the Haar cascade face detector (bundled in assets/, with an OpenCV fallback)."""
    candidates = [config.ROOT / "assets" / "haarcascade_frontalface_default.xml"]
    if hasattr(cv2, "data"):
        candidates.append(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    for path in candidates:
        if path.exists():
            detector = cv2.CascadeClassifier(str(path))
            if not detector.empty():
                return detector
    raise RuntimeError("Could not load the face detector file "
                       "(assets/haarcascade_frontalface_default.xml).")


def classify_face(model, face_bgr):
    """Return (label, confidence) for a cropped BGR face image."""
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, config.IMG_SIZE)
    batch = np.expand_dims(rgb.astype("float32"), axis=0)  # raw 0-255
    p_no_mask = float(model.predict(batch, verbose=0)[0][0])
    if p_no_mask > 0.5:
        return "No Mask", p_no_mask
    return "Mask", 1.0 - p_no_mask


def annotate(frame, model, detector):
    """Detect faces in a BGR frame, draw boxes + labels, return (frame, results)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    results = []
    h, w = frame.shape[:2]
    for (x, y, fw, fh) in faces:
        pad = int(0.1 * fw)  # a little margin around the face
        x1, y1 = max(x - pad, 0), max(y - pad, 0)
        x2, y2 = min(x + fw + pad, w), min(y + fh + pad, h)
        label, conf = classify_face(model, frame[y1:y2, x1:x2])
        color = GREEN if label == "Mask" else RED
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)
        cv2.putText(frame, f"{label} {conf * 100:.0f}%", (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        results.append((label, conf, (x, y, fw, fh)))
    return frame, results


def run_webcam(model, detector, camera=0):
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam. Try --camera 1.")
    print("Webcam running - press 'q' in the video window to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame, _ = annotate(frame, model, detector)
        cv2.imshow("Face Mask Detection (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="path to an image instead of using the webcam")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    model = tf.keras.models.load_model(config.MODEL_PATH)
    detector = load_face_detector()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"Could not read image: {args.image}")
        frame, results = annotate(frame, model, detector)
        config.RESULTS_DIR.mkdir(exist_ok=True)
        out = config.RESULTS_DIR / "annotated.jpg"
        cv2.imwrite(str(out), frame)
        print(f"{len(results)} face(s) found:", [(r[0], round(r[1], 3)) for r in results])
        print(f"Saved {out}")
    else:
        run_webcam(model, detector, args.camera)


if __name__ == "__main__":
    main()
