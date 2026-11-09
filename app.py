"""Flask interface for counting adult spotted lanternflies in one photo."""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent
WEIGHTS_SETTING = Path(os.environ.get("LANTERNFLY_WEIGHTS", "models/best.pt"))
WEIGHTS = WEIGHTS_SETTING if WEIGHTS_SETTING.is_absolute() else ROOT / WEIGHTS_SETTING

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 30_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

try:
    CONFIDENCE = float(os.environ.get("LANTERNFLY_CONFIDENCE", "0.20"))
except ValueError:
    CONFIDENCE = 0.20
if not 0 <= CONFIDENCE <= 1:
    CONFIDENCE = 0.20

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


def load_detector():
    """Load one model at startup; keep the page available if weights are absent."""
    if not WEIGHTS.is_file():
        return None, f"Model file not found. Place best.pt at: {WEIGHTS}"
    try:
        from ultralytics import YOLO
        detector = YOLO(str(WEIGHTS))
        names = detector.names
        names = names.values() if isinstance(names, dict) else names
        if not any("lanternfly" in str(name).lower() for name in names):
            return None, "The selected model does not contain a lanternfly class."
        return detector, None
    except Exception as exc:
        app.logger.exception("Model loading failed")
        return None, f"The model could not be loaded: {exc}"


model, model_error = load_detector()


@app.get("/")
def index():
    return render_template("index.html", model_ready=model is not None)


@app.post("/predict")
def predict():
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        return jsonify(error="Choose or take a photo first."), 400

    payload = upload.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        return jsonify(error="Choose a photo smaller than 12 MB."), 413

    try:
        with Image.open(io.BytesIO(payload)) as source:
            if source.format not in ALLOWED_FORMATS:
                return jsonify(error="Use a JPEG, PNG, or WebP photo."), 400
            if source.width * source.height > MAX_IMAGE_PIXELS:
                return jsonify(error="This image is too large to process."), 400
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
    except Exception:
        return jsonify(error="The selected file is not a readable image."), 400

    if model is None:
        return jsonify(error=model_error or "The model is not ready."), 503

    try:
        result = model.predict(
            image, imgsz=640, conf=CONFIDENCE, iou=0.45,
            max_det=1000, verbose=False
        )[0]

        names = result.names
        if isinstance(names, dict):
            target_ids = {int(i) for i, name in names.items()
                          if "lanternfly" in str(name).lower()}
        else:
            target_ids = {i for i, name in enumerate(names)
                          if "lanternfly" in str(name).lower()}

        detections = []
        if result.boxes is not None:
            for box, score, class_id in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                result.boxes.cls.cpu().tolist()
            ):
                if int(class_id) in target_ids:
                    detections.append((box, float(score)))

        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        line_width = max(2, round(max(image.size) / 400))
        for box, score in detections:
            coords = tuple(round(value) for value in box)
            label = f"Spotted lanternfly {score:.0%}"
            draw.rectangle(coords, outline="#75c66a", width=line_width)
            text_box = draw.textbbox((coords[0], coords[1]), label)
            draw.rectangle(
                (text_box[0] - 3, text_box[1] - 3,
                 text_box[2] + 3, text_box[3] + 3),
                fill="#173d2d"
            )
            draw.text((coords[0], coords[1]), label, fill="white")

        output = io.BytesIO()
        annotated.save(output, format="JPEG", quality=88)
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify(error="The model could not analyze this photo. Try another image."), 500

    return jsonify(
        count=len(detections),
        detections=[{"box": [round(float(v), 2) for v in box],
                     "confidence": round(score, 4)}
                    for box, score in detections],
        annotated_image=f"data:image/jpeg;base64,{encoded}"
    )


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error="Choose a photo smaller than 12 MB."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
