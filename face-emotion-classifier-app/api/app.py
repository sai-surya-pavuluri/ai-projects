# api/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
from PIL import Image
import numpy as np
import datetime, json, io

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# ---------------- Paths ----------------
BASE_DIR = Path(__file__).resolve().parent                # .../face-emotion-classifier-app/api
PROJECT_ROOT = BASE_DIR.parent                            # .../face-emotion-classifier-app
MODEL_DIR = PROJECT_ROOT / "model"
# Pick which checkpoint you want to serve:
WEIGHTS_PATH = MODEL_DIR / "finetuned.weights.h5"         # or MODEL_DIR / "best.weights.h5"
TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"            # to derive label order

# ---------------- Flask ----------------
app = Flask(__name__)
CORS(app)

# ---------------- Labels ----------------
if TRAIN_DIR.exists():
    # Keras used alphabetical folder order during training
    emotion_labels = sorted(d.name for d in TRAIN_DIR.iterdir() if d.is_dir())
else:
    # Fallback to your dataset’s folder order
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

NUM_CLASSES = len(emotion_labels)
CROP = 40

# ---------------- Model (must match training architecture) ----------------
def build_model(weight_decay=1e-4, dropout1=0.5, dropout2=0.5):
    L2 = regularizers.l2(weight_decay)
    inp = keras.Input(shape=(CROP, CROP, 1), dtype='float32')

    def conv_block(x, f):
        x = layers.Conv2D(f, 3, padding='same', use_bias=False, kernel_regularizer=L2)(x)
        x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        x = layers.Conv2D(f, 3, padding='same', use_bias=False, kernel_regularizer=L2)(x)
        x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x)
        return x

    x = conv_block(inp, 64)
    x = conv_block(x, 128)
    x = conv_block(x, 256)
    x = conv_block(x, 512)

    x = layers.Flatten()(x)
    x = layers.Dense(512, kernel_regularizer=L2)(x); x = layers.ReLU()(x); x = layers.Dropout(dropout1)(x)
    x = layers.Dense(256, kernel_regularizer=L2)(x); x = layers.ReLU()(x); x = layers.Dropout(dropout2)(x)
    logits = layers.Dense(NUM_CLASSES, kernel_regularizer=L2)(x)
    out = layers.Softmax(dtype='float32')(logits)
    return keras.Model(inp, out, name="FER2013_VGG_variant")

# Build and load weights
if not WEIGHTS_PATH.exists():
    raise FileNotFoundError(f"Weights file not found at: {WEIGHTS_PATH}")

model = build_model()
# compile not required for inference but harmless:
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.load_weights(str(WEIGHTS_PATH))

# ---------------- Preprocess ----------------
def preprocess_to_40x40_gray(file_storage) -> np.ndarray:
    """
    Convert uploaded file to grayscale, resize 48x48, center-crop 40x40,
    normalize to [0,1], return shape (1, 40, 40, 1).
    """
    img = Image.open(io.BytesIO(file_storage.read())).convert('L')
    img = img.resize((48, 48))
    off = (48 - 40) // 2  # 4
    img = img.crop((off, off, off + 40, off + 40))
    arr = np.array(img).astype('float32') / 255.0
    return np.expand_dims(arr, axis=(0, -1))

# ---------------- Routes ----------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "weights_path": str(WEIGHTS_PATH),
        "input_shape": str(model.input_shape),
        "labels": emotion_labels
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    try:
        x = preprocess_to_40x40_gray(request.files["file"])
        probs = model.predict(x, verbose=0)[0]
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        emotion = emotion_labels[class_id] if 0 <= class_id < NUM_CLASSES else "unknown"
        return jsonify({"emotion": emotion, "confidence": confidence}), 200
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route("/diary", methods=["POST"])
def diary():
    data = request.get_json(silent=True) or {}
    entry = (data.get("entry") or "").strip()
    emotion = data.get("emotion")
    if not entry:
        return jsonify({"error": "Diary entry is empty"}), 400

    ts = datetime.datetime.now().isoformat(timespec="seconds")
    out_dir = PROJECT_ROOT / "diary_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{ts}.json", "w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "emotion": emotion, "entry": entry}, f, ensure_ascii=False)
    return jsonify({"status": "saved"}), 200

if __name__ == "__main__":
    app.run(debug=True, port=5000)
