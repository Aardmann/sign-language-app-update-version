from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import base64
import cv2

app = Flask(__name__)
CORS(app)

# ── Load both models at startup ──────────────────────────────
print("Loading models...")
models_store = {
    "alphabets": tf.keras.models.load_model("sign_language_model.h5"),
    "digits":    tf.keras.models.load_model("sign_digits_model.h5")
}
print("Both models loaded!")

# ── Warm up models ────────────────────────────────────────────
# The first call to a Keras model builds/traces its inference graph,
# which is slow (can be 1-2s). Do that once now, at startup, instead
# of on whichever request happens to hit it first.
print("Warming up models...")
_dummy = np.zeros((1, 64, 64, 3), dtype=np.float32)
for _name, _model in models_store.items():
    _model(_dummy, training=False)
print("Warm-up complete!")

# ── Class labels ─────────────────────────────────────────────
CLASS_NAMES = {
    "alphabets": [
        'A','B','C','D','del','E','F','G','H','I',
        'J','K','L','M','N','nothing','O','P','Q','R',
        'S','space','T','U','V','W','X','Y','Z'
    ],
    "digits": ['0','1','2','3','4','5','6','7','8','9']
}

IMG_SIZE = 64

# ── Routes ───────────────────────────────────────────────────
@app.route('/')
def home():
    return jsonify({'status': 'Sign Language API running', 'models': list(models_store.keys())})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data  = request.json
        mode  = data.get('mode', 'alphabets')   # 'alphabets' or 'digits'
        img_b64 = data['image']

        if mode not in models_store:
            return jsonify({'success': False, 'error': f'Unknown mode: {mode}'}), 400

        # Decode base64 image
        img_bytes = base64.b64decode(img_b64.split(',')[1])
        np_arr    = np.frombuffer(img_bytes, np.uint8)
        img       = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Preprocess
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img / 255.0).astype(np.float32)
        img = np.expand_dims(img, axis=0)

        # Predict with selected model.
        # Calling the model directly (rather than .predict()) skips the
        # overhead Keras adds for building a batch/data-pipeline on every
        # call -- for single-image, low-latency serving this is noticeably
        # faster.
        preds      = models_store[mode](img, training=False).numpy()
        pred_idx   = int(np.argmax(preds))
        confidence = float(preds[0][pred_idx])
        label      = CLASS_NAMES[mode][pred_idx]

        return jsonify({
            'success':    True,
            'label':      label,
            'confidence': round(confidence * 100, 1),
            'mode':       mode
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/models', methods=['GET'])
def get_models():
    return jsonify({
        'available': list(models_store.keys()),
        'classes':   CLASS_NAMES
    })

if __name__ == '__main__':
    # threaded=True lets Flask's dev server handle a new /predict request
    # while a previous one is still finishing, instead of queueing them
    # one-at-a-time -- important since the frontend polls every ~300ms.
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)