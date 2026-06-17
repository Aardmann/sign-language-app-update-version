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
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict with selected model
        preds      = models_store[mode].predict(img, verbose=0)
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
    app.run(debug=False, host='0.0.0.0', port=5000)