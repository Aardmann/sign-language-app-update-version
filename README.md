Using js51828829@gmail.com for supabase login. 
Usual password to login.
Encrypt for database: xadnyw-tehkyf-0nuBqi
# Sign Language Detection 

A real-time American Sign Language (ASL) detection web app that uses a webcam and deep learning to recognise hand signs — supporting both **alphabets (A–Z)** and **numbers (0–9)**. Built with Python, TensorFlow, Flask, and plain HTML/JS.

---

## Table of Contents

- [How the System Works](#how-the-system-works)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start — Running Locally](#quick-start--running-locally)
- [Building the Models Yourself](#building-the-models-yourself)
  - [Alphabets Model](#alphabets-model-a--z)
  - [Numbers Model](#numbers-model-0--9)
- [Web App — Backend & Frontend](#web-app--backend--frontend)
- [Switching Between Models](#switching-between-models)
- [Deploying Online](#deploying-online)
- [Troubleshooting](#troubleshooting)
- [Libraries Reference](#libraries-reference)

---

## How the System Works

```
User opens browser → clicks Start Camera
         ↓
Browser captures webcam frame every 300ms
         ↓
Frame is cropped to the hand ROI box
         ↓
Frame sent as Base64 image to Flask backend
         ↓
Flask decodes image → resizes to 64×64 → runs CNN model
         ↓
Model returns predicted sign + confidence %
         ↓
Browser displays letter/number on screen
         ↓
Same sign held for ~20 frames → added to output text
```

### Architecture Overview

```
sign-language-app/
├── frontend/          ← HTML page the user sees in browser
│   └── index.html
└── backend/           ← Python server that runs the AI models
    ├── app.py
    ├── requirements.txt
    ├── sign_language_model.h5   ← Alphabets model (A–Z)
    └── sign_digits_model.h5     ← Numbers model (0–9)
```

The **frontend** (HTML/JS) handles the camera, UI, and displaying results.  
The **backend** (Flask/Python) loads both trained CNN models and runs predictions.  
They communicate over HTTP — the frontend posts a camera frame, the backend replies with a prediction.

---

## Project Structure

```
sign-language-app/
│
├── backend/
│   ├── app.py                    ← Flask API server
│   ├── requirements.txt          ← Python dependencies
│   ├── sign_language_model.h5    ← Trained alphabets model
│   └── sign_digits_model.h5      ← Trained numbers model
│
├── frontend/
│   └── index.html                ← Web interface
│
└── notebooks/
    ├── SignLanguageMl.ipynb 
    ├── 02_camera_detection.ipynb
    └── signLanguageNum.ipynb
```

> **Note:** The `sign_env/` virtual environment folder lives outside this project folder. It only stores Python packages — not your project files. 
> Optional to create this but necessary if your python version is above 3.12

---

## Prerequisites

Make sure the following are installed on your machine before starting.

### 1. Python 3.9 or higher

Check if you have it:

```bash
python --version
```

Download from [python.org](https://www.python.org/downloads/) if needed.

### 2. pip (Python package manager)

Usually comes with Python. Check:

```bash
pip --version
```

### 3. A webcam

Required for live camera detection in the web app.

---

## Quick Start

Follow these steps exactly in order to run the full app on your machine.

### Step 1 — Clone or download the project

```bash
# If using git
git clone https://github.com/your-username/sign-language-app.git
cd sign-language-app

# Or just download and unzip the project folder
```

### Step 2 — Create a virtual environment

```bash
python -m venv sign_env
```

### Step 3 — Activate the virtual environment

```bash
# Mac / Linux
source sign_env/bin/activate

# Windows
sign_env\Scripts\activate
```

You should see `(sign_env)` at the start of your terminal line.

### Step 4 — Install backend dependencies

```bash
cd backend
pip install flask flask-cors tensorflow numpy opencv-python-headless gunicorn
```

### Step 5 — Make sure both model files are in the backend folder

```
backend/
├── app.py
├── requirements.txt
├── sign_language_model.h5 
└── sign_digits_model.h5
```

If you have not trained the models yet, see [Building the Models Yourself](#building-the-models-yourself) below.

### Step 6 — Start the Flask backend

```bash
# Make sure you are inside the backend/ folder
python app.py
```

You should see:

```
Loading models...
Both models loaded!
 * Running on http://localhos:5000 or
 * Running on http://0.0.0.0:5000
```

**Leave this terminal open.**

### Step 7 — Open a second terminal and serve the frontend (Not necessary just open the html file an it'll connect automatically)

```bash
# Activate venv again in this new terminal
source sign_env/bin/activate      # Mac/Linux
sign_env\Scripts\activate         # Windows

cd sign-language-app/frontend
python -m http.server 3000
```

### Step 8 — Open the app in your browser

```
http://localhost:3000
```

Use the **Alphabets / Numbers** toggle in the header to switch models.  
Click **Start Camera**, place your hand in the dashed box, and start signing.

---

## Building the Models Yourself

You need to train two separate CNN models — one for alphabets and one for numbers. Both are built in **JupyterLab**.

### Setting Up JupyterLab

**Install JupyterLab and all training dependencies:**

```bash
source sign_env/bin/activate   # activate venv first like I said for python versions above 3.12

pip install jupyterlab notebook numpy pandas matplotlib seaborn \
            opencv-python mediapipe tensorflow keras scikit-learn \
            pillow tqdm
```

**Launch JupyterLab:**

```bash
cd sign-language-app
jupyter lab
```

JupyterLab opens in your browser at `http://localhost:8888`.  
Use the **file browser** on the left to navigate, and the **Launcher** to create new notebooks.

> **Important:** Always run notebook cells from top to bottom. If you restart the kernel, re-run Cell 1 (imports) before anything else. Use **Kernel → Restart Kernel and Run All Cells** to run everything cleanly.

---

### Alphabets Model (A – Z)

**Dataset:** ASL Alphabet by Akash — [kaggle.com/datasets/grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

- 87,000 images · 200×200px · 29 classes (A–Z + SPACE, DELETE, NOTHING)
- Download and save the ZIP anywhere on your machine - preferably in the directory you're working in

**Create a new notebook called `02_train_alphabets.ipynb` and add these cells:**

#### Cell 1 — Imports

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, zipfile, warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

print("TensorFlow:", tf.__version__)
print("All imports successful!")
```

#### Cell 2 — Extract ZIP

```python
zip_path   = "asl_alphabet.zip"   # rename to match the file you downloaded
extract_to = "data/alphabets/"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("Extracted!")
for item in os.listdir(extract_to):
    print(" -", item)
```

#### Cell 3 — Verify Structure

```python
TRAIN_DIR = "data/alphabets/asl_alphabet_train/asl_alphabet_train"
TEST_DIR  = "data/alphabets/asl_alphabet_test/asl_alphabet_test"

classes = sorted(os.listdir(TRAIN_DIR))
print(f"Classes ({len(classes)}): {classes}")

for cls in classes:
    count = len(os.listdir(os.path.join(TRAIN_DIR, cls)))
    print(f"  {cls:10s} → {count} images")
```

#### Cell 4 — Data Generators

```python
IMG_SIZE    = 64
BATCH_SIZE  = 32
NUM_CLASSES = 29

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.15,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False   # Never flip sign language images
)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
print("Class mapping:", train_gen.class_indices)
```

#### Cell 5 — Build CNN Model

```python
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', padding='same',
                  input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()
```

#### Cell 6 — Train

```python
early_stop = callbacks.EarlyStopping(
    monitor='val_accuracy', patience=5,
    restore_best_weights=True, verbose=1
)
checkpoint = callbacks.ModelCheckpoint(
    'sign_language_model.h5',
    monitor='val_accuracy', save_best_only=True, verbose=1
)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5,
    patience=3, min_lr=1e-6, verbose=1
)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

print("Done! Model saved to sign_language_model.h5") ## This can take upto several hours, go chill just make sure your machine screen is not off
```

#### Cell 7 — Plot Results

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['accuracy'],     label='Train')
ax1.plot(history.history['val_accuracy'], label='Val')
ax1.set_title('Accuracy')
ax1.legend()
ax1.grid(True)

ax2.plot(history.history['loss'],     label='Train')
ax2.plot(history.history['val_loss'], label='Val')
ax2.set_title('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```

#### Cell 8 — Evaluate

```python
val_gen.reset()
y_pred = np.argmax(model.predict(val_gen), axis=1)
y_true = val_gen.classes

print(classification_report(
    y_true, y_pred,
    target_names=[idx_to_class[i] for i in range(NUM_CLASSES)]
))
```

After training, **copy `sign_language_model.h5` into your `backend/` folder**.

---

### Numbers Model (0 – 9)

**Dataset:** Sign Language for Numbers by Muhammad Khalid — [kaggle.com/datasets/muhammadkhalid/sign-language-for-numbers](https://www.kaggle.com/datasets/muhammadkhalid/sign-language-for-numbers)

- Folder-based dataset with 10 classes (0–9) · colour images
- Download and save the ZIP anywhere on your machine - preferably in the directory you're working in

**Create a new notebook called `03_train_digits.ipynb` and add these cells:**

#### Cell 1 — Imports

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, zipfile, warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

print("TensorFlow:", tf.__version__)
print("All imports successful!")
```

#### Cell 2 — Extract ZIP

```python
zip_path   = "sign_numbers.zip"   # rename to match the file you downloaded and make sure you don't have two zip files in the directory with the same names
extract_to = "data/digits/"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("Extracted!")
for root, dirs, files in os.walk(extract_to):
    level = root.replace(extract_to, '').count(os.sep)
    if level < 3:
        print(' ' * 2 * level + os.path.basename(root) + '/')
```

#### Cell 3 — Verify Structure

```python
# Update this path based on what Cell 2 printed
TRAIN_DIR   = "data/digits/Sign Language for Numbers"
IMG_SIZE    = 64
BATCH_SIZE  = 32
NUM_CLASSES = 11

classes = sorted(os.listdir(TRAIN_DIR))
print(f"Classes ({len(classes)}): {classes}")

for cls in classes:
    cls_path = os.path.join(TRAIN_DIR, cls)
    if os.path.isdir(cls_path):
        count = len(os.listdir(cls_path))
        print(f"  Class {cls:3s} → {count} images")
```

#### Cell 4 — Data Generators

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.15,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    horizontal_flip=False   # Never flip — digit signs are direction-sensitive
)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
print("Class mapping:", train_gen.class_indices)
```

#### Cell 5 — Build CNN Model

```python
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', padding='same',
                  input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()
```

#### Cell 6 — Train

```python
early_stop = callbacks.EarlyStopping(
    monitor='val_accuracy', patience=7,
    restore_best_weights=True, verbose=1
)
checkpoint = callbacks.ModelCheckpoint(
    'sign_digits_model.h5',
    monitor='val_accuracy', save_best_only=True, verbose=1
)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5,
    patience=3, min_lr=1e-6, verbose=1
)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,    # more epochs since dataset is smaller
    callbacks=[early_stop, checkpoint, reduce_lr]
)

print("Done! Model saved to sign_digits_model.h5") # Should be less than 2 hours, if not there's something wrong with your pc
```

#### Cell 7 — Plot Results

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['accuracy'],     label='Train')
ax1.plot(history.history['val_accuracy'], label='Val')
ax1.set_title('Accuracy')
ax1.legend()
ax1.grid(True)

ax2.plot(history.history['loss'],     label='Train')
ax2.plot(history.history['val_loss'], label='Val')
ax2.set_title('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```

#### Cell 8 — Evaluate

```python
val_gen.reset()
y_pred = np.argmax(model.predict(val_gen), axis=1)
y_true = val_gen.classes

print(classification_report(
    y_true, y_pred,
    target_names=[idx_to_class[i] for i in range(NUM_CLASSES)]
))
```

After training, **copy `sign_digits_model.h5` into your `backend/` folder**.

---

## Web App — Backend & Frontend -- run venv first

### Backend (`backend/app.py`)

The Flask server loads both models at startup and exposes a single `/predict` endpoint.

**How it works:**

1. Receives a POST request with a Base64-encoded image and a `mode` field (`"alphabets"` or `"digits"`)
2. Decodes the image using OpenCV
3. Resizes it to 64×64 pixels and normalises pixel values to `[0, 1]`
4. Passes it through the selected CNN model
5. Returns the predicted label and confidence percentage as JSON

**Run the backend:**

```bash
cd backend
python app.py
```

**Test the backend is running:**

Open `http://localhost:5000` in your browser — you should see:

```json
{"models": ["alphabets", "digits"], "status": "Sign Language API running"}
```

### Frontend (`frontend/index.html`)

A single HTML file — no framework, no build step needed.

**How it works:**

1. Accesses the device webcam via `getUserMedia`
2. Every 300ms, draws a cropped frame onto a hidden canvas
3. Converts the canvas to a Base64 JPEG string
4. POSTs it to the Flask backend with the current mode
5. Displays the returned letter/number and confidence bar
6. Uses a **stable prediction counter** — only adds a sign to the output text after it has been predicted consistently for 20 consecutive frames (about 6 seconds)

**Serve the frontend:**

```bash
cd frontend
python -m http.server 3000
```

Open `http://localhost:3000` in your browser.

---

## Switching Between Models

Use the toggle in the top-right corner of the header:

| Toggle | What it does |
|--------|-------------|
| **Alphabets** | Switches to the A–Z model · blue theme · SPACE and DEL signs active |
| **Numbers** | Switches to the 0–9 model · orange theme · no special signs |

When you switch modes, the detection state resets automatically so stale predictions from the previous mode do not carry over.


## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Run `pip install flask flask-cors tensorflow numpy opencv-python-headless` |
| `Address already in use` | Another process is using port 5000 — change `port=5000` in `app.py` to `5001` and update `API_URL` in `index.html` to match |
| Camera access denied | Make sure you open `http://localhost:3000` and **not** `file://` — browsers block camera on `file://` |
| CORS error in browser console | Make sure `flask-cors` is installed and `CORS(app)` is in `app.py` |
| Model not found error | Make sure both `.h5` files are inside the `backend/` folder |
| Low prediction accuracy | Improve lighting, keep hand fully inside the dashed ROI box, avoid busy backgrounds |
| API warning banner shows | The Flask backend is not running — start it with `python app.py` in the `backend/` folder |
| `NameError: plt is not defined` | You skipped the imports cell — go back and run Cell 1 first |
| Jupyter kernel died mid-training | Restart kernel, run Cell 1, then skip straight to Cell 6 to resume from the saved checkpoint |

---

## Libraries Reference

| Library | Purpose |
|---------|---------|
| `tensorflow` / `keras` | Builds and trains the CNN models |
| `numpy` | Handles image arrays and numerical operations |
| `opencv-python` | Reads and preprocesses camera frames |
| `opencv-python-headless` | Lightweight OpenCV for the server (no display needed) |
| `flask` | Python web server that serves the prediction API |
| `flask-cors` | Allows the browser frontend to call the Flask API |
| `gunicorn` | Production server for Flask when deployed on Render |
| `matplotlib` | Plots training accuracy and loss graphs |
| `seaborn` | Draws the confusion matrix heatmap |
| `scikit-learn` | Provides classification report and confusion matrix tools |
| `pillow` | Loads image files from disk |
| `tqdm` | Progress bars for dataset loading |
| `jupyterlab` | Browser-based IDE for writing and running training notebooks |

---

## Datasets

| Model | Dataset | Classes | Images | Link |
|-------|---------|---------|--------|------|
| Alphabets | ASL Alphabet (Akash) | 29 (A–Z + space/del/nothing) | 87,000 | [Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) |
| Numbers | Sign Language for Numbers (Muhammad Khalid) | 10 (0–9) | varies | [Kaggle](https://www.kaggle.com/datasets/muhammadkhalid/sign-language-for-numbers) |

---

## Quick Command Reference

```bash
# Activate virtual environment
source sign_env/bin/activate          # Mac/Linux
sign_env\Scripts\activate             # Windows

# Install all dependencies
pip install flask flask-cors tensorflow numpy opencv-python-headless \
            gunicorn jupyterlab notebook matplotlib seaborn scikit-learn \
            pillow tqdm mediapipe opencv-python

# Launch JupyterLab (for training)
jupyter lab

# Start Flask backend
cd backend && python app.py

# Serve frontend
cd frontend && python -m http.server 3000

# Open app in browser
# → http://localhost:3000
```

---

*Built with Python · TensorFlow · Flask · OpenCV · Font Awesome · JupyterLab*