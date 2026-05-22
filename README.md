# Speech Analysis System

A web-based speech analysis system that classifies speaker **gender**, **age group**, and **voice intensity** using 1D CNN deep learning models trained on MFCC audio features.

## Features

- **Gender classification** — female / male
- **Age group classification** — young / middle / old
- **Voice intensity classification** — quiet / loud
- **Microphone recording** — Start/Stop button controlled
- **File upload** — WAV and MP3 support with automatic resampling
- **Session history** — last 50 analyses logged in the UI

## Tech Stack

| Area | Libraries |
|---|---|
| Deep Learning | TensorFlow 2.15, Keras |
| Audio Processing | Librosa 0.11, SoundDevice 0.5 |
| Web Interface | Streamlit 1.41 |
| ML Utilities | scikit-learn, NumPy |
| Visualization | Matplotlib, Seaborn |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

- **Microphone**: click **Start Recording**, speak, then click **Stop & Analyze**
- **File**: upload a WAV or MP3 file and click **Analyze File**

Results show predicted class and confidence for each of the three classifiers.

## Model Training

Each classifier is trained independently. Pre-trained models are included in `models/`.

| Script | Dataset Required |
|---|---|
| `train_gender.py` | `datasets/audio_dataset/females/`, `.../males/` |
| `train_age.py` | `datasets/final_age_dataset/young/`, `.../middle/`, `.../old/` |
| `train_intensity_avid.py` | `datasets/AVID/` (AVID SENT corpus) |

```bash
python train_gender.py
python train_age.py
python train_intensity_avid.py
```

Trained models are saved to `models/`. Confusion matrices and metrics plots are saved to `plots/`. Extracted features are cached in `cache/` to speed up retraining.

## Model Architecture

All three models use a 1D CNN:

- `Conv1D` → `BatchNormalization` → `MaxPooling1D` (×2–3 blocks)
- `GlobalAveragePooling1D`
- `Dense` + `Dropout`
- `Softmax` output

**Feature config:**

| Model | MFCCs | Delta | Max Frames |
|---|---|---|---|
| Gender | 40 | No | 200 |
| Age | 40 | No | 200 |
| Intensity | 120 (40 x3 with delta + delta2) | Yes | 100 |

Training uses Adam optimizer with categorical crossentropy loss and early stopping (patience = 12).
