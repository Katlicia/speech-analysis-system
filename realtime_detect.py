"""
Real-time speech analysis
Gender / Age Group / Voice Intensity
Microphone -> 4-second sliding window
"""

import sys
import datetime
import numpy as np
import sounddevice as sd
import tensorflow as tf

from features import extract_mfcc

# ── Config ───────────────────────────────────────────────────────────────────
SR          = 16000
CHUNK_SEC   = 4       # recording window in seconds
SILENCE_THR = 0.01    # RMS threshold — below this is treated as silence

MODELS_CFG = {
    "gender": {
        "path":    "models/gender_model.h5",
        "classes": ["female", "male"],
        "max_len": 200,
        "delta":   False,
        "label":   "Gender",
    },
    "age": {
        "path":    "models/age_model.h5",
        "classes": ["middle", "old", "young"],
        "max_len": 200,
        "delta":   False,
        "label":   "Age Group",
    },
    "intensity": {
        "path":    "models/intensity_avid_model.h5",
        "classes": ["loud", "quiet"],
        "max_len": 100,
        "delta":   True,
        "label":   "Intensity",
    },
}

# ── Model loading ─────────────────────────────────────────────────────────────
def load_models():
    models = {}
    for key, cfg in MODELS_CFG.items():
        try:
            models[key] = tf.keras.models.load_model(cfg["path"], compile=False)
            print(f"  + {cfg['label']} model loaded")
        except Exception as e:
            print(f"  x {cfg['label']} model failed: {e}")
            models[key] = None
    return models

# ── Inference ─────────────────────────────────────────────────────────────────
def predict(audio: np.ndarray, model, cfg: dict):
    feat = extract_mfcc(
        audio,
        max_len=cfg["max_len"],
        sr=SR,
        include_delta=cfg["delta"],
    )
    if feat is None:
        return None, 0.0

    x     = feat[np.newaxis, ...]          # (1, max_len, n_feat)
    probs = model.predict(x, verbose=0)[0]
    idx   = int(np.argmax(probs))
    return cfg["classes"][idx], float(probs[idx])

# ── Output ────────────────────────────────────────────────────────────────────
def print_results(results: dict):
    now = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"\n{'─' * 40}")
    print(f"  {now}")
    for key, cfg in MODELS_CFG.items():
        if results.get(key) is None:
            print(f"  {cfg['label']:<12}: —")
        else:
            label, conf = results[key]
            bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
            print(f"  {cfg['label']:<12}: {label:<16} {bar}  {conf*100:.0f}%")
    print(f"{'─' * 40}")

# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 40)
    print("  Real-time Speech Analysis")
    print("=" * 40)

    print("\nLoading models...")
    models = load_models()

    active = {k: v for k, v in models.items() if v is not None}
    if not active:
        print("No models loaded. Exiting.")
        sys.exit(1)

    print(f"\nMicrophone started — {CHUNK_SEC}s window")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            audio = sd.rec(
                int(CHUNK_SEC * SR),
                samplerate=SR,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            audio = audio[:, 0]   # (N,) mono

            rms = float(np.sqrt(np.mean(audio ** 2)))
            if rms < SILENCE_THR:
                print(f"  [{datetime.datetime.now().strftime('%H:%M:%S')}] No speech detected (RMS={rms:.4f})")
                continue

            results = {}
            for key, model in active.items():
                cfg = MODELS_CFG[key]
                label, conf = predict(audio, model, cfg)
                results[key] = (label, conf) if label else None

            print_results(results)

    except KeyboardInterrupt:
        print("\n\nStopped.")

if __name__ == "__main__":
    main()
