"""
Streamlit web interface for real-time speech analysis.
Background thread handles recording/inference; UI polls every 500ms.
"""

import threading
import datetime
import numpy as np
import sounddevice as sd
import streamlit as st
import tensorflow as tf

from features import extract_mfcc

# Config
SR          = 16000
CHUNK_SEC   = 4
SILENCE_THR = 0.01

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

@st.cache_resource
def load_models():
    models = {}
    for key, cfg in MODELS_CFG.items():
        try:
            models[key] = tf.keras.models.load_model(cfg["path"], compile=False)
        except Exception:
            models[key] = None
    return models

@st.cache_resource
def get_shared():
    return {
        "results": {},
        "status":  "starting",
        "updated": None,
        "log":     [],
    }

def predict(audio, model, cfg):
    feat = extract_mfcc(audio, max_len=cfg["max_len"], sr=SR, include_delta=cfg["delta"])
    if feat is None:
        return None, 0.0
    probs = model.predict(feat[np.newaxis, ...], verbose=0)[0]
    idx   = int(np.argmax(probs))
    return cfg["classes"][idx], float(probs[idx])

def recording_loop(models, shared):
    active = {k: v for k, v in models.items() if v is not None}
    buf    = []

    def callback(indata, *_):
        buf.extend(indata[:, 0].tolist())
        if len(buf) < SR * CHUNK_SEC:
            return

        audio_4s = np.array(buf[:SR * CHUNK_SEC])
        del buf[:SR * CHUNK_SEC]

        if float(np.sqrt(np.mean(audio_4s ** 2))) < SILENCE_THR:
            shared["status"]  = "silence"
            shared["updated"] = datetime.datetime.now()
            return

        results = {}
        for key, model in active.items():
            label, conf = predict(audio_4s, model, MODELS_CFG[key])
            if label:
                results[key] = (label, conf)

        if results:
            shared["results"] = results
            shared["status"]  = "active"
            shared["updated"] = datetime.datetime.now()
            entry = {"time": shared["updated"].strftime("%H:%M:%S")} | {
                k: v[0] for k, v in results.items()
            }
            shared["log"].insert(0, entry)
            shared["log"] = shared["log"][:50]

    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        blocksize=int(SR * 0.05), callback=callback):
        threading.Event().wait()

@st.fragment(run_every=0.5)
def main_panel(shared):
    status  = shared["status"]
    updated = shared["updated"]

    if status == "starting":
        st.info("Starting microphone...")
    elif status == "silence":
        st.warning("No speech detected")
    else:
        ts = updated.strftime("%H:%M:%S") if updated else ""
        st.success(f"Listening — last update: {ts}")

    st.divider()

    results = shared["results"]
    cols = st.columns(3)
    for i, (key, cfg) in enumerate(MODELS_CFG.items()):
        with cols[i]:
            st.subheader(cfg["label"])
            if key in results:
                label, conf = results[key]
                st.metric(label=cfg["label"], value=label, label_visibility="collapsed")
                st.progress(conf, text=f"{conf*100:.0f}%")
            else:
                st.metric(label=cfg["label"], value="—", label_visibility="collapsed")
                st.progress(0.0)

    st.divider()

    if shared["log"]:
        st.subheader("Log")
        st.dataframe(shared["log"], use_container_width=True)

def main():
    st.set_page_config(page_title="Speech Analysis", page_icon="🎙️", layout="centered")
    st.title("🎙️ Real-time Speech Analysis")

    models = load_models()
    shared = get_shared()

    if "thread_started" not in st.session_state:
        threading.Thread(target=recording_loop, args=(models, shared), daemon=True).start()
        st.session_state.thread_started = True

    main_panel(shared)
    st.caption(f"Window: {CHUNK_SEC}s  |  SR: {SR} Hz  |  Silence: {SILENCE_THR} RMS")

if __name__ == "__main__":
    main()
