"""
Streamlit web interface for speech analysis.
Record from microphone or upload a WAV/MP3 — results shown on the same page.
"""

import io
import threading
import datetime
import librosa
import numpy as np
import sounddevice as sd
import streamlit as st
import tensorflow as tf

from features import extract_mfcc

SR = 16000

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
        "is_recording": False,
        "results":      {},
        "status":       "idle",   # idle | recording | analyzing | done
        "duration":     0.0,
        "log":          [],
    }

# Inference helpers

def predict(audio, model, cfg):
    feat = extract_mfcc(audio, max_len=cfg["max_len"], sr=SR, include_delta=cfg["delta"])
    if feat is None:
        return None, 0.0
    probs = model.predict(feat[np.newaxis, ...], verbose=0)[0]
    idx   = int(np.argmax(probs))
    return cfg["classes"][idx], float(probs[idx])

def run_inference(audio: np.ndarray, models: dict) -> tuple[dict, float]:
    duration = len(audio) / SR
    active  = {k: v for k, v in models.items() if v is not None}
    results = {}
    for key, model in active.items():
        label, conf = predict(audio, model, MODELS_CFG[key])
        if label:
            results[key] = (label, conf)
    return results, duration

def add_to_log(shared, results, duration, source="mic"):
    entry = (
        {"time": datetime.datetime.now().strftime("%H:%M:%S"),
         "source": source,
         "duration": f"{duration:.1f}s"}
        | {k: v[0] for k, v in results.items()}
    )
    shared["log"].insert(0, entry)
    shared["log"] = shared["log"][:50]

def show_results(results: dict):
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

# Recording thread

def record_and_analyze(models, shared):
    buf = []

    def callback(indata, *_):
        if shared["is_recording"]:
            buf.extend(indata[:, 0].tolist())

    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        blocksize=int(SR * 0.05), callback=callback):
        while shared["is_recording"]:
            threading.Event().wait(timeout=0.05)

    if not buf:
        shared["status"] = "idle"
        return

    shared["status"] = "analyzing"
    audio = np.array(buf, dtype="float32")
    results, duration  = run_inference(audio, models)
    shared["results"]  = results
    shared["duration"] = duration
    shared["status"]   = "done"

    if results:
        add_to_log(shared, results, duration, source="mic")

# Recording section (fragment for live updates)
@st.fragment(run_every=0.5)
def recording_section(shared):
    status = shared["status"]
    prev   = st.session_state.get("_prev_status")

    # Full rerun when analysis finishes so button state updates
    if prev == "analyzing" and status == "done":
        st.session_state["_prev_status"] = status
        st.rerun()
    st.session_state["_prev_status"] = status

    if status == "recording":
        st.success("Recording... speak into the microphone")
    elif status == "analyzing":
        st.info("Analyzing audio...")
    elif status == "done" and shared["results"]:
        st.success(f"Microphone — {shared['duration']:.1f}s recorded")
        show_results(shared["results"])
    elif status == "done":
        st.warning("No speech detected in recording")

def main():
    st.set_page_config(page_title="Speech Analysis", layout="centered")
    st.title("Speech Analysis")

    models = load_models()
    shared = get_shared()

    # Microphone
    st.subheader("Microphone")
    status = shared["status"]
    if status in ("idle", "done"):
        if st.button("Start Recording", type="primary", use_container_width=True):
            shared["is_recording"] = True
            shared["status"]       = "recording"
            shared["results"]      = {}
            threading.Thread(target=record_and_analyze, args=(models, shared), daemon=True).start()
            st.rerun()
    elif status == "recording":
        if st.button("Stop & Analyze", type="secondary", use_container_width=True):
            shared["is_recording"] = False
            st.rerun()
    else:
        st.button("Analyzing...", disabled=True, use_container_width=True)

    recording_section(shared)

    # File Upload
    st.divider()
    st.subheader("Upload File")

    uploaded = st.file_uploader(
        "WAV or MP3, any sample rate",
        type=["wav", "mp3"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        # Clear stale results when a different file is chosen
        if st.session_state.get("upload_name") != uploaded.name:
            st.session_state.pop("upload_results", None)
            st.session_state["upload_name"] = uploaded.name

        st.audio(uploaded)

        if st.button("Analyze File", type="primary", use_container_width=True):
            with st.spinner("Loading and analyzing audio..."):
                audio, file_sr = librosa.load(io.BytesIO(uploaded.getvalue()), sr=None, mono=True)
                if file_sr != SR:
                    audio = librosa.resample(audio, orig_sr=file_sr, target_sr=SR)
                results, duration = run_inference(audio, models)
            st.session_state["upload_results"] = (results, duration)
            if results:
                add_to_log(shared, results, duration, source=uploaded.name)

        if "upload_results" in st.session_state:
            results, duration = st.session_state["upload_results"]
            if results:
                st.success(f"File — {duration:.1f}s")
                show_results(results)
            else:
                st.warning("No speech detected in the file")
    else:
        st.session_state.pop("upload_results", None)
        st.session_state.pop("upload_name", None)

    # History
    if shared["log"]:
        st.divider()
        st.subheader("History")
        st.dataframe(shared["log"], use_container_width=True)

    st.caption(f"SR: {SR} Hz")

if __name__ == "__main__":
    main()
