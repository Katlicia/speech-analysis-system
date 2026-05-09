"""
Voice intensity classification — AVID SENT + MFCC delta + 1D CNN
Classes : quiet (soft + normal)
          loud  (loud + veryloud)
Dataset : AVID/Repository 2/SENT/*.wav
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential                           # type: ignore
from tensorflow.keras.layers import (                                    # type: ignore
    Conv1D, MaxPooling1D, Dropout,
    Dense, BatchNormalization,
    GlobalAveragePooling1D,
)
from tensorflow.keras.utils import to_categorical                        # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore

from features import extract_mfcc

# ── Config ───────────────────────────────────────────────────────────────────
AVID_CSV    = os.path.join("AVID", "Repository 2", "Metadata_with_labels_SENT.csv")
AVID_AUDIO  = os.path.join("AVID", "Repository 2", "SENT")
MAX_LEN     = 100
EPOCHS      = 100
BATCH       = 32
CACHE_X     = "cache/avid_sent_X.npy"
CACHE_Y     = "cache/avid_sent_y.npy"

LABEL_MAP = {
    "soft":     "quiet",
    "normal":   "quiet",
    "loud":     "loud",
    "veryloud": "loud",
}

# ── Data loading ──────────────────────────────────────────────────────────────
def load_dataset():
    if os.path.exists(CACHE_X) and os.path.exists(CACHE_Y):
        print("Loading from cache...")
        return np.load(CACHE_X), np.load(CACHE_Y, allow_pickle=True)

    df = pd.read_csv(AVID_CSV)
    df["binary_label"] = df["Intensity_category_label"].map(LABEL_MAP)

    X, y = [], []
    ok, fail = 0, 0

    for _, row in df.iterrows():
        path = os.path.join(AVID_AUDIO, row["Filename"])
        mfcc = extract_mfcc(path, max_len=MAX_LEN, include_delta=True)
        if mfcc is not None:
            X.append(mfcc)
            y.append(row["binary_label"])
            ok += 1
        else:
            fail += 1

    if fail:
        print(f"  [warning] {fail} files could not be loaded")
    print(f"  Loaded: {ok} files")

    X, y = np.array(X), np.array(y)
    os.makedirs("cache", exist_ok=True)
    np.save(CACHE_X, X)
    np.save(CACHE_Y, y)
    return X, y

# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(input_shape):
    model = Sequential([
        Conv1D(32, 3, activation="relu", padding="same", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),

        Conv1D(64, 3, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.25),

        Conv1D(128, 3, activation="relu", padding="same"),
        BatchNormalization(),
        Dropout(0.3),

        GlobalAveragePooling1D(),

        Dense(64, activation="relu"),
        Dropout(0.4),
        Dense(2, activation="softmax"),
    ])
    return model

# ── Plots ─────────────────────────────────────────────────────────────────────
def save_plots(history, y_true, y_pred, classes, prefix="plots/intensity_avid"):
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Intensity Classification — AVID SENT", fontsize=14)

    axes[0, 0].plot(history.history["accuracy"],     label="Train")
    axes[0, 0].plot(history.history["val_accuracy"], label="Val")
    axes[0, 0].set_title("Accuracy"); axes[0, 0].legend(); axes[0, 0].grid(alpha=.3)

    axes[0, 1].plot(history.history["loss"],     label="Train")
    axes[0, 1].plot(history.history["val_loss"], label="Val")
    axes[0, 1].set_title("Loss"); axes[0, 1].legend(); axes[0, 1].grid(alpha=.3)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=axes[1, 0])
    axes[1, 0].set_title("Confusion Matrix")
    axes[1, 0].set_xlabel("Predicted"); axes[1, 0].set_ylabel("Actual")

    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(metrics))
    w = 0.3
    for i, cls in enumerate(classes):
        vals = [report[cls][m] for m in metrics]
        offset = (i - 0.5) * w
        axes[1, 1].bar(x + offset, vals, w, label=cls)
        for j, v in enumerate(vals):
            axes[1, 1].text(j + offset, v + .02, f"{v:.2f}", ha="center", fontsize=8)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(["Precision", "Recall", "F1"])
    axes[1, 1].set_ylim(0, 1.15)
    axes[1, 1].set_title("Class Metrics"); axes[1, 1].legend(); axes[1, 1].grid(axis="y", alpha=.3)

    plt.tight_layout()
    plt.savefig(f"{prefix}_results.png", dpi=150)
    plt.close()
    print(f"Plot saved -> {prefix}_results.png")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("INTENSITY (AVID SENT) — MFCC delta + 1D CNN")
    print("Classes: quiet (soft+normal)  /  loud (loud+veryloud)")
    print("=" * 55)

    print("\n[1/5] Loading data...")
    X, y = load_dataset()
    print(f"  Total: {len(X)}  |  Shape: {X.shape}")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    {u}: {c}")

    print("\n[2/5] Encoding labels & splitting...")
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_cat = to_categorical(y_enc)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y_enc
    )
    print(f"  Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")
    print(f"  Classes: {le.classes_}")

    print("\n[3/5] Building model...")
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
    ]

    print("\n[4/5] Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH,
        callbacks=callbacks,
        verbose=1,
    )

    print("\n[5/5] Evaluation...")
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = le.inverse_transform(np.argmax(model.predict(X_test, verbose=0), axis=1))
    y_true = le.inverse_transform(np.argmax(y_test, axis=1))

    print(classification_report(y_true, y_pred))
    print(f"Test accuracy: {acc * 100:.2f}%")

    os.makedirs("models", exist_ok=True)
    model.save("models/intensity_avid_model.h5")
    print("Model saved -> models/intensity_avid_model.h5")

    save_plots(history, y_true, y_pred, list(le.classes_))
