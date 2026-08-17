"""
CCS3440 Artificial Intelligence Coursework | Group 02
Option C: Disease Risk Classification - SmartCare Hospital
Task 05b – Deep Learning Model (Bonus): Feedforward Neural Network

Trains a small MLP on the same leakage-free, 15-feature train/test split used
by every classical model in Task 05, using the same explicit TARGET_MAP
(Low=0, Medium=1, High=2) rather than an alphabetically-sorted encoder — the
same class-label discipline applied throughout this project, since a neural
network is exactly as vulnerable to a silent label-order bug as any other
model if class names are hardcoded instead of imported from source.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_fscore_support, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)

from feature_engineering import INV_TARGET_MAP

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def run_task05b():
    print("==================================================")
    print("  Task 05b: Deep Learning Model (Bonus) - MLP")
    print("==================================================")

    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")

    # Class names read from the SAME source-of-truth TARGET_MAP as every
    # other task — never hardcoded, since an alphabetical guess here would
    # reproduce the exact class-label bug fixed elsewhere in this project.
    target_names = [INV_TARGET_MAP[i] for i in range(3)]
    print("Verified class order:", list(enumerate(target_names)))
    print("Train:", X_train.shape, " Test:", X_test.shape)

    # Architecture: 2 hidden layers (64, 32 units, ReLU) with dropout
    # regularisation, sized for a dataset of this scale (800 training rows).
    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(3, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    trainable_params = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
    print("Trainable parameters:", trainable_params)

    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_split=0.2, epochs=150, batch_size=32,
        callbacks=[early_stop], verbose=0
    )
    epochs_run = len(history.history["loss"])
    print(f"Training stopped after {epochs_run} epochs (early stopping, patience=10)")

    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\nTest Accuracy: {acc:.4f} | Precision(macro): {prec:.4f} | "
          f"Recall(macro): {rec:.4f} | F1(macro): {f1:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

    p_c, r_c, f_c, s_c = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1, 2])
    per_class = {
        target_names[i]: {
            "precision": round(float(p_c[i]), 4),
            "recall": round(float(r_c[i]), 4),
            "f1": round(float(f_c[i]), 4),
            "support": int(s_c[i]),
        } for i in range(3)
    }

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss over epochs"); axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="Train Accuracy")
    axes[1].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy over epochs"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "nn_training_curves.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Neural Network — Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "nn_confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Persist model + results, consistent with every other task's outputs
    model.save(MODELS_DIR / "neural_network_model.keras")
    results = {
        "trainable_params": trainable_params,
        "epochs_run": epochs_run,
        "test_accuracy": round(float(acc), 4),
        "test_precision_macro": round(float(prec), 4),
        "test_recall_macro": round(float(rec), 4),
        "test_f1_macro": round(float(f1), 4),
        "per_class": per_class,
    }
    with open(REPORTS_DIR / "task05b_neural_network_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[SUCCESS] Task 05b completed! Model, plots, and results saved to: {REPORTS_DIR}\n")
    return results


if __name__ == "__main__":
    run_task05b()
