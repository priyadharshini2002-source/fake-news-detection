
import os
import json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

DATA_DIR = "data"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

FAKE_PATH = os.path.join(DATA_DIR, "Fake.csv")
TRUE_PATH = os.path.join(DATA_DIR, "True.csv")

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

print("\nLoading datasets...")

fake_df = pd.read_csv(FAKE_PATH)
true_df = pd.read_csv(TRUE_PATH)

fake_df["label"] = 0
true_df["label"] = 1

df = pd.concat([fake_df, true_df], ignore_index=True)

print(f"Fake news records: {len(fake_df):,}")
print(f"Real news records: {len(true_df):,}")
print(f"Total records: {len(df):,}")

# --------------------------------------------------
# 3. PREPROCESS DATA
# --------------------------------------------------

# Standard dataset contains a 'text' column.
# Use title as a fallback if text is unavailable.

if "text" in df.columns:
    df["text"] = df["text"].fillna("").astype(str)
elif "title" in df.columns:
    df["text"] = df["title"].fillna("").astype(str)
else:
    raise ValueError(
        "Dataset must contain either a 'text' or 'title' column."
    )

df["text"] = df["text"].str.strip()
df = df[df["text"].str.len() > 0].copy()

# Remove duplicate articles to reduce data leakage.
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

print(f"Records after cleaning: {len(df):,}")
print("\nClass distribution:")
print(df["label"].map({0: "Fake", 1: "Real"}).value_counts())

X = df["text"]
y = df["label"]

# --------------------------------------------------
# 4. TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {len(X_train):,}")
print(f"Testing samples: {len(X_test):,}")

# --------------------------------------------------
# 5. BUILD TF-IDF + LOGISTIC REGRESSION PIPELINE
# --------------------------------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

# --------------------------------------------------
# 6. TRAIN MODEL
# --------------------------------------------------

print("\nTraining model. Please wait...")

model.fit(X_train, y_train)

print("Training completed successfully!")

# --------------------------------------------------
# 7. EVALUATE MODEL
# --------------------------------------------------

y_pred = model.predict(X_test)

metrics = {
    "model": "TF-IDF + Logistic Regression",
    "training_samples": int(len(X_train)),
    "testing_samples": int(len(X_test)),
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(
        precision_score(y_test, y_pred, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, y_pred, zero_division=0)
    ),
    "f1_score": float(
        f1_score(y_test, y_pred, zero_division=0)
    ),
    "confusion_matrix": confusion_matrix(
        y_test, y_pred
    ).tolist(),
    "classification_report": classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Fake", "Real"],
        zero_division=0,
        output_dict=True
    )
}

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)
print(f"Accuracy : {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall   : {metrics['recall']:.4f}")
print(f"F1-Score : {metrics['f1_score']:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Fake", "Real"],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --------------------------------------------------
# 8. SAVE MODEL AND METRICS
# --------------------------------------------------

model_path = os.path.join(MODEL_DIR, "news_model.pkl")
metrics_path = os.path.join(MODEL_DIR, "metrics.json")

joblib.dump(model, model_path)

with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print("\nModel saved to:", model_path)
print("Metrics saved to:", metrics_path)
print("\nSTEP COMPLETED SUCCESSFULLY!")
