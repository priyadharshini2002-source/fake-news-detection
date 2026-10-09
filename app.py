
from pathlib import Path

import json
import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "news_model.pkl"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
FAKE_PATH = BASE_DIR / "data" / "Fake.csv"
TRUE_PATH = BASE_DIR / "data" / "True.csv"

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide",
)

# --------------------------------------------------
# LOAD MODEL AND METRICS
# --------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return {}
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# --------------------------------------------------
# LOAD DATASETS ONLY WHEN AVAILABLE
# --------------------------------------------------
@st.cache_data
def load_dataset():
    if not FAKE_PATH.exists() or not TRUE_PATH.exists():
        return None

    fake_df = pd.read_csv(FAKE_PATH)
    true_df = pd.read_csv(TRUE_PATH)

    fake_df["label"] = "Fake"
    true_df["label"] = "True"

    return pd.concat(
        [fake_df, true_df],
        ignore_index=True,
    )


# --------------------------------------------------
# APP HEADER
# --------------------------------------------------
st.title("📰 Fake News Detection System")
st.write(
    "An NLP and Machine Learning application that "
    "classifies news text as potentially Fake or Real."
)

try:
    model = load_model()
    metrics = load_metrics()
except Exception as error:
    st.error("Could not load the trained model.")
    st.code(f"{type(error).__name__}: {error}")
    st.info(
        "Ensure models/news_model.pkl is committed to "
        "your GitHub repository."
    )
    st.stop()

# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Predict News",
        "EDA Dashboard",
        "Model Performance",
    ],
)

# --------------------------------------------------
# HOME
# --------------------------------------------------
if page == "Home":
    st.subheader("Welcome")
    st.markdown(
        """
        This project uses Natural Language Processing (NLP)
        and a trained machine learning pipeline to analyse
        news text.

        **Features**
        - Fake or Real news prediction
        - Confidence score when supported by the model
        - Exploratory Data Analysis dashboard
        - Model performance metrics
        """
    )

    st.warning(
        "Predictions are model estimates, not a guarantee "
        "that a news article is true or false. Verify important "
        "claims using reliable sources."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model", type(model).__name__)
    with col2:
        st.metric(
            "Dataset",
            "Available" if (
                FAKE_PATH.exists() and TRUE_PATH.exists()
            ) else "Not uploaded",
        )

# --------------------------------------------------
# PREDICT NEWS
# --------------------------------------------------
elif page == "Predict News":
    st.subheader("🔍 Analyse a News Article")
    st.write("Paste the headline or article text below.")

    news_text = st.text_area(
        "News text",
        height=220,
        placeholder="Enter a news headline or article here...",
    )

    if st.button("Analyse News", type="primary"):
        if not news_text.strip():
            st.warning("Please enter some news text first.")
        else:
            try:
                prediction = model.predict([news_text])[0]
                label = str(prediction).strip()

                st.subheader("Prediction")

                normalised = label.lower()
                if normalised in ("fake", "false", "0"):
                    st.error("⚠️ Prediction: Fake News")
                elif normalised in ("true", "real", "1"):
                    st.success("✅ Prediction: Real News")
                else:
                    st.info(f"Model prediction: {label}")

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(
                        [news_text]
                    )[0]
                    confidence = float(max(probabilities))
                    st.metric(
                        "Model confidence",
                        f"{confidence:.2%}",
                    )
                    st.caption(
                        "Confidence is the model's estimated "
                        "probability, not proof of factual accuracy."
                    )

            except Exception as error:
                st.error("Prediction failed.")
                st.code(f"{type(error).__name__}: {error}")

# --------------------------------------------------
# EDA DASHBOARD
# --------------------------------------------------
elif page == "EDA Dashboard":
    st.subheader("📊 Exploratory Data Analysis")

    dataset = load_dataset()

    if dataset is None:
        st.warning(
            "Dataset files are not available on the server. "
            "Upload data/Fake.csv and data/True.csv to GitHub "
            "to enable this dashboard. News prediction can "
            "still work without these CSV files."
        )
    else:
        st.metric("Total articles", f"{len(dataset):,}")

        st.subheader("Class Distribution")
        counts = dataset["label"].value_counts()
        st.bar_chart(counts)

        st.subheader("Dataset Preview")
        st.dataframe(dataset.head(20), width="stretch")

        if "text" in dataset.columns:
            dataset["word_count"] = (
                dataset["text"]
                .fillna("")
                .astype(str)
                .str.split()
                .str.len()
            )

            st.subheader("Article Word Count")
            st.bar_chart(
                dataset.groupby("label")["word_count"].mean()
            )

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------
elif page == "Model Performance":
    st.subheader("📈 Model Performance")

    if not metrics:
        st.warning(
            "metrics.json is unavailable. Add "
            "models/metrics.json to your GitHub repository."
        )
    else:
        metric_keys = [
            ("Accuracy", "accuracy"),
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("F1 Score", "f1_score"),
        ]

        columns = st.columns(4)
        for column, (title, key) in zip(columns, metric_keys):
            value = metrics.get(key)
            with column:
                if isinstance(value, (int, float)):
                    st.metric(title, f"{value:.2%}")
                else:
                    st.metric(title, "N/A")

        st.subheader("Additional Metrics")

        if "confusion_matrix" in metrics:
            st.write("Confusion Matrix")
            st.dataframe(
                pd.DataFrame(metrics["confusion_matrix"]),
                width="stretch",
            )

        if "classification_report" in metrics:
            st.write("Classification Report")
            report = metrics["classification_report"]

            if isinstance(report, dict):
                st.dataframe(
                    pd.DataFrame(report).transpose(),
                    width="stretch",
                )
            else:
                st.text(str(report))

        with st.expander("View raw metrics"):
            st.json(metrics)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.sidebar.divider()
st.sidebar.caption("Fake News Detection | NLP + Machine Learning")
