
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Fake News Detection | AI Analytics",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Use paths relative to app.py
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "news_model.pkl"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
FAKE_PATH = BASE_DIR / "data" / "Fake.csv"
TRUE_PATH = BASE_DIR / "data" / "True.csv"


# =====================================================
# CUSTOM STYLING
# =====================================================

st.markdown("""
<style>
.main-title {
    font-size: 2.4rem;
    font-weight: 750;
    letter-spacing: -1px;
}
.subtitle {
    color: #64748b;
    font-size: 1rem;
}
div[data-testid="stMetric"] {
    background-color: rgba(128,128,128,0.08);
    padding: 16px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.18);
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD MODEL AND METRICS
# =====================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.is_file():
        raise FileNotFoundError(
            f"Metrics file not found: {METRICS_PATH}"
        )

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_dataset():
    if not FAKE_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {FAKE_PATH}")

    if not TRUE_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {TRUE_PATH}")

    fake_df = pd.read_csv(FAKE_PATH)
    true_df = pd.read_csv(TRUE_PATH)

    fake_df["label"] = 0
    true_df["label"] = 1

    df = pd.concat([fake_df, true_df], ignore_index=True)

    for column in ["text", "title", "subject"]:
        if column not in df.columns:
            df[column] = ""

    df["text"] = df["text"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["subject"] = df["subject"].fillna("Unknown").astype(str)

    df["label_name"] = df["label"].map({0: "Fake", 1: "Real"})
    df["text_length"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()

    return df.drop_duplicates(subset=["text"]).reset_index(drop=True)


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown("## 🔎 NewsGuard AI")
    st.caption("Fake News Detection & Analytics")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Home", "EDA Dashboard", "Predict News", "Model Performance"]
    )

    st.divider()
    st.caption("NLP + Machine Learning")
    st.caption("TF-IDF Text Classification")
    st.caption("Version 1.0")


# =====================================================
# LOAD PROJECT FILES
# =====================================================

try:
    model = load_model()
    metrics = load_metrics()

except Exception as error:
    st.error("Unable to load the model or metrics.")
    st.code(f"{type(error).__name__}: {error}")

    st.info(
        "Check that models/news_model.pkl and "
        "models/metrics.json are committed to GitHub "
        "inside the models folder."
    )
    st.stop()


# =====================================================
# HOME
# =====================================================

if page == "Home":

    st.markdown(
        '<div class="main-title">🔎 Fake News Detection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">An NLP-powered application '
        'that classifies news text as Fake or Real.</div>',
        unsafe_allow_html=True
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Test Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    c2.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
    c3.metric("Recall", f"{metrics['recall'] * 100:.2f}%")
    c4.metric("F1-Score", f"{metrics['f1_score'] * 100:.2f}%")

    st.subheader("Project Overview")

    left, right = st.columns(2)

    with left:
        st.markdown("""
        This application uses Natural Language Processing and
        supervised machine learning to classify news text.

        **Workflow**
        1. Load labelled news data.
        2. Convert text into numerical features.
        3. Train and evaluate a machine learning pipeline.
        4. Load the saved model.
        5. Predict the class of new text.
        """)

    with right:
        st.markdown("""
        **Technology Stack**
        - Python
        - Pandas
        - Scikit-learn
        - TF-IDF
        - Logistic Regression
        - Streamlit
        - Matplotlib and Seaborn
        """)

    st.warning(
        "This is an experimental text classifier, not a fact-checking "
        "service. Predictions may be wrong, particularly for recent "
        "events, satire, and unfamiliar topics."
    )


# =====================================================
# EDA DASHBOARD
# =====================================================

elif page == "EDA Dashboard":

    st.title("📊 Exploratory Data Analysis")

    try:
        df = load_dataset()
    except Exception as error:
        st.error("Could not load the datasets.")
        st.code(f"{type(error).__name__}: {error}")
        st.info("Check data/Fake.csv and data/True.csv in GitHub.")
        st.stop()

    fake_count = int((df["label"] == 0).sum())
    real_count = int((df["label"] == 1).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Unique Articles", f"{len(df):,}")
    c2.metric("Fake Articles", f"{fake_count:,}")
    c3.metric("Real Articles", f"{real_count:,}")

    left, right = st.columns(2)

    with left:
        st.subheader("News Class Distribution")
        counts = df["label_name"].value_counts().reindex(
            ["Fake", "Real"], fill_value=0
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("News Class")
        ax.set_ylabel("Number of Articles")
        ax.tick_params(axis="x", rotation=0)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Article Length Distribution")
        plot_df = df[
            df["text_length"] <= df["text_length"].quantile(0.98)
        ]

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(
            data=plot_df,
            x="text_length",
            hue="label_name",
            bins=35,
            ax=ax
        )
        ax.set_xlabel("Characters per Article")
        ax.set_ylabel("Number of Articles")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    left, right = st.columns(2)

    with left:
        st.subheader("Articles by Subject")
        subject_counts = (
            df.groupby(["subject", "label_name"])
            .size()
            .unstack(fill_value=0)
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        subject_counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("Subject")
        ax.set_ylabel("Number of Articles")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Average Article Length")
        avg_length = df.groupby("label_name")["word_count"].mean()

        fig, ax = plt.subplots(figsize=(7, 4))
        avg_length.plot(kind="bar", ax=ax)
        ax.set_ylabel("Average Word Count")
        ax.tick_params(axis="x", rotation=0)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Most Frequent Words")

    selected_class = st.selectbox(
        "Select news class", ["All", "Fake", "Real"]
    )

    word_df = df if selected_class == "All" else df[
        df["label_name"] == selected_class
    ]

    sample_texts = word_df["text"].head(12000).tolist()

    if sample_texts:
        vectorizer = CountVectorizer(
            stop_words="english",
            max_features=20,
            token_pattern=r"(?u)\b[a-zA-Z]{3,}\b"
        )

        try:
            matrix = vectorizer.fit_transform(sample_texts)
            frequencies = matrix.sum(axis=0).A1

            word_freq = pd.DataFrame({
                "Word": vectorizer.get_feature_names_out(),
                "Frequency": frequencies
            }).sort_values("Frequency", ascending=False)

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(data=word_freq, x="Frequency", y="Word", ax=ax)
            ax.set_title(f"Frequent Words — {selected_class} News")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        except ValueError:
            st.info("Not enough usable text to display word frequencies.")

    with st.expander("Preview dataset"):
        st.dataframe(
            df[["title", "subject", "label_name", "text_length"]].head(100),
            width="stretch",
            hide_index=True
        )


# =====================================================
# PREDICT NEWS
# =====================================================

elif page == "Predict News":

    st.title("🧠 Fake / Real News Prediction")

    news_text = st.text_area(
        "Enter a news headline or article",
        height=220,
        placeholder="Paste news text here..."
    )

    predict_button = st.button(
        "Analyze News",
        type="primary",
        width="stretch"
    )

    if predict_button:
        if not news_text.strip():
            st.warning("Please enter some news text first.")

        elif len(news_text.strip()) < 20:
            st.warning("Please provide at least 20 characters.")

        else:
            try:
                prediction = int(model.predict([news_text])[0])

                probabilities = None
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba([news_text])[0]

                st.divider()
                st.subheader("Prediction Result")

                if prediction == 0:
                    st.error("Predicted Class: FAKE NEWS")
                else:
                    st.success("Predicted Class: REAL NEWS")

                if probabilities is not None:
                    class_scores = {
                        int(label): float(score)
                        for label, score in zip(model.classes_, probabilities)
                    }

                    fake_score = class_scores.get(0, 0.0)
                    real_score = class_scores.get(1, 0.0)

                    c1, c2 = st.columns(2)
                    c1.metric("Fake-class model score", f"{fake_score * 100:.2f}%")
                    c2.metric("Real-class model score", f"{real_score * 100:.2f}%")

                    st.progress(
                        max(0.0, min(1.0, real_score)),
                        text="Model score for the Real class"
                    )

                st.info(
                    "Model scores are not proof that a news story is "
                    "factually true or false."
                )

                with st.expander("View submitted text"):
                    st.write(news_text)

            except Exception as error:
                st.error("Prediction failed.")
                st.code(f"{type(error).__name__}: {error}")


# =====================================================
# MODEL PERFORMANCE
# =====================================================

elif page == "Model Performance":

    st.title("📈 Model Performance")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    c2.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
    c3.metric("Recall", f"{metrics['recall'] * 100:.2f}%")
    c4.metric("F1-Score", f"{metrics['f1_score'] * 100:.2f}%")

    st.divider()
    st.subheader("Confusion Matrix")

    cm = metrics.get("confusion_matrix")

    if cm is not None and len(cm) == 2:
        fig, ax = plt.subplots(figsize=(6, 4))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Fake", "Real"],
            yticklabels=["Fake", "Real"],
            ax=ax
        )

        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Confusion matrix is not available in metrics.json.")

    st.subheader("Classification Report")

    report = metrics.get("classification_report", {})
    report_rows = []

    for name in ["Fake", "Real", "macro avg", "weighted avg"]:
        if name in report:
            item = report[name]
            report_rows.append({
                "Class": name,
                "Precision": item.get("precision", 0),
                "Recall": item.get("recall", 0),
                "F1-Score": item.get("f1-score", 0),
                "Support": item.get("support", 0)
            })

    if report_rows:
        report_df = pd.DataFrame(report_rows)
        st.dataframe(
            report_df.style.format({
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1-Score": "{:.4f}",
                "Support": "{:.0f}"
            }),
            width="stretch",
            hide_index=True
        )

    st.subheader("Metric Definitions")

    with st.expander("What do these metrics mean?"):
        st.markdown("""
        - **Accuracy:** Fraction of all test predictions that were correct.
        - **Precision:** How many predictions for a class were correct.
        - **Recall:** How many actual items in a class were identified.
        - **F1-score:** Harmonic mean of precision and recall.
        - **Confusion matrix:** Counts of correct and incorrect predictions.
        """)

    if "training_samples" in metrics and "testing_samples" in metrics:
        st.caption(
            f"Training samples: {metrics['training_samples']:,} | "
            f"Test samples: {metrics['testing_samples']:,}"
        )

    st.warning(
        "Metrics reflect performance on the recorded test split and "
        "do not guarantee accuracy on future or unseen news."
    )
