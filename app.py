
import os
import json
import re
from collections import Counter

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

MODEL_PATH = "models/news_model.pkl"
METRICS_PATH = "models/metrics.json"
FAKE_PATH = "data/Fake.csv"
TRUE_PATH = "data/True.csv"


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
    .section-title {
        font-size: 1.4rem;
        font-weight: 650;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.18);
    }
</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD TRAINED MODEL AND METRICS
# =====================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. "
            "Run py train_model.py first."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError(
            f"Metrics not found: {METRICS_PATH}"
        )

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_dataset():
    fake_df = pd.read_csv(FAKE_PATH)
    true_df = pd.read_csv(TRUE_PATH)

    fake_df["label"] = 0
    true_df["label"] = 1

    df = pd.concat(
        [fake_df, true_df],
        ignore_index=True
    )

    df["text"] = df["text"].fillna("").astype(str)
    df["title"] = (
        df["title"].fillna("").astype(str)
        if "title" in df.columns
        else ""
    )

    df["subject"] = (
        df["subject"].fillna("Unknown").astype(str)
        if "subject" in df.columns
        else "Unknown"
    )

    df["label_name"] = df["label"].map({
        0: "Fake",
        1: "Real"
    })

    df["text_length"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()

    return df.drop_duplicates(subset=["text"]).reset_index(
        drop=True
    )


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown("## 🔎 NewsGuard AI")
    st.caption("Fake News Detection & Analytics")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Home",
            "EDA Dashboard",
            "Predict News",
            "Model Performance"
        ]
    )

    st.divider()
    st.caption("Machine Learning Project")
    st.caption("TF-IDF + Logistic Regression")
    st.caption("Version 1.0")


# =====================================================
# LOAD RESOURCES
# =====================================================

try:
    model = load_model()
    metrics = load_metrics()
except Exception as error:
    st.error(f"Unable to load project files: {error}")
    st.info(
        "Check that models/news_model.pkl and "
        "models/metrics.json exist."
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
        '<div class="subtitle">'
        'An NLP-powered machine learning application for '
        'classifying news articles as Fake or Real.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Test Accuracy",
        f"{metrics['accuracy'] * 100:.2f}%"
    )
    c2.metric(
        "Precision",
        f"{metrics['precision'] * 100:.2f}%"
    )
    c3.metric(
        "Recall",
        f"{metrics['recall'] * 100:.2f}%"
    )
    c4.metric(
        "F1-Score",
        f"{metrics['f1_score'] * 100:.2f}%"
    )

    st.markdown("### Project Overview")

    left, right = st.columns([1.25, 1])

    with left:
        st.markdown("""
        This application uses **Natural Language Processing (NLP)**
        and supervised machine learning to classify news text.

        **Workflow**
        1. Load labelled fake and real news articles.
        2. Convert text into numerical TF-IDF features.
        3. Train a Logistic Regression classifier.
        4. Evaluate the model on a held-out test set.
        5. Use the saved pipeline to classify new text.
        """)

    with right:
        st.markdown("#### Technology Stack")
        st.markdown("""
        - Python
        - Pandas and NumPy
        - Scikit-learn
        - TF-IDF text representation
        - Logistic Regression
        - Streamlit
        - Matplotlib and Seaborn
        """)

    st.divider()
    st.markdown("### How it works")

    step1, step2, step3 = st.columns(3)

    step1.markdown(
        "**01 · Input**\n\n"
        "Paste a news article or headline."
    )
    step2.markdown(
        "**02 · Analyze**\n\n"
        "The trained NLP pipeline processes the text."
    )
    step3.markdown(
        "**03 · Predict**\n\n"
        "View the predicted class and model score."
    )

    st.warning(
        "This is an experimental text classifier, not a factual "
        "verification service. Predictions may be wrong, especially "
        "for recent events, unfamiliar topics, satire, or edited text."
    )


# =====================================================
# EDA DASHBOARD
# =====================================================

elif page == "EDA Dashboard":

    st.title("📊 Exploratory Data Analysis")
    st.write(
        "Explore the dataset's class distribution, text length, "
        "subjects and frequent words."
    )

    try:
        df = load_dataset()
    except Exception as error:
        st.error(f"Could not load datasets: {error}")
        st.stop()

    fake_count = int((df["label"] == 0).sum())
    real_count = int((df["label"] == 1).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Unique Articles", f"{len(df):,}")
    c2.metric("Fake Articles", f"{fake_count:,}")
    c3.metric("Real Articles", f"{real_count:,}")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("News Class Distribution")

        counts = (
            df["label_name"]
            .value_counts()
            .reindex(["Fake", "Real"], fill_value=0)
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("News Class")
        ax.set_ylabel("Number of Articles")
        ax.set_title("Fake vs Real News")
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
            element="step",
            stat="count",
            common_norm=False,
            ax=ax
        )
        ax.set_xlabel("Characters per Article")
        ax.set_ylabel("Number of Articles")
        ax.set_title("Article Length by Class")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Articles by Subject")

        subject_counts = (
            df.groupby(["subject", "label_name"])
            .size()
            .unstack(fill_value=0)
        )

        subject_counts = subject_counts.sort_values(
            by=list(subject_counts.columns),
            ascending=False
        ).head(12)

        fig, ax = plt.subplots(figsize=(8, 5))
        subject_counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("Subject")
        ax.set_ylabel("Number of Articles")
        ax.set_title("News Subjects by Class")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Average Article Length")

        avg_length = (
            df.groupby("label_name")["word_count"]
            .mean()
            .reindex(["Fake", "Real"])
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        avg_length.plot(kind="bar", ax=ax)
        ax.set_xlabel("News Class")
        ax.set_ylabel("Average Word Count")
        ax.set_title("Average Words per Article")
        ax.tick_params(axis="x", rotation=0)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    st.subheader("Most Frequent Words")

    selected_class = st.selectbox(
        "Select news class",
        ["All", "Fake", "Real"]
    )

    word_df = df

    if selected_class != "All":
        word_df = df[df["label_name"] == selected_class]

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
            sns.barplot(
                data=word_freq,
                x="Frequency",
                y="Word",
                ax=ax
            )
            ax.set_title(f"Frequent Words — {selected_class} News")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        except ValueError:
            st.info("Not enough usable text to display word frequencies.")

    with st.expander("Preview dataset"):
        st.dataframe(
            df[["title", "subject", "label_name", "text_length"]].head(100),
            use_container_width=True
        )

    st.caption(
        "EDA uses the available labelled dataset. Patterns in this "
        "dataset may not represent news sources outside it."
    )


# =====================================================
# PREDICT NEWS
# =====================================================

elif page == "Predict News":

    st.title("🧠 Fake / Real News Prediction")
    st.write(
        "Enter a news headline or article below to obtain a "
        "model-generated classification."
    )

    news_text = st.text_area(
        "News headline or article",
        height=220,
        placeholder=(
            "Paste the news headline or article text here..."
        )
    )

    predict_button = st.button(
        "Analyze News",
        type="primary",
        use_container_width=True
    )

    if predict_button:

        if not news_text.strip():
            st.warning("Please enter some news text first.")

        elif len(news_text.strip()) < 20:
            st.warning(
                "Please provide more text for a more meaningful "
                "classification. Short text may be unreliable."
            )

        else:
            with st.spinner("Analyzing text..."):
                prediction = int(model.predict([news_text])[0])

                probabilities = None
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(
                        [news_text]
                    )[0]

            st.divider()
            st.subheader("Prediction Result")

            if prediction == 0:
                st.error("Predicted Class: FAKE NEWS")
            else:
                st.success("Predicted Class: REAL NEWS")

            if probabilities is not None:
                classes = list(model.classes_)
                class_scores = {
                    int(label): float(score)
                    for label, score in zip(classes, probabilities)
                }

                fake_score = class_scores.get(0, 0.0)
                real_score = class_scores.get(1, 0.0)

                c1, c2 = st.columns(2)

                c1.metric(
                    "Fake-class model score",
                    f"{fake_score * 100:.2f}%"
                )
                c2.metric(
                    "Real-class model score",
                    f"{real_score * 100:.2f}%"
                )

                st.progress(
                    max(0.0, min(1.0, real_score)),
                    text="Model score for the Real class"
                )

            st.info(
                "These scores reflect the classifier's learned "
                "class probabilities, not the verified probability "
                "that the news is factually true or false."
            )

            with st.expander("View submitted text"):
                st.write(news_text)


# =====================================================
# MODEL PERFORMANCE
# =====================================================

elif page == "Model Performance":

    st.title("📈 Model Performance")
    st.write(
        "Evaluation metrics from the held-out test dataset."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Accuracy",
        f"{metrics['accuracy'] * 100:.2f}%"
    )
    c2.metric(
        "Precision",
        f"{metrics['precision'] * 100:.2f}%"
    )
    c3.metric(
        "Recall",
        f"{metrics['recall'] * 100:.2f}%"
    )
    c4.metric(
        "F1-Score",
        f"{metrics['f1_score'] * 100:.2f}%"
    )

    st.divider()

    st.subheader("Confusion Matrix")

    cm = metrics.get("confusion_matrix")

    if cm and len(cm) == 2:
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
        ax.set_title("Confusion Matrix")
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

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
            use_container_width=True,
            hide_index=True
        )

    st.subheader("Metric Definitions")

    with st.expander("What do these metrics mean?"):
        st.markdown("""
        - **Accuracy:** Fraction of all test predictions that were correct.
        - **Precision:** Of the items predicted as a class, how many were correct.
        - **Recall:** Of the actual items in a class, how many were identified.
        - **F1-score:** Harmonic mean of precision and recall.
        - **Confusion matrix:** Counts of correct and incorrect predictions.
        """)

    st.caption(
        f"Training samples: {metrics['training_samples']:,} | "
        f"Test samples: {metrics['testing_samples']:,}"
    )

    st.warning(
        "Performance is measured on this dataset's test split. "
        "It does not guarantee equivalent performance on unseen "
        "publishers, future events, or deliberately misleading text."
    )
