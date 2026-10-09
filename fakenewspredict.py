# ============================================================
# FAKE NEWS DETECTION - COMPLETE EDA
# ============================================================

# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from collections import Counter

import warnings
warnings.filterwarnings("ignore")

# Plot style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11


# ------------------------------------------------------------
# 2. LOAD DATASETS
# ------------------------------------------------------------

fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

print("Fake Dataset Shape:", fake_df.shape)
print("True Dataset Shape:", true_df.shape)


# ------------------------------------------------------------
# 3. DISPLAY BASIC INFORMATION
# ------------------------------------------------------------

print("\n========== FAKE DATASET ==========")
print(fake_df.head())

print("\n========== TRUE DATASET ==========")
print(true_df.head())

print("\nFake Dataset Columns:")
print(fake_df.columns.tolist())

print("\nTrue Dataset Columns:")
print(true_df.columns.tolist())


# ------------------------------------------------------------
# 4. ADD LABELS
# ------------------------------------------------------------

# 0 = Fake
# 1 = True

fake_df["label"] = 0
true_df["label"] = 1


# ------------------------------------------------------------
# 5. COMBINE DATASETS
# ------------------------------------------------------------

df = pd.concat([fake_df, true_df], ignore_index=True)

print("\nCombined Dataset Shape:", df.shape)

print("\nCombined Dataset:")
print(df.head())


# ------------------------------------------------------------
# 6. CREATE HUMAN-READABLE LABEL
# ------------------------------------------------------------

df["label_name"] = df["label"].map({
    0: "Fake",
    1: "True"
})

print("\nLabel Distribution:")
print(df["label_name"].value_counts())


# ------------------------------------------------------------
# 7. DATASET INFORMATION
# ------------------------------------------------------------

print("\n========== DATASET INFO ==========")
print(df.info())


# ------------------------------------------------------------
# 8. STATISTICAL SUMMARY
# ------------------------------------------------------------

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe(include="all"))


# ============================================================
# GRAPH 1
# FAKE VS TRUE NEWS DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 6))

sns.countplot(
    data=df,
    x="label_name",
    hue="label_name",
    palette=["#e74c3c", "#2ecc71"],
    legend=False
)

plt.title("Distribution of Fake and True News")
plt.xlabel("News Type")
plt.ylabel("Number of Articles")

plt.tight_layout()
plt.show()


# Print percentages
print("\n========== CLASS DISTRIBUTION ==========")

class_distribution = df["label_name"].value_counts()
class_percentage = df["label_name"].value_counts(normalize=True) * 100

print(class_distribution)

print("\nPercentage:")
print(class_percentage.round(2))


# ------------------------------------------------------------
# 9. MISSING VALUE ANALYSIS
# ------------------------------------------------------------

print("\n========== MISSING VALUES ==========")

missing_values = df.isnull().sum()

print(missing_values)


# ============================================================
# GRAPH 2
# MISSING VALUES
# ============================================================

missing_nonzero = missing_values[missing_values > 0]

if len(missing_nonzero) > 0:

    plt.figure(figsize=(10, 6))

    sns.barplot(
        x=missing_nonzero.values,
        y=missing_nonzero.index,
        hue=missing_nonzero.index,
        palette="viridis",
        legend=False
    )

    plt.title("Missing Values by Column")
    plt.xlabel("Number of Missing Values")
    plt.ylabel("Column")

    plt.tight_layout()
    plt.show()

else:
    print("No missing values found.")


# ------------------------------------------------------------
# 10. DUPLICATE ANALYSIS
# ------------------------------------------------------------

print("\n========== DUPLICATE ANALYSIS ==========")

print("Duplicate rows:", df.duplicated().sum())
print("Duplicate titles:", df["title"].duplicated().sum())
print("Duplicate texts:", df["text"].duplicated().sum())


# ============================================================
# GRAPH 3
# DUPLICATE ANALYSIS
# ============================================================

duplicate_counts = pd.Series({
    "Duplicate Rows": df.duplicated().sum(),
    "Duplicate Titles": df["title"].duplicated().sum(),
    "Duplicate Texts": df["text"].duplicated().sum()
})

plt.figure(figsize=(8, 5))

sns.barplot(
    x=duplicate_counts.index,
    y=duplicate_counts.values,
    hue=duplicate_counts.index,
    palette="magma",
    legend=False
)

plt.title("Duplicate Records Analysis")
plt.xlabel("Duplicate Type")
plt.ylabel("Count")

plt.xticks(rotation=15)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 11. TEXT CLEANING FOR EDA
# ------------------------------------------------------------

# Fill missing text/title values
df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")


# ------------------------------------------------------------
# 12. CREATE TEXT FEATURES
# ------------------------------------------------------------

# Character count
df["title_char_count"] = df["title"].str.len()
df["text_char_count"] = df["text"].str.len()

# Word count
df["title_word_count"] = df["title"].str.split().str.len()
df["text_word_count"] = df["text"].str.split().str.len()

# Sentence count
df["sentence_count"] = (
    df["text"]
    .str.count(r"[.!?]")
)

# Capital letters
df["capital_count"] = (
    df["text"]
    .str.count(r"[A-Z]")
)

# Number of exclamation marks
df["exclamation_count"] = (
    df["text"]
    .str.count("!")
)

# Number of question marks
df["question_count"] = (
    df["text"]
    .str.count(r"\?")
)

# Number of digits
df["digit_count"] = (
    df["text"]
    .str.count(r"\d")
)

# Number of URLs
df["url_count"] = (
    df["text"]
    .str.count(r"http\S+|www\S+")
)


# ------------------------------------------------------------
# 13. DISPLAY TEXT STATISTICS
# ------------------------------------------------------------

text_features = [
    "title_char_count",
    "title_word_count",
    "text_char_count",
    "text_word_count",
    "sentence_count",
    "capital_count",
    "exclamation_count",
    "question_count",
    "digit_count",
    "url_count"
]

print("\n========== TEXT STATISTICS ==========")

print(
    df.groupby("label_name")[text_features]
    .mean()
    .round(2)
)


# ============================================================
# GRAPH 4
# TEXT WORD COUNT DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="text_word_count",
    hue="label_name",
    bins=50,
    kde=True,
    palette=["#e74c3c", "#2ecc71"],
    alpha=0.45
)

plt.title("Text Word Count Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Number of Articles")

plt.xlim(0, df["text_word_count"].quantile(0.99))

plt.tight_layout()
plt.show()


# ============================================================
# GRAPH 5
# TEXT LENGTH BOXPLOT
# ============================================================

plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x="label_name",
    y="text_word_count",
    hue="label_name",
    palette=["#e74c3c", "#2ecc71"],
    legend=False
)

plt.title("Text Length Comparison: Fake vs True")
plt.xlabel("News Type")
plt.ylabel("Number of Words")

plt.ylim(
    0,
    df["text_word_count"].quantile(0.99)
)

plt.tight_layout()
plt.show()


# ============================================================
# GRAPH 6
# TITLE WORD COUNT COMPARISON
# ============================================================

plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x="label_name",
    y="title_word_count",
    hue="label_name",
    palette=["#e74c3c", "#2ecc71"],
    legend=False
)

plt.title("Title Length Comparison: Fake vs True")
plt.xlabel("News Type")
plt.ylabel("Number of Words")

plt.tight_layout()
plt.show()


# ============================================================
# GRAPH 7
# TITLE CHARACTER COUNT
# ============================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="title_char_count",
    hue="label_name",
    bins=40,
    kde=True,
    palette=["#e74c3c", "#2ecc71"],
    alpha=0.5
)

plt.title("Title Character Count Distribution")
plt.xlabel("Title Character Count")
plt.ylabel("Number of Articles")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 14. SUBJECT ANALYSIS
# ------------------------------------------------------------

print("\n========== SUBJECT DISTRIBUTION ==========")

print(df["subject"].value_counts())


# ============================================================
# GRAPH 8
# SUBJECT DISTRIBUTION
# ============================================================

plt.figure(figsize=(12, 7))

subject_counts = df["subject"].value_counts()

sns.barplot(
    x=subject_counts.values,
    y=subject_counts.index,
    hue=subject_counts.index,
    palette="Set2",
    legend=False
)

plt.title("Distribution of News by Subject")
plt.xlabel("Number of Articles")
plt.ylabel("Subject")

plt.tight_layout()
plt.show()


# ============================================================
# GRAPH 9
# SUBJECT VS FAKE/TRUE
# ============================================================

plt.figure(figsize=(12, 7))

sns.countplot(
    data=df,
    y="subject",
    hue="label_name",
    palette=["#e74c3c", "#2ecc71"]
)

plt.title("Fake vs True News by Subject")
plt.xlabel("Number of Articles")
plt.ylabel("Subject")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 15. SUBJECT VS LABEL PERCENTAGE
# ------------------------------------------------------------

subject_label_percentage = (
    pd.crosstab(
        df["subject"],
        df["label_name"],
        normalize="index"
    ) * 100
)

print("\n========== SUBJECT VS LABEL (%) ==========")
print(subject_label_percentage.round(2))


# ============================================================
# GRAPH 10
# SUBJECT LABEL HEATMAP
# ============================================================

plt.figure(figsize=(10, 7))

sns.heatmap(
    subject_label_percentage,
    annot=True,
    fmt=".1f",
    cmap="RdYlGn"
)

plt.title("Fake vs True Percentage by Subject")
plt.xlabel("News Type")
plt.ylabel("Subject")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 16. DATE ANALYSIS
# ------------------------------------------------------------

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

print("\n========== DATE INFORMATION ==========")

print("Minimum date:", df["date"].min())
print("Maximum date:", df["date"].max())

print("Invalid dates:", df["date"].isnull().sum())


# Create date features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["month_name"] = df["date"].dt.month_name()


# ============================================================
# GRAPH 11
# NEWS BY YEAR
# ============================================================

year_counts = (
    df.groupby(["year", "label_name"])
    .size()
    .reset_index(name="count")
)

plt.figure(figsize=(12, 6))

sns.lineplot(
    data=year_counts,
    x="year",
    y="count",
    hue="label_name",
    marker="o",
    palette=["#e74c3c", "#2ecc71"]
)

plt.title("Fake vs True News Over the Years")
plt.xlabel("Year")
plt.ylabel("Number of Articles")

plt.tight_layout()
plt.show()


# ============================================================
# GRAPH 12
# MONTHLY DISTRIBUTION
# ============================================================

month_counts = (
    df.groupby(["month", "label_name"])
    .size()
    .reset_index(name="count")
)

plt.figure(figsize=(12, 6))

sns.lineplot(
    data=month_counts,
    x="month",
    y="count",
    hue="label_name",
    marker="o",
    palette=["#e74c3c", "#2ecc71"]
)

plt.title("Fake vs True News by Month")
plt.xlabel("Month")
plt.ylabel("Number of Articles")

plt.xticks(range(1, 13))

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 17. COMBINE TITLE + TEXT
# ------------------------------------------------------------

df["full_text"] = (
    df["title"] + " " + df["text"]
)


# ------------------------------------------------------------
# 18. TEXT CLEANING FUNCTION
# ------------------------------------------------------------

def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove HTML
    text = re.sub(r"<.*?>", "", text)

    # Keep alphabetic characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["clean_text"] = df["full_text"].apply(clean_text)


# ------------------------------------------------------------
# 19. STOPWORDS
# ------------------------------------------------------------

try:

    import nltk

    nltk.download("stopwords", quiet=True)

    from nltk.corpus import stopwords

    stop_words = set(stopwords.words("english"))

except:

    stop_words = set()


def tokenize(text):

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
        and len(word) > 2
    ]

    return words


# ------------------------------------------------------------
# 20. TOP WORDS - FAKE NEWS
# ------------------------------------------------------------

fake_words = Counter()

for text in df.loc[df["label"] == 0, "clean_text"]:

    fake_words.update(
        tokenize(text)
    )


# ------------------------------------------------------------
# 21. TOP WORDS - TRUE NEWS
# ------------------------------------------------------------

true_words = Counter()

for text in df.loc[df["label"] == 1, "clean_text"]:

    true_words.update(
        tokenize(text)
    )


print("\n========== TOP FAKE NEWS WORDS ==========")
print(fake_words.most_common(20))

print("\n========== TOP TRUE NEWS WORDS ==========")
print(true_words.most_common(20))


# ============================================================
# GRAPH 13
# TOP WORDS IN FAKE NEWS
# ============================================================

fake_top = pd.DataFrame(
    fake_words.most_common(20),
    columns=["word", "count"]
)

plt.figure(figsize=(12, 8))

sns.barplot(
    data=fake_top,
    x="count",
    y="word",
    hue="word",
    palette="Reds_r",
    legend=False
)

plt.title("Top 20 Words in Fake News")
plt.xlabel("Frequency")
plt.ylabel("Word")

plt.tight_layout()
plt.show()


# ============================================================
# GRAPH 14
# TOP WORDS IN TRUE NEWS
# ============================================================

true_top = pd.DataFrame(
    true_words.most_common(20),
    columns=["word", "count"]
)

plt.figure(figsize=(12, 8))

sns.barplot(
    data=true_top,
    x="count",
    y="word",
    hue="word",
    palette="Greens_r",
    legend=False
)

plt.title("Top 20 Words in True News")
plt.xlabel("Frequency")
plt.ylabel("Word")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 22. BIGRAM ANALYSIS
# ------------------------------------------------------------

from sklearn.feature_extraction.text import CountVectorizer


# Fake news bigrams
fake_texts = df.loc[
    df["label"] == 0,
    "clean_text"
]

true_texts = df.loc[
    df["label"] == 1,
    "clean_text"
]


fake_vectorizer = CountVectorizer(
    stop_words="english",
    ngram_range=(2, 2),
    max_features=20
)

fake_matrix = fake_vectorizer.fit_transform(fake_texts)

fake_bigram_counts = np.asarray(
    fake_matrix.sum(axis=0)
).flatten()

fake_bigrams = pd.DataFrame({
    "bigram": fake_vectorizer.get_feature_names_out(),
    "count": fake_bigram_counts
}).sort_values(
    "count",
    ascending=False
)


true_vectorizer = CountVectorizer(
    stop_words="english",
    ngram_range=(2, 2),
    max_features=20
)

true_matrix = true_vectorizer.fit_transform(true_texts)

true_bigram_counts = np.asarray(
    true_matrix.sum(axis=0)
).flatten()

true_bigrams = pd.DataFrame({
    "bigram": true_vectorizer.get_feature_names_out(),
    "count": true_bigram_counts
}).sort_values(
    "count",
    ascending=False
)


# ============================================================
# GRAPH 15
# TOP BIGRAMS IN FAKE NEWS
# ============================================================

plt.figure(figsize=(12, 8))

sns.barplot(
    data=fake_bigrams,
    x="count",
    y="bigram",
    hue="bigram",
    palette="Oranges_r",
    legend=False
)

plt.title("Top 20 Bigrams in Fake News")
plt.xlabel("Frequency")
plt.ylabel("Bigram")

plt.tight_layout()
plt.show()


# ============================================================
# GRAPH 16
# TOP BIGRAMS IN TRUE NEWS
# ============================================================

plt.figure(figsize=(12, 8))

sns.barplot(
    data=true_bigrams,
    x="count",
    y="bigram",
    hue="bigram",
    palette="Blues_r",
    legend=False
)

plt.title("Top 20 Bigrams in True News")
plt.xlabel("Frequency")
plt.ylabel("Bigram")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 23. PUNCTUATION ANALYSIS
# ------------------------------------------------------------

punctuation_features = [
    "exclamation_count",
    "question_count",
    "capital_count",
    "digit_count",
    "url_count"
]

punctuation_mean = (
    df.groupby("label_name")[punctuation_features]
    .mean()
)

print("\n========== PUNCTUATION / STYLE STATISTICS ==========")
print(punctuation_mean.round(2))


# ============================================================
# GRAPH 17
# EXCLAMATION MARKS
# ============================================================

plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x="label_name",
    y="exclamation_count",
    hue="label_name",
    palette=["#e74c3c", "#2ecc71"],
    legend=False
)

plt.title("Exclamation Marks: Fake vs True")
plt.xlabel("News Type")
plt.ylabel("Number of Exclamation Marks")

plt.ylim(
    0,
    df["exclamation_count"].quantile(0.99) + 1
)

plt.tight_layout()
plt.show()


# ============================================================
# GRAPH 18
# QUESTION MARKS
# ============================================================

plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x="label_name",
    y="question_count",
    hue="label_name",
    palette=["#e74c3c", "#2ecc71"],
    legend=False
)

plt.title("Question Marks: Fake vs True")
plt.xlabel("News Type")
plt.ylabel("Number of Question Marks")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 24. CORRELATION ANALYSIS
# ------------------------------------------------------------

numeric_features = [
    "label",
    "title_char_count",
    "title_word_count",
    "text_char_count",
    "text_word_count",
    "sentence_count",
    "capital_count",
    "exclamation_count",
    "question_count",
    "digit_count",
    "url_count"
]

correlation = df[numeric_features].corr()


# ============================================================
# GRAPH 19
# CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Between Text Features")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 25. REUTERS SOURCE ANALYSIS
# ------------------------------------------------------------

df["contains_reuters"] = (
    df["text"]
    .str.contains(
        "Reuters",
        case=False,
        na=False
    )
)


print("\n========== REUTERS ANALYSIS ==========")

print(
    pd.crosstab(
        df["contains_reuters"],
        df["label_name"]
    )
)


# ============================================================
# GRAPH 20
# REUTERS MENTION VS LABEL
# ============================================================

reuters_counts = (
    df.groupby(
        ["contains_reuters", "label_name"]
    )
    .size()
    .reset_index(name="count")
)

reuters_counts["contains_reuters"] = (
    reuters_counts["contains_reuters"]
    .map({
        True: "Contains Reuters",
        False: "Does Not Contain Reuters"
    })
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=reuters_counts,
    x="contains_reuters",
    y="count",
    hue="label_name",
    palette=["#e74c3c", "#2ecc71"]
)

plt.title("Reuters Mention in Fake vs True News")
plt.xlabel("Reuters Mention")
plt.ylabel("Number of Articles")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 26. CHECK FOR SOURCE / DATASET BIAS
# ------------------------------------------------------------

print("\n========== DATASET BIAS CHECK ==========")

print("\nSubject distribution by label:")
print(
    pd.crosstab(
        df["subject"],
        df["label_name"],
        normalize="columns"
    ).round(3)
)


# ------------------------------------------------------------
# 27. SUMMARY STATISTICS BY CLASS
# ------------------------------------------------------------

summary = (
    df.groupby("label_name")[
        [
            "title_word_count",
            "text_word_count",
            "sentence_count",
            "capital_count",
            "exclamation_count",
            "question_count",
            "digit_count",
            "url_count"
        ]
    ]
    .agg(["mean", "median", "std"])
    .round(2)
)

print("\n========== FINAL TEXT STATISTICS ==========")
print(summary)


# ------------------------------------------------------------
# 28. FINAL DATASET SUMMARY
# ------------------------------------------------------------

print("\n")
print("=" * 60)
print("FINAL EDA SUMMARY")
print("=" * 60)

print("Total articles:", len(df))
print("Fake articles:", (df["label"] == 0).sum())
print("True articles:", (df["label"] == 1).sum())

print(
    "Duplicate rows:",
    df.duplicated().sum()
)

print(
    "Missing values:",
    df.isnull().sum().sum()
)

print(
    "Date range:",
    df["date"].min(),
    "to",
    df["date"].max()
)

print(
    "Number of subjects:",
    df["subject"].nunique()
)

print(
    "Average text words:",
    round(df["text_word_count"].mean(), 2)
)

print(
    "Average title words:",
    round(df["title_word_count"].mean(), 2)
)

print("=" * 60)
print("EDA COMPLETED")
print("=" * 60)
