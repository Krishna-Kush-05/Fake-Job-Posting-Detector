"""Train the Fake Job Posting Detector.

Final corrected training pipeline:
1. Load the raw Kaggle dataset.
2. Drop columns intentionally excluded from the deployed application.
3. Remove duplicates AFTER the final feature subset is defined.
4. Build the SAME text input collected by Streamlit: title + description.
5. Clean the text using the same preprocessing used by Streamlit.
6. Split into train/test BEFORE fitting TF-IDF to prevent leakage.
7. Fit TF-IDF only on training text and transform test text.
8. Append the same three binary job-posting features.
9. Train and compare four classifiers.
10. Select the model with the highest fraud-class F1 score.
11. Save the selected model, vectorizer, metrics, and training metadata.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import nltk
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "fake_job_postings.csv"

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "fake_job_detector.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
METRICS_PATH = REPORT_DIR / "model_metrics.csv"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


# =========================================================
# CONFIGURATION
# =========================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

TEXT_COLUMNS = ["title", "description"]

BINARY_COLUMNS = [
    "telecommuting",
    "has_company_logo",
    "has_questions",
]

DROP_COLUMNS = [
    "job_id",
    "salary_range",
    "department",
]


# =========================================================
# NLTK
# =========================================================

def ensure_nltk_resources() -> None:
    """Download the NLTK resources required by the text cleaner."""
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]

    for resource_path, package_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            print(f"Downloading NLTK resource: {package_name}")
            nltk.download(package_name, quiet=True)


# =========================================================
# TEXT PREPROCESSING
# =========================================================

def clean_text(
    text: str,
    stop_words: set[str],
    lemmatizer: WordNetLemmatizer,
) -> str:
    """Apply the same preprocessing logic used in app.py."""
    text = str(text).lower()

    # Remove URLs.
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove HTML tags.
    text = re.sub(r"<.*?>", " ", text)

    # Keep alphabetic characters and whitespace.
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    ensure_nltk_resources()

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    # -----------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------

    print("=" * 70)
    print("FAKE JOB POSTING DETECTOR - FINAL TRAINING")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at:\n{DATA_PATH}\n\n"
            "Place fake_job_postings.csv inside the data/ folder."
        )

    print("\nLoading dataset...")
    df = pd.read_csv(DATA_PATH)

    print(
        f"Raw dataset: {len(df):,} rows x "
        f"{len(df.columns)} columns"
    )

    # -----------------------------------------------------
    # 2. Drop columns excluded from the final model
    # -----------------------------------------------------

    missing_drop_columns = [
        col for col in DROP_COLUMNS if col not in df.columns
    ]

    if missing_drop_columns:
        raise KeyError(
            "Expected columns are missing from the dataset: "
            f"{missing_drop_columns}"
        )

    df = df.drop(columns=DROP_COLUMNS).copy()

    # Check duplicates at the same stage as the EDA
    duplicate_count = int(df.duplicated().sum())

    print(
        f"\nDuplicates after dropping excluded columns: "
        f"{duplicate_count:,}"
    )

    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    print(
        f"Dataset after duplicate removal: "
        f"{len(df):,} rows"
    )

    required_columns = (
        TEXT_COLUMNS
        + BINARY_COLUMNS
        + ["fraudulent"]
    )

    missing_required_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_required_columns:
        raise KeyError(
            "Required columns are missing from the dataset: "
            f"{missing_required_columns}"
        )

    df = df[required_columns].copy()
    # -----------------------------------------------------
    # 4. Handle missing values
    # -----------------------------------------------------

    for column in TEXT_COLUMNS:
        df[column] = df[column].fillna("").astype(str)

    for column in BINARY_COLUMNS:
        df[column] = (
            pd.to_numeric(df[column], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    df["fraudulent"] = (
        pd.to_numeric(df["fraudulent"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    # -----------------------------------------------------
    # 5. Build the SAME text input used by Streamlit
    # -----------------------------------------------------

    df["text"] = (
        df["title"]
        + " "
        + df["description"]
    ).str.strip()

    df["text"] = df["text"].apply(
        lambda value: clean_text(
            value,
            stop_words,
            lemmatizer,
        )
    )

    # Remove rows whose text becomes completely empty after
    # preprocessing. This is safer than training on blank text.
    empty_text_count = int((df["text"].str.strip() == "").sum())

    if empty_text_count > 0:
        print(
            f"Rows with empty cleaned text removed: "
            f"{empty_text_count:,}"
        )
        df = df[df["text"].str.strip() != ""].reset_index(drop=True)

    # -----------------------------------------------------
    # 6. Prepare X and y
    # -----------------------------------------------------

    X_text = df["text"]
    X_binary = df[BINARY_COLUMNS].to_numpy(dtype=np.int8)
    y = df["fraudulent"]

    print("\nClass distribution:")
    print(
        y.value_counts()
        .rename(index={0: "Genuine", 1: "Fraudulent"})
    )

    print("\nClass percentages:")
    print(
        (y.value_counts(normalize=True) * 100)
        .round(2)
        .rename(index={0: "Genuine", 1: "Fraudulent"})
    )

    # -----------------------------------------------------
    # 7. Train/test split BEFORE TF-IDF
    # -----------------------------------------------------

    (
        X_text_train,
        X_text_test,
        X_bin_train,
        X_bin_test,
        y_train,
        y_test,
    ) = train_test_split(
        X_text,
        X_binary,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\nTraining rows: {len(y_train):,}")
    print(f"Testing rows : {len(y_test):,}")

    # -----------------------------------------------------
    # 8. TF-IDF - FIT ONLY ON TRAINING TEXT
    # -----------------------------------------------------

    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=3,
    )

    X_train_text = tfidf.fit_transform(X_text_train)
    X_test_text = tfidf.transform(X_text_test)

    print(
        f"\nTF-IDF vocabulary size: "
        f"{len(tfidf.vocabulary_):,}"
    )

    # -----------------------------------------------------
    # 9. Append the three binary features
    # -----------------------------------------------------

    X_train = hstack(
        [
            X_train_text,
            csr_matrix(X_bin_train),
        ],
        format="csr",
    )

    X_test = hstack(
        [
            X_test_text,
            csr_matrix(X_bin_test),
        ],
        format="csr",
    )

    print(f"Final training matrix: {X_train.shape}")
    print(f"Final testing matrix : {X_test.shape}")

    # -----------------------------------------------------
    # 10. Define models
    # -----------------------------------------------------

    models = {
        "Naive Bayes": MultinomialNB(),

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),

        "Linear SVM": LinearSVC(
            random_state=RANDOM_STATE,
        ),

        "Balanced Linear SVM": LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    # -----------------------------------------------------
    # 11. Train + evaluate
    # -----------------------------------------------------

    results = []
    trained_models = {}

    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    for name, model in models.items():
        print(f"\n{name}")
        print("-" * 70)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        trained_models[name] = model

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0,
        )
        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0,
        )
        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        results.append(
            {
                "Model": name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
            }
        )

        print(
            classification_report(
                y_test,
                y_pred,
                target_names=[
                    "Genuine",
                    "Fraudulent",
                ],
                zero_division=0,
            )
        )

        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))

    # -----------------------------------------------------
    # 12. Compare models
    # -----------------------------------------------------

    results_df = (
        pd.DataFrame(results)
        .sort_values(
            by="F1 Score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "Accuracy": "{:.4f}".format,
                "Precision": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1 Score": "{:.4f}".format,
            },
        )
    )

    # -----------------------------------------------------
    # 13. Select final model
    # -----------------------------------------------------
    #
    # Primary criterion = fraud F1.
    # Accuracy is deliberately NOT used as the selection criterion
    # because the dataset is strongly imbalanced.
    # -----------------------------------------------------

    best_name = str(results_df.iloc[0]["Model"])
    best_model = trained_models[best_name]

    selected_row = results_df.iloc[0]

    print(
        f"\nSelected model by highest fraudulent-class F1: "
        f"{best_name}"
    )

    print("\nSelected model metrics:")
    for metric in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
    ]:
        print(
            f"{metric}: "
            f"{float(selected_row[metric]):.4f}"
        )

    # -----------------------------------------------------
    # 14. Save final artifacts
    # -----------------------------------------------------

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(tfidf, VECTORIZER_PATH)

    results_df.to_csv(
        METRICS_PATH,
        index=False,
    )

    metadata = {
        "model": best_name,
        "text_features": "title + description",
        "binary_features": BINARY_COLUMNS,
        "dropped_columns": DROP_COLUMNS,
        "tfidf_max_features": 5000,
        "tfidf_ngram_range": [1, 2],
        "tfidf_min_df": 3,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "duplicate_rows_removed_after_feature_selection": duplicate_count,
        "modeling_rows": int(len(df)),
        "training_rows": int(len(y_train)),
        "testing_rows": int(len(y_test)),
        "fraudulent_rows": int((y == 1).sum()),
        "genuine_rows": int((y == 0).sum()),
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    print("\nSaved artifacts:")
    print(f"Model     : {MODEL_PATH}")
    print(f"Vectorizer: {VECTORIZER_PATH}")
    print(f"Metrics   : {METRICS_PATH}")
    print(f"Metadata  : {METADATA_PATH}")

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
