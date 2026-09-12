import json
import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix, hstack
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Fake Job Posting Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_PATH = MODEL_DIR / "fake_job_detector.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
METRICS_PATH = REPORT_DIR / "model_metrics.csv"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


# =========================================================
# CUSTOM UI
# =========================================================

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 2rem;
        }

        .hero {
            padding: 1.4rem 1.5rem;
            border-radius: 18px;
            border: 1px solid rgba(128,128,128,.20);
            background: linear-gradient(
                135deg,
                rgba(35, 102, 180, .15),
                rgba(35, 160, 120, .08)
            );
            margin-bottom: 1.2rem;
        }

        .hero-title {
            font-size: 2.15rem;
            font-weight: 750;
            margin-bottom: .35rem;
        }

        .hero-subtitle {
            color: #a9b0bb;
            font-size: 1rem;
            line-height: 1.5;
        }

        .section-title {
            font-size: 1.35rem;
            font-weight: 700;
            margin-top: .25rem;
            margin-bottom: .2rem;
        }

        .helper {
            color: #9da5b1;
            font-size: .9rem;
            margin-bottom: .7rem;
        }

        .result-card {
            padding: 1.2rem 1.25rem;
            border-radius: 16px;
            margin-top: .8rem;
            border: 1px solid rgba(128,128,128,.20);
        }

        .result-good {
            background: rgba(20, 140, 80, .13);
            border-color: rgba(45, 190, 115, .30);
        }

        .result-bad {
            background: rgba(190, 50, 60, .13);
            border-color: rgba(245, 90, 100, .32);
        }

        .result-label {
            font-size: 1.45rem;
            font-weight: 750;
        }

        .risk-low {
            color: #65d99b;
            font-weight: 700;
            font-size: 1.15rem;
        }

        .risk-high {
            color: #ff6c79;
            font-weight: 700;
            font-size: 1.15rem;
        }

        .info-box {
            padding: .95rem 1rem;
            border-radius: 14px;
            background: rgba(70, 120, 190, .10);
            border: 1px solid rgba(70, 120, 190, .25);
            margin-top: .8rem;
        }

        .metric-card {
            padding: 1rem;
            border-radius: 14px;
            background: rgba(128,128,128,.07);
            border: 1px solid rgba(128,128,128,.16);
        }

        .footer {
            text-align: center;
            color: #818995;
            font-size: .82rem;
            margin-top: 1.6rem;
        }

        div[data-testid="stCheckbox"] {
            padding-top: .25rem;
        }

        .stButton > button {
            min-height: 2.8rem;
            font-weight: 650;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD MODEL AND VECTORIZER
# =========================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(f"TF-IDF vectorizer not found: {VECTORIZER_PATH}")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


try:
    model, vectorizer = load_model()
except Exception as exc:
    st.error("The trained model could not be loaded.")
    st.code(str(exc))
    st.stop()


# =========================================================
# LOAD METADATA / METRICS
# =========================================================

metadata = {}

if METADATA_PATH.exists():
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            metadata = json.load(file)
    except (json.JSONDecodeError, OSError):
        metadata = {}


def load_metrics():
    if not METRICS_PATH.exists():
        return None

    try:
        df = pd.read_csv(METRICS_PATH)
    except Exception:
        return None

    if df.empty:
        return None

    model_column = next(
        (c for c in ("Model", "model") if c in df.columns),
        None,
    )

    if model_column:
        selected_name = str(metadata.get("model", "")).strip()
        if selected_name:
            selected = df[
                df[model_column].astype(str).str.strip() == selected_name
            ]
            if not selected.empty:
                return selected.iloc[0]

    return df.iloc[0]


metrics = load_metrics()


# =========================================================
# NLP SETUP
# =========================================================

import nltk

# Download required NLTK resources on Streamlit Cloud
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# =========================================================
# TEXT PREPROCESSING
# MUST MATCH TRAINING PIPELINE
# =========================================================

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    words = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    ]

    return " ".join(words)


# =========================================================
# PREDICTION
# =========================================================

def predict_job(
    job_title: str,
    job_description: str,
    telecommuting: int,
    has_company_logo: int,
    has_questions: int,
):
    full_text = f"{job_title} {job_description}".strip()
    cleaned = clean_text(full_text)

    if not cleaned.strip():
        raise ValueError(
            "The entered text does not contain enough usable words."
        )

    text_vector = vectorizer.transform([cleaned])

    binary_features = csr_matrix(
        [[
            int(telecommuting),
            int(has_company_logo),
            int(has_questions),
        ]]
    )

    final_vector = hstack(
        [text_vector, binary_features],
        format="csr",
    )

    expected_features = getattr(model, "n_features_in_", None)

    if expected_features is not None:
        actual_features = final_vector.shape[1]
        if actual_features != expected_features:
            raise ValueError(
                "Model/vectorizer feature mismatch. "
                f"Model expects {expected_features} features, "
                f"but prediction produced {actual_features}. "
                "Run train_model.py again and keep the new model and vectorizer together."
            )

    prediction = int(model.predict(final_vector)[0])

    decision_score = None
    if hasattr(model, "decision_function"):
        decision_score = float(model.decision_function(final_vector)[0])

    return prediction, decision_score


# =========================================================
# HELPERS
# =========================================================

def metric_value(row, names):
    if row is None:
        return "N/A"

    for name in names:
        if name in row.index and pd.notna(row[name]):
            try:
                return f"{float(row[name]) * 100:.2f}%"
            except (TypeError, ValueError):
                return str(row[name])

    return "N/A"


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🛡️ AI Fake Job Posting Detector</div>
        <div class="hero-subtitle">
            Analyze a job posting using an NLP machine-learning model and
            identify whether it shows patterns associated with a potentially
            fraudulent posting.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# INPUT SECTION
# =========================================================

st.markdown('<div class="section-title">🔎 Analyze a Job Posting</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="helper">Enter the job information exactly as it appears on a recruitment platform such as LinkedIn Jobs.</div>',
    unsafe_allow_html=True,
)

job_title = st.text_input(
    "💼 Job Title",
    placeholder="e.g. Software Engineer",
)

job_description = st.text_area(
    "📝 Job Description",
    height=260,
    placeholder=(
        "Paste the complete job description here...\n\n"
        "Tip: include responsibilities, requirements, company information, "
        "salary/benefits, contact instructions, and application details when available."
    ),
)

col1, col2, col3 = st.columns(3)

with col1:
    telecommuting = st.checkbox("🏠 Remote Job")

with col2:
    has_company_logo = st.checkbox("🏢 Company Logo Available")

with col3:
    has_questions = st.checkbox("❓ Screening Questions Included")

st.caption(
    "These three options are additional binary features used by the trained model."
)


# =========================================================
# ANALYZE
# =========================================================

analyze = st.button(
    "🔍 Analyze Job Posting",
    use_container_width=True,
    type="primary",
)

if analyze:
    title_clean = job_title.strip()
    description_clean = job_description.strip()

    # Input-quality validation.
    if not title_clean:
        st.warning("Please enter a Job Title.")
        st.stop()

    if len(title_clean) < 5:
        st.warning("Please enter a more specific Job Title.")
        st.stop()

    if not description_clean:
        st.warning("Please enter a Job Description.")
        st.stop()

    if len(description_clean) < 30:
        st.warning(
            "Please enter a meaningful Job Description (at least 30 characters)."
        )
        st.stop()

    cleaned_preview = clean_text(f"{title_clean} {description_clean}")

    if len(cleaned_preview.split()) < 8:
        st.warning(
            "The entered text is too short or does not contain enough meaningful "
            "job-posting information for a reliable analysis."
        )
        st.stop()

    try:
        with st.spinner("Analyzing the job posting..."):
            prediction, decision_score = predict_job(
                job_title=title_clean,
                job_description=description_clean,
                telecommuting=telecommuting,
                has_company_logo=has_company_logo,
                has_questions=has_questions,
            )

        st.divider()
        st.markdown('<div class="section-title">📌 Prediction Result</div>', unsafe_allow_html=True)

        if prediction == 0:
            st.markdown(
                """
                <div class="result-card result-good">
                    <div class="result-label">✅ Genuine Job Posting</div>
                    <div style="margin-top:.35rem;">
                        The model did not detect strong fraudulent patterns in the
                        submitted posting.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            r1, r2 = st.columns([1, 2])

            with r1:
                st.markdown("**Risk Level**")
                st.markdown('<div class="risk-low">🟢 LOW</div>', unsafe_allow_html=True)

            with r2:
                st.markdown(
                    """
                    <div class="info-box">
                        <b>Recommended checks</b><br>
                        ✓ Verify the company website and official careers page.<br>
                        ✓ Apply through an official recruitment portal.<br>
                        ✓ Check the recruiter email domain.<br>
                        ✓ Avoid sharing sensitive information unnecessarily.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:
            st.markdown(
                """
                <div class="result-card result-bad">
                    <div class="result-label">🚨 Fraudulent Job Posting Detected</div>
                    <div style="margin-top:.35rem;">
                        The model detected patterns associated with fraudulent
                        job postings.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            r1, r2 = st.columns([1, 2])

            with r1:
                st.markdown("**Risk Level**")
                st.markdown('<div class="risk-high">🔴 HIGH</div>', unsafe_allow_html=True)

            with r2:
                st.markdown(
                    """
                    <div class="info-box">
                        <b>Important precautions</b><br>
                        ✗ Do not pay registration, training, or security fees.<br>
                        ✗ Verify the recruiter and company through official sources.<br>
                        ✗ Do not share Aadhaar, PAN, bank details, OTPs, or passwords.<br>
                        ✗ Report suspicious postings to the relevant job platform.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if decision_score is not None:
            with st.expander("ℹ️ Technical Model Details"):
                st.write(f"Linear SVM decision margin: `{decision_score:.4f}`")
                st.caption(
                    "This is a model decision score, not a probability. "
                    "The application therefore does not display a percentage probability of fraud."
                )

    except Exception as exc:
        st.error("Prediction could not be completed.")
        st.code(str(exc))


# =========================================================
# PROJECT / MODEL INFORMATION
# =========================================================

st.divider()
st.markdown('<div class="section-title">📊 Project Information</div>', unsafe_allow_html=True)

info1, info2 = st.columns(2)

with info1:
    st.markdown(
        """
        **Machine Learning**

        - **Algorithm:** Linear Support Vector Machine
        - **Text representation:** TF-IDF
        - **Text input:** Job Title + Job Description
        - **Additional features:** Remote Job, Company Logo, Screening Questions
        - **Task:** Binary Classification
        """
    )

with info2:
    st.markdown("**Model Performance**")

    if metrics is not None:
        m1, m2 = st.columns(2)
        m3, m4 = st.columns(2)

        with m1:
            st.metric("Accuracy", metric_value(metrics, ["Accuracy", "accuracy"]))

        with m2:
            st.metric("Fraud Precision", metric_value(metrics, ["Precision", "precision"]))

        with m3:
            st.metric("Fraud Recall", metric_value(metrics, ["Recall", "recall"]))

        with m4:
            st.metric("Fraud F1", metric_value(metrics, ["F1 Score", "F1", "f1"]))

    else:
        st.info("Run train_model.py to generate model_metrics.csv.")


# =========================================================
# TRAINING DETAILS
# =========================================================

training_text = metadata.get("text_features")
binary_features = metadata.get("binary_features")
model_name = metadata.get("model")

with st.expander("🧠 Training Details"):
    if model_name:
        st.write(f"**Selected model:** {model_name}")

    if training_text:
        st.write(f"**Text features:** {training_text}")

    if binary_features:
        st.write(
            "**Binary features:** " + ", ".join(map(str, binary_features))
        )

    st.write("**TF-IDF:** 5,000 features, min_df=3")
    st.write("**Split:** 80% training / 20% testing, stratified, random_state=42")
    st.write("**Evaluation focus:** Fraud Precision, Fraud Recall, Fraud F1")


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Built with Streamlit • Scikit-learn • NLTK • Pandas • SciPy
        <br>
        <span style="opacity:.75;">
            Prediction is a risk indicator and should not be treated as absolute proof of fraud.
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
