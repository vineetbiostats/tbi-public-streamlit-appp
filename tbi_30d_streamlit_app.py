import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


st.set_page_config(
    page_title="30-Day Mortality Risk Calculator",
    layout="wide",
)


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = Path(r"C:\Users\nihar\OneDrive\Desktop\TBI EXTERNAL\Tbi external.dta")
TARGET_CENTER = "aiims_delhi_prospective"
OUTCOME = "in_hospital_mortality"
RANDOM_STATE = 42

PREDICTOR_COLUMNS = [
    "Age",
    "gender",
    "motor_score",
    "pupil_reactivity",
    "limb_movement",
    "extracranial_injury",
    "hypoxia",
    "hypotension",
    "midline_shift",
    "skull_fracture",
    "mass_effect",
    "contusion",
    "subdural_hematoma",
    "epidural_hematoma",
    "dot_hemorrhages",
    "basal_cisterns_effaced",
    "traumatic_sah",
    "intraventricular_hemorrhage",
    "non_evacuated_hematoma",
    "evacuated_hematoma",
    "decompressive_craniectomy",
    "hemoglobin",
    "blood_glucose",
    "sodium",
    "serum_creatinine",
]

CONTINUOUS_COLUMNS = [
    "Age",
    "hemoglobin",
    "blood_glucose",
    "sodium",
    "serum_creatinine",
]

CATEGORICAL_COLUMNS = [
    column_name for column_name in PREDICTOR_COLUMNS if column_name not in CONTINUOUS_COLUMNS
]

OUTPUT_DIR = APP_DIR / "tbi_30d_streamlit_artifacts"
MODEL_PATH = OUTPUT_DIR / "lasso_30d_risk_calculator.joblib"
METADATA_PATH = OUTPUT_DIR / "ui_metadata.json"
ROOT_MODEL_PATH = APP_DIR / "lasso_30d_risk_calculator.joblib"
ROOT_METADATA_PATH = APP_DIR / "ui_metadata.json"

INPUT_GROUPS = [
    ("Demographics and Primary Injury", ["Age", "gender", "extracranial_injury"]),
    ("Neurologic Examination", ["motor_score", "pupil_reactivity", "limb_movement"]),
    (
        "Physiology and Early Secondary Insults",
        ["hypoxia", "hypotension"],
    ),
    (
        "CT Findings",
        [
            "midline_shift",
            "skull_fracture",
            "mass_effect",
            "contusion",
            "subdural_hematoma",
            "epidural_hematoma",
            "dot_hemorrhages",
            "basal_cisterns_effaced",
            "traumatic_sah",
            "intraventricular_hemorrhage",
            "non_evacuated_hematoma",
            "evacuated_hematoma",
            "decompressive_craniectomy",
        ],
    ),
    ("Laboratory Values", ["hemoglobin", "blood_glucose", "sodium", "serum_creatinine"]),
]

RISK_CATEGORY_RULES = [
    {"max": 0.10, "label": "Low Risk", "color": "#1d8f6a", "message": "Estimated mortality risk is low."},
    {
        "max": 0.30,
        "label": "Intermediate Risk",
        "color": "#d08a00",
        "message": "Estimated mortality risk is intermediate and warrants closer attention.",
    },
    {"max": 1.01, "label": "High Risk", "color": "#c0392b", "message": "Estimated mortality risk is high."},
]

GENDER_LABELS = {
    "0": "Female",
    "1": "Male",
    "0.0": "Female",
    "1.0": "Male",
}

YES_NO_COLUMNS = {
    "extracranial_injury",
    "hypoxia",
    "hypotension",
    "midline_shift",
    "skull_fracture",
    "mass_effect",
    "contusion",
    "subdural_hematoma",
    "epidural_hematoma",
    "dot_hemorrhages",
    "basal_cisterns_effaced",
    "traumatic_sah",
    "intraventricular_hemorrhage",
    "non_evacuated_hematoma",
    "evacuated_hematoma",
    "decompressive_craniectomy",
}


def parse_sklearn_major_minor(version_string):
    parts = []
    for token in version_string.split(".")[:2]:
        numeric = ""
        for char in token:
            if char.isdigit():
                numeric += char
            else:
                break
        parts.append(int(numeric or 0))
    while len(parts) < 2:
        parts.append(0)
    return tuple(parts)


SKLEARN_VERSION = parse_sklearn_major_minor(sklearn.__version__)


def make_onehot_encoder():
    if SKLEARN_VERSION >= (1, 2):
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return OneHotEncoder(handle_unknown="ignore", sparse=False)


def cast_to_string(x):
    if hasattr(x, "astype"):
        return x.astype(str)
    return np.asarray(x).astype(str)


def build_preprocessor():
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("to_string", FunctionTransformer(cast_to_string, validate=False)),
            ("encoder", make_onehot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, CONTINUOUS_COLUMNS),
            ("cat", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def build_pipeline():
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                LogisticRegressionCV(
                    Cs=np.logspace(-3, 3, 50),
                    cv=5,
                    penalty="l1",
                    solver="saga",
                    scoring="roc_auc",
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                    refit=True,
                ),
            ),
        ]
    )


def coerce_binary_outcome(series, series_name):
    text_mapping = {
        "0": 0,
        "1": 1,
        "no": 0,
        "yes": 1,
        "false": 0,
        "true": 1,
        "alive": 0,
        "dead": 1,
        "survived": 0,
        "died": 1,
    }

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
    else:
        lower_str = series.astype(str).str.strip().str.lower()
        numeric = pd.to_numeric(lower_str, errors="coerce")
        missing_numeric = numeric.isna() & series.notna()
        numeric.loc[missing_numeric] = lower_str.loc[missing_numeric].map(text_mapping)

    unique_non_missing = set(pd.Series(numeric).dropna().unique())
    if not unique_non_missing.issubset({0, 1}):
        raise ValueError(
            f"Outcome column '{series_name}' is not binary after coercion. "
            f"Observed values: {sorted(unique_non_missing)}"
        )

    return pd.Series(numeric, index=series.index)


@st.cache_data(show_spinner=False)
def load_ui_metadata():
    metadata_path = METADATA_PATH if METADATA_PATH.exists() else ROOT_METADATA_PATH
    if not metadata_path.exists():
        raise ValueError(
            f"Metadata file not found at {METADATA_PATH} or {ROOT_METADATA_PATH}. "
            "Create or copy the packaged app artifacts before deployment."
        )
    return json.loads(metadata_path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_training_dataframe():
    df = pd.read_stata(DATA_PATH)

    required_columns = ["center_name", OUTCOME] + PREDICTOR_COLUMNS
    missing_required = [column_name for column_name in required_columns if column_name not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    df = df.loc[df["center_name"].astype(str).str.strip() == TARGET_CENTER].copy()
    if df.empty:
        raise ValueError(f"No rows found for center_name == '{TARGET_CENTER}'")

    df[OUTCOME] = coerce_binary_outcome(df[OUTCOME], OUTCOME)
    df = df.dropna(subset=[OUTCOME]).copy()
    df[OUTCOME] = df[OUTCOME].astype(int)

    for column_name in CONTINUOUS_COLUMNS:
        df[column_name] = pd.to_numeric(df[column_name], errors="coerce")

    return df


@st.cache_resource(show_spinner="Loading or training the LASSO model...")
def load_or_train_model():
    model_path = MODEL_PATH if MODEL_PATH.exists() else ROOT_MODEL_PATH
    if model_path.exists():
        return joblib.load(model_path), f"loaded from disk: {model_path.name}"

    if not DATA_PATH.exists():
        raise ValueError(
            f"Model file is missing at {MODEL_PATH} and {ROOT_MODEL_PATH}, "
            f"and local training data was not found at {DATA_PATH}."
        )

    training_df = load_training_dataframe()
    X = training_df[PREDICTOR_COLUMNS].copy()
    y = training_df[OUTCOME].copy()

    pipeline = build_pipeline()
    pipeline.fit(X, y)

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
        return pipeline, f"trained and saved to {MODEL_PATH}"
    except PermissionError:
        return pipeline, "trained in memory only"


def predict_single_patient(model, patient_dict):
    patient_df = pd.DataFrame([patient_dict], columns=PREDICTOR_COLUMNS)
    for column_name in CONTINUOUS_COLUMNS:
        patient_df[column_name] = pd.to_numeric(patient_df[column_name], errors="coerce")
    return float(model.predict_proba(patient_df)[0, 1])


def categorical_options(ui_metadata, column_name):
    options = ui_metadata["categorical_options"].get(column_name, [])
    return options if options else [""]


def numeric_default(ui_metadata, column_name):
    value = ui_metadata["continuous_defaults"].get(column_name, 0.0)
    if pd.isna(value):
        return 0.0
    return float(value)


def format_categorical_label(column_name, raw_value):
    raw_text = str(raw_value)

    if column_name == "gender":
        return GENDER_LABELS.get(raw_text, raw_text)

    if column_name in YES_NO_COLUMNS:
        if raw_text in {"0", "0.0"}:
            return "No"
        if raw_text in {"1", "1.0"}:
            return "Yes"

    return raw_text


def get_risk_category(risk):
    for rule in RISK_CATEGORY_RULES:
        if risk < rule["max"]:
            return rule
    return RISK_CATEGORY_RULES[-1]


def inject_app_styles():
    st.markdown(
        """
        <style>
        .main {
            background:
                radial-gradient(circle at top left, rgba(17, 104, 108, 0.10), transparent 28%),
                radial-gradient(circle at top right, rgba(208, 138, 0, 0.08), transparent 22%),
                linear-gradient(180deg, #f6f3eb 0%, #fbfaf7 100%);
        }
        .hero-card,
        .result-card,
        .group-card {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(26, 35, 42, 0.08);
            border-radius: 22px;
            box-shadow: 0 18px 40px rgba(26, 35, 42, 0.08);
        }
        .hero-card {
            padding: 1.4rem 1.6rem;
            margin-bottom: 1rem;
        }
        .hero-eyebrow {
            color: #8d5d12;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .hero-title {
            color: #16313b;
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.15;
            margin: 0.25rem 0 0.45rem 0;
        }
        .hero-copy {
            color: #42525b;
            font-size: 1rem;
            line-height: 1.5;
            margin: 0;
        }
        .group-card {
            padding: 1rem 1rem 0.25rem 1rem;
            margin-bottom: 1rem;
        }
        .group-title {
            color: #16313b;
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .group-copy {
            color: #66757d;
            font-size: 0.9rem;
            margin-bottom: 0.8rem;
        }
        .result-card {
            padding: 1.3rem;
            min-height: 380px;
        }
        .result-title {
            color: #16313b;
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 0.6rem;
        }
        .risk-pill {
            display: inline-block;
            padding: 0.4rem 0.8rem;
            border-radius: 999px;
            color: white;
            font-weight: 700;
            font-size: 0.92rem;
            margin-top: 0.35rem;
            margin-bottom: 0.75rem;
        }
        .risk-note {
            color: #42525b;
            font-size: 0.96rem;
            line-height: 1.45;
        }
        .risk-gauge-wrap {
            display: flex;
            justify-content: center;
            margin: 0.3rem 0 1rem 0;
        }
        .risk-gauge {
            width: 230px;
            height: 230px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }
        .risk-gauge::before {
            content: "";
            width: 166px;
            height: 166px;
            border-radius: 50%;
            background: #fffdf9;
            box-shadow: inset 0 0 0 1px rgba(26, 35, 42, 0.06);
        }
        .risk-gauge-content {
            position: absolute;
            text-align: center;
        }
        .risk-gauge-value {
            color: #16313b;
            font-size: 2.2rem;
            font-weight: 800;
            line-height: 1;
        }
        .risk-gauge-label {
            color: #6a7880;
            font-size: 0.88rem;
            margin-top: 0.25rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-eyebrow">AIIMS Delhi Prospective Cohort</div>
            <div class="hero-title">30-Day Mortality LASSO Calculator</div>
            <p class="hero-copy">
                Estimate patient-level mortality probability from the fitted LASSO logistic regression model.
                The calculator uses the saved pipeline, so preprocessing and coefficients stay aligned.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_gauge(risk, category):
    bounded_risk = min(max(float(risk), 0.0), 1.0)
    degrees = bounded_risk * 360
    st.markdown(
        f"""
        <div class="risk-gauge-wrap">
            <div
                class="risk-gauge"
                style="background: conic-gradient({category['color']} 0deg {degrees:.1f}deg, #e8ece8 {degrees:.1f}deg 360deg);"
            >
                <div class="risk-gauge-content">
                    <div class="risk-gauge-value">{bounded_risk:.1%}</div>
                    <div class="risk-gauge-label">Predicted Risk</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_single_patient_form(ui_metadata):
    patient_values = {}
    for group_title, group_columns in INPUT_GROUPS:
        st.markdown(
            f"""
            <div class="group-card">
                <div class="group-title">{group_title}</div>
                <div class="group-copy">Enter the variables exactly as defined in the training dataset.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        left_column, right_column = st.columns(2)
        for index, column_name in enumerate(group_columns):
            container = left_column if index % 2 == 0 else right_column
            with container:
                if column_name in CONTINUOUS_COLUMNS:
                    patient_values[column_name] = st.number_input(
                        column_name,
                        value=numeric_default(ui_metadata, column_name),
                        step=0.1,
                        format="%.3f",
                    )
                else:
                    options = categorical_options(ui_metadata, column_name)
                    patient_values[column_name] = st.selectbox(
                        column_name,
                        options=options,
                        index=0,
                        format_func=lambda raw_value, col=column_name: format_categorical_label(col, raw_value),
                    )

    return patient_values


def main():
    inject_app_styles()
    render_hero()

    try:
        ui_metadata = load_ui_metadata()
        model, model_status = load_or_train_model()
    except Exception as exc:
        st.error(f"App setup failed: {type(exc).__name__}: {exc}")
        st.stop()

    st.sidebar.header("Model Summary")
    st.sidebar.write(f"Center: `{ui_metadata['target_center']}`")
    st.sidebar.write(f"Outcome: `{ui_metadata['outcome']}`")
    st.sidebar.write(f"Model status: `{model_status}`")
    st.sidebar.write(f"Patients: `{ui_metadata['sample_size']}`")
    st.sidebar.write(f"Deaths: `{ui_metadata['deaths']}`")
    st.sidebar.write(f"Prevalence: `{ui_metadata['prevalence']:.1%}`")
    st.sidebar.write(f"Model file locations: `{MODEL_PATH.name}` or `{ROOT_MODEL_PATH.name}`")
    st.sidebar.write(f"Metadata file locations: `{METADATA_PATH.name}` or `{ROOT_METADATA_PATH.name}`")

    st.info(
        "Binary fields are shown as Yes/No, and gender is displayed as Female/Male for easier entry. "
        "For batch CSV prediction, the uploaded file must still contain the original predictor columns."
    )
    st.caption("Gender is currently displayed as Female = 0 and Male = 1 based on the stored source coding.")

    tab_single, tab_batch = st.tabs(["Single Patient", "Batch CSV"])

    with tab_single:
        input_column, result_column = st.columns([1.45, 1.0], gap="large")

        with input_column:
            with st.form("single_patient_form"):
                patient_values = build_single_patient_form(ui_metadata)
                submitted = st.form_submit_button("Calculate 30-day mortality risk", use_container_width=True)

        with result_column:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<div class="result-title">Prediction Summary</div>', unsafe_allow_html=True)

            if submitted:
                risk = predict_single_patient(model, patient_values)
                category = get_risk_category(risk)

                render_probability_gauge(risk, category)
                st.markdown(
                    f'<div class="risk-pill" style="background:{category["color"]};">{category["label"]}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="risk-note">{category["message"]}</div>',
                    unsafe_allow_html=True,
                )
                st.metric("Predicted 30-day mortality probability", f"{risk:.1%}")
                st.caption(
                    "Risk categories are display aids in this app. You can adjust the cut points in the code if your team uses different thresholds."
                )
                st.dataframe(
                    pd.DataFrame([patient_values]),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.markdown(
                    """
                    <div class="risk-note">
                        Enter patient values on the left and submit the form to generate the predicted
                        mortality probability, visual gauge, and risk category.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)

    with tab_batch:
        uploaded_file = st.file_uploader(
            "Upload a CSV file with the same predictor columns",
            type=["csv"],
        )

        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
            except Exception as exc:
                st.error(f"Could not read CSV: {type(exc).__name__}: {exc}")
            else:
                missing_columns = [
                    column_name for column_name in PREDICTOR_COLUMNS if column_name not in batch_df.columns
                ]
                if missing_columns:
                    st.error(f"Missing columns in uploaded CSV: {missing_columns}")
                else:
                    batch_input = batch_df[PREDICTOR_COLUMNS].copy()
                    for column_name in CONTINUOUS_COLUMNS:
                        batch_input[column_name] = pd.to_numeric(
                            batch_input[column_name],
                            errors="coerce",
                        )

                    batch_df["predicted_30d_mortality_risk"] = model.predict_proba(batch_input)[:, 1]
                    batch_df["risk_category"] = batch_df["predicted_30d_mortality_risk"].apply(
                        lambda risk: get_risk_category(risk)["label"]
                    )

                    st.dataframe(batch_df, use_container_width=True)
                    st.download_button(
                        "Download predictions CSV",
                        batch_df.to_csv(index=False).encode("utf-8"),
                        file_name="tbi_30d_mortality_predictions.csv",
                        mime="text/csv",
                    )


if __name__ == "__main__":
    main()
