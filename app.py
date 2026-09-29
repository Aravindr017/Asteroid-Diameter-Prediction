from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model


ROOT = Path(__file__).resolve().parent
MODEL_FILES = {
    "Baseline": "best_model_baseline.keras",
    "Learning-rate tuned": "best_model_lr_tuned.keras",
    "Dropout": "best_model_dropout.keras",
}
FEATURE_COLUMNS = (
    "neo",
    "pha",
    "H",
    "albedo",
    "e",
    "a",
    "q",
    "i",
    "om",
    "w",
    "ma",
    "ad",
    "n",
    "tp",
    "per",
    "per_y",
    "moid",
    "moid_ld",
    "class",
    "rms",
)

st.set_page_config(
    page_title="Asteroid diameter estimator",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_assets():
    models_dir = ROOT / "Trained Models"
    preprocessor_path = models_dir / "preprocessor.pkl"
    missing_files = [
        str(path)
        for path in [
            preprocessor_path,
            *(models_dir / name for name in MODEL_FILES.values()),
        ]
        if not path.is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Missing required model files: " + ", ".join(missing_files)
        )

    with preprocessor_path.open("rb") as file:
        preprocessor = pickle.load(file)

    actual_columns = tuple(preprocessor.feature_names_in_)
    if actual_columns != FEATURE_COLUMNS:
        raise ValueError(
            "The saved preprocessor feature schema does not match the app. "
            f"Expected {FEATURE_COLUMNS}, received {actual_columns}."
        )

    expected_input_size = len(preprocessor.get_feature_names_out())
    models = {}
    for label, filename in MODEL_FILES.items():
        model = load_model(models_dir / filename)
        if model.input_shape[-1] != expected_input_size:
            raise ValueError(
                f"{filename} expects {model.input_shape[-1]} inputs, "
                "but the saved "
                f"preprocessor produces {expected_input_size}."
            )
        models[label] = model

    classes = preprocessor.named_transformers_["cat"].named_steps[
        "onehot"
    ].categories_[0].tolist()
    return models, preprocessor, classes


def render_prediction_page():
    st.title("Asteroid diameter estimator", icon=":material/public:")
    st.caption(
        "Estimate diameter from the orbital and physical fields used to train "
        "the saved models."
    )

    try:
        models, preprocessor, asteroid_classes = load_assets()
    except Exception as error:
        st.error(f"The prediction assets could not be loaded: {error}")
        st.caption(
            "Check the files in `Trained Models` and install dependencies "
            "from `requirements.txt`."
        )
        return

    with st.form("asteroid_prediction", border=True):
        st.subheader("Orbital elements")
        orbital_left, orbital_right = st.columns(2)
        with orbital_left:
            a = st.number_input(
                "Semi-major axis a (AU)",
                min_value=0.001,
                value=2.75495,
                step=0.01,
            )
            e = st.number_input(
                "Eccentricity e",
                min_value=0.0,
                max_value=0.999999,
                value=0.13892,
                step=0.01,
            )
            q = st.number_input(
                "Perihelion distance q (AU)",
                min_value=0.0,
                value=2.36860,
                step=0.01,
            )
            inclination = st.number_input(
                "Inclination i (degrees)",
                min_value=0.0,
                max_value=180.0,
                value=9.33569,
                step=0.1,
            )
            om = st.number_input(
                "Longitude of ascending node om (degrees)",
                min_value=0.0,
                max_value=360.0,
                value=160.30134,
                step=0.1,
            )
            w = st.number_input(
                "Argument of perihelion w (degrees)",
                min_value=0.0,
                max_value=360.0,
                value=183.50950,
                step=0.1,
            )
            ma = st.number_input(
                "Mean anomaly ma (degrees)",
                min_value=0.0,
                max_value=360.0,
                value=188.27529,
                step=0.1,
            )

        with orbital_right:
            ad = st.number_input(
                "Aphelion distance ad (AU)",
                min_value=0.001,
                value=3.17351,
                step=0.01,
            )
            n = st.number_input(
                "Mean motion n (degrees/day)",
                min_value=0.0,
                value=0.21554,
                step=0.001,
            )
            tp = st.number_input(
                "Time of perihelion tp (Julian day)",
                min_value=2_300_000.0,
                max_value=2_600_000.0,
                value=2_459_007.39,
                step=1.0,
            )
            per = st.number_input(
                "Orbital period per (days)",
                min_value=0.0,
                value=1670.20,
                step=1.0,
            )
            per_y = st.number_input(
                "Orbital period per_y (years)",
                min_value=0.0,
                value=4.57276,
                step=0.1,
            )
            moid = st.number_input(
                "Earth MOID (AU)", min_value=0.0, value=1.38958, step=0.01
            )
            moid_ld = st.number_input(
                "Earth MOID (lunar distances)",
                min_value=0.0,
                value=540.78285,
                step=1.0,
            )
            rms = st.number_input(
                "Orbit-fit residual rms",
                min_value=0.0,
                value=0.54418,
                step=0.01,
            )

        st.subheader("Physical properties and classification")
        physical_left, physical_right = st.columns(2)
        with physical_left:
            absolute_magnitude = st.number_input(
                "Absolute magnitude H",
                min_value=-10.0,
                max_value=50.0,
                value=15.2,
                step=0.1,
            )
            albedo = st.number_input(
                "Geometric albedo",
                min_value=0.0,
                max_value=1.0,
                value=0.079,
                step=0.01,
            )
            asteroid_class = st.selectbox(
                "Orbital class",
                asteroid_classes,
                index=asteroid_classes.index("MBA"),
            )
        with physical_right:
            neo_label = st.selectbox("Near-Earth object (neo)", ["No", "Yes"])
            pha_label = st.selectbox(
                "Potentially hazardous (pha)", ["No", "Yes"]
            )
            selected_model = st.selectbox(
                "Prediction model", list(MODEL_FILES)
            )

        submitted = st.form_submit_button(
            "Estimate diameter",
            type="primary",
            icon=":material/rocket_launch:",
            width="stretch",
        )

    if not submitted:
        st.caption(
            "The default values are dataset medians. Select the fields for an "
            "object, then estimate."
        )
        return

    input_values = {
        "neo": int(neo_label == "Yes"),
        "pha": int(pha_label == "Yes"),
        "H": absolute_magnitude,
        "albedo": albedo,
        "e": e,
        "a": a,
        "q": q,
        "i": inclination,
        "om": om,
        "w": w,
        "ma": ma,
        "ad": ad,
        "n": n,
        "tp": tp,
        "per": per,
        "per_y": per_y,
        "moid": moid,
        "moid_ld": moid_ld,
        "class": asteroid_class,
        "rms": rms,
    }

    try:
        input_frame = pd.DataFrame(
            [input_values], columns=preprocessor.feature_names_in_
        )
        transformed = preprocessor.transform(input_frame)
        if not np.isfinite(transformed).all():
            raise ValueError(
                "Preprocessing produced non-finite values. "
                "Check the entered fields."
            )

        prediction = float(
            models[selected_model].predict(transformed, verbose=0).ravel()[0]
        )
        if not np.isfinite(prediction):
            raise ValueError(
                "The model returned a non-finite diameter estimate."
            )
    except Exception as error:
        st.error(f"Prediction failed: {error}")
        return

    st.success("Estimate complete")
    result_column, model_column = st.columns(2)
    with result_column:
        st.metric("Estimated diameter", f"{prediction:,.4f} km")
    with model_column:
        st.metric("Model", selected_model)
    st.caption(
        "This is a point estimate from the selected model; a confidence "
        "interval is not available."
    )

    with st.expander("Review submitted inputs"):
        st.dataframe(
            pd.DataFrame(
                {
                    "Input": list(input_values),
                    "Value": [str(value) for value in input_values.values()],
                }
            ),
            hide_index=True,
            width="stretch",
        )


def render_model_library_page():
    st.title("Model library", icon=":material/monitoring:")
    st.caption(
        "Details read from the saved Keras models and fitted preprocessing "
        "pipeline."
    )

    try:
        models, preprocessor, asteroid_classes = load_assets()
    except Exception as error:
        st.error(f"The model assets could not be loaded: {error}")
        return

    numeric_transformer = preprocessor.named_transformers_["num"]
    numeric_features = list(numeric_transformer.feature_names_in_)
    encoded_class_count = len(
        preprocessor.named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out()
    )
    st.metric("Available models", len(models))

    architecture = []
    for label, model in models.items():
        layers = [
            f"Dense({layer.units})"
            if hasattr(layer, "units")
            else f"Dropout({layer.rate:g})"
            for layer in model.layers
        ]
        architecture.append(
            {
                "Model": label,
                "Input features": model.input_shape[-1],
                "Architecture": " → ".join(layers),
                "Parameters": f"{model.count_params():,}",
            }
        )
    st.dataframe(pd.DataFrame(architecture), hide_index=True, width="stretch")

    st.subheader("Fitted input pipeline")
    first, second = st.columns(2)
    with first:
        st.metric("Numeric fields", len(numeric_features))
        st.caption(", ".join(numeric_features))
    with second:
        st.metric("Encoded class fields", encoded_class_count)
        st.caption("Orbital classes: " + ", ".join(asteroid_classes))

    st.info(
        "Benchmark scores are not shown because the project does not include "
        "the training dataset "
        "or saved held-out evaluation results."
    )


def render_project_page():
    st.title("Project information", icon=":material/folder_open:")
    st.write(
        "The Streamlit app performs inference from the checked-in Keras "
        "models "
        "and fitted preprocessor. The original training CSV is not included, "
        "so retraining and independent "
        "evaluation are not available from this folder alone."
    )

    expected_paths = [
        ("app.py", ROOT / "app.py"),
        ("Model/Asteriod.ipynb", ROOT / "Model" / "Asteriod.ipynb"),
        ("Dataset/dataset.csv", ROOT / "Dataset" / "dataset.csv"),
        (
            "Trained Models/preprocessor.pkl",
            ROOT / "Trained Models" / "preprocessor.pkl",
        ),
        *(
            (f"Trained Models/{name}", ROOT / "Trained Models" / name)
            for name in MODEL_FILES.values()
        ),
    ]
    status = pd.DataFrame(
        {
            "Path": [label for label, _ in expected_paths],
            "Status": [
                "Available" if path.is_file() else "Not included"
                for _, path in expected_paths
            ],
        }
    )
    st.dataframe(status, hide_index=True, width="stretch")
    st.caption(
        "`Dataset/Sample.txt` is an empty placeholder. It is not required to "
        "run predictions."
    )

    with st.expander("Input fields used by the model"):
        st.code(", ".join(FEATURE_COLUMNS), language="text")


with st.sidebar:
    st.title("Asteroid lab", anchor=False)
    page = st.radio(
        "Workspace",
        ["Predict diameter", "Model library", "Project information"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Asteroid diameter prediction · DNN regression")

if page == "Predict diameter":
    render_prediction_page()
elif page == "Model library":
    render_model_library_page()
else:
    render_project_page()
