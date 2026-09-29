# Asteroid Diameter Prediction

A Streamlit application for estimating asteroid diameter with three saved Keras regression models. Predictions use the fitted preprocessing pipeline included with the model artifacts.

## Run the application

From the project root in PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. The saved preprocessor was created with scikit-learn 1.6.1, so keep that version when loading `preprocessor.pkl`.

## Project structure

```text
.
├── app.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
├── Dataset/
│   └── Sample.txt              # Empty placeholder; training CSV is not included
├── Model/
│   └── Asteriod.ipynb          # Training and evaluation notebook
└── Trained Models/
    ├── best_model_baseline.keras
    ├── best_model_dropout.keras
    ├── best_model_lr_tuned.keras
    └── preprocessor.pkl
```

The trained artifacts are enough to run predictions. The original training CSV and held-out evaluation results are not present, so the application does not report benchmark scores or support retraining from this checkout.

## Prediction inputs

The app collects the same 20 raw fields recorded by the fitted pipeline: 19 numeric fields (`neo`, `pha`, `H`, `albedo`, `e`, `a`, `q`, `i`, `om`, `w`, `ma`, `ad`, `n`, `tp`, `per`, `per_y`, `moid`, `moid_ld`, and `rms`) plus the orbital `class`. It transforms each row with the saved preprocessor before passing it to a selected model.