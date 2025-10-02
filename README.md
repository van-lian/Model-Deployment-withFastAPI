# Obesity Prediction

Local-first Streamlit app to predict obesity categories from lifestyle and demographic inputs using pre-trained models stored in `model/`.

## Project Structure

```
.
├── streamlit_app.py        # Local Streamlit UI (runs fully offline)
├── main.py                 # Optional FastAPI backend (if you choose to serve an API)
├── model/                  # Pickled artifacts: model + encoders + expected_features
├── requirements.txt
└── README.md
```

## Setup

1) Create a Python environment and install dependencies

```bash
pip install -r requirements.txt
```

2) Ensure artifacts exist

Place the following files inside `model/`:

- `best_rf_model.pkl`
- `age_scaler.pkl`
- `weight_scaler.pkl`
- `onehot_encoder.pkl`
- `ordinal_encoder.pkl`
- `label_encoder.pkl`
- `expected_features.pkl`

## Run the Streamlit app (local-only)

```bash
streamlit run app.py
```

The app loads artifacts from `model/` and predicts locally. No cloud/Azure required. You can also run `streamlit_app.py` which presents a similar local UI.

## Optional: Run the FastAPI server

If you want an API, you can still run the FastAPI app:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health`
- `POST /predict`
- `GET /model-info`

## Notes

- The Streamlit app currently collects `Height` but the preprocessing uses the model’s `expected_features`. If `Height` was not used during training, it will be dropped when aligning to `expected_features`.
- The stored `label_encoder` is an `OrdinalEncoder` and expects 2D arrays. Handled internally.

## Streamlit Cloud deployment checklist

- Main file: set to `app.py` in app settings
- Python version: add `runtime.txt` with `python-3.11`
- Ensure `model/` exists in the repo with these files:
  - `best_rf_model.pkl`, `age_scaler.pkl`, `weight_scaler.pkl`, `onehot_encoder.pkl`, `ordinal_encoder.pkl`, `label_encoder.pkl`, `expected_features.pkl`
- Avoid Git LFS for these small artifacts or enable LFS fetch in Streamlit
- On first load, choose "Local" mode in the sidebar
- For FastAPI mode, provide a public base URL (not localhost); the app checks `/health` and includes clear timeouts/errors
