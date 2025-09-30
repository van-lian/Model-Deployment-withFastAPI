import streamlit as st
import os
import pandas as pd
import pickle
import requests
from typing import List

st.set_page_config(page_title="Obesity Prediction App (Local)", page_icon="🍏", layout="centered")

st.sidebar.title("About")
st.sidebar.info(
    """
    This app predicts obesity category using a remote FastAPI backend.
    
    - Fill in the form on the main page.
    - Click **Predict** to call the FastAPI `/predict` endpoint.
    - Configure the backend URL below or via `FASTAPI_URL`.
    """
)

# Inference mode
FASTAPI_URL_DEFAULT = os.getenv("FASTAPI_URL", "")
mode = st.sidebar.radio("Inference mode", ["Local", "FastAPI"], index=0)
fastapi_url = None
if mode == "FastAPI":
    fastapi_url = st.sidebar.text_input("FastAPI base URL", FASTAPI_URL_DEFAULT or "http://127.0.0.1:8000")
    def check_api_health(base_url: str):
        try:
            r = requests.get(f"{base_url.rstrip('/')}/health", timeout=5)
            return r.ok
        except Exception:
            return False
if fastapi_url:
    healthy = check_api_health(fastapi_url)
    if healthy:
        st.sidebar.success("Backend healthy")
    else:
        st.sidebar.warning("Backend not reachable (check URL and /health)")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

REQUIRED_ARTIFACTS: List[str] = [
    'best_rf_model.pkl',
    'age_scaler.pkl',
    'weight_scaler.pkl',
    'onehot_encoder.pkl',
    'ordinal_encoder.pkl',
    'label_encoder.pkl',
    'expected_features.pkl',
]

def list_missing_artifacts(model_dir: str) -> List[str]:
    missing = []
    for filename in REQUIRED_ARTIFACTS:
        if not os.path.exists(os.path.join(model_dir, filename)):
            missing.append(filename)
    return missing

# Artifact availability notice (for Local mode)
missing_artifacts = list_missing_artifacts(MODEL_DIR)
if mode == "Local":
    if missing_artifacts:
        st.sidebar.error("Missing model artifacts in `model/`:")
        for f in missing_artifacts:
            st.sidebar.code(f)
        st.stop()

@st.cache_resource(show_spinner=False)
def get_predictor():
    return ObesityPredictor()

class ObesityPredictor:
    def __init__(self):
        self.components = {}
        for name in ['best_rf_model', 'age_scaler', 'weight_scaler', 'onehot_encoder',
                    'ordinal_encoder', 'label_encoder', 'expected_features']:
            with open(os.path.join(MODEL_DIR, f'{name}.pkl'), 'rb') as f:
                self.components[name] = pickle.load(f)

    def preprocess(self, data):
        df = pd.DataFrame([data])
        df['Age'] = self.components['age_scaler'].transform(df[['Age']])
        df['Weight'] = self.components['weight_scaler'].transform(df[['Weight']])

        onehot_cols = ['MTRANS', 'Gender']
        if all(col in df.columns for col in onehot_cols):
            onehot_df = pd.DataFrame(
                self.components['onehot_encoder'].transform(df[onehot_cols]),
                columns=self.components['onehot_encoder'].get_feature_names_out(onehot_cols)
            )
            df = df.drop(columns=onehot_cols)
            df = pd.concat([df.reset_index(drop=True), onehot_df.reset_index(drop=True)], axis=1)

        ordinal_cols = ['CALC', 'CAEC']
        present_ordinal_cols = [col for col in ordinal_cols if col in df.columns]
        if len(present_ordinal_cols) == 2:
            try:
                ordinal_transformed = self.components['ordinal_encoder'].transform(df[ordinal_cols])
                df[ordinal_cols] = ordinal_transformed
            except Exception:
                calc_mapping = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
                caec_mapping = {'Sometimes': 0, 'Frequently': 1, 'no': 2, 'Always': 3}
                df['CALC'] = df['CALC'].map(calc_mapping).fillna(0).astype(int)
                df['CAEC'] = df['CAEC'].map(caec_mapping).fillna(0).astype(int)
        else:
            for col in ordinal_cols:
                if col not in df.columns:
                    df[col] = 0

        binary_cols = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].map({'yes': 1, 'no': 0, 'Yes': 1, 'No': 0})

        expected_features = self.components['expected_features']
        for feat in expected_features:
            if feat not in df.columns:
                df[feat] = 0
        df = df[expected_features]
        return df

    def predict(self, data):
        X = self.preprocess(data)
        pred = self.components['best_rf_model'].predict(X)
        try:
            if pred.ndim == 1:
                pred_2d = pred.reshape(-1, 1)
            else:
                pred_2d = pred
            result = self.components['label_encoder'].inverse_transform(pred_2d)[0][0]
            return result
        except Exception:
            return pred[0]

def user_input_form():
    with st.form("obesity_form"):
        st.subheader("Personal Information")
        col1, col2 = st.columns(2)
        with col1:
            Age = st.number_input("Age", min_value=0.0, max_value=120.0, value=25.0, help="Enter your age in years.")
            Weight = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0, value=70.0, help="Enter your weight in kilograms.")
            Height = st.number_input("Height (m)", min_value=0.0, max_value=2.5, value=1.7, help="Enter your height in meters.")
            Gender = st.selectbox("Gender", ["Male", "Female"], help="Select your gender.")
        with col2:
            family_history_with_overweight = st.selectbox("Family history with overweight", ["yes", "no"], help="Does your family have a history of overweight?")
            SMOKE = st.selectbox("Do you smoke?", ["yes", "no"], help="Do you currently smoke?")
            SCC = st.selectbox("Do you monitor your calorie consumption? (SCC)", ["yes", "no"], help="Do you monitor your daily calorie intake?")
            MTRANS = st.selectbox("Transportation method (MTRANS)", ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"], help="Your main mode of transportation.")

        st.markdown("---")
        st.subheader("Lifestyle & Eating Habits")
        col3, col4 = st.columns(2)
        with col3:
            FCVC = st.number_input("Frequency of vegetable consumption (FCVC)", min_value=0.0, max_value=3.0, value=2.0, help="How often do you eat vegetables? (0-3)")
            NCP = st.number_input("Number of main meals (NCP)", min_value=1.0, max_value=5.0, value=3.0, help="How many main meals do you have per day?")
            FAF = st.number_input("Physical activity frequency (FAF)", min_value=0.0, max_value=3.0, value=1.0, help="How often do you exercise per week? (0-3)")
            TUE = st.number_input("Time using technology devices (TUE)", min_value=0.0, max_value=2.0, value=1.0, help="Average hours per day using technology devices.")
        with col4:
            FAVC = st.selectbox("Frequent high caloric food consumption (FAVC)", ["yes", "no"], help="Do you often eat high-calorie foods?")
            CALC = st.selectbox("Alcohol consumption (CALC)", ["no", "Sometimes", "Frequently", "Always"], help="How often do you consume alcohol?")
            CAEC = st.selectbox("Consumption of food between meals (CAEC)", ["no", "Sometimes", "Frequently", "Always"], help="How often do you eat between meals?")

        st.markdown(":information_source: **All information is confidential and used only for prediction purposes.**")
        submitted = st.form_submit_button("Predict")
        reset = st.form_submit_button("Reset")

        if reset:
            st.rerun()

        if submitted:
            input_data = {
                "Age": Age,
                "Weight": Weight,
                "Height": Height,
                "FCVC": FCVC,
                "NCP": NCP,
                "FAF": FAF,
                "TUE": TUE,
                "family_history_with_overweight": family_history_with_overweight,
                "FAVC": FAVC,
                "SMOKE": SMOKE,
                "SCC": SCC,
                "MTRANS": MTRANS,
                "Gender": Gender,
                "CALC": CALC,
                "CAEC": CAEC
            }
            if mode == "FastAPI":
                if not fastapi_url:
                    st.error("Please provide a FastAPI base URL in the sidebar.")
                else:
                    with st.spinner("Calling FastAPI backend..."):
                        try:
                            url = f"{fastapi_url.rstrip('/')}/predict"
                            resp = requests.post(url, json=input_data, timeout=20)
                            resp.raise_for_status()
                            result = resp.json()
                            prediction = result.get("prediction", "No result")
                            st.success("Prediction received!")
                            st.markdown(f"""
                            <div style='background-color:#e6ffe6;padding:20px;border-radius:10px;'>
                            <h3 style='color:#228B22;'>Prediction Result</h3>
                            <p style='font-size:22px'><b>{prediction}</b></p>
                            </div>
                            """, unsafe_allow_html=True)
                        except requests.exceptions.Timeout:
                            st.error("Backend request timed out (20s). Check server load/network.")
                        except requests.exceptions.ConnectionError:
                            st.error("Could not connect to backend. Verify the URL and that the server is running.")
                        except requests.exceptions.HTTPError as e:
                            try:
                                detail = resp.json().get('detail', str(e))
                            except Exception:
                                detail = str(e)
                            st.error(f"Backend returned an error: {detail}")
                        except Exception as e:
                            st.error(f"Prediction failed: {e}")
            else:
                with st.spinner("Predicting using local model..."):
                    try:
                        predictor = get_predictor()
                        prediction = predictor.predict(input_data)
                        st.success("Prediction received!")
                        st.markdown(f"""
                        <div style='background-color:#e6ffe6;padding:20px;border-radius:10px;'>
                        <h3 style='color:#228B22;'>Prediction Result</h3>
                        <p style='font-size:22px'><b>{prediction}</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

user_input_form()
