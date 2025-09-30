import streamlit as st
import os
import pandas as pd
import pickle

# Local model directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

# Predictor (local)
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

st.set_page_config(page_title="Obesity Prediction App", page_icon="🍏", layout="centered")

st.sidebar.title("About")
st.sidebar.info(
    """
    This app predicts obesity category based on your input data using a local machine learning model.\n\n- Fill in the form on the main page.\n- Click **Predict** to get the result.\n- The prediction runs locally.
    """
)

st.title("Obesity Prediction App (Local)")
st.write("""
Enter your details below. All fields are required. The prediction result will appear after you click **Predict**.
""")

st.markdown("""
    <style>
    .stApp {background-color: #262730;}
    h1, h2, h3, h4, h5, h6, .stMarkdown, .stTextInput label, .stSelectbox label, .stNumberInput label {
        color: #fff !important;
    }
    .stButton>button {background-color: #228B22; color: white;}
    </style>
""", unsafe_allow_html=True)

def user_input_form():
    with st.form("obesity_form"):
        st.header("Personal & Demographic Information")
        col1, col2 = st.columns(2)
        with col1:
            Age = st.number_input("Age", min_value=0.0, max_value=120.0, value=25.0, help="Enter your age in years.")
            Weight = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0, value=70.0, help="Enter your weight in kilograms.")
            Height = st.number_input("Height (m)", min_value=0.0, max_value=2.5, value=1.7, help="Enter your height in meters.")
            Gender = st.selectbox("Gender", ["Male", "Female"], help="Select your gender.")
            MTRANS = st.selectbox("Transportation method (MTRANS)", ["Public_Transportation", "Walking", "Automobile", "Bike"], help="Your main mode of transportation.")
        with col2:
            family_history_with_overweight = st.selectbox("Family history with overweight", ["yes", "no"], help="Does your family have a history of overweight?")
            SMOKE = st.selectbox("Do you smoke?", ["yes", "no"], help="Do you currently smoke?")
            SCC = st.selectbox("Do you monitor your calorie consumption? (SCC)", ["yes", "no"], help="Do you monitor your daily calorie intake?")
            FAVC = st.selectbox("Frequent high caloric food consumption (FAVC)", ["yes", "no"], help="Do you often eat high-calorie foods?")
        
        st.markdown("---")
        st.header("Lifestyle & Eating Habits")
        col3, col4 = st.columns(2)
        with col3:
            FCVC = st.number_input("Frequency of vegetable consumption (FCVC)", min_value=0.0, max_value=3.0, value=2.0, help="How often do you eat vegetables? (0-3)")
            NCP = st.number_input("Number of main meals (NCP)", min_value=1.0, max_value=5.0, value=3.0, help="How many main meals do you have per day?")
            CH2O = st.number_input("Daily water consumption (CH2O, liters)", min_value=0.0, max_value=5.0, value=2.0, help="How many liters of water do you drink per day?")
        with col4:
            FAF = st.number_input("Physical activity frequency (FAF)", min_value=0.0, max_value=3.0, value=1.0, help="How often do you exercise per week? (0-3)")
            TUE = st.number_input("Time using technology devices (TUE)", min_value=0.0, max_value=2.0, value=1.0, help="Average hours per day using technology devices.")
        
        st.markdown("---")
        st.header("Eating & Drinking Habits")
        col5, col6 = st.columns(2)
        with col5:
            CALC = st.selectbox("Alcohol consumption (CALC)", ["no", "Sometimes", "Frequently", "Always"], help="How often do you consume alcohol?")
        with col6:
            CAEC = st.selectbox("Consumption of food between meals (CAEC)", ["no", "Sometimes", "Frequently", "Always"], help="How often do you eat between meals?")
        
        st.markdown(":information_source: **All information is confidential and used only for prediction purposes.**")
        submitted = st.form_submit_button("Predict", use_container_width=True)
        reset = st.form_submit_button("Reset", use_container_width=True)

        if reset:
            st.rerun()

        if submitted:
            input_data = {
                "Age": Age,
                "Weight": Weight,
                "Height": Height,
                "FCVC": FCVC,
                "NCP": NCP,
                "CH2O": CH2O,
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
            
            with st.spinner("Predicting using local model..."):
                try:
                    predictor = ObesityPredictor()
                    prediction = predictor.predict(input_data)
                    st.markdown(f"""
                    <div style='background: #222; padding: 20px 28px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.12); margin-top: 24px; max-width: 480px; margin-left: auto; margin-right: auto; border: 2px solid #228B22;'>
                        <h3 style='color:#cae00d; margin-bottom: 10px; font-size: 1.7rem;'>Prediction Result</h3>
                        <p style='font-size:1.3rem; color:#fff; font-weight:600; letter-spacing:1px; margin: 0;'>{prediction.replace('_', ' ')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

user_input_form()

