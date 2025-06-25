import streamlit as st
import requests
import os

# Set your Azure Web App URL here
# Replace 'your-webapp-name' with your actual Azure Web App name
API_URL = os.getenv("API_URL", "model-dep-hpcscafbhgbcdnbe.canadacentral-01.azurewebsites.net")

st.set_page_config(page_title="Obesity Prediction App (Azure)", page_icon="🍏", layout="centered")

st.sidebar.title("About")
st.sidebar.info(
    """
    This app predicts obesity category based on your input data using a machine learning model deployed on Azure.\n\n- Fill in the form on the main page.\n- Click **Predict** to get the result.\n- The prediction is powered by a cloud-based machine learning model.
    """
)

st.title("Obesity Prediction App (Azure)")
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

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        return response.status_code == 200
    except:
        return False

def user_input_form():
    # Check API health first
    if not check_api_health():
        st.error(f"⚠️ Cannot connect to the API at {API_URL}. Please check if the service is running.")
        st.info("Make sure your Azure Web App is deployed and running.")
        return
    else:
        st.success("✅ Connected to Azure API")

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
            
            with st.spinner("Predicting using Azure model..."):
                try:
                    url = f"{API_URL}/predict"
                    response = requests.post(url, json=input_data, timeout=30)
                    response.raise_for_status()
                    result = response.json()
                    prediction = result.get("prediction", "No result")
                    
                    st.markdown(f"""
                    <div style='background: #222; padding: 20px 28px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.12); margin-top: 24px; max-width: 480px; margin-left: auto; margin-right: auto; border: 2px solid #228B22;'>
                        <h3 style='color:#cae00d; margin-bottom: 10px; font-size: 1.7rem;'>Prediction Result</h3>
                        <p style='font-size:1.3rem; color:#fff; font-weight:600; letter-spacing:1px; margin: 0;'>{prediction.replace('_', ' ')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {e}")
                    st.info("Please check if your Azure Web App is running and accessible.")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

user_input_form()

# Add footer with API info
st.markdown("---")
st.markdown(f"**API Endpoint:** `{API_URL}`")
