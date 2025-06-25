import streamlit as st
import requests

st.set_page_config(page_title="Obesity Prediction App", page_icon="🍏", layout="centered")

# Sidebar with instructions
st.sidebar.title("About")
st.sidebar.info(
    """
    This app predicts obesity category based on your input data using a machine learning model.
    
    - Fill in the form on the main page.
    - Click **Predict** to get the result.
    - The prediction is powered by a FastAPI backend.
    """
)

st.title("Obesity Prediction App")
st.write("""
Enter your details below. All fields are required. The prediction result will appear after you click **Predict**.
""")

# Grouped input form
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
            st.experimental_rerun()

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
            with st.spinner("Sending data to backend and waiting for prediction..."):
                try:
                    url = "http://localhost:8000/predict"
                    response = requests.post(url, json=input_data)
                    if response.status_code == 200:
                        result = response.json()
                        st.success("Prediction received!")
                        st.markdown(f"""
                        <div style='background-color:#e6ffe6;padding:20px;border-radius:10px;'>
                        <h3 style='color:#228B22;'>Prediction Result</h3>
                        <p style='font-size:22px'><b>{result['prediction']}</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(f"Error: {response.json()['detail']}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")

user_input_form()
