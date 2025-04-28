import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Health Prediction App",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f0f4f8;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #00bcd4;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #008c9e;
    }
    .stNumberInput label, .stSelectbox label {
        font-size: 20px;
        color: white;
    }
    .result-box {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
        text-align: center;
    }
    .header {
        font-size: 34px;
        color: #00bcd4;
        text-align: center;
        margin-bottom: 10px;
    }
    .subheader {
        font-size: 18px;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    .warning {
        color: #d32f2f;
        font-weight: bold;
    }
    .success {
        color: #388e3c;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Health Prediction")
    st.markdown("Select a dataset to predict health outcomes.")
    
    dataset = st.selectbox("Select Prediction Type", ["Diabetes", "Hospital Readmission", "Kidney Disease"])
    
    st.markdown("---")
    st.markdown("""
    **Instructions:**
    - Select a dataset.
    - Enter the required details.
    - Click 'Predict' to see results.
    
    **Note**: Predictions are for informational purposes. Consult a doctor for medical advice.
    """)
    st.markdown("Built with Streamlit")

# Main app
st.markdown('<div class="header">Health Prediction App</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Assess your health risk with a few clicks</div>', unsafe_allow_html=True)

# Load model based on dataset
model_path = {"Diabetes": "diabetes-model.pkl", 
              "Hospital Readmission": "hospital-model.pkl", 
              "Kidney Disease": "kidney-model.pkl"}[dataset]
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found! Please place it in the same directory.")
    st.stop()

try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Input form based on dataset
st.subheader("Enter Your Details")
with st.form(key="prediction_form"):
    if dataset == "Diabetes":
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1, help="Number of pregnancies")
            glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, max_value=300.0, value=100.0, step=1.0, help="Blood sugar level")
            blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=200.0, value=70.0, step=1.0, help="Diastolic blood pressure")
            skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0, step=1.0, help="Triceps skin fold thickness")
        with col2:
            insulin = st.number_input("Insulin (µU/mL)", min_value=0.0, max_value=1000.0, value=80.0, step=1.0, help="Serum insulin level")
            bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0, step=0.1, help="Body Mass Index")
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, step=0.01, help="Genetic diabetes score")
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=30, step=1, help="Age in years")
        
        # Input validation
        input_data = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [dpf],
            'Age': [age]
        })
        validation = glucose == 0 or bmi == 0

    elif dataset == "Hospital Readmission":
        col1, col2 = st.columns(2)
        with col1:
            age = st.selectbox("Age Group", ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'], help="Age range")
            time_in_hospital = st.number_input("Time in Hospital (days)", min_value=1, max_value=14, value=1, step=1, help="Days spent in hospital")
            n_lab_procedures = st.number_input("Lab Procedures", min_value=0, max_value=100, value=0, step=1, help="Number of lab tests")
            n_procedures = st.number_input("Procedures", min_value=0, max_value=10, value=0, step=1, help="Number of procedures")
            n_medications = st.number_input("Medications", min_value=1, max_value=100, value=1, step=1, help="Number of medications")
            n_outpatient = st.number_input("Outpatient Visits", min_value=0, max_value=50, value=0, step=1, help="Outpatient visits in last year")
            n_inpatient = st.number_input("Inpatient Admissions", min_value=0, max_value=50, value=0, step=1, help="Inpatient admissions in last year")
            n_emergency = st.number_input("Emergency Visits", min_value=0, max_value=50, value=0, step=1, help="Emergency visits in last year")
        
        with col2:
            medical_specialty = st.selectbox("Medical Specialty", ['InternalMedicine', 'Cardiology', 'Surgery', 'Other', 'Unknown'], help="Primary medical specialty")
            diag_1 = st.text_input("Primary Diagnosis (ICD9)", value="250.01", help="Primary diagnosis code")
            diag_2 = st.text_input("Secondary Diagnosis (ICD9)", value="401", help="Secondary diagnosis code")
            diag_3 = st.text_input("Additional Diagnosis (ICD9)", value="428", help="Additional diagnosis code")
            glucose_test = st.selectbox("Glucose Test Result", ['normal', 'abnormal', 'not tested'], help="Serum glucose test result")
            a1c_test = st.selectbox("A1C Test Result", ['normal', 'abnormal', 'not tested'], help="HbA1c test result")
            change = st.selectbox("Medication Change", ['yes', 'no'], help="Change in medication")
            diabetes_med = st.selectbox("Diabetes Medication", ['yes', 'no'], help="Prescribed diabetes medication")
        
        # Input data
        input_data = pd.DataFrame({
            'age': [age],
            'time_in_hospital': [time_in_hospital],
            'n_lab_procedures': [n_lab_procedures],
            'n_procedures': [n_procedures],
            'n_medications': [n_medications],
            'n_outpatient': [n_outpatient],
            'n_inpatient': [n_inpatient],
            'n_emergency': [n_emergency],
            'medical_specialty': [medical_specialty],
            'diag_1': [diag_1],
            'diag_2': [diag_2],
            'diag_3': [diag_3],
            'glucose_test': [glucose_test],
            'A1Ctest': [a1c_test],
            'change': [change],
            'diabetes_med': [diabetes_med]
        })
        validation = False  # No specific validation for readmission

    else:  # Kidney Disease
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=30, step=1, help="Age in years")
            blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=200.0, value=70.0, step=1.0, help="Blood pressure")
            specific_gravity = st.number_input("Specific Gravity", min_value=1.0, max_value=1.05, value=1.02, step=0.005, help="Urine specific gravity")
            albumin = st.number_input("Albumin", min_value=0.0, max_value=5.0, value=0.0, step=1.0, help="Albumin level in urine")
            sugar = st.number_input("Sugar", min_value=0.0, max_value=5.0, value=0.0, step=1.0, help="Sugar level in urine")
            red_blood_cells = st.selectbox("Red Blood Cells", ['normal', 'abnormal'], help="Red blood cells in urine")
            pus_cell = st.selectbox("Pus Cell", ['normal', 'abnormal'], help="Pus cells in urine")
            pus_cell_clumps = st.selectbox("Pus Cell Clumps", ['present', 'not present'], help="Presence of pus cell clumps")
            bacteria = st.selectbox("Bacteria", ['present', 'not present'], help="Presence of bacteria in urine")
            blood_glucose_random = st.number_input("Blood Glucose Random (mg/dL)", min_value=0.0, max_value=500.0, value=100.0, step=1.0, help="Random blood glucose level")
            blood_urea = st.number_input("Blood Urea (mg/dL)", min_value=0.0, max_value=200.0, value=30.0, step=1.0, help="Blood urea level")
            serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0, max_value=20.0, value=1.0, step=0.1, help="Serum creatinine level")
        
        with col2:
            sodium = st.number_input("Sodium (mEq/L)", min_value=0.0, max_value=200.0, value=135.0, step=1.0, help="Sodium level")
            potassium = st.number_input("Potassium (mEq/L)", min_value=0.0, max_value=10.0, value=4.0, step=0.1, help="Potassium level")
            hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=20.0, value=12.0, step=0.1, help="Hemoglobin level")
            packed_cell_volume = st.number_input("Packed Cell Volume (%)", min_value=0.0, max_value=60.0, value=40.0, step=1.0, help="Packed cell volume")
            white_blood_cell_count = st.number_input("White Blood Cell Count (cells/cmm)", min_value=0.0, max_value=20000.0, value=7000.0, step=100.0, help="White blood cell count")
            red_blood_cell_count = st.number_input("Red Blood Cell Count (millions/cmm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1, help="Red blood cell count")
            hypertension = st.selectbox("Hypertension", ['yes', 'no'], help="Presence of hypertension")
            diabetes_mellitus = st.selectbox("Diabetes Mellitus", ['yes', 'no'], help="Presence of diabetes")
            coronary_artery_disease = st.selectbox("Coronary Artery Disease", ['yes', 'no'], help="Presence of coronary artery disease")
            appetite = st.selectbox("Appetite", ['good', 'poor'], help="Appetite status")
            pedal_edema = st.selectbox("Pedal Edema", ['yes', 'no'], help="Presence of pedal edema")
            anemia = st.selectbox("Anemia", ['yes', 'no'], help="Presence of anemia")
        
        # Input data
        input_data = pd.DataFrame({
            'Age': [age],
            'Blood_Pressure': [blood_pressure],
            'Specific_Gravity': [specific_gravity],
            'Albumin': [albumin],
            'Sugar': [sugar],
            'Red_Blood_Cells': [red_blood_cells],
            'Pus_Cell': [pus_cell],
            'Pus_Cell_Clumps': [pus_cell_clumps],
            'Bacteria': [bacteria],
            'Blood_Glucose_Random': [blood_glucose_random],
            'Blood_Urea': [blood_urea],
            'Serum_Creatinine': [serum_creatinine],
            'Sodium': [sodium],
            'Potassium': [potassium],
            'Hemoglobin': [hemoglobin],
            'Packed_Cell_Volume': [packed_cell_volume],
            'White_Blood_Cell_Count': [white_blood_cell_count],
            'Red_Blood_Cell_Count': [red_blood_cell_count],
            'Hypertension': [hypertension],
            'Diabetes_Mellitus': [diabetes_mellitus],
            'Coronary_Artery_Disease': [coronary_artery_disease],
            'Appetite': [appetite],
            'Pedal_Edema': [pedal_edema],
            'Anemia': [anemia]
        })
        validation = blood_glucose_random == 0 or serum_creatinine == 0  # Basic validation

    # Submit button
    submit_button = st.form_submit_button(label="Predict")

# Prediction
if submit_button:
    if validation:
        st.warning("Please enter valid values (e.g., Blood Glucose and Serum Creatinine cannot be 0).")
    else:
        try:
            # Predict
            prediction = model.predict(input_data)[0]
            confidence = model.predict_proba(input_data)[0]
            confidence_score = max(confidence) * 100

            # Show result
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.subheader("Prediction Result")
            if dataset == "Diabetes":
                if prediction == 1:
                    st.markdown('<p class="warning">⚠️ Diabetes Risk Detected</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="success">✅ No Diabetes Risk</p>', unsafe_allow_html=True)
            elif dataset == "Hospital Readmission":
                if prediction == 1:
                    st.markdown('<p class="warning">⚠️ Hospital Readmission Likely</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="success">✅ Hospital Readmission Unlikely</p>', unsafe_allow_html=True)
            else:  # Kidney Disease
                if prediction == 1:
                    st.markdown('<p class="warning">⚠️ Kidney Disease Risk Detected</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="success">✅ No Kidney Disease Risk</p>', unsafe_allow_html=True)
            
            st.write(f"**Confidence**: {confidence_score:.2f}%")
            st.write("**Disclaimer**: This is a prediction. Consult a healthcare professional for a diagnosis.")
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# Footer
st.markdown("---")
st.markdown('<p style="text-align: center; color: #666;">Health Prediction App | Powered by Streamlit</p>', unsafe_allow_html=True)