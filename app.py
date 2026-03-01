import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ── PAGE CONFIGURATION ──
st.set_page_config(
    page_title="CardioSense AI | Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
)

# ── CUSTOM CSS FOR MEDICAL THEME ──
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #1e3d59;
    }
    footer {
        visibility: hidden;
    }
    .footer-custom {
        text-align: center;
        padding: 20px;
        color: #6c757d;
        border-top: 1px solid #dee2e6;
        margin-top: 50px;
    }
</style>
""", unsafe_allow_html=True)

# ── HEADER ──
st.title("🫀 CardioSense AI")
st.markdown("### Clinical Heart Disease Prediction & Risk Assessment")
st.markdown("---")

# ── HELPER FUNCTIONS ──

@st.cache_resource
def load_models():
    """Load the trained models and scaler."""
    try:
        svm = joblib.load('svm_model.pkl')
        rf = joblib.load('rf_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return svm, rf, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

def engineer_features(data_dict):
    """Implement 7 engineered features based on the pipeline."""
    # Base features
    age = data_dict['Age']
    bp = data_dict['BP']
    chol = data_dict['Cholesterol']
    cp = data_dict['Chest pain type']
    thal = data_dict['Thallium']
    max_hr = data_dict['Max HR']
    fbs = data_dict['FBS over 120']
    angina = data_dict['Exercise angina']

    # 1. Age Group
    if age <= 45: age_group = 1
    elif age <= 60: age_group = 2
    else: age_group = 3

    # 2. Hypertension (BP > 140)
    hypertension = 1 if bp > 140 else 0

    # 3. High Cholesterol (> 240)
    high_cholesterol = 1 if chol > 240 else 0

    # 4. HR Reserve
    hr_reserve = (220 - age) - max_hr

    # 5. Asymptomatic CP
    asymptomatic_cp = 1 if cp == 4 else 0

    # 6. Reversible Defect
    reversible_defect = 1 if thal == 7 else 0

    # 7. Risk Score
    risk_score = hypertension + high_cholesterol + fbs + angina + asymptomatic_cp

    # Add to dictionary
    data_dict['Age_Group'] = age_group
    data_dict['Hypertension'] = hypertension
    data_dict['High_Cholesterol'] = high_cholesterol
    data_dict['HR_Reserve'] = hr_reserve
    data_dict['Asymptomatic_CP'] = asymptomatic_cp
    data_dict['Reversible_Defect'] = reversible_defect
    data_dict['Risk_Score'] = risk_score
    
    return data_dict

# ── LOGIC ──

svm, rf, scaler = load_models()

if not svm:
    st.stop()

# ── SIDEBAR INPUTS ──
st.sidebar.header("📋 Clinical Parameters")

# Organizing features for easier access
# FEATURE_COLS: ['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina', 'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium']

def get_user_inputs():
    inputs = {}
    inputs['Age'] = st.sidebar.number_input("Age", 1, 120, 50)
    inputs['Sex'] = st.sidebar.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    inputs['Chest pain type'] = st.sidebar.selectbox("Chest Pain Type", options=[1, 2, 3, 4], 
                                                format_func=lambda x: {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-anginal Pain", 4: "Asymptomatic"}[x])
    inputs['BP'] = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    inputs['Cholesterol'] = st.sidebar.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
    inputs['FBS over 120'] = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[1, 0], format_func=lambda x: "True" if x == 1 else "False")
    inputs['EKG results'] = st.sidebar.selectbox("Resting EKG Results", options=[0, 1, 2], 
                                            format_func=lambda x: {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}[x])
    inputs['Max HR'] = st.sidebar.slider("Maximum Heart Rate Achieved", 60, 220, 150)
    inputs['Exercise angina'] = st.sidebar.selectbox("Exercise Induced Angina", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    inputs['ST depression'] = st.sidebar.number_input("ST Depression Induced by Exercise", 0.0, 10.0, 1.0, 0.1)
    inputs['Slope of ST'] = st.sidebar.selectbox("Slope of Peak Exercise ST Segment", options=[1, 2, 3],
                                            format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x])
    inputs['Number of vessels fluro'] = st.sidebar.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
    inputs['Thallium'] = st.sidebar.selectbox("Thallium Stress Test Result", options=[3, 6, 7],
                                         format_func=lambda x: {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}[x])
    return inputs

inputs = get_user_inputs()

# ── PREDICTION & RESULTS ──

if st.sidebar.button("💡 Analyze Heart Health"):
    # Apply Feature Engineering
    full_data = engineer_features(inputs)
    
    # Matching the training columns order
    # FEATURE_COLS_ENGINEERED = FEATURE_COLS + ['Age_Group', 'Hypertension', 'High_Cholesterol', 'HR_Reserve', 'Asymptomatic_CP', 'Reversible_Defect', 'Risk_Score']
    cols = ['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina', 'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium', 
            'Age_Group', 'Hypertension', 'High_Cholesterol', 'HR_Reserve', 'Asymptomatic_CP', 'Reversible_Defect', 'Risk_Score']
    
    input_df = pd.DataFrame([full_data])[cols]
    
    # Scale data for SVM
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prob = svm.predict_proba(input_scaled)[0][1]
    pred = int(prob > 0.5)
    
    # Display Layout
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("📊 Analysis Results")
        
        # Risk Calculation
        if prob < 0.3:
            risk_label = "Low"
            color = "green"
        elif prob < 0.7:
            risk_label = "Moderate"
            color = "orange"
        else:
            risk_label = "High"
            color = "red"
            
        st.write(f"Based on clinical data, the patient's heart disease risk is:")
        st.markdown(f"<h2 style='color:{color}; text-align:center;'>{risk_label} ({prob*100:.1f}%)</h2>", unsafe_allow_html=True)
        
        st.progress(prob)
        
        if pred == 1:
            st.warning("⚠️ High diagnostic signal for Heart Disease detected. Clinical follow-up recommended.")
        else:
            st.success("✅ Diagnostic markers appear within normal ranges.")

    with col2:
        st.subheader("📈 Diagnostic Insights")
        # Feature importance from RF (more interpretable)
        importances = rf.feature_importances_
        feat_imp = pd.Series(importances, index=cols).sort_values(ascending=False).head(8)
        
        fig, ax = plt.subplots()
        sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="viridis", ax=ax)
        ax.set_title("Top 8 Risk Factors for this Profile")
        st.pyplot(fig)

    # ── ENGINEERED FEATURES DISPLAY ──
    st.markdown("---")
    st.subheader("⚙️ Derivative Biomarkers")
    cols_der = st.columns(4)
    cols_der[0].metric("Risk Score", f"{full_data['Risk_Score']}/5")
    cols_der[1].metric("HR Reserve", full_data['HR_Reserve'])
    cols_der[2].metric("Hypertension", "Yes" if full_data['Hypertension'] else "No")
    cols_der[3].metric("Age Group", f"Tier {full_data['Age_Group']}")

else:
    # Initial landing view
    st.info("👈 Enter clinical parameters in the sidebar and click 'Analyze Heart Health' to generate a report.")
    
    # Welcome Visual
    st.image("https://img.freepik.com/free-photo/doctor-examining-heart-beating-medical-stethoschope-health-care-concept_1150-12822.jpg", caption="Medical Diagnosis Assistance", use_container_width=True)

# ── FOOTER ──
st.markdown("""
<div class="footer-custom">
    <p>CardioSense AI v1.0.0 | © 2024 Medical ML Systems | Not a replacement for professional medical advice.</p>
</div>
""", unsafe_allow_html=True)
