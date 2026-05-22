import streamlit as st
import mlflow
import pandas as pd
import numpy as np

st.set_page_config(page_title="Diabetes Classifier", page_icon="🩺", layout="centered")

mlflow.set_tracking_uri("sqlite:///mlflow.db")

@st.cache_resource
def load_model():
    return mlflow.pyfunc.load_model("models:/Diabetes_RandomForest/1")

model = load_model()

st.markdown("""
    <style>
        .stApp { background: #f8f9fa; }
        .main > div { padding: 2rem; }
        h1 { color: #1a237e; font-size: 2.2rem; }
        .stButton > button { background: #1a237e; color: white; width: 100%; border-radius: 8px; padding: 0.6rem; font-size: 1.1rem; }
        .stButton > button:hover { background: #283593; }
        div[data-testid="metric-container"] { background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 Diabetes Risk Classifier")
st.markdown("*MLflow Model Registry — RandomForest v1*")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose", 0, 300, 120)
    blood_pressure = st.number_input("BloodPressure", 0, 200, 80)
    skin_thickness = st.number_input("SkinThickness", 0, 100, 20)

with col2:
    insulin = st.number_input("Insulin", 0, 900, 30)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0, step=0.1)
    dpf = st.number_input("DiabetesPedigreeFunction", 0.0, 3.0, 0.5, step=0.01)
    age = st.number_input("Age", 1, 120, 30)

st.markdown("---")

if st.button("🔍 Predict Diabetes Risk"):
    input_df = pd.DataFrame([[
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, dpf, age
    ]], columns=[
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ])

    raw_model = model.get_raw_model()
    prob = raw_model.predict_proba(input_df)[0]
    pred = int(prob[1] >= 0.5)
    risk_pct = round(prob[1] * 100, 1)
    no_risk_pct = round(prob[0] * 100, 1)

    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Prediction", "🟢 Non-Diabetic" if pred == 0 else "🔴 Diabetic")
    col_b.metric("Risk Probability", f"{risk_pct}%")
    col_c.metric("Confidence", f"{no_risk_pct}%" if pred == 0 else f"{risk_pct}%")

    if risk_pct > 70:
        st.warning(f"⚠️ High risk ({risk_pct}%). Consult a healthcare professional.")
    elif risk_pct > 40:
        st.info(f"⚠️ Moderate risk ({risk_pct}%). Lifestyle review recommended.")
    else:
        st.success(f"✅ Low risk ({risk_pct}%). Maintain healthy habits.")

    st.markdown("### Feature Impact")
    importances = raw_model.feature_importances_
    feat_df = pd.DataFrame({"Feature": input_df.columns, "Importance": importances}).sort_values("Importance", ascending=False)
    st.bar_chart(feat_df.set_index("Feature"), height=300)
