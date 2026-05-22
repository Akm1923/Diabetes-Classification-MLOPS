import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Diabetes Classifier", page_icon="🩺", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("models/rf_model.pkl")

model = load_model()

FEATURES = {
    "Pregnancies":         {"min": 0,  "max": 20, "val": 1,  "unit": "count",     "icon": "🤰", "desc": "Number of pregnancies"},
    "Glucose":             {"min": 0,  "max": 250,"val": 120,"unit": "mg/dL",     "icon": "🍬", "desc": "Plasma glucose (OGTT)"},
    "BloodPressure":       {"min": 0,  "max": 130,"val": 70, "unit": "mm Hg",     "icon": "❤️", "desc": "Diastolic blood pressure"},
    "SkinThickness":       {"min": 0,  "max": 70, "val": 20, "unit": "mm",        "icon": "📏", "desc": "Triceps skin fold thickness"},
    "Insulin":             {"min": 0,  "max": 600,"val": 30, "unit": "mu U/ml",   "icon": "💉", "desc": "2-Hour serum insulin"},
    "BMI":                 {"min": 0.0,"max": 60.0,"val":25.0,"unit": "kg/m²",    "icon": "⚖️", "desc": "Body mass index"},
    "DiabetesPedigreeFunction": {"min": 0.0, "max": 2.5, "val": 0.5, "unit": "",  "icon": "🧬", "desc": "Genetic diabetes risk score"},
    "Age":                 {"min": 1,  "max": 120,"val": 30, "unit": "years",     "icon": "🎂", "desc": "Patient age"},
}

st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    .main > div { padding: 1.5rem 2rem; }
    h1 { color: #1a237e; font-weight: 700; }
    .subtitle { color: #5c6bc0; font-size: 1rem; margin-top: -0.5rem; margin-bottom: 1.5rem; }
    
    .feat-card {
        background: white; padding: 1rem 1.2rem 0.8rem; border-radius: 14px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06); margin-bottom: 0.8rem;
        border: 1px solid #eef0f5; transition: all 0.2s;
        height: 100%;
    }
    .feat-card:hover { box-shadow: 0 6px 28px rgba(0,0,0,0.10); border-color: #c5cae9; }
    .feat-card label { font-weight: 600; color: #1a237e; font-size: 0.9rem; display: flex; align-items: center; gap: 6px; }
    .feat-card .desc { font-size: 0.75rem; color: #78909c; margin: 2px 0 6px; }
    .feat-card .unit { font-size: 0.7rem; color: #90a4ae; }
    
    div[data-testid="stNumberInput"] input { border: 1px solid #e0e0e0; border-radius: 8px; padding: 0.4rem; }
    
    .stButton > button {
        background: linear-gradient(135deg, #1a237e, #283593);
        color: white; border: none; border-radius: 12px;
        padding: 0.7rem 1.5rem; font-size: 1.1rem; font-weight: 600;
        box-shadow: 0 4px 15px rgba(26,35,126,0.25);
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 25px rgba(26,35,126,0.35);
    }
    .result-card { background: white; padding: 1.5rem; border-radius: 16px; box-shadow: 0 6px 30px rgba(0,0,0,0.08); border-left: 5px solid #1a237e; }
    .metric-box { background: #f8f9ff; padding: 0.8rem; border-radius: 12px; text-align: center; border: 1px solid #e8eaf6; }
    .metric-box .val { font-size: 1.3rem; font-weight: 700; color: #1a237e; }
    .metric-box .lbl { font-size: 0.75rem; color: #78909c; }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 Diabetes Risk Classifier")
st.markdown('<p class="subtitle">Pima Indians Diabetes — RandomForest | MLOps Pipeline</p>', unsafe_allow_html=True)

with st.form("input_form"):
    st.markdown("### 📋 Patient Health Metrics")
    
    keys = list(FEATURES.keys())
    for row_idx in range(0, len(keys), 2):
        cols = st.columns(2)
        for offset in range(2):
            if row_idx + offset < len(keys):
                name = keys[row_idx + offset]
                info = FEATURES[name]
                with cols[offset]:
                    st.markdown(f"""
                        <div class="feat-card">
                            <label>{info['icon']} {name}</label>
                            <div class="desc">{info['desc']} <span class="unit">({info['min']}–{info['max']} {info['unit']})</span></div>
                    """, unsafe_allow_html=True)
                    if name in ("BMI", "DiabetesPedigreeFunction"):
                        st.number_input("", min_value=info["min"], max_value=info["max"], value=info["val"], step=0.1, key=name, label_visibility="collapsed")
                    else:
                        st.number_input("", min_value=info["min"], max_value=info["max"], value=info["val"], step=1, key=name, label_visibility="collapsed")
                    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    submitted = st.form_submit_button("🔍 Predict Diabetes Risk", use_container_width=True)

if submitted:
    input_df = pd.DataFrame([[
        st.session_state.Pregnancies, st.session_state.Glucose,
        st.session_state.BloodPressure, st.session_state.SkinThickness,
        st.session_state.Insulin, st.session_state.BMI,
        st.session_state.DiabetesPedigreeFunction, st.session_state.Age
    ]], columns=list(FEATURES.keys()))

    prob = model.predict_proba(input_df)[0]
    pred = int(prob[1] >= 0.5)
    risk_pct = round(prob[1] * 100, 1)
    safe_pct = round(prob[0] * 100, 1)

    st.markdown("---")
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Prediction Result")
    
    r = st.columns([1, 1, 1, 2])
    r[0].markdown(f'<div class="metric-box"><div class="val">{"🟢 Non-Diabetic" if pred == 0 else "🔴 Diabetic"}</div><div class="lbl">Diagnosis</div></div>', unsafe_allow_html=True)
    r[1].markdown(f'<div class="metric-box"><div class="val">{risk_pct}%</div><div class="lbl">Risk Probability</div></div>', unsafe_allow_html=True)
    r[2].markdown(f'<div class="metric-box"><div class="val">{safe_pct if pred == 0 else risk_pct}%</div><div class="lbl">Confidence</div></div>', unsafe_allow_html=True)

    if risk_pct > 70:
        r[3].error(f"⚠️ High risk ({risk_pct}%) — Consult a healthcare professional.")
    elif risk_pct > 40:
        r[3].warning(f"⚠️ Moderate risk ({risk_pct}%) — Lifestyle review recommended.")
    else:
        r[3].success(f"✅ Low risk ({risk_pct}%) — Maintain healthy habits.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 🔬 Feature Importance")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": list(FEATURES.keys()), "Importance": importances}).sort_values("Importance", ascending=True)
    st.bar_chart(feat_df.set_index("Feature"), height=350)

    with st.expander("📋 Input Summary"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)
