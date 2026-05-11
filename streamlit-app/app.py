import streamlit as st
import pandas as pd
import joblib
import json
from datetime import datetime
import sqlite3
from pathlib import Path
import plotly.express as px

st.set_page_config(
    page_title="DefectGuard AI",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    .stButton>button {
        background: linear-gradient(90deg, #ff4d4d, #ff8c00);
        color: white;
        border-radius: 12px;
        height: 3.2em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Paths & Model
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR.parent / "models" / "random_forest_defect.pkl"
FEATURE_PATH = BASE_DIR.parent / "models" / "feature_info.json"
DB_PATH = BASE_DIR / "streamlit_predictions.db"

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_PATH) as f:
        feature_data = json.load(f)
    return model, feature_data["features"]

model, feature_names = load_model()

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            features_json TEXT,
            prediction INTEGER,
            probability REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/factory.png", width=80)
    st.title("DefectGuard AI")
    st.markdown("**AI-Powered Quality Control**")
    
    mode = st.radio("**Main Menu**", 
                   ["Manual Input", "Batch Prediction", "History"])
    
    st.success("✅ Model Loaded")
    st.caption("Random Forest • 95.06% Accuracy")

#  MAIN TITLE 
st.markdown('<h1 class="main-header">Manufacturing Defect Prediction System</h1>', unsafe_allow_html=True)
st.markdown("Real-time Quality Control using Machine Learning")

#  MANUAL INPUT 
if mode == "Manual Input":
    st.subheader("Enter Production Parameters")
    
    input_data = {}
    cols = st.columns(4)
    
    for i, feature in enumerate(feature_names):
        with cols[i % 4]:
            input_data[feature] = st.number_input(f"**{feature}**", value=50.0, format="%.4f")

    if st.button("🔮 Predict Defect Risk", type="primary", use_container_width=True):
        with st.spinner("Analyzing with Random Forest Model..."):
            input_df = pd.DataFrame([input_data])
            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][1]
            
            # Save prediction
            conn = sqlite3.connect(DB_PATH)
            conn.execute(
                "INSERT INTO predictions (created_at, features_json, prediction, probability) VALUES (?, ?, ?, ?)",
                (datetime.now().isoformat(), json.dumps(input_data), int(pred), float(proba))
            )
            conn.commit()
            conn.close()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("**Prediction**", "DEFECT LIKELY" if pred == 1 else "NO DEFECT")
            with col2:
                st.metric("**Probability**", f"{proba:.1%}")
            with col3:
                st.metric("**Confidence**", "High" if proba > 0.7 else "Medium")

            if pred == 1:
                st.error("⚠️ HIGH DEFECT RISK — Recommend Quality Inspection")
            else:
                st.success("✅ Low risk. Production appears safe.")

#  BATCH & HISTORY (same as before) 
elif mode == "Batch Prediction":
    st.subheader("Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("**Preview:**", df.head())
        
        if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                df['Predicted_Defect'] = model.predict(df[feature_names])
                df['Defect_Probability'] = model.predict_proba(df[feature_names])[:, 1]
                
                st.success(f"✅ {len(df)} rows processed!")
                st.dataframe(df, use_container_width=True)
                
                csv = df.to_csv(index=False).encode()
                st.download_button("📥 Download Results", csv, "predictions.csv", "text/csv")

elif mode == "History":
    st.subheader("Prediction History")
    conn = sqlite3.connect(DB_PATH)
    history = pd.read_sql("SELECT * FROM predictions ORDER BY id DESC LIMIT 100", conn)
    conn.close()
    
    if not history.empty:
        st.dataframe(history, use_container_width=True)
        history['created_at'] = pd.to_datetime(history['created_at'])
        fig = px.line(history, x='created_at', y='probability', title="Defect Probability Trend", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No predictions yet.")
