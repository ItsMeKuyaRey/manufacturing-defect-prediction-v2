import streamlit as st
import pandas as pd
import joblib
import json
from datetime import datetime
import sqlite3
from pathlib import Path
import plotly.express as px

#  PAGE CONFIG 
st.set_page_config(
    page_title="DefectGuard AI",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

#  CUSTOM CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    .sub-header { 
        color: #b0b0b0; 
        font-size: 1.2rem; 
        margin-bottom: 2rem; 
    }
    .metric-card {
        background: #1e1e2e;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #333;
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(90deg, #ff4d4d, #ff8c00);
        color: white;
        border-radius: 12px;
        height: 3.2em;
        font-weight: bold;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

#  LOAD MODEL & PATHS 
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

#  DATABASE 
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

#  SIDEBAR 
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/factory.png", width=80)
    st.title("DefectGuard AI")
    st.markdown("**AI-Powered Quality Control**")
    
    mode = st.radio("**Main Menu**", 
                   ["Dashboard Overview", "Manual Input", "Batch Prediction", "History"])
    
    st.success("✅ Model Loaded")
    st.caption("Random Forest • 95.06% Accuracy")

#  DASHBOARD OVERVIEW 
if mode == "Dashboard Overview":
    st.markdown('<h1 class="main-header">Manufacturing Defect Prediction System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time Quality Intelligence Dashboard</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Predictions", "248", "↑ 12%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Defect Rate", "18.4%", "↓ 3.2%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg Risk Score", "42.7%", "↓ 8%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Last Prediction", "12 min ago", "Safe")
        st.markdown('</div>', unsafe_allow_html=True)

#  MANUAL INPUT 
elif mode == "Manual Input":
    st.markdown('<h1 class="main-header">Manual Prediction</h1>', unsafe_allow_html=True)
    
    input_data = {}
    cols = st.columns(4)
    
    for i, feature in enumerate(feature_names):
        with cols[i % 4]:
            input_data[feature] = st.number_input(
                f"**{feature}**", 
                value=50.0, 
                format="%.4f"
            )

    if st.button(" Predict Defect Risk", type="primary", use_container_width=True):
        with st.spinner("Analyzing..."):
            input_df = pd.DataFrame([input_data])
            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][1]
            
            # Save to database
            conn = sqlite3.connect(DB_PATH)
            conn.execute(
                "INSERT INTO predictions (created_at, features_json, prediction, probability) VALUES (?, ?, ?, ?)",
                (datetime.now().isoformat(), json.dumps(input_data), int(pred), float(proba))
            )
            conn.commit()
            conn.close()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("**Status**", "DEFECT LIKELY" if pred == 1 else "NO DEFECT")
            with col2:
                st.metric("**Probability**", f"{proba:.1%}")
            with col3:
                st.metric("**Confidence**", "High" if proba > 0.7 else "Medium")

            if pred == 1:
                st.error(" HIGH DEFECT RISK DETECTED — Recommend immediate quality inspection")
            else:
                st.success("Production parameters look safe")

#  BATCH PREDICTION 
elif mode == "Batch Prediction":
    st.markdown('<h1 class="main-header">Batch Prediction</h1>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("**Preview:**", df.head())
        
        if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
            with st.spinner("Processing batch..."):
                predictions = model.predict(df[feature_names])
                probabilities = model.predict_proba(df[feature_names])[:, 1]
                
                df['Predicted_Defect'] = predictions
                df['Defect_Probability'] = probabilities
                
                st.success(f" {len(df)} rows processed successfully!")
                st.dataframe(df, use_container_width=True)
                
                csv = df.to_csv(index=False).encode()
                st.download_button("📥 Download Results", csv, "batch_predictions.csv", "text/csv")

#  HISTORY 
    st.markdown('<h1 class="main-header">Prediction History</h1>', unsafe_allow_html=True)
    
    conn = sqlite3.connect(DB_PATH)
    history = pd.read_sql("SELECT * FROM predictions ORDER BY id DESC LIMIT 100", conn)
    conn.close()
    
    if not history.empty:
        st.dataframe(history, use_container_width=True)
        
        history['created_at'] = pd.to_datetime(history['created_at'])
        fig = px.line(history, x='created_at', y='probability', 
                      title="Defect Probability Trend Over Time",
                      markers=True, line_shape="linear")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No predictions recorded yet.")
