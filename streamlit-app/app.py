import streamlit as st
import pandas as pd
import joblib
import json
from datetime import datetime
import sqlite3
from pathlib import Path

# ====================== CONFIG ======================
st.set_page_config(page_title="Defect Prediction", layout="wide")
st.title("🛠️ Manufacturing Defect Prediction System")
st.markdown("**Random Forest Model** | Accuracy: **95.06%**")

# Paths
MODEL_PATH = Path("../models/random_forest_defect.pkl")
FEATURE_PATH = Path("../models/feature_info.json")
DB_PATH = Path("streamlit_predictions.db")

# Load model and features
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_PATH) as f:
        feature_data = json.load(f)
    return model, feature_data["features"]

model, feature_names = load_model()

# ====================== DATABASE ======================
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

# ====================== SIDEBAR ======================
mode = st.sidebar.radio("Choose Mode", ["Manual Input", "Batch Prediction (CSV)", "History"])

# ====================== MANUAL INPUT ======================
if mode == "Manual Input":
    st.subheader("Enter Production Parameters")
    
    input_data = {}
    cols = st.columns(4)
    
    for i, feature in enumerate(feature_names):
        with cols[i % 4]:
            input_data[feature] = st.number_input(
                feature, 
                value=float(50), 
                format="%.4f"
            )
    
    if st.button("🔮 Predict Defect", type="primary", use_container_width=True):
        input_df = pd.DataFrame([input_data])
        
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
        
        if pred == 1:
            st.error("🚨 DEFECT LIKELY")
            st.metric("Defect Probability", f"{proba:.1%}")
        else:
            st.success("✅ No Defect Predicted")
            st.metric("Defect Probability", f"{proba:.1%}")
        
        # Save to database
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO predictions (created_at, features_json, prediction, probability) VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(), json.dumps(input_data), int(pred), float(proba))
        )
        conn.commit()
        conn.close()

# ====================== BATCH CSV ======================
elif mode == "Batch Prediction (CSV)":
    st.subheader("Upload CSV for Batch Prediction")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:", df.head())
        
        if st.button("Predict All Rows"):
            predictions = model.predict(df[feature_names])
            probabilities = model.predict_proba(df[feature_names])[:, 1]
            
            df['Predicted_Defect'] = predictions
            df['Defect_Probability'] = probabilities
            
            st.success(f"✅ Batch Prediction Complete! {len(df)} rows processed.")
            st.dataframe(df, use_container_width=True)
            
            # Download button
            csv = df.to_csv(index=False).encode()
            st.download_button("📥 Download Predictions", csv, "defect_predictions.csv", "text/csv")

# ====================== HISTORY ======================
elif mode == "History":
    st.subheader("Prediction History")
    conn = sqlite3.connect(DB_PATH)
    history = pd.read_sql("SELECT * FROM predictions ORDER BY id DESC LIMIT 50", conn)
    conn.close()
    
    if not history.empty:
        st.dataframe(history, use_container_width=True)
    else:
        st.info("No predictions yet.")

st.sidebar.success("Model Loaded Successfully ✅")