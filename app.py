# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 11:40:48 2025
@author: Bibhu

GO-Infused Concrete Compressive Strength Predictor
Publication-ready Streamlit Interface
===========================================================
"""

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ---------------------- Page Config ----------------------
st.set_page_config(
    page_title="GO-Infused Concrete Strength Predictor",
    page_icon="🧱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------- Title Section ----------------------
st.markdown(
    """
    <div style="background-color:#2c3e50;padding:18px;border-radius:8px;margin-top:10px;">
        <h2 style="color:white;text-align:center;margin-bottom:0;">
        GO-Infused Concrete Compressive Strength Predictor
        </h2>
        <p style="color:#bdc3c7;text-align:center;margin-top:4px;font-size:15px;">
        A Machine Learning-based Framework for Graphene Oxide Concrete Mix Design
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")  # spacing

# ---------------------- Load Dataset ----------------------
file_path = "Graphene Oxide Dataset_py.xlsx"
df = pd.read_excel(file_path)

X = df.drop(columns=['CS'], errors='ignore')
y = df['CS']

# Split & Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)
r2 = r2_score(y_test, model.predict(X_test))

# ---------------------- Sidebar ----------------------
st.sidebar.header("📘 Model Summary")
st.sidebar.markdown(
    f"""
    **Algorithm:** SOA-Optimized Random Forest (SOA-RF)  
    **Dataset Size:** {len(df)} samples  
    **Features Used:** {len(X.columns)}  
    **R² Score:** {r2:.3f}  
    **Research Year:** 2025  
    """
)

st.sidebar.markdown("---")
st.sidebar.info("Developed for research on Graphene Oxide-infused concrete (GOIC) reliability analysis.")

# ---------------------- Input Section ----------------------
st.subheader("🔹 Enter Mix Design Parameters")

field_units = {
    "Cement (kg/m³)": "kg/m³",
    "Water (kg/m³)": "kg/m³",
    "Fine Aggregate (kg/m³)": "kg/m³",
    "Coarse Aggregate (kg/m³)": "kg/m³",
    "Superplasticizer (% of binder)": "%",
    "Fly Ash (% of binder)": "%",
    "Silica Fume (% of binder)": "%",
    "Steel Fiber (% by volume)": "%",
    "Graphene Oxide (% by wt. of cement)": "%",
    "Curing Duration (days)": "days"
}

cols = st.columns(2)
inputs = []

for i, (param, unit) in enumerate(field_units.items()):
    with cols[i % 2]:
        val = st.number_input(f"{param}", min_value=0.0, step=0.1, format="%.3f")
        st.caption(f"Unit: **{unit}**")
        inputs.append(val)

st.write("")  # spacing

# ---------------------- Predict Button ----------------------
predict_button = st.button("🔹 Predict Compressive Strength", use_container_width=True)

if predict_button:
    try:
        prediction = model.predict([inputs])[0]
        st.success(f"**Predicted Compressive Strength: {prediction:.2f} MPa**")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

# ---------------------- Visual Divider ----------------------
st.markdown("<hr style='margin:30px 0;'>", unsafe_allow_html=True)

# ---------------------- Footer ----------------------
st.markdown(
    """
    <div style="text-align:center;">
        <p style="color:gray;font-size:13px;">
        <b>Developed by:</b> Bibhu Prasad Mishra (2025) <br>
        <a href="mailto:bibhumi2121@gmail.com" style="color:gray;text-decoration:none;">
        bibhumi2121@gmail.com
        </a><br>
        <span style="font-size:12px;">GO-Concrete Research | Machine Learning in Sustainable Materials</span>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
