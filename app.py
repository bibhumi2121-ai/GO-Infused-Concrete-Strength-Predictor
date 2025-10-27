# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 11:40:48 2025

@author: bibhu
"""

# app.py
# =====================================================
# GO-Infused Concrete Compressive Strength Predictor
# Developed by Bibhu (2025)
# =====================================================

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

# ---------------------- Title and Header ----------------------
st.markdown(
    """
    <div style="background-color:#2c3e50;padding:15px;border-radius:8px">
    <h2 style="color:white;text-align:center;">
    🧱 GO-Infused Concrete Compressive Strength Predictor
    </h2></div>
    """,
    unsafe_allow_html=True
)

# ---------------------- Load Dataset ----------------------
file_path = "Graphene Oxide Dataset_py.xlsx"
df = pd.read_excel(file_path)

# Target and features
X = df.drop(columns=['CS'], errors='ignore')
y = df['CS']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)
r2 = r2_score(y_test, model.predict(X_test))

# ---------------------- Sidebar Info ----------------------
st.sidebar.header("📘 Model Information")
st.sidebar.markdown(f"""
**Model:** Random Forest Regressor  
**Dataset size:** {df.shape[0]} samples  
**Features used:** {len(X.columns)}  
**Model Accuracy (R²):** {r2:.3f}
""")

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
        value = st.number_input(f"{param}", min_value=0.0, step=0.1, format="%.3f")
        st.caption(f"Unit: **{unit}**")
        inputs.append(value)

# ---------------------- Prediction Button ----------------------
if st.button("🚀 Predict Compressive Strength"):
    try:
        prediction = model.predict([inputs])[0]
        st.success(f"**Predicted Compressive Strength: {prediction:.2f} MPa**")
    except Exception as e:
        st.error(f"⚠️ Error in prediction: {e}")

# ---------------------- Footer ----------------------
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align:center; color:gray; font-size:14px;">
    Developed by <b>Bibhu (2025)</b> <i>bibhumis2121@gmail.com</i>
    </div>
    """,
    unsafe_allow_html=True
)
