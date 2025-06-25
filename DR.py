import matplotlib.pyplot as plt
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap

# Load model
model = joblib.load('DR.pkl')

# Define features with units
features = [
    "TC (Total Cholesterol, mmol/L)",
    "BUN (Blood Urea Nitrogen, mmol/L)",
    "FIB (Fibrinogen, g/L)",
    "GLU (Glucose, mmol/L)"
]
default_values = [5.96, 6.9, 2.55, 5.7]

# Page setup
st.title("Diabetic Retinopathy Risk Assessment")

# User input section
st.header("Input Clinical Parameters")
input_data = []
for i, feature in enumerate(features):
    input_data.append(st.number_input(
        f"{feature}:",
        min_value=0.0,
        max_value=100.0,
        value=default_values[i],
        step=0.01,
        format="%.2f",
        help=f"Enter {feature.split('(')[0].strip()} value in specified units"
    ))

# Prediction button
if st.button("Run Assessment"):
    # Prepare data
    input_array = np.array([input_data])
    df_input = pd.DataFrame(input_array, columns=[f.split(' ')[0] for f in features])

    # Make prediction
    prediction = model.predict(input_array)[0]
    proba = model.predict_proba(input_array)[0]

    # Display results
    st.header("Assessment Results")
    if prediction == 1:
        st.error(f"Result: High risk of diabetic retinopathy (Probability: {proba[1] * 100:.1f}%)")
    else:
        st.success(f"Result: Low risk of diabetic retinopathy (Probability: {proba[0] * 100:.1f}%)")

    # SHAP explanation
    st.subheader("Model Interpretation")
    st.write("The SHAP plot below shows how each feature contributes to the prediction:")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_input)

    # Handle binary classification SHAP values
    if isinstance(explainer.expected_value, np.ndarray):
        expected_value = explainer.expected_value[1]
        shap_value = shap_values[1]
    else:
        expected_value = explainer.expected_value
        shap_value = shap_values

    plt.figure(figsize=(10, 4))
    shap.force_plot(
        expected_value,
        shap_value,
        df_input,
        matplotlib=True,
        show=False,
        feature_names=[f.split(' ')[0] for f in features]
    )

    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

    # Additional explanation
    st.markdown("""
    **How to interpret this plot:**
    - Features pushing the prediction to the right (red) increase retinopathy risk
    - Features pushing to the left (blue) decrease risk
    - Base value is the model's average prediction
    - Final prediction is where all feature contributions sum up
    """)
