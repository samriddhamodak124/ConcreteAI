
import streamlit as st
import pandas as pd
import joblib
import json
import shap
import matplotlib.pyplot as plt

# ---------------------------------
# 1. Load model and supporting files
# ---------------------------------

model = joblib.load("concrete_strength_model.pkl")

with open("metrics.json", "r") as f:
    metrics = json.load(f)

with open("feature_ranges.json", "r") as f:
    feature_ranges = json.load(f)

# ---------------------------------
# 2. Application configuration
# ---------------------------------

st.set_page_config(
    page_title="ConcreteAI",
    page_icon="🏗️",
    layout="wide"
)

st.title("🏗️ ConcreteAI")
st.subheader("Explainable Concrete Compressive Strength Prediction")

st.write(
    "Predict concrete compressive strength using a "
    "Random Forest machine learning model and understand "
    "individual predictions using SHAP explainability."
)

st.info("Mix quantities are in kg/m³. Age is in days.")

# ---------------------------------
# 3. Input parameters
# ---------------------------------

feature_names = list(feature_ranges.keys())

short_names = [
    "Cement",
    "Blast Furnace Slag",
    "Fly Ash",
    "Water",
    "Superplasticizer",
    "Coarse Aggregate",
    "Fine Aggregate",
    "Age"
]

labels = {
    feature_names[0]: "Cement (kg/m³)",
    feature_names[1]: "Blast Furnace Slag (kg/m³)",
    feature_names[2]: "Fly Ash (kg/m³)",
    feature_names[3]: "Water (kg/m³)",
    feature_names[4]: "Superplasticizer (kg/m³)",
    feature_names[5]: "Coarse Aggregate (kg/m³)",
    feature_names[6]: "Fine Aggregate (kg/m³)",
    feature_names[7]: "Age (days)"
}

st.header("Concrete Mix Parameters")

inputs = {}

col1, col2 = st.columns(2)

for i, feature in enumerate(feature_names):

    limits = feature_ranges[feature]

    minimum = float(limits["min"])
    maximum = float(limits["max"])
    default = float(limits["median"])

    with col1 if i < 4 else col2:
        inputs[feature] = st.number_input(
            labels[feature],
            min_value=minimum,
            max_value=maximum,
            value=default,
            step=1.0,
            key=feature
        )

st.caption(
    "Input limits are based on the training dataset."
)

# ---------------------------------
# 4. Prediction and SHAP explanation
# ---------------------------------

if st.button("Predict Compressive Strength", type="primary"):

    input_df = pd.DataFrame(
        [inputs],
        columns=feature_names
    )

    prediction = model.predict(input_df)[0]

    st.success("Prediction completed successfully!")

    st.metric(
        "Predicted Compressive Strength",
        f"{prediction:.2f} MPa"
    )

    st.divider()

    # SHAP explanation
    st.header("Explainable AI - SHAP Analysis")

    with st.spinner("Generating SHAP explanation..."):

        explainer = shap.TreeExplainer(model)

        shap_values = explainer(input_df)

        # Use shorter names for graph readability
        explanation = shap.Explanation(
            values=shap_values.values[0],
            base_values=shap_values.base_values[0],
            data=input_df.iloc[0].values,
            feature_names=short_names
        )

    st.subheader("Individual Prediction Explanation")

    st.write(
        "The graph shows how each input parameter moves "
        "the prediction above or below the model's baseline."
    )

    fig = plt.figure(figsize=(10, 7))

    shap.plots.waterfall(
        explanation,
        show=False
    )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Red contributions increase the prediction relative "
        "to the baseline; blue contributions decrease it."
    )

    # ---------------------------------
    # 5. Model performance
    # ---------------------------------

    st.divider()

    st.header("Model Performance")

    c1, c2, c3 = st.columns(3)

    c1.metric("MAE", f'{metrics["MAE"]:.3f} MPa')
    c2.metric("RMSE", f'{metrics["RMSE"]:.3f} MPa')
    c3.metric("R² Score", f'{metrics["R2"]:.4f}')

    # ---------------------------------
    # 6. Overall feature importance
    # ---------------------------------

    st.divider()

    st.header("Overall Feature Importance")

    importance_df = pd.DataFrame({
        "Parameter": short_names,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    st.bar_chart(
        importance_df.set_index("Parameter")
    )

    st.caption(
        "Feature importance indicates the model's relative "
        "reliance on each parameter. It does not prove causation."
    )
