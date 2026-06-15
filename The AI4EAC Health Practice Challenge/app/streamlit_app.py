"""
Depression Risk Screening Tool
--------------------------------
A Streamlit app that uses a trained XGBoost model (top-15 SHAP features)
to estimate the likelihood that a survey respondent meets the
epidemiological threshold for moderate depression, based on routine
household survey data (Busara Center / Zindi AI4EAC Health Challenge).

IMPORTANT: This tool is for educational/portfolio purposes only.
It is NOT a diagnostic instrument and should not be used to make
clinical decisions. Predictions should be used, at most, to help
prioritize outreach by community health workers.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

# -----------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Depression Risk Screening Tool",
    page_icon="🧠",
    layout="centered"
)

# -----------------------------------------------------------------------
# Load model bundle
# -----------------------------------------------------------------------
# Get the directory this script lives in, then go to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "models", "depression_model_bundle.pkl")

@st.cache_resource
def load_model_bundle(path=MODEL_PATH):
    return joblib.load(path)


bundle = load_model_bundle()
pipeline = bundle["pipeline"]
features = bundle["features"]
feature_descriptions = bundle["feature_descriptions"]

model = pipeline.named_steps["model"]
explainer = shap.TreeExplainer(model)

# -----------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------
st.title("🧠 Depression Risk Screening Tool")

st.markdown(
    """
    This tool estimates the likelihood that an individual meets the
    epidemiological threshold for **moderate depression**, based on
    routine household survey indicators (economic, food security,
    health, and demographic data).

    It is built on data from the Busara Center for Behavioral Economics
    / Zindi **AI4EAC Health Practice Challenge**, using an XGBoost
    classifier trained on the top 15 features identified via SHAP
    feature importance.
    """
)

st.warning(
    "**This is not a diagnostic tool.** Results are intended to support "
    "prioritization of outreach by community health workers or NGOs, "
    "not to replace clinical assessment by a qualified mental health "
    "professional."
)

st.divider()

# -----------------------------------------------------------------------
# Input form
# -----------------------------------------------------------------------
st.header("Respondent Information")

input_data = {}

col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics")

    input_data["age"] = st.slider(
        feature_descriptions["age"], min_value=18, max_value=90, value=35
    )

    input_data["edu"] = st.slider(
        feature_descriptions["edu"], min_value=0, max_value=20, value=8
    )

    married_label = st.radio(
        feature_descriptions["married"], options=["No", "Yes"], horizontal=True
    )
    input_data["married"] = 1 if married_label == "Yes" else 0

    st.subheader("Food Security")

    enoughtom_label = st.radio(
        feature_descriptions["fs_enoughtom"], options=["No", "Yes"], horizontal=True
    )
    input_data["fs_enoughtom"] = 1 if enoughtom_label == "Yes" else 0

    sleephun_label = st.radio(
        feature_descriptions["fs_sleephun"], options=["No", "Yes"], horizontal=True
    )
    input_data["fs_sleephun"] = 1 if sleephun_label == "Yes" else 0

    input_data["fs_adwholed_often"] = st.slider(
        feature_descriptions["fs_adwholed_often"],
        min_value=0, max_value=30, value=0,
        help="Number of whole days in the past month with no food eaten by adults in the household"
    )

with col2:
    st.subheader("Household Economics")

    input_data["asset_durable"] = st.number_input(
        feature_descriptions["asset_durable"], min_value=0.0, value=100.0, step=10.0
    )

    input_data["asset_savings"] = st.number_input(
        feature_descriptions["asset_savings"], min_value=0.0, value=0.0, step=5.0
    )

    input_data["asset_livestock"] = st.number_input(
        feature_descriptions["asset_livestock"], min_value=0.0, value=50.0, step=10.0
    )

    input_data["cons_ed"] = st.number_input(
        feature_descriptions["cons_ed"], min_value=0.0, value=0.0, step=1.0
    )

    saved_mpesa_label = st.radio(
        feature_descriptions["saved_mpesa"], options=["No", "Yes"], horizontal=True
    )
    input_data["saved_mpesa"] = 1 if saved_mpesa_label == "Yes" else 0

    st.subheader("Farming & Health")

    ownfarm_label = st.radio(
        feature_descriptions["ent_ownfarm"], options=["No", "Yes"], horizontal=True
    )
    input_data["ent_ownfarm"] = 1 if ownfarm_label == "Yes" else 0

    input_data["ent_farmexpenses"] = st.number_input(
        feature_descriptions["ent_farmexpenses"], min_value=0.0, value=0.0, step=1.0
    )

    input_data["med_portion_sickinjured"] = st.slider(
        feature_descriptions["med_portion_sickinjured"],
        min_value=0.0, max_value=1.0, value=0.0, step=0.05,
        help="Proportion of household members who were sick or injured in the past month"
    )

    input_data["med_sickdays_hhave"] = st.slider(
        feature_descriptions["med_sickdays_hhave"],
        min_value=0.0, max_value=30.0, value=0.0, step=1.0,
        help="Average number of sick days per household member in the past month"
    )

st.divider()

# -----------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------
if st.button("Assess Risk", type="primary", use_container_width=True):

    # build input dataframe in the correct feature order
    input_df = pd.DataFrame([input_data])[features]

    proba = pipeline.predict_proba(input_df)[0, 1]
    prediction = pipeline.predict(input_df)[0]

    # define risk tiers
    if proba < 0.33:
        risk_tier = "Low"
        risk_color = "green"
    elif proba < 0.66:
        risk_tier = "Medium"
        risk_color = "orange"
    else:
        risk_tier = "High"
        risk_color = "red"

    st.header("Results")

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.metric("Estimated Probability", f"{proba:.1%}")

    with res_col2:
        st.markdown(
            f"### Risk Tier: <span style='color:{risk_color}'>{risk_tier}</span>",
            unsafe_allow_html=True
        )

    if prediction == 1:
        st.error(
            "The model flags this individual's responses as **consistent with "
            "moderate depression**. Consider follow-up by a community health "
            "worker or referral for further assessment."
        )
    else:
        st.success(
            "The model does **not** flag this individual's responses as "
            "consistent with moderate depression based on the indicators provided."
        )

    # -------------------------------------------------------------------
    # SHAP explanation for this individual
    # -------------------------------------------------------------------
    st.subheader("What drove this prediction?")

    shap_values_instance = explainer.shap_values(input_df)

    # handle both array and list outputs depending on shap/xgboost version
    if isinstance(shap_values_instance, list):
        shap_vals = shap_values_instance[1][0]
    else:
        shap_vals = shap_values_instance[0]

    contrib_df = pd.DataFrame({
        "Feature": [feature_descriptions.get(f, f) for f in features],
        "Value": [input_data[f] for f in features],
        "SHAP Contribution": shap_vals
    }).sort_values("SHAP Contribution", key=abs, ascending=False)

    st.markdown(
        "The chart below shows which factors pushed the prediction "
        "**toward** higher depression risk (red) or **away from** it (blue), "
        "for this specific individual."
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in contrib_df["SHAP Contribution"]]
    ax.barh(contrib_df["Feature"], contrib_df["SHAP Contribution"], color=colors)
    ax.set_xlabel("SHAP value (impact on prediction)")
    ax.set_title("Top Contributing Factors")
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)

    with st.expander("See raw input values"):
        st.dataframe(contrib_df[["Feature", "Value"]].reset_index(drop=True))

st.divider()

st.caption(
    "Model: XGBoost classifier (top 15 SHAP-ranked features) | "
    "Data: Busara Center for Behavioral Economics, via the Zindi "
    "AI4EAC Health Practice Challenge | "
    "Built for portfolio/educational purposes."
)

