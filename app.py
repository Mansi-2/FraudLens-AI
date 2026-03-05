import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from model import load_or_train_model, prepare_input
import seaborn as sns

st.set_page_config(page_title="FraudLens AI", layout="wide")

model, explainer, scaler, columns, df = load_or_train_model()

# Sidebar Styling
st.sidebar.markdown("""
<style>
section[data-testid="stSidebar"] {
    background-color: #0e1117;
    border-right: 2px solid #00c8ff;
}

div.stButton > button {
    width: 100%;
    padding: 12px;
    border-radius: 12px;
    border: 2px solid #00c8ff;
    background-color: transparent;
    color: white;
    font-weight: 600;
}

div.stButton > button:hover {
    background-color: #00c8ff;
    color: black;
}
</style>
""", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "Home"

pages = ["Home", "Scenario Simulator", "Local Explanation", "Global Insights", "Fraud Comparison"]

for p in pages:
    if st.sidebar.button(p):
        st.session_state.page = p

page = st.session_state.page


# ---------------- HOME ----------------
if page == "Home":

    st.title("💳 FraudLens AI – Interpretable Credit Card Fraud Detection")

    st.markdown("""
    ### What This Project Does

    Traditional fraud detection systems only say:
    **Fraud** or **Not Fraud**

    That’s not enough.

    FraudLens AI explains:

    • Why a transaction is considered risky  
    • Which features increase or decrease fraud probability  
    • How transaction amount and timing influence risk  
    • Global behavioral patterns learned by the model  

    ### Why Interpretability Matters

    In finance, decisions must be explainable.
    Banks cannot block transactions without justification.

    This system uses SHAP (Shapley Additive Explanations),
    a game-theory based method that explains how each feature
    contributes to a prediction.

    Instead of mathematical components,
    we use human-readable features like:

    • Transaction Amount  
    • Hour of Transaction  
    • Night Transaction Indicator  
    • High Amount Flag  

    The result:
    A fraud detection system that is transparent,
    interpretable, and trustworthy.
    """)


# ---------------- SCENARIO SIMULATOR ----------------
if page == "Scenario Simulator":

    st.title("⚡ Transaction Scenario Simulator")

    amount = st.slider("Transaction Amount ($)", 0.0, 5000.0, 150.0)
    hour = st.slider("Hour of Day", 0, 23, 14)

    user_input = prepare_input(amount, hour, scaler, columns, df)

    probability = model.predict_proba(user_input)[0][1]

    st.metric("Fraud Probability", f"{probability*100:.2f}%")

    if probability > 0.5:
        st.error("⚠️ High Fraud Risk")
    else:
        st.success("✅ Low Fraud Risk")


# ---------------- LOCAL EXPLANATION ----------------
if page == "Local Explanation":

    st.title("🔍 Why Is This Transaction Risky?")

    amount = st.slider("Transaction Amount ($)", 0.0, 5000.0, 150.0, key="local_amt")
    hour = st.slider("Hour of Day", 0, 23, 14, key="local_hr")

    user_input = prepare_input(amount, hour, scaler, columns, df)

    shap_values = explainer.shap_values(user_input)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    impact = pd.DataFrame({
        "Feature": columns,
        "Impact": shap_values.flatten()
    })

    impact = impact.sort_values(by="Impact", key=abs, ascending=False).head(8)

    fig, ax = plt.subplots(figsize=(5,3))
    colors = ["red" if val > 0 else "green" for val in impact["Impact"]]
    ax.barh(impact["Feature"], impact["Impact"], color=colors)
    ax.set_xlabel("Impact on Fraud Probability")
    st.pyplot(fig)

    st.markdown("""
    Red bars increase fraud probability.
    Green bars decrease fraud probability.

    This explains the exact reasoning behind the model's decision.
    """)


# ---------------- GLOBAL INSIGHTS ----------------
if page == "Global Insights":

    st.header("🌍 Global Fraud Insights")


    sample_df = df.sample(2000, random_state=42)

    X_sample = sample_df.drop("Class", axis=1)
    X_sample_scaled = scaler.transform(X_sample)

    shap_values = explainer.shap_values(X_sample_scaled)

    st.subheader("Top Features Influencing Fraud (Global View)")
    fig, ax = plt.subplots(figsize=(6,4))
    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        max_display=10,
        show=False
)

    st.pyplot(fig)
    plt.close(fig)

# ---------------- FRAUD COMPARISON ----------------
if page == "Fraud Comparison":

    st.title("📊 Fraud vs Non-Fraud Feature Comparison")

    feature = st.selectbox(
        "Select Feature",
        ["Amount", "Hour", "Log_Amount"]
    )

    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5,3))

    sns.kdeplot(
        data=df[df["Class"] == 0],
        x=feature,
        ax=ax,
        label="Non-Fraud",
        fill=True
    )

    sns.kdeplot(
        data=df[df["Class"] == 1],
        x=feature,
        ax=ax,
        label="Fraud",
        fill=True
    )

    ax.legend()
    ax.set_xlabel(feature)

    st.pyplot(fig)