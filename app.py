import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from model import load_or_train_model, prepare_input
import seaborn as sns
from streamlit_echarts import st_echarts
import math

st.set_page_config(page_title="FraudRadar", layout="wide")

model, explainer, scaler, columns, df = load_or_train_model()

# ---------------- AI VISUAL COMPONENTS ----------------

def fraud_risk_meter(prob):

    if prob is None:
        prob = 0

    prob = float(prob)
    percent = round(prob * 100, 2)

    option = {
        "series": [
            {
                "type": "gauge",

                "startAngle": 180,
                "endAngle": 0,

                "min": 0,
                "max": 100,

                "radius": "100%",
                "center": ["50%", "75%"],

                "axisLine": {
                    "lineStyle": {
                        "width": 20,
                        "color": [
                            [0.3, "#00ff99"],
                            [0.7, "#ffcc00"],
                            [1, "#ff4d4d"]
                        ]
                    }
                },

                "pointer": {
                    "length": "60%",
                    "width": 6
                },

                "progress": {
                    "show": True,
                    "width": 20
                },

                "axisTick": {"show": False},
                "splitLine": {"show": False},
                "axisLabel": {"show": False},

                "detail": {
                    "formatter": "{value}%",
                    "fontSize": 22,
                    "color": "white",
                    "offsetCenter": [0, "-10%"]
                },

                "data": [
                    {"value": percent}
                ]
            }
        ]
    }

    st_echarts(options=option, height="260px")

def risk_badge(prob):

    if prob < 0.3:
        color = "#00ff99"
        label = "LOW RISK"
    elif prob < 0.7:
        color = "#ffcc00"
        label = "MEDIUM RISK"
    else:
        color = "#ff4d4d"
        label = "HIGH RISK"

    st.markdown(
        f"""
        <div style="
        padding:10px;
        border-radius:10px;
        background:{color};
        color:black;
        font-weight:bold;
        text-align:center;
        width:200px;
        ">
        {label}
        </div>
        """,
        unsafe_allow_html=True
    )


def generate_explanation(prob, amount, hour):

    if prob < 0.3:
        risk = "low"
    elif prob < 0.7:
        risk = "moderate"
    else:
        risk = "high"

    text = f"""
    The AI model estimates a **{risk} probability of fraud**.

    The transaction amount **₹{amount:.2f}** occurring at **hour {hour}**
    is compared with historical behaviour patterns.

    Transactions at unusual times or unusually high amounts
    increase fraud likelihood, while normal behaviour reduces risk.
    """

    st.markdown(
    f"""
    <div style="
    padding:18px;
    border-radius:12px;
    background:#071b2b;
    border:1px solid #00c8ff;
    color:white;
    ">
    {text}
    </div>
    """,
    unsafe_allow_html=True
)

# -------------- Sidebar Styling---------------------------------
st.markdown("""
<style>

/* Sidebar background */
section[data-testid="stSidebar"]{
    background: linear-gradient(180deg,#020617,#030b1a,#020617);
    border-right: 2px solid #00c8ff;
}

/* Sidebar navigation buttons */
.stButton>button{
    width: 100%;
    height: 70px;
    border-radius: 16px;
    border: 2px solid #00c8ff;
    background-color: transparent;
    color: white;
    font-size: 18px;
    font-weight: 500;
    margin-top: 12px;
    transition: all 0.3s ease;
}

/* Hover effect */
.stButton>button:hover{
    background: rgba(0,200,255,0.15);
    box-shadow: 0px 0px 12px #00c8ff;
    transform: scale(1.02);
}

/* Sidebar title */
.sidebar-title{
    font-size: 26px;
    font-weight: 700;
    color: white;
    margin-bottom: 20px;
}

/* Section spacing */
.sidebar-section{
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)

with st.sidebar:

    st.sidebar.image("data/FRLogo.png",width=260)
    st.sidebar.markdown("---")

    if st.button("🏠 Home"):
        st.session_state.page = "Home"

    if st.button("🧪 Scenario Simulator"):
        st.session_state.page = "Scenario Simulator"

    if st.button("🔍 Local Explanation"):
        st.session_state.page = "Local Explanation"

    if st.button("🌍 Global Insights"):
        st.session_state.page = "Global Insights"

    if st.button("📊 Fraud Comparison"):
        st.session_state.page = "Fraud Comparison"

if "page" not in st.session_state:
    st.session_state.page = "Home"
page = st.session_state.page


# ---------------- HOME ----------------
if page == "Home":

    st.title("💳 FraudRadar – Explainable Credit Card Fraud Detection System")

    st.markdown("""
    ### What This Project Does

    Traditional fraud detection systems only say:
    **Fraud** or **Not Fraud**

    That’s not enough.

    FraudRadar explains:

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

    # DASHBOARD LAYOUT
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Fraud Probability", f"{probability*100:.2f}%")

    with col2:
        st.subheader("Risk Level")
        risk_badge(probability)

    with col3:
        st.subheader("Fraud Risk Meter")
        fraud_risk_meter(probability)

    st.subheader("AI Explanation")
    generate_explanation(probability, amount, hour)


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

    fig, ax = plt.subplots(figsize=(4,2))
    colors = ["red" if val > 0 else "green" for val in impact["Impact"]]
    ax.barh(impact["Feature"], impact["Impact"], color=colors)
    ax.set_xlabel("Impact on Fraud Probability")
    st.pyplot(fig, use_container_width=False)

    st.markdown("""
<div style="
background-color:#111827;
border:1px solid #374151;
padding:18px;
border-radius:10px;
margin-top:15px;
font-size:16px;
line-height:1.6;
color:#e5e7eb;
">

<b>Interpretation Guide</b><br><br>

<span style="color:#ff4d4d; font-weight:1000;">Red bars</span> increase the probability that the transaction is fraudulent.<br>

<span style="color:#00ff99; font-weight:1000;">Green bars</span> decrease the probability of fraud.<br><br>

This visualization explains the <b>exact reasoning behind the model's decision</b> by showing which features influenced the fraud prediction the most.

</div>
""", unsafe_allow_html=True)




# ---------------- GLOBAL INSIGHTS ----------------
if page == "Global Insights":

    st.header("🌍 Global Fraud Insights")


    sample_df = df.sample(2000, random_state=42)

    X_sample = sample_df.drop("Class", axis=1)
    X_sample_scaled = scaler.transform(X_sample)

    shap_values = explainer.shap_values(X_sample_scaled)

    st.subheader("Top Features Influencing Fraud (Global View)")
    fig, ax = plt.subplots(figsize=(4,3))
    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        max_display=10,
        show=False
)

    st.pyplot(fig, use_container_width=False)
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

    fig, ax = plt.subplots(figsize=(4,2.5))

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

    st.pyplot(fig, use_container_width=False)

    