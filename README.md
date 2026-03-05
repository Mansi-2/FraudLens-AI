# 💳 Credit Card Fraud Explanation System

An **AI-powered explainable fraud detection dashboard** that predicts whether a credit card transaction is fraudulent and **explains the reasoning behind the prediction in a human-understandable way**.

This project combines **Machine Learning + Explainable AI (XAI) + Interactive Visualization** to make fraud detection transparent instead of a black-box model.

The system allows users to **simulate transactions**, adjust **transaction amount and time**, and observe how these changes influence fraud probability.

---

# 🚀 Project Overview

Financial fraud detection models often work as **black boxes**. They output a probability but provide **no reasoning** behind the prediction.

This project solves that problem using **Explainable AI (SHAP)**.

The system:

- Predicts fraud probability for a transaction
- Explains which features influenced the decision
- Shows global patterns in fraud data
- Allows **interactive scenario simulation**

The result is a **human-readable fraud analysis system** that could be used by:

- Fraud analysts
- Financial institutions
- Risk analysts
- Data scientists studying model interpretability

---

# 🧠 Core Idea

Instead of simply predicting fraud:
Transaction → ML Model → Fraud Probability


We extend the pipeline:
Transaction → ML Model → SHAP Explainability → Human-readable Insights


This allows us to answer:

- Why was this transaction predicted as fraud?
- Which features influence fraud most?
- How does changing amount or time affect fraud probability?

---

# ⚙️ Tech Stack
Python
Streamlit
XGBoost
SHAP (Explainable AI)
Scikit-learn
Pandas
NumPy
Matplotlib


---

# 📊 Dataset Used

Dataset: **Credit Card Fraud Detection Dataset**

This dataset contains **European cardholder transactions** over two days.

Key characteristics:

- Total transactions: **284,807**
- Fraud cases: **492**
- Fraud rate: **~0.17%**

This is a **highly imbalanced dataset**, which is common in real-world fraud detection.

---

# 🧾 Dataset Features Explained

## Dataset

The dataset is not included in this repository due to GitHub file size limits.

Download it from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

After downloading, place the file here:

FraudLens-AI/data/creditcard.csv

The dataset contains **31 features**.

## 1️⃣ Time

- Seconds elapsed since the first transaction in the dataset.
- Represents **when the transaction occurred**.

Example:
7200 seconds = 2 hours after first recorded transaction

In our system this is converted to:
Hour of Day (0–23) so humans can understand the timing.

---

## 2️⃣ Amount

Transaction value in euros.

Examples:
$5 coffee purchase
$150 online shopping
$2000 electronics purchase

Fraud transactions often show **abnormal spending patterns**.

---

## 3️⃣ PCA Features (V1 – V28)
V1
V2
V3
...
V28

These are **Principal Component Analysis (PCA) transformed features**.

Why PCA was used:

- To protect sensitive financial information
- To anonymize customer data

These features represent **hidden patterns in transaction behavior** such as:

- unusual spending patterns
- abnormal merchant behavior
- unusual location/time combinations
- card usage anomalies

Because the original features were confidential, they were transformed into these **numerical latent variables**.

Even though they look abstract, the ML model can still detect fraud patterns from them.

---

## 4️⃣ Class (Target Variable)

Transaction label:
0 → Normal Transaction
1 → Fraud Transaction

This is the **prediction target for the machine learning model**.

---

# 🔧 Feature Engineering (Added in This Project)

To make the model more interpretable we create additional features.

### Hour
Hour = Time // 3600

Converts seconds to **hour of day (0–23)**.

Fraud often occurs during unusual hours.

---

### Log Amount
Log_Amount = log(Amount)

Helps stabilize very large transaction values.

---

### Night Transaction Flag
Is_Night

Flag indicating:
1 → Transaction occurred between 10 PM – 5 AM
0 → Otherwise

Fraud patterns are often higher during nighttime.

---

### High Amount Flag
High_Amount

Marks transactions in the **top 5% of spending values**.

Large sudden purchases can indicate fraud.

---

# 🧪 Model Used

The system uses: XGBoost Classifier

Reasons for using XGBoost:

- Excellent performance on tabular datasets
- Handles imbalanced data well
- Works perfectly with SHAP explainability

---

# 🔍 Explainable AI (SHAP)

We use **SHAP (SHapley Additive exPlanations)**.

SHAP is based on **cooperative game theory**.

It answers: How much did each feature contribute to the final prediction?

Example explanation:
Fraud Probability: 82%

Top contributing factors:
- High transaction amount
- Night-time purchase
- Unusual PCA pattern

Instead of just predicting fraud, the model **explains the reasoning**.

---

# 🖥️ Application Features

## 🏠 1. Project Overview Page

The first page explains:

- What fraud detection is
- Why explainable AI is important
- How this system works

This makes the dashboard understandable even to **non-technical users**.

---

## 🎛️ 2. Scenario Simulator

Interactive sliders allow users to simulate transactions.

Users can adjust:
- Transaction Amount
- Transaction Time (Hour)

The system instantly updates:
- Fraud Probability
- Local SHAP explanation

This acts like a **What-if Simulator**

Example:
$50 at 2 PM → low fraud probability
$5000 at 3 AM → high fraud probability

---

## 📊 3. Fraud vs Non-Fraud Comparison

Shows **distribution differences** between fraud and normal transactions.

Visualized features:
- Amount
- Hour
- Log Amount

This reveals patterns such as:

- Fraud transactions often occur at unusual hours
- Fraud may involve unusually high or unusual amounts

---

## 🌍 4. Global Fraud Insights

Uses **SHAP Global Feature Importance**.

This reveals: Which features influence fraud detection the most overall

Example insights:
- Transaction Amount
- Time of Transaction
- Certain PCA patterns

This helps understand **how the model behaves globally**.

---

# 📉 Visualization Design

Charts are intentionally **small and clean**.

Reasons:

- Dashboard readability
- Professional UI
- Faster rendering
- Avoid overwhelming the user

---

# 🎯 Learning Outcomes

This project demonstrates:

- Machine Learning for fraud detection
- Explainable AI (SHAP)
- Interactive ML dashboards
- Real-world financial analytics
- Model interpretability techniques

---

# 💡 Future Improvements

Possible enhancements:

- Real-time fraud detection API
- Deep learning models
- Graph-based fraud detection
- Transaction network visualization
- Real bank dataset integration
- Live fraud monitoring dashboard

---

# ⭐ Project Purpose

This project was built to demonstrate:
- Explainable Machine Learning
- Applied AI in Finance
- Interactive Data Science Systems

It serves as a **portfolio-grade AI project combining ML + XAI + dashboard engineering**.