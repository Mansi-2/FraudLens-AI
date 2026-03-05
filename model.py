import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import os
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# -------------------------------
# Feature Engineering
# -------------------------------
def feature_engineering(df):
    df = df.copy()

    # Convert seconds to hour
    df["Hour"] = (df["Time"] // 3600) % 24

    # Night flag
    df["Is_Night"] = df["Hour"].apply(lambda x: 1 if x >= 22 or x <= 5 else 0)

    # Log amount
    df["Log_Amount"] = np.log1p(df["Amount"])

    # High amount flag (top 5%)
    df["High_Amount"] = (df["Amount"] > df["Amount"].quantile(0.95)).astype(int)

    return df
# -------------------------------
# Data Config
# -------------------------------

DATA_PATH = "data/creditcard.csv"
DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

# -------------------------------
# Load & Train Model
# -------------------------------

def download_dataset():
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(DATA_PATH):
        print("Downloading dataset...")
        r = requests.get(DATA_URL)
        with open(DATA_PATH, "wb") as f:
            f.write(r.content)

def load_or_train_model():

    download_dataset()

    df = pd.read_csv(DATA_PATH)

    df = feature_engineering(df)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/Test Split (important)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    # Handle class imbalance
    fraud_count = sum(y_train == 1)
    normal_count = sum(y_train == 0)
    scale_pos_weight = normal_count / fraud_count

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    # Optional sanity check (prints in terminal)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    explainer = shap.TreeExplainer(model)

    return model, explainer, scaler, X.columns.tolist(), df


# -------------------------------
# Prepare Input 
# -------------------------------
def prepare_input(amount, hour, scaler, columns, df):

    # Controlled simulation logic
    # High risk scenario -> sample fraud-like pattern
    if amount > 3000 or hour <= 4:
        baseline_pool = df[df["Class"] == 1]
        if len(baseline_pool) > 0:
            baseline = baseline_pool.sample(1, random_state=None).copy()
        else:
            baseline = df.drop("Class", axis=1).sample(1).copy()
    else:
        baseline = df[df["Class"] == 0].sample(1, random_state=None).copy()

    baseline = baseline.drop("Class", axis=1, errors="ignore")

    # Modify only user-controlled features
    baseline["Amount"] = amount
    baseline["Time"] = hour * 3600
    baseline["Hour"] = hour
    baseline["Is_Night"] = 1 if hour >= 22 or hour <= 5 else 0
    baseline["Log_Amount"] = np.log1p(amount)
    baseline["High_Amount"] = 1 if amount > df["Amount"].quantile(0.95) else 0

    baseline = baseline[columns]

    scaled = scaler.transform(baseline)

    return scaled