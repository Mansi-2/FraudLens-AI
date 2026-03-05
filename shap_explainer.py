import joblib
import shap
import matplotlib.pyplot as plt

model = joblib.load("fraud_model.pkl")
explainer = joblib.load("shap_explainer.pkl")

def get_shap_values(input_df):
    shap_values = explainer.shap_values(input_df)
    return shap_values

def plot_shap(input_df):
    shap_values = explainer.shap_values(input_df)
    shap.force_plot(
        explainer.expected_value,
        shap_values,
        input_df,
        matplotlib=True
    )
    plt.tight_layout()
    plt.show()