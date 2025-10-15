import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from utils.preprocess import preprocess_churn, preprocess_tenure

st.set_page_config(layout="wide")
st.title("📊 Customer Churn & Tenure Prediction Dashboard")

# Load models
churn_model = joblib.load("models/churn_logreg_pipeline.pkl")
tenure_model = joblib.load("models/tenure_pipeline.pkl")  # this includes preprocessing + XGBoost

# File uploader
uploaded_file = st.file_uploader("Upload your customer data CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📁 Preview of Uploaded Data")
    st.dataframe(df.head())

    # Run predictions
    churn_X = preprocess_churn(df)
    tenure_X = preprocess_tenure(df)

    churn_probs = churn_model.predict_proba(churn_X)[:, 1]
    churn_preds = churn_model.predict(churn_X)
    tenure_preds = tenure_model.predict(tenure_X)

    df_result = df.copy()
    df_result["Churn Probability"] = churn_probs
    df_result["Churn Prediction"] = churn_preds
    df_result["Predicted Tenure (Months)"] = tenure_preds

    st.subheader("📈 Prediction Results")
    st.dataframe(df_result)

    # --- Filter Options ---
    st.sidebar.header("🔍 Filter Options")
    contract_filter = st.sidebar.multiselect("Filter by Contract", options=df_result["Contract"].unique())
    if contract_filter:
        df_result = df_result[df_result["Contract"].isin(contract_filter)]

    # --- Visualizations ---
    st.subheader("📊 Visual Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Churn Distribution**")
        sns.countplot(x="Churn Prediction", data=df_result)
        st.pyplot(plt.gcf())
        plt.clf()

    with col2:
        st.markdown("**Predicted Tenure Distribution**")
        sns.histplot(df_result["Predicted Tenure (Months)"], kde=True)
        st.pyplot(plt.gcf())
        plt.clf()

    # --- Download ---
    csv = df_result.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Predictions as CSV", csv, "churn_tenure_predictions.csv", "text/csv")
