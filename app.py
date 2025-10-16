import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LinearRegression

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from preprocess import preprocess_churn, preprocess_tenure

st.set_page_config(layout="wide")
st.title("📊 Customer Churn & Tenure Prediction Dashboard")

# Load models
churn_model = joblib.load("Results/churn_logreg_pipeline.pkl")
tenure_model = joblib.load("Results/tenure_pipeline.pkl")

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
    tenure_preds = np.round(tenure_model.predict(tenure_X), 1)

    df_result = df.copy()
    df_result["Churn Probability"] = (churn_probs * 100).round(1).astype(str) + "%"
    df_result["Churn Prediction"] = pd.Series(churn_preds).map({0: "No", 1: "Yes"})
    df_result["Predicted Tenure (Months)"] = tenure_preds

    st.subheader("📈 Prediction Results")
    st.dataframe(df_result)

    # --- Filter Options ---
    st.sidebar.header("🔍 Filter Options")
    contract_filter = st.sidebar.multiselect("Filter by Contract", options=df_result["Contract"].unique())
    if contract_filter:
        df_result = df_result[df_result["Contract"].isin(contract_filter)]

    # --- Visual Insights ---
    st.subheader("📊 Visual Insights")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Total Churn Distribution**")
        fig, ax = plt.subplots()
        ax = sns.countplot(x="Churn", data=df_result)
        ax.set_ylabel("Number of Customers")
        ax.set_title("Total Churn Distribution")

        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=10, color='black', xytext=(0, 10),
                        textcoords='offset points')

        st.pyplot(fig)
        st.markdown("This bar chart shows the overall number of customers who churned vs. those who didn’t.")

        st.markdown("**Churn Prediction Accuracy**")
        if "Churn" in df_result.columns and df_result["Churn"].nunique() == 2:
            try:
                y_true = df_result["Churn"].map({"No": 0, "Yes": 1})
                y_pred = pd.Series(churn_preds)
                cm = confusion_matrix(y_true, y_pred, normalize='all')
                cm_percent = cm * 100
                fig_cm, ax_cm = plt.subplots()
                disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=["No Churn", "Churn"])
                disp.plot(ax=ax_cm, cmap="Blues", include_values=False)
                for i in range(cm_percent.shape[0]):
                    for j in range(cm_percent.shape[1]):
                        value = f"{cm_percent[i, j]:.1f}%"
                        ax_cm.text(j, i, value, ha="center", va="center", color="black", fontsize=12)
                st.pyplot(fig_cm)
                st.markdown("This confusion matrix shows prediction performance. Each percentage represents the share of total predictions. Values on the diagonal indicate correct predictions.")
                plt.clf()
            except Exception as e:
                st.error(f"Could not plot confusion matrix: {e}")
        else:
            st.info("True churn labels not available. Showing basic prediction count.")
            sns.countplot(x="Churn Prediction", data=df_result)
            st.pyplot(plt.gcf())
            plt.clf()

    with col2:
        st.markdown("**Total Tenure Prediction**")
        if "tenure" in df_result.columns and "Predicted Tenure (Months)" in df_result.columns:
            fig3, ax3 = plt.subplots()
            ax3.scatter(df_result["tenure"], df_result["Predicted Tenure (Months)"], alpha=0.4, label="Predictions")
            ax3.plot([0, 80], [0, 80], 'r--', label="Ideal Fit")

            # Add trend line
            model = LinearRegression()
            X = df_result["tenure"].values.reshape(-1, 1)
            y = df_result["Predicted Tenure (Months)"].values
            model.fit(X, y)
            trend = model.predict(X)
            ax3.plot(df_result["tenure"], trend, color="green", label="Trend Line")

            ax3.set_xlabel("True Tenure (Months)")
            ax3.set_ylabel("Predicted Tenure (Months)")
            ax3.set_title("True vs. Predicted Tenure")
            ax3.legend()
            st.pyplot(fig3)
            st.markdown("This scatter plot compares actual tenure vs. predicted. The dashed red line shows perfect predictions. The green trend line shows the overall fit.")
            plt.clf()
        else:
            st.info("True tenure not available. Showing distribution.")
            sns.histplot(df_result["Predicted Tenure (Months)"], kde=True)
            st.pyplot(plt.gcf())
            plt.clf()

    # --- Download ---
    csv = df_result.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Predictions as CSV", csv, "churn_tenure_predictions.csv", "text/csv")
