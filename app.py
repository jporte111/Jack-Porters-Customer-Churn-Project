import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from preprocess import preprocess_churn, preprocess_tenure

st.set_page_config(layout="wide")
st.title("📊 Customer Churn & Tenure Prediction Dashboard")

# Load models
churn_model = joblib.load("Results/churn_logreg_pipeline.pkl")
tenure_model = joblib.load("Results/tenure_pipeline.pkl") # this includes preprocessing + XGBoost

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
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

with col1:
    st.markdown("**Confusion Matrix: Churn Prediction Accuracy**")
    if "Churn" in df_result.columns and df_result["Churn"].nunique() == 2:
        try:
            y_true = df_result["Churn"].map({"No": 0, "Yes": 1})
            y_pred = df_result["Churn Prediction"]
            cm = confusion_matrix(y_true, y_pred, normalize='all')
            cm_percent = cm * 100
            fig_cm, ax_cm = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=["No Churn", "Churn"])
            disp.plot(ax=ax_cm, cmap="viridis")
            for i in range(cm_percent.shape[0]):
                for j in range(cm_percent.shape[1]):
                    value = f"{cm_percent[i, j]:.1f}%"
                    ax_cm.text(j, i, value, ha="center", va="center", color="white", fontsize=12)

            st.pyplot(fig_cm)
            plt.clf()
        except Exception as e:
            st.error(f"Could not plot confusion matrix: {e}")
    else:
        st.info("True churn labels not available. Showing basic prediction count.")
        sns.countplot(x="Churn Prediction", data=df_result)
        st.pyplot(plt.gcf())
        plt.clf()

with col2:
    st.markdown("**True vs. Predicted Tenure**")
    if "tenure" in df_result.columns and "Predicted Tenure (Months)" in df_result.columns:
        fig3, ax3 = plt.subplots()
        ax3.scatter(df_result["tenure"], df_result["Predicted Tenure (Months)"], alpha=0.4)
        ax3.plot([0, 80], [0, 80], 'r--')
        ax3.set_xlabel("True Tenure (Months)")
        ax3.set_ylabel("Predicted Tenure (Months)")
        ax3.set_title("True vs. Predicted Tenure")
        st.pyplot(fig3)
        plt.clf()
    else:
        st.info("True tenure not available. Showing distribution.")
        sns.histplot(df_result["Predicted Tenure (Months)"], kde=True)
        st.pyplot(plt.gcf())
        plt.clf()


    # --- Download ---
    csv = df_result.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Predictions as CSV", csv, "churn_tenure_predictions.csv", "text/csv")
