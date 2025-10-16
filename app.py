import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, r2_score
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
    tenure_preds = tenure_model.predict(tenure_X)

    df_result = df.copy()
    df_result["Churn Probability"] = churn_probs * 100
    df_result["Churn Prediction"] = pd.Series(churn_preds).map({0: "No", 1: "Yes"})
    df_result["Predicted Tenure (Months)"] = np.round(tenure_preds, 1)

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

        # Dynamic churn insight
        total_customers = len(df_result)
        total_churned = (df_result["Churn Prediction"] == "Yes").sum()
        churn_rate = (total_churned / total_customers) * 100
        df_result["Churn Prediction"] = pd.Series(churn_preds).map({0: "No", 1: "Yes"})
        top_contract = df_result[df_result["Churn Prediction"] == "Yes"]["Contract"].mode()[0]
        top_payment = df_result[df_result["Churn Prediction"] == "Yes"]["PaymentMethod"].mode()[0]

        st.markdown(f"""
        **Insight:**
        Out of **{total_customers}** customers, **{total_churned} ({churn_rate:.1f}%)** are predicted to churn.  
        Churn is most common among customers with **{top_contract}** contracts and those using **{top_payment}**.
        """)

        st.markdown("**Churn Prediction Accuracy**")
        if "Churn" in df_result.columns and df_result["Churn"].nunique() == 2:
            try:
                y_true = df_result["Churn"].map({"No": 0, "Yes": 1})
                y_pred = df_result["Churn Prediction"].map({"No": 0, "Yes": 1})
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

            model = LinearRegression()
            model.fit(df_result[["tenure"]], df_result[["Predicted Tenure (Months)"]])
            trend_line = model.predict(df_result[["tenure"]])
            ax3.plot(df_result["tenure"], trend_line, color='green', label="Trend Line")
            ax3.set_xlabel("True Tenure (Months)")
            ax3.set_ylabel("Predicted Tenure (Months)")
            ax3.set_title("True vs. Predicted Tenure")
            ax3.legend()
            st.pyplot(fig3)
            plt.clf()

            r2 = r2_score(df_result["tenure"], df_result["Predicted Tenure (Months)"])
            avg_predicted = df_result["Predicted Tenure (Months)"].mean()
            top_contract_tenure = df_result.groupby("Contract")["Predicted Tenure (Months)"].mean().idxmax()
            top_internet = df_result.groupby("InternetService")["Predicted Tenure (Months)"].mean().idxmin()

            st.markdown(f"""
            **Insight:**
            This chart shows the relationship between actual and predicted tenure.  
            The **green line** represents the model's trend in predicting tenure.  
            The **red dashed line** is an ideal 1:1 prediction. In this case, **R² Score**: **{r2:.2f}** —  
            The model has a reasonable correlation with actual tenure.

            The average predicted tenure is **{avg_predicted:.1f} months**.  
            Customers with **{top_contract_tenure}** contracts tend to stay the longest.  
            Customers with **{top_internet}** internet service have the shortest predicted tenure.
            """)

        else:
            st.info("True tenure not available. Showing distribution.")
            sns.histplot(df_result["Predicted Tenure (Months)"], kde=True)
            st.pyplot(plt.gcf())
            plt.clf()

    # --- Download ---
    csv = df_result.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Predictions as CSV", csv, "churn_tenure_predictions.csv", "text/csv")
