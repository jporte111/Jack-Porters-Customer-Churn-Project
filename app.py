import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import confusion_matrix, r2_score
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

    # Preprocess data
    churn_X = preprocess_churn(df)
    tenure_X = preprocess_tenure(df)

    # Make predictions
    churn_probs = churn_model.predict_proba(churn_X)[:, 1]
    churn_preds = churn_model.predict(churn_X)
    tenure_preds = tenure_model.predict(tenure_X)

    df_result = df.copy()
    df_result["Churn Probability"] = (churn_probs * 100).round(1)
    df_result["Churn Prediction"] = pd.Series(churn_preds).map({0: "No", 1: "Yes"})
    df_result["Predicted Tenure (Months)"] = np.round(tenure_preds, 1)

    st.subheader("📈 Prediction Results")
    st.dataframe(df_result)

    # --- Filters ---
    st.sidebar.header("🔍 Filter Options")
    filter_columns = ["Contract", "gender", "InternetService", "PaymentMethod", "Churn Prediction"]
    for col in filter_columns:
        if col in df_result.columns:
            selected = st.sidebar.multiselect(f"Filter by {col}", options=df_result[col].unique(), default=df_result[col].unique())
            df_result = df_result[df_result[col].isin(selected)]

    # --- Visualizations ---
    st.subheader("📊 Visual Insights")
    col1, col2 = st.columns(2)

    # Churn Bar Chart + Insights
    with col1:
        st.markdown("### Total Churn Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x="Churn Prediction", data=df_result, ax=ax)
        ax.set_ylabel("Number of Customers")
        ax.set_title("Total Churn Distribution")
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=10, color='black', xytext=(0, 10),
                        textcoords='offset points')
        st.pyplot(fig)

        # Churn Insights + Next Steps
        total_customers = len(df_result)
        total_churned = (df_result["Churn Prediction"] == "Yes").sum()
        churn_rate = (total_churned / total_customers) * 100 if total_customers else 0

        top_contract = df_result[df_result["Churn Prediction"] == "Yes"]["Contract"].mode()[0] if not df_result[df_result["Churn Prediction"] == "Yes"].empty else "N/A"
        top_payment = df_result[df_result["Churn Prediction"] == "Yes"]["PaymentMethod"].mode()[0] if not df_result[df_result["Churn Prediction"] == "Yes"].empty else "N/A"

        st.markdown("""
        ### 🔍 Churn Insight:
        - **Total customers:** {0}
        - **Predicted to churn:** {1} ({2:.1f}%)
        - **Most churners are on:** *{3}* contracts
        - **Most common payment method among churners:** *{4}*

        ### ✅ Next Steps:
        - Offer loyalty incentives or discounts for customers on *{3}* contracts.
        - Improve payment experience for *{4}* users.
        - Implement proactive retention strategies for high churn probability segments.
        """.format(total_customers, total_churned, churn_rate, top_contract, top_payment))

    # Tenure Scatter Plot + Insights
    with col2:
        st.markdown("### Total Tenure Prediction")
        fig, ax = plt.subplots()
        ax.scatter(df_result["tenure"], df_result["Predicted Tenure (Months)"], alpha=0.4, label="Predictions")
        ax.plot([0, 80], [0, 80], 'r--', label="Ideal Fit")
        z = np.polyfit(df_result["tenure"], df_result["Predicted Tenure (Months)"], 1)
        p = np.poly1d(z)
        ax.plot(df_result["tenure"], p(df_result["tenure"]), "g-", label="Trend Line")
        ax.set_xlabel("True Tenure (Months)")
        ax.set_ylabel("Predicted Tenure (Months)")
        ax.set_title("True vs. Predicted Tenure")
        ax.legend()
        st.pyplot(fig)

        r2 = r2_score(df_result["tenure"], df_result["Predicted Tenure (Months)"])

        st.markdown("""
        ### 🧠 Tenure Insight:
        - The green line shows the model's trend in predicting tenure.
        - The red dashed line is the ideal 1:1 prediction.
        - **R² Score:** {0:.2f} — Higher is better.

        ### ✅ Next Steps:
        - Review predicted short tenure customers for churn risk.
        - Tailor retention campaigns by tenure group.
        - Use predicted tenure for more personalized support or upsell timing.
        """.format(r2))

    # Confusion Matrix
    if "ChurnFlag" in df_result.columns:
        st.subheader("Confusion Matrix - Churn Prediction (%)")
        cm = confusion_matrix(df_result["ChurnFlag"], churn_preds)
        cm_percent = cm / cm.sum() * 100
        fig, ax = plt.subplots()
        sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues",
                    xticklabels=["No Churn", "Churn"],
                    yticklabels=["No Churn", "Churn"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix - Churn Prediction (%)")
        st.pyplot(fig)

    # Download predictions
    csv = df_result.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Predictions as CSV", csv, "churn_tenure_predictions.csv", "text/csv")
