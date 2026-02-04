import streamlit as st
import sklearn
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from preprocess import preprocess_churn, preprocess_tenure
from sklearn.metrics import r2_score, confusion_matrix
import numpy as np
import site
import sys
sys.path.append(site.getusersitepackages())

st.write("✅ Scikit-learn version (patched):", sklearn.__version__)

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
    df_result["Churn Probability"] = (churn_probs * 100).round(1)
    df_result["Churn Prediction"] = pd.Series(churn_preds).map({0: "No", 1: "Yes"})
    df_result["Predicted Tenure (Months)"] = np.round(tenure_preds, 1)

    st.subheader("📈 Prediction Results")
    st.dataframe(df_result)

    # --- Filter Options ---
    st.sidebar.header("🔍 Filter Options")
    filter_columns = ["Contract", "gender", "InternetService", "PaymentMethod", "Churn Prediction"]
    for col in filter_columns:
        if col in df_result.columns:
            selected = st.sidebar.multiselect(f"Filter by {col}", options=df_result[col].unique(), default=df_result[col].unique())
            df_result = df_result[df_result[col].isin(selected)]

    # --- Visualizations ---
    st.subheader("📊 Visual Insights")
    col1, col2 = st.columns(2)

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

    st.markdown("### Confusion Matrix - Churn Prediction (%)")
    cm = confusion_matrix(df["Churn"].map({"No": 0, "Yes": 1}), churn_preds)
    cm_percent = cm / cm.sum() * 100

    fig, ax = plt.subplots()
    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Insights Columns
    col3, col4 = st.columns(2)

    with col3:
        total_customers = len(df_result)
        total_churned = (df_result["Churn Prediction"] == "Yes").sum()
        churn_rate = (total_churned / total_customers) * 100 if total_customers else 0

        top_contract = df_result[df_result["Churn Prediction"] == "Yes"]["Contract"].mode()[0] if not df_result[df_result["Churn Prediction"] == "Yes"].empty else "N/A"
        top_payment = df_result[df_result["Churn Prediction"] == "Yes"]["PaymentMethod"].mode()[0] if not df_result[df_result["Churn Prediction"] == "Yes"].empty else "N/A"

        st.markdown("### Churn Insight")
        st.markdown(f"""
        - **Total customers:** {total_customers}
        - **Predicted to churn:** {total_churned} ({churn_rate:.1f}%)
        - Most churners are on **{top_contract}** contracts.
        - Most common payment method among churners: **{top_payment}**

        **Next Steps:**
        - Target at-risk customers with loyalty offers or incentives.
        - Improve support and satisfaction for month-to-month users.
        - Investigate issues with the '{top_payment}' method users.
        """)

    with col4:
        r2 = r2_score(df_result["tenure"], df_result["Predicted Tenure (Months)"])
        st.markdown("### Tenure Insight")
        st.markdown(f"""
        - The green line shows the model's trend in predicting tenure.
        - The red dashed line is the ideal 1:1 prediction.
        - **R² Score:** {r2:.2f} — Higher is better. Indicates the model's accuracy.

        **Next Steps:**
        - Focus on customers with lower predicted tenure for retention efforts.
        - Use predicted tenure to forecast long-term revenue trends.
        - Combine this with churn probability for better targeting.
        """)

    # --- Download ---
    csv = df_result.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Predictions as CSV", csv, "churn_tenure_predictions.csv", "text/csv")