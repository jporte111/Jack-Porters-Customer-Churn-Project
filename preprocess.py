import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Columns needed for churn and tenure
CHURN_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

TENURE_FEATURES = [
    col for col in CHURN_FEATURES if col != "tenure"
]

# Preprocessing for churn
def preprocess_churn(df):
    # Define the exact columns used during model training
    expected_columns = [
        "gender", "SeniorCitizen", "Partner", "Dependents",
        "tenure", "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "MonthlyCharges", "TotalCharges"
    ]

    # Add any missing columns with default values
    for col in expected_columns:
        if col not in df.columns:
            df[col] = np.nan

    # Subset to only expected columns
    df = df[expected_columns]

    # Ensure TotalCharges is numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Return clean data
    return df


# Preprocessing for tenure
# Assumes the model already includes preprocessing steps
# Here, we just return the required columns

def preprocess_tenure(df):
    return df[TENURE_FEATURES].copy()
