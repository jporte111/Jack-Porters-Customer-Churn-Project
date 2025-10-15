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
import numpy as np

def preprocess_tenure(df):
    expected_columns = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges"
    ]

    # Add missing columns with NaNs (will be handled by model's pipeline)
    for col in expected_columns:
        if col not in df.columns:
            df[col] = np.nan

    # Subset to expected columns only
    df = df[expected_columns]

    # Fix types
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df

def preprocess_churn(df):
    expected_columns = CHURN_FEATURES.copy()

    # Add missing columns with NaNs (for safety)
    for col in expected_columns:
        if col not in df.columns:
            df[col] = np.nan

    # Subset to expected columns
    df = df[expected_columns]

    # Fix types
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df


# Preprocessing for tenure
# Assumes the model already includes preprocessing steps
# Here, we just return the required columns

def preprocess_tenure(df):
    return df[TENURE_FEATURES].copy()
