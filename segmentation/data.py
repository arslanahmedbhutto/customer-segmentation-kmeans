"""Load and tidy the Mall Customers dataset."""

from pathlib import Path

import pandas as pd

from segmentation import DATA_PATH

INCOME, SPEND, AGE = "income_k", "spending_score", "age"
FEATURES_2D = [INCOME, SPEND]
FEATURES_3D = [AGE, INCOME, SPEND]


def load_customers(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"CustomerID": str})
    df = df.rename(columns={
        "CustomerID": "customer_id", "Genre": "gender", "Age": AGE,
        "Annual Income (k$)": INCOME, "Spending Score (1-100)": SPEND,
    })
    if df["customer_id"].duplicated().any():
        raise ValueError("duplicate customer IDs")
    if df[[AGE, INCOME, SPEND]].isna().any().any():
        raise ValueError("missing values in numeric columns")
    return df
