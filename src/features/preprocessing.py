# src/features/preprocessing.py

import pandas as pd
import joblib
import os


def preprocess(df):
    print("Starting preprocessing...")

    # =========================
    # 1. SELECT REQUIRED COLUMNS
    # =========================
    required_cols = ["InvoiceDate", "StockCode", "Quantity"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df[required_cols].copy()

    # =========================
    # 2. REMOVE MISSING VALUES
    # =========================
    print("Handling missing values...")
    df = df.dropna()

    # =========================
    # 3. REMOVE RETURNS (NEGATIVE SALES)
    # =========================
    print("Removing returns (negative quantity)...")
    df = df[df["Quantity"] > 0]

    # =========================
    # 4. CONVERT DATE COLUMN
    # =========================
    print("Converting InvoiceDate to datetime...")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # =========================
    # 5. EXTRACT DATE (REMOVE TIME)
    # =========================
    print("Extracting date...")
    df["date"] = df["InvoiceDate"].dt.date
    df["date"] = pd.to_datetime(df["date"])

    # =========================
    # 6. AGGREGATE TO DAILY SALES PER ITEM
    # =========================
    print("Aggregating daily sales per item...")

    df_daily = (
        df.groupby(["date", "StockCode"])["Quantity"]
        .sum()
        .reset_index()
    )

    # =========================
    # 7. RENAME COLUMNS
    # =========================
    df_daily = df_daily.rename(
        columns={
            "StockCode": "item_id",
            "Quantity": "sales"
        }
    )

    # =========================
    # 🔥 8. CREATE ITEM MAPPING (IMPORTANT)
    # =========================
    print("Creating item_id mapping...")

    categories = df_daily["item_id"].astype("category").cat.categories
    item_mapping = dict(enumerate(categories))

    # Save mapping for later use (UI)
    os.makedirs("models", exist_ok=True)
    joblib.dump(item_mapping, "models/item_mapping.pkl")

    # =========================
    # 🔥 9. ENCODE item_id
    # =========================
    print("Encoding item_id...")
    df_daily["item_id"] = df_daily["item_id"].astype("category").cat.codes

    # =========================
    # 10. SORT DATA
    # =========================
    df_daily = df_daily.sort_values(by=["item_id", "date"])

    # =========================
    # 11. REMOVE LOW-FREQUENCY ITEMS
    # =========================
    print("Filtering low-frequency items...")

    item_counts = df_daily["item_id"].value_counts()
    valid_items = item_counts[item_counts > 30].index

    df_daily = df_daily[df_daily["item_id"].isin(valid_items)]

    # =========================
    # 12. RESET INDEX
    # =========================
    df_daily = df_daily.reset_index(drop=True)

    print("Preprocessing completed.")
    print(f"Final shape: {df_daily.shape}")

    return df_daily