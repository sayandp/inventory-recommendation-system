# src/features/feature_engineering.py

import pandas as pd


def create_features(df):
    print("Starting feature engineering...")

    # =========================
    # 1. VALIDATION
    # =========================
    required_cols = ["date", "item_id", "sales"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # =========================
    # 2. SORT DATA (CRITICAL)
    # =========================
    df = df.sort_values(by=["item_id", "date"]).copy()

    # =========================
    # 3. CREATE TIME FEATURES
    # =========================
    print("Creating time-based features...")

    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month

    # =========================
    # 4. CREATE LAG FEATURES
    # =========================
    print("Creating lag features...")

    df["lag_1"] = df.groupby("item_id")["sales"].shift(1)

    df["lag_7"] = df.groupby("item_id")["sales"].shift(7)
    df["lag_14"] = df.groupby("item_id")["sales"].shift(14)
    df["lag_28"] = df.groupby("item_id")["sales"].shift(28)
    # =========================
    # 5. CREATE ROLLING FEATURES (NO LEAKAGE)
    # =========================
    print("Creating rolling features...")

    df["rolling_mean_7"] = (
        df.groupby("item_id")["sales"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    df["rolling_mean_28"] = (
        df.groupby("item_id")["sales"]
        .shift(1)
        .rolling(28)
        .mean()
    )

    # =========================
    # 6. HANDLE MISSING VALUES
    # =========================
    print("Dropping NaN values from lag/rolling...")

    df = df.dropna()

    # =========================
    # 7. FINAL SORT + RESET
    # =========================
    df = df.sort_values(by=["item_id", "date"]).reset_index(drop=True)

    print("Feature engineering completed.")
    print(f"Final shape: {df.shape}")

    return df