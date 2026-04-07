# main.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

from src.features.preprocessing import preprocess
from src.features.feature_engineering import create_features
from src.features.external_data import add_holidays, add_weather

from src.models.split import split_data
from src.models.train import train_model, save_model
from src.models.evaluate import evaluate
from src.models.inventory import recommend_stock


# =========================
# LOGGING SETUP
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():

    # =========================
    # 1. LOAD DATA
    # =========================
    logging.info("Loading data...")
    df = pd.read_csv("data/raw/online_retail.csv", encoding="latin1")
    logging.info(f"Raw data shape: {df.shape}")

    # =========================
    # 2. PREPROCESSING
    # =========================
    logging.info("Preprocessing data...")
    df = preprocess(df)
    logging.info(f"After preprocessing: {df.shape}")

    # =========================
    # 3. ADD EXTERNAL DATA
    # =========================
    logging.info("Adding external data...")
    df = add_holidays(df)
    df = add_weather(df)

    # DEBUG CHECK
    print("\nWeather data sample:")
    print(df[["temperature", "rain"]].head())

    # =========================
    # 4. FEATURE ENGINEERING
    # =========================
    logging.info("Creating features...")
    df = create_features(df)
    logging.info(f"After feature engineering: {df.shape}")
    
    # =========================
    # 5. TRAIN TEST SPLIT
    # =========================
    logging.info("Splitting data...")
    train, test = split_data(df)

    # =========================
    # 6. TRAIN MODEL
    # =========================
    features = [
    "item_id",   # 🔥 ADD THIS
    "day_of_week", "week_of_year", "month",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_28",
    "is_holiday"
   ]


    X_train = train[features]
    y_train = train["sales"]

    X_test = test[features]
    y_test = test["sales"]

    logging.info("Training model...")
    model = train_model(X_train, y_train)

    # =========================
    # MLOPS: SAVE MODEL
    # =========================
    save_model(model)

    # =========================
    # 7. PREDICT + EVALUATE
    # =========================
    logging.info("Making predictions...")
    preds = model.predict(X_test)

    logging.info("Evaluating model...")
    metrics = evaluate(y_test, preds)

    logging.info(f"Model Performance: {metrics}")

    # =========================
    # 🔥 8. INVENTORY SYSTEM (SMART)
    # =========================
    logging.info("Generating inventory recommendations...")

    recommended_stock = recommend_stock(preds, method="balanced")

    print("\nSample Inventory Recommendations:")
    for i in range(5):
        print(f"Predicted: {preds[i]:.2f} → Recommended: {recommended_stock[i]:.2f}")

    # OPTIONAL: Save results
    results = X_test.copy()
    results["actual"] = y_test.values
    results["predicted"] = preds
    results["recommended_stock"] = recommended_stock

    results.to_csv("inventory_recommendations.csv", index=False)

    # =========================
    # 9. FEATURE IMPORTANCE
    # =========================
    logging.info("Plotting feature importance...")

    importance = model.feature_importances_
    feature_names = features

    indices = np.argsort(importance)

    plt.figure(figsize=(10, 6))
    plt.barh(np.array(feature_names)[indices], importance[indices])

    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.title("Feature Importance")

    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.show()

    # =========================
    # 10. FINAL OUTPUT CHECK
    # =========================
    print("\nTrain sample:")
    print(train.head())

    print("\nTest sample:")
    print(test.head())

    print("\nColumns:")
    print(df.columns)


if __name__ == "__main__":
    main()