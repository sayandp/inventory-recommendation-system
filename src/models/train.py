# src/models/train.py

from xgboost import XGBRegressor
import joblib
import os


def train_model(X_train, y_train):

    print("Training model...")

    model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
  )

    model.fit(X_train, y_train)

    print("Model training completed.")

    return model


# =========================
# SAVE MODEL FUNCTION
# =========================
def save_model(model, path="models/model.pkl"):

    # 🔥 Ensure folder exists (important fix)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    joblib.dump(model, path)

    print(f"Model saved at {path}")