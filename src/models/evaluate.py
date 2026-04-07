# src/models/evaluate.py
# src/models/evaluate.py

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import logging


def evaluate(y_true, y_pred):
    """
    Evaluate model performance using RMSE and MAE
    """

    # =========================
    # VALIDATION
    # =========================
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    # =========================
    # METRICS
    # =========================
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # =========================
    # LOG RESULTS
    # =========================
    logging.info(f"RMSE: {rmse:.2f}")
    logging.info(f"MAE: {mae:.2f}")

    return {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2)
    }