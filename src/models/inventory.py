# src/models/inventory.py

import numpy as np


def recommend_stock(preds, method="balanced"):
    """
    Convert predicted demand into recommended stock.

    Parameters:
    - preds: model predictions (array)
    - method: strategy for safety stock
        "balanced" → moderate safety stock (recommended)
        "conservative" → higher stock (avoid stockouts)
        "aggressive" → lower stock (reduce overstock)

    Returns:
    - recommended stock values
    """

    preds = np.array(preds)

    # =========================
    # METHOD 1: BALANCED (DEFAULT)
    # =========================
    if method == "balanced":
        demand_std = np.std(preds)
        return preds + (0.5 * demand_std)

    # =========================
    # METHOD 2: CONSERVATIVE
    # =========================
    elif method == "conservative":
        demand_std = np.std(preds)
        return preds + demand_std

    # =========================
    # METHOD 3: AGGRESSIVE
    # =========================
    elif method == "aggressive":
        return preds * 1.1  # 10% buffer

    # =========================
    # DEFAULT FALLBACK
    # =========================
    else:
        return preds