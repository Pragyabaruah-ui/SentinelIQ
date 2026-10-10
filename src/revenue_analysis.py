
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_orders() -> pd.DataFrame:
    """Load the original Olist orders dataset."""
    file_path = RAW_DATA_DIR / "olist_orders_dataset.csv"
    return pd.read_csv(file_path)


def load_payments() -> pd.DataFrame:
    """Load the original Olist payment dataset."""
    file_path = RAW_DATA_DIR / "olist_order_payments_dataset.csv"
    return pd.read_csv(file_path)


def calculate_daily_revenue() -> pd.DataFrame:
    """Calculate daily payment revenue with missing dates filled with zero."""
    orders = load_orders()
    payments = load_payments()

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"],
        errors="coerce"
    )

    order_dates = orders[
        ["order_id", "order_purchase_timestamp", "order_status"]
    ]

    merged = payments.merge(
        order_dates,
        on="order_id",
        how="left",
        validate="many_to_one"
    )

    merged = merged.dropna(subset=["order_purchase_timestamp"])
    merged["date"] = merged["order_purchase_timestamp"].dt.normalize()

    daily_revenue = (
        merged.groupby("date")["payment_value"]
        .sum()
        .sort_index()
    )

    full_dates = pd.date_range(
        start=daily_revenue.index.min(),
        end=daily_revenue.index.max(),
        freq="D"
    )

    daily_revenue = daily_revenue.reindex(
        full_dates,
        fill_value=0
    )

    result = daily_revenue.rename("daily_revenue").reset_index()
    result = result.rename(columns={"index": "date"})
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")

    return result


def save_daily_revenue() -> Path:
    """Save daily revenue results to the processed data directory."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    daily_revenue = calculate_daily_revenue()
    output_path = PROCESSED_DATA_DIR / "daily_revenue.csv"

    daily_revenue.to_csv(output_path, index=False)

    return output_path
