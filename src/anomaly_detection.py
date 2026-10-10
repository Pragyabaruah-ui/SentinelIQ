
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_daily_revenue() -> pd.DataFrame:
    """Load the prepared daily revenue dataset."""
    file_path = PROCESSED_DATA_DIR / "daily_revenue.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Revenue data not found: {file_path}")

    df = pd.read_csv(file_path, parse_dates=["date"])

    if df.empty:
        raise ValueError("The daily revenue dataset is empty.")

    if df["daily_revenue"].isna().any():
        raise ValueError("Daily revenue contains missing values.")

    return df.sort_values("date").reset_index(drop=True)


def detect_revenue_anomalies(
    contamination: float = 0.03,
    random_state: int = 42
) -> pd.DataFrame:
    """Detect unusual daily revenue using Isolation Forest."""
    if not 0 < contamination < 0.5:
        raise ValueError("Contamination must be between 0 and 0.5.")

    df = load_daily_revenue()
    features = df[["daily_revenue"]]

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state
    )

    model.fit(features)

    df["anomaly_prediction"] = model.predict(features)
    df["anomaly_score"] = model.decision_function(features)
    df["is_anomaly"] = df["anomaly_prediction"] == -1

    df["anomaly_type"] = "Normal"
    df.loc[
        df["is_anomaly"] & (df["daily_revenue"] == 0),
        "anomaly_type"
    ] = "Zero-revenue anomaly"

    df.loc[
        df["is_anomaly"] & (df["daily_revenue"] > 0),
        "anomaly_type"
    ] = "Unusual revenue"

    return df


def save_anomaly_results() -> Path:
    """Save anomaly predictions to CSV."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    results = detect_revenue_anomalies()
    output_path = PROCESSED_DATA_DIR / "revenue_anomalies.csv"

    results.to_csv(output_path, index=False)

    return output_path


def summarize_anomalies() -> dict:
    """Generate business summary metrics from anomaly predictions."""
    df = detect_revenue_anomalies()
    anomalies = df[df["is_anomaly"]]

    return {
        "days_analyzed": len(df),
        "anomalies_detected": len(anomalies),
        "anomaly_percentage": round(
            len(anomalies) / len(df) * 100, 2
        ),
        "total_revenue": round(df["daily_revenue"].sum(), 2),
        "average_daily_revenue": round(
            df["daily_revenue"].mean(), 2
        ),
        "highest_anomaly_revenue": round(
            anomalies["daily_revenue"].max(), 2
        ) if not anomalies.empty else 0,
        "lowest_anomaly_revenue": round(
            anomalies["daily_revenue"].min(), 2
        ) if not anomalies.empty else 0,
    }


def save_anomaly_summary() -> Path:
    """Export the business summary to a CSV report."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    summary = summarize_anomalies()
    summary_df = pd.DataFrame(
        list(summary.items()),
        columns=["metric", "value"]
    )

    output_path = PROCESSED_DATA_DIR / "anomaly_summary.csv"
    summary_df.to_csv(output_path, index=False)

    return output_path
