
from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ORDERS_FILE = RAW_DIR / "olist_orders_dataset.csv"
DAILY_REVENUE_FILE = PROCESSED_DIR / "daily_revenue.csv"
OUTPUT_FILE = PROCESSED_DIR / "monthly_revenue_analysis.csv"

# A month with fewer than this many orders is flagged for review.
# This is a review threshold, not proof of incomplete data.
MIN_ORDERS_FOR_REVIEW = 100


# ============================================================
# LOAD SOURCE DATA
# ============================================================

def load_data():
    """Load daily revenue and original order data."""
    if not DAILY_REVENUE_FILE.exists():
        raise FileNotFoundError(
            f"Daily revenue file not found: {DAILY_REVENUE_FILE}"
        )

    if not ORDERS_FILE.exists():
        raise FileNotFoundError(
            f"Orders file not found: {ORDERS_FILE}"
        )

    daily = pd.read_csv(
        DAILY_REVENUE_FILE,
        parse_dates=["date"],
    )

    orders = pd.read_csv(
        ORDERS_FILE,
        usecols=["order_id", "order_purchase_timestamp"],
    )

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"],
        errors="coerce",
    )

    orders = orders.dropna(
        subset=["order_purchase_timestamp"]
    )

    if daily.empty or orders.empty:
        raise ValueError(
            "Daily revenue or order data is empty."
        )

    return daily, orders


# ============================================================
# MONTHLY REVENUE AND ORDER COUNTS
# ============================================================

def calculate_monthly_analysis():
    """Calculate monthly revenue, order counts, and growth."""
    daily, orders = load_data()

    daily["month"] = daily["date"].dt.to_period("M").astype(str)

    monthly_revenue = (
        daily.groupby("month", as_index=False)
        .agg(
            total_revenue=("daily_revenue", "sum"),
            average_daily_revenue=("daily_revenue", "mean"),
            active_days=("daily_revenue", "count"),
        )
    )

    orders["month"] = (
        orders["order_purchase_timestamp"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly_orders = (
        orders.groupby("month", as_index=False)
        .agg(
            order_count=("order_id", "nunique"),
            latest_order_date=("order_purchase_timestamp", "max"),
        )
    )

    monthly = monthly_revenue.merge(
        monthly_orders,
        on="month",
        how="left",
        validate="one_to_one",
    )

    monthly["order_count"] = (
        monthly["order_count"].fillna(0).astype(int)
    )

    monthly["latest_order_date"] = pd.to_datetime(
        monthly["latest_order_date"],
        errors="coerce",
    )

    # Flag months with unusually low order counts for review.
    # This is not a definitive completeness test.
    monthly["coverage_status"] = "Review required"

    monthly.loc[
        monthly["order_count"] >= MIN_ORDERS_FOR_REVIEW,
        "coverage_status",
    ] = "Order activity recorded"

    monthly.loc[
        monthly["order_count"] == 0,
        "coverage_status",
    ] = "No recorded orders"

    # Month-over-month revenue growth.
    monthly["previous_month_revenue"] = (
        monthly["total_revenue"].shift(1)
    )

    # Only compare against a meaningful previous-month baseline
    # and exclude months flagged for review from growth highlights.
    valid_baseline = (
        monthly["previous_month_revenue"] >= 1000
    )

    monthly["mom_growth_pct"] = float("nan")

    monthly.loc[valid_baseline, "mom_growth_pct"] = (
        (
            monthly.loc[valid_baseline, "total_revenue"]
            - monthly.loc[valid_baseline, "previous_month_revenue"]
        )
        / monthly.loc[valid_baseline, "previous_month_revenue"]
        * 100
    )

    monthly.loc[
        monthly["coverage_status"] != "Order activity recorded",
        "mom_growth_pct",
    ] = float("nan")

    numeric_columns = [
        "total_revenue",
        "average_daily_revenue",
        "previous_month_revenue",
        "mom_growth_pct",
    ]

    monthly[numeric_columns] = monthly[numeric_columns].round(2)

    return monthly


# ============================================================
# SAVE MONTHLY RESULTS
# ============================================================

def save_monthly_analysis(monthly):
    """Save monthly analysis to a CSV file."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    monthly.to_csv(OUTPUT_FILE, index=False)

    print(f"\nMonthly analysis saved to: {OUTPUT_FILE}")


# ============================================================
# PRINT SUMMARY
# ============================================================

def main():
    monthly = calculate_monthly_analysis()
    save_monthly_analysis(monthly)

    print("\nMONTHLY REVENUE SUMMARY")
    print("=" * 65)
    print(f"Months analyzed: {len(monthly)}")
    print(
        f"Total revenue: "
        f"{monthly['total_revenue'].sum():,.2f}"
    )

    # Revenue totals are shown separately from coverage assessments.
    highest = monthly.loc[
        monthly["total_revenue"].idxmax()
    ]

    print(
        f"Highest-revenue month: {highest['month']} "
        f"({highest['total_revenue']:,.2f})"
    )

    eligible = monthly[
        (monthly["coverage_status"] == "Order activity recorded")
        & (monthly["total_revenue"] > 0)
    ]

    if not eligible.empty:
        highest_eligible = eligible.loc[
            eligible["total_revenue"].idxmax()
        ]

        lowest_eligible = eligible.loc[
            eligible["total_revenue"].idxmin()
        ]

        print(
            f"Highest revenue among months with sufficient "
            f"order activity: {highest_eligible['month']} "
            f"({highest_eligible['total_revenue']:,.2f})"
        )

        print(
            f"Lowest revenue among months with sufficient "
            f"order activity: {lowest_eligible['month']} "
            f"({lowest_eligible['total_revenue']:,.2f})"
        )

    valid_growth = monthly.dropna(
        subset=["mom_growth_pct"]
    )

    if not valid_growth.empty:
        strongest = valid_growth.loc[
            valid_growth["mom_growth_pct"].idxmax()
        ]

        weakest = valid_growth.loc[
            valid_growth["mom_growth_pct"].idxmin()
        ]

        print("\nMONTH-OVER-MONTH GROWTH HIGHLIGHTS")
        print("=" * 65)

        print(
            f"Strongest growth: {strongest['month']} "
            f"({strongest['mom_growth_pct']:.2f}%)"
        )

        print(
            f"Sharpest decline: {weakest['month']} "
            f"({weakest['mom_growth_pct']:.2f}%)"
        )

    print("\nMONTHLY DATA COVERAGE REVIEW")
    print("=" * 65)

    print(
        monthly[
            [
                "month",
                "order_count",
                "latest_order_date",
                "coverage_status",
            ]
        ].tail(6).to_string(index=False)
    )

    print(
        "\nInterpretation: 'Review required' identifies months "
        "with low recorded order counts. It does not prove why "
        "the counts are low or establish dataset completeness."
    )


if __name__ == "__main__":
    main()
