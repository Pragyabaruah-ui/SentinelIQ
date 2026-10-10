from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

st.set_page_config(
    page_title="SentinelIQ | Revenue Intelligence",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_data():
    revenue_path = PROCESSED_DIR / "daily_revenue.csv"
    anomalies_path = PROCESSED_DIR / "revenue_anomalies.csv"

    if not revenue_path.exists():
        raise FileNotFoundError(
            "daily_revenue.csv not found. Run revenue processing first."
        )
    if not anomalies_path.exists():
        raise FileNotFoundError(
            "revenue_anomalies.csv not found. Run anomaly detection first."
        )

    revenue = pd.read_csv(revenue_path, parse_dates=["date"])
    anomalies = pd.read_csv(anomalies_path, parse_dates=["date"])
    return revenue, anomalies


st.title("SentinelIQ")
st.subheader("E-commerce Revenue Intelligence Dashboard")
st.caption(
    "Monitor daily payment revenue and investigate unusual revenue patterns."
)


# ============================================================
# REVENUE MONITORING AND ANOMALY DETECTION
# ============================================================

try:
    revenue, anomalies = load_data()

    min_date = revenue["date"].min().date()
    max_date = revenue["date"].max().date()

    st.sidebar.header("Dashboard Filters")
    selected_dates = st.sidebar.date_input(
        "Select date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date, end_date = min_date, max_date

    start_timestamp = pd.Timestamp(start_date)
    end_timestamp = pd.Timestamp(end_date) + pd.Timedelta(days=1)

    filtered_revenue = revenue[
        (revenue["date"] >= start_timestamp)
        & (revenue["date"] < end_timestamp)
    ].copy()

    filtered_anomalies = anomalies[
        (anomalies["date"] >= start_timestamp)
        & (anomalies["date"] < end_timestamp)
    ].copy()

    total_revenue = filtered_revenue["daily_revenue"].sum()
    average_revenue = (
        filtered_revenue["daily_revenue"].mean()
        if not filtered_revenue.empty
        else 0
    )
    flagged = filtered_anomalies[
        filtered_anomalies["is_anomaly"].astype(bool)
    ].copy()

    anomaly_count = len(flagged)
    anomaly_percentage = (
        anomaly_count / len(filtered_anomalies) * 100
        if not filtered_anomalies.empty
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"{total_revenue:,.2f}")
    col2.metric("Average Daily Revenue", f"{average_revenue:,.2f}")
    col3.metric("Anomalies Detected", anomaly_count)
    col4.metric("Anomaly Rate", f"{anomaly_percentage:.2f}%")

    st.caption(f"Selected period: {start_date} to {end_date}")
    st.divider()

    st.subheader("Daily Revenue Trend")
    revenue_chart = px.line(
        filtered_revenue,
        x="date",
        y="daily_revenue",
        title="Daily Payment Revenue",
        labels={"date": "Date", "daily_revenue": "Revenue"},
    )
    st.plotly_chart(
        revenue_chart,
        use_container_width=True,
        key="daily_revenue_trend",
    )

    st.subheader("Revenue Anomalies")
    if not flagged.empty:
        anomaly_chart = px.scatter(
            flagged,
            x="date",
            y="daily_revenue",
            color="anomaly_type",
            hover_data=["anomaly_score"],
            title="Detected Unusual Revenue Days",
            labels={
                "date": "Date",
                "daily_revenue": "Revenue",
                "anomaly_type": "Anomaly Type",
            },
        )
        st.plotly_chart(
            anomaly_chart,
            use_container_width=True,
            key="revenue_anomaly_scatter",
        )

        st.dataframe(
            flagged[
                ["date", "daily_revenue", "anomaly_score", "anomaly_type"]
            ].sort_values("anomaly_score"),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Business Insights")
        highest = flagged.loc[flagged["daily_revenue"].idxmax()]
        lowest = flagged.loc[flagged["daily_revenue"].idxmin()]

        st.write(
            f"Highest flagged revenue: {highest['daily_revenue']:,.2f} "
            f"on {highest['date'].strftime('%Y-%m-%d')}."
        )
        st.write(
            f"Lowest flagged revenue: {lowest['daily_revenue']:,.2f} "
            f"on {lowest['date'].strftime('%Y-%m-%d')}."
        )
        st.write(
            "Isolation Forest flagged these dates because their revenue "
            "patterns differ from patterns learned by the model. Review "
            "the underlying transactions before drawing business conclusions."
        )
    else:
        st.info("No anomalies were detected in the selected period.")

    st.caption(
        "Anomaly flags are model-generated alerts, not confirmed fraud "
        "or proof of a business problem."
    )

except (FileNotFoundError, ValueError, KeyError) as error:
    st.error(str(error))


# ============================================================
# PAYMENT METHOD ANALYSIS
# ============================================================

st.divider()
st.header("Payment Method Analysis")
st.write(
    "Analyze payment methods by transaction volume, revenue contribution, "
    "and their presence on dates flagged by the anomaly detection model."
)

try:
    payment_path = PROCESSED_DIR / "payment_method_analysis.csv"
    anomaly_rate_path = PROCESSED_DIR / "payment_method_anomaly_rates.csv"

    if not payment_path.exists():
        raise FileNotFoundError(
            f"Payment analysis file not found: {payment_path}"
        )
    if not anomaly_rate_path.exists():
        raise FileNotFoundError(
            f"Payment anomaly-rate file not found: {anomaly_rate_path}"
        )

    payment_df = pd.read_csv(payment_path)
    anomaly_rate_df = pd.read_csv(anomaly_rate_path)

    st.subheader("Payment Method Overview")
    st.dataframe(payment_df, use_container_width=True, hide_index=True)

    st.subheader("Transaction Volume by Payment Method")
    if "transaction_count" in payment_df.columns:
        transaction_chart = px.bar(
            payment_df,
            x="payment_type",
            y="transaction_count",
            title="Transactions by Payment Method",
            labels={
                "payment_type": "Payment Method",
                "transaction_count": "Transaction Count",
            },
        )
        st.plotly_chart(
            transaction_chart,
            use_container_width=True,
            key="payment_transaction_volume",
        )

    if "revenue_share_pct" in payment_df.columns:
        st.subheader("Revenue Share by Payment Method")
        payment_share_chart = px.bar(
            payment_df,
            x="payment_type",
            y="revenue_share_pct",
            title="Revenue Share by Payment Method",
            labels={
                "payment_type": "Payment Method",
                "revenue_share_pct": "Revenue Share (%)",
            },
        )
        st.plotly_chart(
            payment_share_chart,
            use_container_width=True,
            key="payment_revenue_share",
        )

    st.subheader("Flagged Transaction Rate by Payment Method")
    required_rate_columns = [
        "payment_type",
        "total_records",
        "flagged_records",
        "flagged_record_rate_pct",
    ]
    st.dataframe(
        anomaly_rate_df[required_rate_columns],
        use_container_width=True,
        hide_index=True,
    )

    rate_chart = px.bar(
        anomaly_rate_df,
        x="payment_type",
        y="flagged_record_rate_pct",
        title="Flagged Transaction Rate by Payment Method",
        labels={
            "payment_type": "Payment Method",
            "flagged_record_rate_pct": "Flagged Record Rate (%)",
        },
    )
    st.plotly_chart(
        rate_chart,
        use_container_width=True,
        key="payment_anomaly_rate",
    )

    st.caption(
        "Flagged transaction rates are based on order purchase dates "
        "identified by the anomaly detector. They do not establish causation "
        "or indicate confirmed fraud."
    )

except FileNotFoundError as error:
    st.error(str(error))
except (ValueError, KeyError) as error:
    st.error(f"Unable to display payment analysis: {error}")


# ============================================================
# PRODUCT CATEGORY ANALYSIS
# ============================================================

st.divider()
st.header("Product Category Performance")
st.write(
    "Explore which product categories contribute the most to e-commerce "
    "item revenue, compare sales volume, and identify categories with "
    "strong revenue contributions."
)

try:
    product_path = PROCESSED_DIR / "product_category_analysis.csv"

    if not product_path.exists():
        raise FileNotFoundError(
            "product_category_analysis.csv not found. "
            "Run: python src/product_analysis.py"
        )

    product_df = pd.read_csv(product_path)
    st.subheader("Category Revenue Overview")
    st.dataframe(product_df, use_container_width=True, hide_index=True)

    st.subheader("Top 10 Categories by Item Revenue")
    top_categories = product_df.nlargest(
        10, "total_item_revenue"
    ).sort_values("total_item_revenue", ascending=True)

    category_revenue_chart = px.bar(
        top_categories,
        x="total_item_revenue",
        y="product_category_name",
        orientation="h",
        labels={
            "total_item_revenue": "Item Revenue",
            "product_category_name": "Product Category",
        },
        title="Top 10 Product Categories by Revenue",
    )
    st.plotly_chart(
        category_revenue_chart,
        use_container_width=True,
        key="product_category_revenue",
    )

    st.subheader("Revenue Share by Category")
    share_categories = product_df.nlargest(
        10, "revenue_share_pct"
    ).sort_values("revenue_share_pct", ascending=True)

    category_share_chart = px.bar(
        share_categories,
        x="revenue_share_pct",
        y="product_category_name",
        orientation="h",
        labels={
            "revenue_share_pct": "Revenue Share (%)",
            "product_category_name": "Product Category",
        },
        title="Top 10 Categories by Revenue Share",
    )
    st.plotly_chart(
        category_share_chart,
        use_container_width=True,
        key="product_category_share",
    )

    st.subheader("Items Sold by Category")
    volume_categories = product_df.nlargest(
        10, "total_items"
    ).sort_values("total_items", ascending=True)

    category_volume_chart = px.bar(
        volume_categories,
        x="total_items",
        y="product_category_name",
        orientation="h",
        labels={
            "total_items": "Order Items",
            "product_category_name": "Product Category",
        },
        title="Top 10 Categories by Order Item Count",
    )
    st.plotly_chart(
        category_volume_chart,
        use_container_width=True,
        key="product_category_volume",
    )

    st.caption(
        "Revenue represents item prices, excluding freight charges. Products "
        "with missing category names are grouped under 'unknown'. Category "
        "revenue is descriptive and does not by itself establish profitability."
    )

except FileNotFoundError as error:
    st.error(str(error))
except (ValueError, KeyError) as error:
    st.error(f"Unable to display product category analysis: {error}")


# ============================================================
# MONTHLY REVENUE, GROWTH, AND DATA COVERAGE
# ============================================================

st.divider()
st.header("Monthly Revenue & Growth Analysis")
st.write(
    "Track monthly revenue trends, compare performance with the previous "
    "month, and review periods with unusually low recorded order activity."
)

try:
    monthly_path = PROCESSED_DIR / "monthly_revenue_analysis.csv"

    if not monthly_path.exists():
        raise FileNotFoundError(
            "monthly_revenue_analysis.csv not found. "
            "Run: python src/monthly_analysis.py"
        )

    monthly_df = pd.read_csv(monthly_path)

    required_monthly_columns = {
        "month",
        "total_revenue",
        "average_daily_revenue",
        "active_days",
        "previous_month_revenue",
        "mom_growth_pct",
        "order_count",
        "latest_order_date",
        "coverage_status",
    }
    missing_columns = required_monthly_columns - set(monthly_df.columns)

    if missing_columns:
        st.error(
            "The monthly CSV is missing these columns: "
            f"{', '.join(sorted(missing_columns))}. "
            "Run `python src/monthly_analysis.py` and refresh the dashboard."
        )
    else:
        monthly_df["month"] = pd.to_datetime(
            monthly_df["month"], format="%Y-%m", errors="coerce"
        )
        monthly_df["latest_order_date"] = pd.to_datetime(
            monthly_df["latest_order_date"], errors="coerce"
        )
        monthly_df = (
            monthly_df.dropna(subset=["month"])
            .sort_values("month")
            .copy()
        )

        # Explicit labels for hover information.
        monthly_df["coverage_label"] = monthly_df["coverage_status"].fillna(
            "Status unavailable"
        )

        st.subheader("Monthly Revenue Trend")
        monthly_revenue_chart = px.line(
            monthly_df,
            x="month",
            y="total_revenue",
            markers=True,
            title="Total Revenue by Month",
            labels={"month": "Month", "total_revenue": "Revenue"},
            custom_data=["order_count", "coverage_label"],
        )
        monthly_revenue_chart.update_traces(
            hovertemplate=(
                "Month: %{x|%B %Y}<br>"
                "Revenue: %{y:,.2f}<br>"
                "Recorded orders: %{customdata[0]}<br>"
                "Coverage status: %{customdata[1]}"
                "<extra></extra>"
            )
        )

        review_points = monthly_df[
            monthly_df["coverage_status"] != "Order activity recorded"
        ].copy()

        # Red markers make months requiring review stand out on the line.
        if not review_points.empty:
            monthly_revenue_chart.add_trace(
                go.Scatter(
                    x=review_points["month"],
                    y=review_points["total_revenue"],
                    mode="markers",
                    name="Coverage review required",
                    marker={
                        "color": "red",
                        "size": 13,
                        "symbol": "diamond",
                        "line": {"color": "white", "width": 1},
                    },
                    customdata=review_points[
                        ["order_count", "coverage_label"]
                    ].to_numpy(),
                    hovertemplate=(
                        "Month: %{x|%B %Y}<br>"
                        "Revenue: %{y:,.2f}<br>"
                        "Recorded orders: %{customdata[0]}<br>"
                        "Coverage status: %{customdata[1]}"
                        "<extra>Coverage review required</extra>"
                    ),
                )
            )

        monthly_revenue_chart.update_layout(
            legend_title_text="Data status",
        )
        st.plotly_chart(
            monthly_revenue_chart,
            use_container_width=True,
            key="monthly_revenue_trend",
        )

        st.info(
            "Red diamond markers identify months with low or no recorded "
            "order activity. These markers signal a need to investigate the "
            "source data; they do not prove the cause of the low revenue."
        )

        st.subheader("Month-over-Month Revenue Growth")
        growth_df = monthly_df.dropna(subset=["mom_growth_pct"]).copy()

        if not growth_df.empty:
            growth_chart = px.bar(
                growth_df,
                x="month",
                y="mom_growth_pct",
                title="Monthly Revenue Growth (%)",
                labels={"month": "Month", "mom_growth_pct": "Growth (%)"},
                color="mom_growth_pct",
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
            )
            st.plotly_chart(
                growth_chart,
                use_container_width=True,
                key="monthly_revenue_growth",
            )
        else:
            st.info(
                "No valid month-over-month growth comparisons are available."
            )

        eligible_months = monthly_df[
            (monthly_df["coverage_status"] == "Order activity recorded")
            & (monthly_df["total_revenue"] > 0)
        ].copy()

        st.subheader("Monthly Performance Highlights")
        if not eligible_months.empty:
            highest_month = eligible_months.loc[
                eligible_months["total_revenue"].idxmax()
            ]
            lowest_month = eligible_months.loc[
                eligible_months["total_revenue"].idxmin()
            ]

            col1, col2 = st.columns(2)
            col1.metric(
                "Highest-Revenue Month",
                highest_month["month"].strftime("%B %Y"),
                f"{highest_month['total_revenue']:,.2f}",
            )
            col2.metric(
                "Lowest-Revenue Month (Eligible Months)",
                lowest_month["month"].strftime("%B %Y"),
                f"{lowest_month['total_revenue']:,.2f}",
            )
        else:
            st.info(
                "No months currently meet the order-activity threshold."
            )

        st.subheader("Monthly Data Coverage Review")
        st.write(
            "Months with low recorded order counts are flagged for review. "
            "This identifies unusual data coverage; it does not prove why "
            "the counts are low or establish that the source is incomplete."
        )

        review_df = monthly_df[
            monthly_df["coverage_status"] != "Order activity recorded"
        ].copy()

        if not review_df.empty:
            st.warning(
                f"{len(review_df)} month(s) have low or no recorded order "
                "activity. Investigate these periods before interpreting "
                "their revenue as a business trend."
            )
            coverage_display = review_df[
                [
                    "month",
                    "order_count",
                    "latest_order_date",
                    "total_revenue",
                    "coverage_status",
                ]
            ].copy()
            coverage_display["month"] = coverage_display["month"].dt.strftime(
                "%Y-%m"
            )
            coverage_display["latest_order_date"] = (
                coverage_display["latest_order_date"].dt.strftime("%Y-%m-%d")
            )
            st.dataframe(
                coverage_display,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success(
                "No months were flagged by the current order-count threshold."
            )

        st.subheader("Monthly Revenue Details")
        display_df = monthly_df.copy()
        display_df["month"] = display_df["month"].dt.strftime("%Y-%m")
        display_df["latest_order_date"] = (
            display_df["latest_order_date"].dt.strftime("%Y-%m-%d")
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Month-over-month growth compares revenue with the previous month. "
            "Low-order-count months are excluded from growth highlights. "
            "Coverage-review flags identify unusual recorded order counts, "
            "not a confirmed explanation for those counts."
        )

except FileNotFoundError as error:
    st.error(str(error))
except (ValueError, KeyError) as error:
    st.error(f"Unable to display monthly analysis: {error}")
    # ============================================================
# MONTHLY DATA COVERAGE REVIEW
# ============================================================

st.divider()
st.header("Monthly Data Coverage Review")

st.write(
    "Identify months with unusually low recorded order counts "
    "that may require further investigation."
)

try:
    coverage_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "monthly_revenue_analysis.csv"
    )

    if not coverage_path.exists():
        st.error(
            "Monthly coverage file not found. "
            "Run: python src/monthly_analysis.py"
        )
    else:
        coverage_df = pd.read_csv(coverage_path)

        required_columns = {
            "month",
            "order_count",
            "latest_order_date",
            "coverage_status",
        }

        if not required_columns.issubset(coverage_df.columns):
            st.error(
                "The monthly CSV is missing required columns. "
                f"Available columns: {list(coverage_df.columns)}"
            )
        else:
            review_df = coverage_df[
                coverage_df["coverage_status"] == "Review required"
            ].copy()

            st.metric("Months Requiring Review", len(review_df))

            st.dataframe(
                review_df[
                    [
                        "month",
                        "order_count",
                        "latest_order_date",
                        "coverage_status",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Review flags indicate unusually low recorded order "
                "counts. They do not establish the cause or prove "
                "that the dataset is incomplete."
            )

except (FileNotFoundError, ValueError, KeyError) as error:
    st.error(f"Unable to display coverage review: {error}")

    
# ============================================================
# BUSINESS RECOMMENDATIONS
# ============================================================

st.divider()
st.header("Business Recommendations")

st.write(
    "Recommended actions based on the revenue, anomaly, "
    "payment-method, product-category, and data-coverage analyses."
)

with st.expander("1. Investigate unusual revenue days", expanded=True):
    st.write(
        "**Finding:** The anomaly detector flagged November 24, 2017 "
        "as a day with unusually high revenue."
    )
    st.write(
        "**Recommended action:** Review the orders and payment records "
        "for that date. Check for promotions, bulk purchases, duplicate "
        "records, or other unusual activity before deciding whether "
        "the result reflects genuine business performance."
    )

with st.expander("2. Review months with low recorded order counts"):
    st.write(
        "**Finding:** September 2016, December 2016, September 2018, "
        "and October 2018 were flagged for coverage review."
    )
    st.write(
        "**Recommended action:** Verify the source files, date coverage, "
        "and order ingestion for these months before using them for "
        "business comparisons or forecasting."
    )

with st.expander("3. Prioritize product-category analysis"):
    st.write(
        "**Finding:** Beauty and personal care (beleza_saude) had the "
        "highest item revenue in the category analysis."
    )
    st.write(
        "**Recommended action:** Review category-level sales trends, "
        "order volumes, and profit margins before making inventory "
        "or marketing decisions. Revenue alone does not establish "
        "profitability."
    )

with st.expander("4. Monitor payment-method performance"):
    st.write(
        "**Finding:** Credit cards accounted for the largest share "
        "of recorded payment revenue in the payment analysis."
    )
    st.write(
        "**Recommended action:** Track payment volume, revenue share, "
        "failed payments, and refunds by payment method. Investigate "
        "unusual patterns without assuming that a payment method "
        "caused a revenue anomaly."
    )

with st.expander("5. Improve data-quality monitoring"):
    st.write(
        "**Finding:** Some months have unusually low recorded order "
        "counts, and the revenue series contains zero-revenue days."
    )
    st.write(
        "**Recommended action:** Establish checks for missing dates, "
        "unexpected drops in order volume, duplicate records, and "
        "missing values. Confirm source-data completeness before "
        "treating these patterns as business changes."
    )

st.caption(
    "These recommendations are investigation priorities based on "
    "the available analysis. Validate the underlying records before "
    "taking operational action."
)

