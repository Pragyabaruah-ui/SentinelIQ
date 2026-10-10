
from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_data():
    """Load product and order-item datasets."""
    products_path = RAW_DIR / "olist_products_dataset.csv"
    items_path = RAW_DIR / "olist_order_items_dataset.csv"

    if not products_path.exists():
        raise FileNotFoundError(f"Product dataset not found: {products_path}")

    if not items_path.exists():
        raise FileNotFoundError(f"Order-items dataset not found: {items_path}")

    products = pd.read_csv(products_path)
    items = pd.read_csv(items_path)

    return products, items


def analyze_product_categories():
    """Calculate revenue and sales metrics for each product category."""
    products, items = load_data()

    # Keep only the product fields required for analysis.
    products = products[["product_id", "product_category_name"]].copy()

    # Replace missing category names without dropping products.
    products["product_category_name"] = (
        products["product_category_name"].fillna("unknown")
    )

    # Connect each order item to its product category.
    merged = items.merge(
        products,
        on="product_id",
        how="left",
        validate="many_to_one",
    )

    # Handle any unmatched product IDs defensively.
    merged["product_category_name"] = (
        merged["product_category_name"].fillna("unknown")
    )

    # Calculate category-level business metrics.
    category_summary = (
        merged.groupby("product_category_name", as_index=False)
        .agg(
            total_items=("order_item_id", "count"),
            unique_products=("product_id", "nunique"),
            total_item_revenue=("price", "sum"),
            average_item_price=("price", "mean"),
            total_freight=("freight_value", "sum"),
        )
    )

    total_revenue = category_summary["total_item_revenue"].sum()

    if total_revenue > 0:
        category_summary["revenue_share_pct"] = (
            category_summary["total_item_revenue"] / total_revenue * 100
        )
    else:
        category_summary["revenue_share_pct"] = 0.0

    # Rank categories by item revenue.
    category_summary = category_summary.sort_values(
        "total_item_revenue",
        ascending=False,
    ).reset_index(drop=True)

    category_summary.insert(
        0,
        "revenue_rank",
        range(1, len(category_summary) + 1),
    )

    # Round monetary and percentage metrics for readability.
    numeric_columns = [
        "total_item_revenue",
        "average_item_price",
        "total_freight",
        "revenue_share_pct",
    ]

    category_summary[numeric_columns] = category_summary[numeric_columns].round(2)

    return category_summary


def save_product_analysis(category_summary):
    """Save the category analysis as a CSV file."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DIR / "product_category_analysis.csv"
    category_summary.to_csv(output_path, index=False)

    print(f"Product category analysis saved to: {output_path}")


def main():
    category_summary = analyze_product_categories()
    save_product_analysis(category_summary)

    print("\nDataset summary")
    print("------------------------------")
    print(f"Categories analyzed: {len(category_summary)}")
    print(f"Total items analyzed: {category_summary['total_items'].sum():,}")
    print(
        "Total item revenue: "
        f"{category_summary['total_item_revenue'].sum():,.2f}"
    )

    print("\nTop 10 categories by item revenue")
    print("---------------------------------")
    print(
        category_summary[
            [
                "revenue_rank",
                "product_category_name",
                "total_items",
                "total_item_revenue",
                "revenue_share_pct",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
