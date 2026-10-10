# 🛡️ SentinelIQ

### Which days did this store's revenue behave strangely, and is it a real signal or just missing data?

SentinelIQ is a revenue-monitoring and analytics project I built using the Brazilian Olist e-commerce dataset. It brings together daily and monthly revenue analysis, anomaly detection, payment-method comparisons, product-category performance, and data-coverage checks in one interactive Streamlit dashboard.

The idea behind it is pretty simple: when revenue suddenly spikes or drops, how do you know whether it's a genuine change in business activity, an unusual transaction pattern, or a problem with the data itself?

I wanted to explore that question instead of stopping at the usual revenue charts. A sudden increase might look great on a dashboard, and a sharp decline might look worrying, but neither tells the whole story on its own. Before drawing conclusions, it's worth checking what happened around that date and whether the underlying data is reliable enough to support the conclusion.

That's what SentinelIQ tries to do. It identifies patterns worth investigating, puts them into business context, and makes it easier to decide what to check next.

---

## Why I built it

Revenue dashboards are useful for seeing what happened. But I was interested in a slightly different question: **what should someone investigate when the numbers don't look normal?**

For example, a day with unusually high revenue could be linked to a promotion, a few large orders, or a genuine change in customer activity. It could also be something that needs closer inspection in the transaction records. Similarly, a month with very little recorded revenue shouldn't automatically be treated as a business slowdown if the underlying data might not cover the full period.

I wanted to bring these questions into the same workflow rather than analyze revenue, anomalies, and data quality separately.

SentinelIQ therefore combines descriptive analysis with anomaly detection and a basic coverage review. The goal isn't to make automatic business decisions. It's to help identify where a closer look might be useful.

---

## What SentinelIQ does

### 1. Watches daily revenue

The project uses order payment records to calculate daily revenue and examine how it changes over time. This creates the baseline for the rest of the analysis.

Looking at the daily series helps reveal ordinary fluctuations as well as days that stand out from the surrounding pattern. It also gives the anomaly detector a consistent set of daily observations to evaluate.

The current processed revenue series contains **774 daily observations**, covering the available dates from September 2016 to October 2018.

### 2. Flags unusual revenue days

I used **Isolation Forest**, an unsupervised machine-learning algorithm, to flag days whose revenue patterns differ from the rest of the series.

In the current analysis, **24 out of 774 days were flagged**. Those dates are highlighted in the dashboard so they can be examined alongside the revenue trend.

The important part is what happens after a day gets flagged. The model identifies an unusual observation; it doesn't explain the reason behind it. A flagged date could reflect a real business event, a concentration of transactions, or something in the data that deserves checking.

For that reason, SentinelIQ treats anomaly flags as investigation prompts rather than confirmed incidents.

### 3. Compares payment methods

The payment analysis looks at how recorded revenue is distributed across payment methods, including credit cards, boleto, debit cards, and vouchers.

This helps answer questions such as:

- Which payment methods contribute the largest share of recorded revenue?
- How does revenue distribution differ between payment methods?
- Which payment methods appear in transactions associated with flagged dates?

The last question needs some care. A transaction occurring on a flagged date does not mean that its payment method caused the anomaly. This analysis is intended to show patterns worth examining, not to establish causation.

### 4. Breaks down product categories

Revenue can look healthy overall while performance differs considerably between product categories. The category analysis breaks down recorded item revenue to make those differences easier to explore.

It helps identify categories that contribute substantially to revenue and provides a starting point for further questions about sales patterns, order volumes, and category performance.

Revenue alone doesn't tell us which categories are the most profitable, since costs and margins are not captured by this comparison. The results are therefore useful for understanding sales contribution, not for making profitability claims on their own.

### 5. Tracks the monthly story

Daily numbers are useful for spotting short-term changes, but monthly analysis gives a broader view of how the business changes over time.

SentinelIQ summarizes monthly revenue and examines month-over-month changes. The dashboard makes it easier to see which months grew, which declined, and where a change might deserve a closer look.

Monthly comparisons also need context. A percentage change can be misleading when the earlier month has very little recorded activity, so I included a separate coverage review rather than treating every large increase or decrease as a meaningful business signal.

### 6. Questions the data before trusting the trend

This is one of the parts I wanted to include from the beginning.

Some months in the dataset contain far fewer recorded orders than others. In the current analysis, September 2016, December 2016, September 2018, and October 2018 were flagged for coverage review because their recorded order counts were unusually low relative to the chosen review threshold.

That doesn't automatically mean the data is incomplete. Low counts could have more than one explanation, and the available analysis alone cannot establish the cause. The point is to make these months visible so that their source records and date coverage can be checked before someone interprets them as genuine changes in business activity.

In other words, SentinelIQ doesn't just ask whether the revenue looks unusual. It also asks whether the data is suitable for the conclusion being drawn from it.

### 7. Turns findings into investigation priorities

The final section of the dashboard brings the analytical results into a business context. It suggests practical next steps, such as reviewing unusual revenue dates, checking months with low recorded order counts, examining category-level performance, and monitoring payment-method trends.

These are investigation priorities, not automatic decisions. The underlying records and relevant business context should be checked before taking action.

---

## A few results from the current analysis

| Metric | Current result |
|---|---:|
| Daily revenue observations | 774 |
| Days flagged by Isolation Forest | 24 |
| Share of days flagged | Approximately 3.1% |
| Total recorded payment revenue | 16,008,872.12 |
| Months flagged for coverage review | 4 |

The total is based on the recorded payment values used in the analysis. It should not be interpreted as profit, and the anomaly count represents model flags rather than confirmed fraud or business incidents.

These figures describe the current dataset and processing setup. They are not intended to represent the performance of Olist as a whole.

---

## Built with

- **Python** — analysis scripts and data-processing workflow
- **Pandas** — loading, joining, cleaning, grouping, and aggregating the data
- **NumPy** — numerical operations
- **Scikit-learn** — Isolation Forest anomaly detection
- **Matplotlib** — visual analysis
- **Streamlit** — interactive dashboard

---

## How the project is organized

```text
SentinelIQ/
├── data/
│   ├── raw/                  # Original Olist CSV files
│   └── processed/            # Analysis outputs used by the dashboard
├── src/
│   ├── revenue_analysis.py   # Daily revenue calculation
│   ├── anomaly_detection.py  # Isolation Forest anomaly flags
│   ├── product_analysis.py   # Product-category analysis
│   ├── monthly_analysis.py   # Monthly trends and coverage review
│   └── dashboard.py          # Interactive Streamlit dashboard
├── main.py
├── requirements.txt
└── README.md
```

The analysis scripts generate processed data files, which the dashboard uses to display the results. Keeping the calculations separate from the dashboard makes the workflow easier to follow and makes individual analysis steps easier to inspect.

---

## The dataset

I used the [Brazilian E-Commerce Public Dataset by Olist on Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

The dataset contains information about Brazilian e-commerce orders, payments, products, and other parts of the order lifecycle. I used the relevant tables to explore recorded payment revenue, payment-method distribution, product-category performance, and monthly order activity.

The analysis combines related records using their available identifiers, such as order IDs, where appropriate. The daily revenue series is based on recorded payment values associated with orders.

**Dataset note:** The results depend on the records and date coverage present in the downloaded dataset. The analysis does not assume that every unusual pattern represents a real-world business event.

---

## Run it locally

If you'd like to explore the dashboard yourself, follow these steps.

### 1. Clone the repository

After publishing the project to GitHub, replace the placeholder URL with your repository URL.

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd SentinelIQ
```

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks environment activation, use the Python executable inside `.venv` directly or resolve the local execution-policy issue before continuing.

### 3. Install the dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Generate the processed analysis files

Run the scripts from the project root:

```bash
python src/revenue_analysis.py
python src/anomaly_detection.py
python src/product_analysis.py
python src/monthly_analysis.py
```

These commands reflect the intended analysis workflow. The scripts should be run in the order required by their data dependencies, and the expected input CSV files must be present under `data/raw/`.

### 5. Start the dashboard

```bash
python -m streamlit run src/dashboard.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to explore the dashboard.

---

## What I learned from building it

One of the main things this project reinforced for me is that finding an interesting pattern is only the beginning of an analysis.

A model can flag an unusual date, but that result becomes much more useful when it's considered alongside the original revenue trend and the quality of the underlying data. Similarly, a monthly decline might look significant until you realize that the month has very few recorded orders and needs further checking.

I also wanted the results to lead somewhere practical. Instead of presenting charts without context, the dashboard connects the analysis to questions someone might investigate next. That makes the project closer to a small decision-support tool than a collection of separate plots.

There are still limitations, but I think being clear about them is part of making an analysis useful. The dashboard should help someone ask better questions, not give them a false sense of certainty.

---

## Limitations and things to keep in mind

- **Anomaly flags are not proof of fraud.** Isolation Forest identifies unusual observations but does not explain why they occurred.
- **Revenue is not profit.** The analysis uses recorded payment values and does not account for all business costs.
- **Association is not causation.** Payment-method patterns around flagged dates do not show that a payment method caused an anomaly.
- **Low order counts need investigation.** A month flagged for coverage review is not automatically incomplete; the source records need to be checked.
- **The analysis depends on the available data.** Missing records, unusual source-data patterns, and processing choices can affect the results.
- **Recommendations require context.** Operational decisions should be based on verified records and additional business information.

---

## What I'd like to improve next

- [ ] Add automated checks for missing dates, duplicate records, and unexpected changes in order volume.
- [ ] Compare Isolation Forest with alternative anomaly-detection approaches.
- [ ] Make the anomaly-detection settings configurable from the dashboard.
- [ ] Add more systematic evaluation of the anomaly results.
- [ ] Improve the documentation of the data-processing pipeline.
- [ ] Deploy the dashboard so it can be explored without running it locally.

---

## Project summary

SentinelIQ started with a practical question: when revenue behaves strangely, how can we tell whether it's a meaningful business signal or something that needs a data-quality check first?

By combining revenue analysis, anomaly detection, payment-method comparisons, product-category analysis, monthly trends, and coverage review, the project brings those questions into one place.

It's a work in progress, but the goal is clear: **make unusual patterns easier to find, easier to investigate, and harder to misinterpret.**

*SentinelIQ · Data Analytics · Anomaly Detection · Business Intelligence*