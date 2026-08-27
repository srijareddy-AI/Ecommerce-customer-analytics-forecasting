# E-Commerce Customer Analytics & Revenue Forecasting

A business intelligence and data science project combining customer segmentation,
sales forecasting, and executive-level reporting — built on real transaction data
from a UK-based online retailer.

## Business Problem

Businesses often treat all customers the same, missing the fact that a small
percentage typically drives a disproportionate share of revenue. Separately, teams
need forward-looking revenue estimates to plan inventory, staffing, and marketing
spend — not just historical reporting.

This project answers two questions any retail or e-commerce business asks:
1. **Which customers matter most, and how should we treat them differently?**
2. **What will our revenue look like over the next quarter?**

## Dataset

- **Source:** UCI Machine Learning Repository — Online Retail Dataset
- **Size:** 541,909 transaction line items from a UK-based online gift retailer,
  Dec 2010 – Dec 2011
- After cleaning (removing missing customer IDs, cancellations, and invalid
  quantity/price entries): **397,884 clean transactions**, **4,338 unique customers**,
  **£8.9M** in total revenue

## Approach

### 1. Data Cleaning
Removed transactions with missing customer identifiers (guest checkouts cannot be
segmented), excluded order cancellations, and filtered out invalid quantity/price
entries — reducing noise before any analysis began.

### 2. Customer Segmentation (RFM + K-Means)
Calculated **Recency, Frequency, and Monetary** value for every customer, then applied
K-Means clustering to group customers into four segments — **Champions, Loyal
Customers, Regular Customers, and At Risk/Lost** — based on their actual purchasing
behavior rather than arbitrary spend thresholds.

### 3. Revenue Forecasting (Prophet)
Aggregated transactions into daily revenue and trained a Prophet time-series model,
capturing weekly seasonality and overall growth trend, to forecast the next 90 days
with a 90% confidence interval — not a single point estimate.

### 4. Executive Dashboard (Streamlit)
Built an interactive dashboard presenting KPIs, segment breakdowns, the forecast
chart, and revenue-by-country — the kind of view a business stakeholder would
actually use to make decisions.

## Key Findings

- **~5% of customers (Champions + Loyal) generate nearly 48% of total revenue** —
  a clear signal for where retention efforts should be prioritized.
- Revenue shows a **clear weekly cycle and an upward trend** through 2011, both
  captured automatically by the forecasting model.
- The UK dominates revenue, but international markets show growth pockets worth
  further investigation.

## Tech Stack

Python, pandas, scikit-learn (K-Means, StandardScaler), Prophet, Streamlit,
Matplotlib

## Files

- `online_retail.csv` — raw source data
- `clean_retail.csv` — cleaned transaction-level data
- `rfm_segmented.csv` — customer-level RFM scores and segment assignments
- `daily_revenue.csv`, `forecast.csv` — time-series data and forecast output
- `prophet_model.pkl` — trained forecasting model
- `app.py` — interactive Streamlit dashboard
- `forecast_chart.png`, `segment_charts.png` — key visualizations

## Running the dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Author

Srija Reddy Annam — MS Artificial Intelligence, Faulkner University
[LinkedIn](https://linkedin.com/in/srijareddyannam) | [GitHub](https://github.com/srijareddy-AI)
