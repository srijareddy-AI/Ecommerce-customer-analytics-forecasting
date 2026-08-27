import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

st.set_page_config(page_title="E-Commerce Customer Analytics", layout="wide")

@st.cache_data
def load_data():
    rfm = pd.read_csv('rfm_segmented.csv')
    daily_revenue = pd.read_csv('daily_revenue.csv')
    daily_revenue['ds'] = pd.to_datetime(daily_revenue['ds'])
    forecast = pd.read_csv('forecast.csv')
    forecast['ds'] = pd.to_datetime(forecast['ds'])
    df = pd.read_csv('clean_retail.csv')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return rfm, daily_revenue, forecast, df

rfm, daily_revenue, forecast, df = load_data()

st.title("📊 E-Commerce Customer Analytics & Revenue Forecasting")
st.markdown("""
Business intelligence dashboard built on **397,884 real transactions** from a UK-based
online retailer (Dec 2010 – Dec 2011). Combines customer segmentation (RFM + K-Means),
90-day revenue forecasting (Prophet), and executive-level KPIs.
""")

# --- KPI ROW ---
st.divider()
total_revenue = df['Revenue'].sum()
total_customers = df['CustomerID'].nunique()
avg_order_value = df.groupby('InvoiceNo')['Revenue'].sum().mean()
total_orders = df['InvoiceNo'].nunique()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Revenue", f"£{total_revenue:,.0f}")
k2.metric("Total Customers", f"{total_customers:,}")
k3.metric("Total Orders", f"{total_orders:,}")
k4.metric("Avg Order Value", f"£{avg_order_value:,.2f}")

# --- CUSTOMER SEGMENTATION ---
st.divider()
st.header("Customer Segmentation (RFM Analysis)")
st.markdown("""
Customers are grouped using **Recency, Frequency, and Monetary (RFM)** analysis with
K-Means clustering — a widely used technique for identifying which customers matter most.
""")

col1, col2 = st.columns([1, 1])
with col1:
    segment_revenue = rfm.groupby('Segment')['Monetary'].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = ['#27AE60', '#2980B9', '#F39C12', '#C0392B']
    ax.bar(segment_revenue.index, segment_revenue.values, color=colors)
    ax.set_title('Revenue by Segment')
    ax.set_ylabel('Revenue (£)')
    plt.xticks(rotation=20)
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    segment_summary = rfm.groupby('Segment').agg(
        Customers=('CustomerID', 'count'),
        Avg_Recency_Days=('Recency', 'mean'),
        Avg_Orders=('Frequency', 'mean'),
        Avg_Spend=('Monetary', 'mean')
    ).round(1).sort_values('Avg_Spend', ascending=False)
    st.dataframe(segment_summary, use_container_width=True)

top_pct = rfm[rfm['Segment'].isin(['Champions', 'Loyal Customers'])].shape[0] / rfm.shape[0] * 100
top_rev_pct = rfm[rfm['Segment'].isin(['Champions', 'Loyal Customers'])]['Monetary'].sum() / rfm['Monetary'].sum() * 100
st.info(f"💡 **Key Insight:** {top_pct:.1f}% of customers (Champions + Loyal) generate {top_rev_pct:.1f}% of total revenue.")

# --- FORECASTING ---
st.divider()
st.header("90-Day Revenue Forecast")
st.markdown("Built with **Prophet**, capturing weekly seasonality and overall growth trend.")

fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(daily_revenue['ds'], daily_revenue['y'], 'o', markersize=2, color='#2C3E50', label='Actual Revenue')
ax2.plot(forecast['ds'], forecast['yhat'], color='#E74C3C', linewidth=2, label='Forecast')
ax2.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='#E74C3C', alpha=0.15, label='90% Confidence Range')
ax2.set_ylabel('Revenue (£)')
ax2.legend()
plt.tight_layout()
st.pyplot(fig2)

next_30 = forecast.tail(90).head(30)['yhat'].sum()
st.info(f"💡 **Forecast Insight:** Projected revenue for the next 30 days: **£{next_30:,.0f}**")

# --- COUNTRY BREAKDOWN ---
st.divider()
st.header("Revenue by Country (Top 10)")
country_revenue = df.groupby('Country')['Revenue'].sum().sort_values(ascending=False).head(10)
fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.barh(country_revenue.index[::-1], country_revenue.values[::-1], color='#2980B9')
ax3.set_xlabel('Revenue (£)')
plt.tight_layout()
st.pyplot(fig3)

st.caption("Built by Srija Reddy Annam | Tools: Python, pandas, scikit-learn, Prophet, Streamlit | Dataset: UCI Online Retail")
