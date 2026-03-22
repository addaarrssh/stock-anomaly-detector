

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Stock Anomaly Detector",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Anomaly Detector")
st.markdown("Detects unusual price and volume behaviour in NSE-listed Indian stocks using Machine Learning.")

st.sidebar.header("Settings")

ticker = st.sidebar.text_input(
    label="NSE Ticker Symbol",
    value="RELIANCE.NS",         # default value
    help="Add .NS at the end for NSE stocks. Example: TCS.NS, INFY.NS, HDFCBANK.NS"
)

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
end_date   = st.sidebar.date_input("End Date",   value=pd.to_datetime("2024-12-31"))

contamination = st.sidebar.slider(
    label="Sensitivity (% of days flagged as anomalies)",
    min_value=0.01,
    max_value=0.05,
    value=0.02,
    step=0.01,
    help="Higher = more anomalies flagged. Lower = only the most extreme days."
)

run_button = st.sidebar.button("Run Analysis", type="primary")


if run_button:

    with st.spinner(f"Downloading data for {ticker}..."):
        df = yf.download(ticker,
                         start=start_date,
                         end=end_date,
                         auto_adjust=True)

    if df.empty:
        st.error("No data found. Check the ticker symbol and try again.")
        st.stop()

    df.columns = df.columns.get_level_values(0)

    df['price_change'] = df['Close'].pct_change() * 100
    df['vol_avg_20']   = df['Volume'].rolling(20).mean()
    df['vol_ratio']    = df['Volume'] / df['vol_avg_20']
    df['day_range']    = (df['High'] - df['Low']) / df['Close'] * 100
    df = df.dropna()

    features = df[['price_change', 'vol_ratio', 'day_range']]
    scaler   = StandardScaler()
    X        = scaler.fit_transform(features)

    model        = IsolationForest(contamination=contamination, random_state=42)
    df['anomaly'] = model.fit_predict(X)

    anomalies = df[df['anomaly'] == -1]

    # ---- STEP 4: Show summary numbers at the top ----
    col1, col2, col3 = st.columns(3)

    col1.metric(
        label="Total Days Analyzed",
        value=len(df)
    )
    col2.metric(
        label="Anomaly Days Found",
        value=len(anomalies)
    )
    col3.metric(
        label="Biggest Single Drop",
        value=f"{anomalies['price_change'].min():.2f}%"
    )

    st.subheader("Price Chart with Anomalies Highlighted")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax1.plot(df.index, df['Close'], color='steelblue', linewidth=1, label='Close Price')
    ax1.scatter(anomalies.index, anomalies['Close'],
                color='red', zorder=5, s=60, label='Anomaly')
    ax1.set_ylabel('Price (₹)')
    ax1.legend()
    ax1.set_title(f'{ticker} — Anomaly Detection')

    ax2.bar(df.index, df['Volume'], color='lightgray', width=1, label='Normal volume')
    ax2.bar(anomalies.index, anomalies['Volume'],
            color='red', width=1, label='Anomaly volume')
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Date')
    ax2.legend()

    plt.tight_layout()

    st.pyplot(fig)

    st.subheader("Anomaly Days — Detailed Breakdown")

    display_df = anomalies[['Close', 'price_change', 'vol_ratio', 'day_range']].copy()

    display_df.columns = ['Close Price (₹)', 'Price Change %', 'Volume Ratio', 'Day Range %']

    display_df = display_df.round(2)

    display_df = display_df.sort_values('Volume Ratio', ascending=False)

    st.dataframe(display_df, use_container_width=True)

    st.info(
        "Tip: Take any flagged date and Google '[Stock name] [date] news' "
        "to find the real world event that caused the anomaly. "
        "Volume spikes almost always have a story behind them."
    )
