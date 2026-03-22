# Stock Anomaly Detector

A machine learning web app that detects unusual price and volume behaviour in NSE-listed Indian stocks using Isolation Forest.

## Live Demo
[Click here to open the app](https://stock-anomaly-detector-azed3b9shjzwceqlbykqhd.streamlit.app/)

## What it does

Most investors don't have time to watch their stocks every day. This app automatically scans any NSE-listed stock and flags the days where something unusual happened — unusually large price moves, abnormal trading volume, or extreme intraday swings. These anomalies almost always have a real news event behind them.

## How it works

1. Downloads up to 3 years of historical OHLCV data for any NSE stock using yfinance
2. Engineers three signals from the raw data:
   - **Price change %** — how much did the stock move that day
   - **Volume ratio** — today's volume divided by 20-day rolling average
   - **Day range %** — intraday high-low spread as a percentage of close
3. Scales all features using StandardScaler so no single feature dominates
4. Runs Isolation Forest — an unsupervised ML algorithm that finds days that look nothing like normal days
5. Displays results as an interactive chart with anomaly dates highlighted in red

## Key Finding — Validation on Reliance Industries (2022–2024)

| Date | Price Change | Volume Ratio | What Actually Happened |
|------|-------------|--------------|----------------------|
| 2022-07-01 | -7.20% | 4.34x | Govt windfall tax on fuel exports — biggest drop in 18 months |
| 2022-05-31 | -1.21% | 3.58x | Toy manufacturing JV announcement + SEBI filing |
| 2024-08-29 | +1.51% | 3.52x | AGM — bonus shares 1:1 + Jio Brain AI launch |
| 2023-09-20 | -2.23% | 3.44x | Block deal + post-AGM uncertainty on Jio/Retail IPO |
| 2024-08-30 | -0.74% | 3.35x | Mixed analyst reaction to AGM announcements |

**Key insight:** The model catches both negative events (crashes) and positive events (announcements) — because it finds unusual activity, not just bad news.

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| yfinance | Downloads NSE stock data from Yahoo Finance |
| pandas | Data manipulation and feature engineering |
| scikit-learn | Isolation Forest + StandardScaler |
| matplotlib | Charts and visualisations |
| Streamlit | Web app framework |

## How to run locally

1. Clone this repository
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Run the app:
```
streamlit run app.py
```
4. Open your browser at `http://localhost:8501`

## How to use

- Type any NSE ticker in the sidebar — for example `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`
- Adjust the date range and sensitivity slider
- Click **Run Analysis**
- Take any flagged date and Google `[Stock name] [date] news` to find what caused it

## What I learned

- Feature engineering matters more than the algorithm — the volume ratio was the most powerful signal
- Unsupervised ML can find events it was never told about — the model flagged the July 2022 windfall tax shock with zero knowledge of the news
- Anomaly detection catches both good and bad surprises equally

## Author

Built by [Adarsh Sahu]
