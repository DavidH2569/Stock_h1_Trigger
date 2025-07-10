import streamlit as st
import yfinance as yf
import pandas as pd
# from awesome_oscillator import calculate_ao

st.set_page_config(page_title="H1 EMA20 Triggers (D1 AO < 0)", layout="wide")
st.title("H1 EMA20 Cross-Up Triggers for Daily AO < 0")

# -- PARAMETERS -------------------------------------------------------------
# TICKERS = [
#     "NVDA", "MSFT", "AAPL", "AMZN", "GOOG", "META", "AVGO", "TSLA", "JPM", "WMT",
    # … add the rest of your list here …
# ]

TICKERS = ["NVDA", "MSFT", "AAPL", "AMZN", "GOOG", "META", "AVGO", "TSLA", "JPM", "WMT",
           "LLY", "V", "ORCL", "NFLX", "MA", "XOM", "COST", "JNJ", "PG", "HD",
           "BAC", "ABBV", "PLTR", "KO", "PM", "UNH", "CSCO", "IBM", "WFC", "CVX",
           "GE", "TMUS", "CRM", "ABT", "MS", "AMD", "AXP", "LIN", "DIS", "INTU",
           "GS", "NOW", "MRK", "MCD", "T", "UBER", "TXN", "RTX", "BX", "CAT",
           "ISRG", "ACN", "BKNG", "PEP", "VZ", "QCOM", "BA", "SCHW", "BLK", "ADBE",
           "SPGI", "C", "AMGN", "TMO", "AMAT", "HON", "BSX", "NEE", "SYK", "PGR",
           "GEV", "PFE", "DHR", "UNP", "ETN", "GILD", "COF", "TJX", "MU", "DE",
           "PANW", "CMCSA", "ANET", "LRCX", "CRWD", "LOW", "ADP", "KKR", "KLAC", "ADI",
           "VRTX", "COP", "APH", "MDT", "CB", "NKE", "SBUX", "LMT", "MMC", "ICE",
           ]
DAYS_LOOKBACK = 90
def calculate_ao(median_df: pd.DataFrame) -> pd.DataFrame:
    """
    Awesome Oscillator: SMA5(median_price) - SMA34(median_price)
    median_df should be a DataFrame where each column is (High+Low)/2 for a ticker.
    """
    sma5  = median_df.rolling(window=5,  min_periods=5).mean()
    sma34 = median_df.rolling(window=34, min_periods=34).mean()
    return sma5 - sma34

# -- STEP 1: FETCH DAILY DATA & AO ------------------------------------------
@st.cache_data(ttl=3600)
def fetch_daily_ao(tickers, days):
    # download daily OHLC for lookback + a bit extra
    df = yf.download(tickers, period=f"{days}d", interval="1d", progress=False, auto_adjust=False)
    # get the median price series for AO (High+Low)/2
    if isinstance(df.columns, pd.MultiIndex):
        median = (df['High'] + df['Low']) / 2
    else:
        # single‐ticker fallback
        median = pd.DataFrame({tickers[0]: (df['High'] + df['Low'])/2})
    ao = calculate_ao(median)
    return ao

daily_ao = fetch_daily_ao(TICKERS, DAYS_LOOKBACK)
# pick tickers whose most-recent AO is < 0
negative_ao_tickers = [
    t for t in TICKERS
    if daily_ao[t].iloc[-1] < 0
]

st.write(f"Tickers with latest Daily AO < 0 (of {len(TICKERS)}): {', '.join(negative_ao_tickers)}")

# -- STEP 2: FOR EACH, FETCH H1 & FIND CROSS‐UPS ----------------------------
@st.cache_data(ttl=1800)
def find_h1_triggers(tickers, days, daily_ao):
    triggers = []
    for t in tickers:
        # Download 1h bars for this ticker
        h1 = yf.download(
            t,
            period=f"{days}d",
            interval="1h",
            progress=False,
            auto_adjust=False
        )
        # If the DataFrame has a MultiIndex on columns, flatten it
        if isinstance(h1.columns, pd.MultiIndex):
            h1.columns = h1.columns.get_level_values(0)

        # Skip if we don’t have enough data
        if h1.empty or len(h1) < 21:
            continue

        # Compute EMA20 on the Close price
        h1['EMA20']      = h1['Close'].ewm(span=20, adjust=False).mean()
        h1['prev_close'] = h1['Close'].shift(1)
        h1['prev_ema20'] = h1['EMA20'].shift(1)

        # Identify the cross-up: prior bar was below EMA20, current bar above
        cross_up = (
            (h1['prev_close']  < h1['prev_ema20']) &
            (h1['Close']       > h1['EMA20'])
        )

        # For each trigger, check the daily AO on that date
        for idx in h1.index[cross_up]:
            # Make sure we compare by calendar date
            dt = idx.tz_localize(None)
            date_str = dt.date().isoformat()
            if date_str in daily_ao.index and daily_ao.at[date_str, t] < 0:
                triggers.append({
                    'Date':   dt.date(),
                    'Time':   dt.time(),
                    'Ticker': t,
                    'Price':  round(h1.at[idx, 'Close'], 4)
                })

    return pd.DataFrame(triggers)


df_triggers = find_h1_triggers(negative_ao_tickers, DAYS_LOOKBACK, daily_ao)

# -- STEP 3: SHOW RESULTS ---------------------------------------------------
if df_triggers.empty:
    st.info("No H1 EMA20 cross-up triggers found in the last 90 days for tickers with daily AO < 0.")
else:
    st.subheader("H1 EMA20 Cross-Up Triggers (Daily AO < 0)")
    st.dataframe(df_triggers.sort_values(['Date', 'Time', 'Ticker']).reset_index(drop=True))
