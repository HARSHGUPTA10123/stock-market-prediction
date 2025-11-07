import streamlit as st
import pandas as pd
import datetime as dt
from functions import *
import plotly.graph_objects as go
from patterns import candlestick_patterns
import ta
import numpy as np
from yahooquery import Ticker

st.set_page_config(page_title="Pattern Recognition", layout="wide")
st.title('Pattern Recognition')
st.write('A pattern is identified by connecting common price points across time. Technical analysts seek these to anticipate future price direction of stocks.')

# Load Symbols
try:
    csv = pd.read_csv('symbols.csv')
    symbol = [s + ".NS" for s in csv['Symbol'].tolist()]
except:
    symbol = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]

default_index = symbol.index('RELIANCE.NS') if 'RELIANCE.NS' in symbol else 0
ticker_input = st.selectbox('Choose NSE Stock', symbol, index=default_index)

# Data Fetcher (Clean timezone handling)
def get_yahooquery_data(symbol, start_date, end_date):
    try:
        stock = Ticker(symbol)
        start = start_date.strftime('%Y-%m-%d')
        end = end_date.strftime('%Y-%m-%d')

        df = stock.history(start=start, end=end)

        if df.empty:
            return pd.DataFrame()

        df = df.reset_index()
        df.rename(columns={
            'date': 'Date', 'open': 'Open', 'high': 'High',
            'low': 'Low', 'close': 'Close', 'volume': 'Volume',
            'adjclose': 'Adj Close'
        }, inplace=True)

        if 'Adj Close' not in df.columns:
            df['Adj Close'] = df['Close']

        # FIX TIMEZONE ALWAYS → tz-naive
        df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_convert(None)

        return df
    except:
        return pd.DataFrame()

# Price Section
show = st.radio("Show Price Chart?", ("Show", "Hide"))

if show == "Show":
    start_input = st.date_input("Start Date", dt.datetime.now() - dt.timedelta(days=120))
    end_input = st.date_input("End Date", dt.datetime.now())

    start_dt = dt.datetime.combine(start_input, dt.time())
    end_dt = dt.datetime.combine(end_input, dt.time())

    df_price = get_yahooquery_data(ticker_input, start_dt, end_dt)

    if not df_price.empty:
        chart = st.radio("Chart Style", ("Candlestick", "Line"))

        if chart == "Line":
            fig = go.Figure(data=[go.Scatter(
                x=df_price["Date"],
                y=df_price["Close"],
                mode="lines"
            )])
        else:
            fig = go.Figure(data=[go.Candlestick(
                x=df_price['Date'],
                open=df_price['Open'],
                high=df_price['High'],
                low=df_price['Low'],
                close=df_price['Close'],
                increasing_line_color='green',
                decreasing_line_color='red',
                increasing_fillcolor='green',
                decreasing_fillcolor='red'
            )])

        fig.update_layout(
            height=600,
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            xaxis_rangeslider_visible=False
        )

        st.plotly_chart(fig, use_container_width=True)


# Pattern Recognition
st.subheader("Candlestick Pattern Scan")

start = dt.datetime.now() - dt.timedelta(days=365)
end = dt.datetime.now()

df = get_yahooquery_data(ticker_input, start, end)

if df.empty:
    st.error("No data available for pattern recognition.")
    st.stop()

open_ = df['Open'].values
high = df['High'].values
low = df['Low'].values
close = df['Close'].values

signals = []
names = list(candlestick_patterns.values())

for pattern_func in candlestick_patterns.keys():
    try:
        result = getattr(ta, pattern_func)(open_, high, low, close)
        last = result[-1]
        if last > 0:
            signals.append("Bullish")
        elif last < 0:
            signals.append("Bearish")
        else:
            signals.append("Neutral")
    except:
        signals.append("Neutral")

signal_df = pd.DataFrame({"Pattern": names, "Signal": signals})

bullish = signal_df[signal_df['Signal'] == "Bullish"]
bearish = signal_df[signal_df['Signal'] == "Bearish"]
neutral = signal_df[signal_df['Signal'] == "Neutral"]

col1, col2, col3 = st.columns(3)
col1.metric("Bullish Signals", len(bullish))
col2.metric("Bearish Signals", len(bearish))
col3.metric("Neutral Signals", len(neutral))

def style(val):
    if val == "Bullish": return "background-color:#d4edda; color:#155724; font-weight:bold;"
    if val == "Bearish": return "background-color:#f8d7da; color:#721c24; font-weight:bold;"
    return "background-color:#e2e3e5; color:#383d41; font-weight:bold;"

st.dataframe(signal_df.style.map(style, subset=['Signal']), use_container_width=True)

st.write("### Bullish Patterns:")
for p in bullish["Pattern"]:
    st.write("🟢", p)

st.write("### Bearish Patterns:")
for p in bearish["Pattern"]:
    st.write("🔻", p)
