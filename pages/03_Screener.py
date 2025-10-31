import streamlit as st
import pandas as pd
import yfinance as yf
import datetime as dt
from functions import *
from millify import millify

st.title('Screener')

# Load symbols with error handling
try:
    csv = pd.read_csv('symbols.csv')
    symbol = csv['Symbol'].tolist()
    for i in range(0, len(symbol)):
        symbol[i] = symbol[i] + ".NS"
except Exception as e:
    st.error(f"Error loading symbols: {e}")
    symbol = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]  # Default fallback

st.write("A stock screener is a set of tools that allow investors to quickly sort through various parameters of the stock")
ticker_input = st.selectbox('Enter or Choose stock', symbol)

start_input = dt.datetime.today() - dt.timedelta(120)
end_input = dt.datetime.today()

# Download data with error handling - USING THE WORKING APPROACH
try:
    stock_data = yf.Ticker(ticker_input)
    df = stock_data.history(start=start_input, end=end_input)
    
    if df.empty:
        st.error(f"No data available for {ticker_input} in the selected date range.")
        st.stop()
    
    # Reset index to get Date as column
    df = df.reset_index()
    df['symbols'] = ticker_input
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Ensure all required columns exist
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in df.columns:
            st.error(f"Missing required column: {col}")
            st.stop()
            
    # Add Adj Close if not present (use Close as fallback)
    if 'Adj Close' not in df.columns:
        df['Adj Close'] = df['Close']
        
except Exception as e:
    st.error(f"Error fetching data for {ticker_input}: {e}")
    st.stop()

stock = yf.Ticker(ticker_input)
info = stock.info or {}  # Handle None case

# Safe data extraction with None checks - USING THE WORKING APPROACH
try:
    closing_price = round((df['Close'].iloc[-1:]), 2)
    opening_price = round(df['Open'].iloc[-1:], 2).astype('str')
    sma_df = calc_moving_average(df, 12)
    sma_df_tail = round(sma_df['sma'].iloc[-1:].astype('int64'), 2)
    ema_df_tail = round(sma_df['ema'].iloc[-1:].astype('int64'), 2)

    macd_df = calc_macd(df)
    ema26_df_tail = round(macd_df['ema26'].iloc[-1:].astype('int64'), 2)
    macd_df_tail = round(macd_df['macd'].iloc[-1:].astype('int64'), 2)
    signal_df_tail = round(macd_df['signal'].iloc[-1:].astype('int64'), 2)

    rsi_df = RSI(df, 14)
    rsi_df_tail = round(rsi_df['RSI'].iloc[-1:].astype('int64'), 2)

    adx_df = ADX(df, 14)
    adx_df_tail = round(adx_df.iloc[-1:].astype('int64'), 2)

    breaking_out = is_breaking_out(df)
    consolidating = is_consolidating(df)
except Exception as e:
    st.error(f"Error calculating indicators: {e}")
    # Set default values
    closing_price = sma_df_tail = ema_df_tail = ema26_df_tail = macd_df_tail = signal_df_tail = rsi_df_tail = adx_df_tail = pd.Series([0])
    breaking_out = consolidating = "N/A"

# Helper function for millify with error handling
def safe_millify(value, precision=2):
    """Safely apply millify with error handling"""
    try:
        if value is None or value == "N/A":
            return "N/A"
        if hasattr(value, 'iloc'):
            value = value.iloc[0] if len(value) > 0 else value
        return millify(float(value), precision=precision)
    except:
        return "N/A"

# metrics at a glance with safe info access
with st.container():
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('52 Week Low', millify(info.get('fiftyTwoWeekLow', 0), 2))
    col2.metric('52 Week High', millify(info.get('fiftyTwoWeekHigh', 0), 2))

    col_1, col_2, col_3, col_4 = st.columns(4)
    col_1.metric('Market Day Low', millify(info.get('regularMarketDayLow', 0), 2))
    col_2.metric('Market Day High', millify(info.get('regularMarketDayHigh', 0), 2))

with st.container():
    co_1, co_2, co_3, co_4 = st.columns(4)
    co_1.metric('EBITDA Margin', info.get('ebitdaMargins', 'N/A'))
    co_2.metric('Profit Margin', info.get('profitMargins', 'N/A'))
    co_3.metric('Gross Margin', info.get('grossMargins', 'N/A'))
    co_4.metric('Operating Margin', info.get('operatingMargins', 'N/A'))

with st.container():
    co_11, co_22, co_33, co_44 = st.columns(4)
    co_11.metric('Current Ratio', info.get('currentRatio', 'N/A'))
    co_22.metric('Return on Assets', info.get('returnOnAssets', 'N/A'))
    co_33.metric('Debt to Equity', info.get('debtToEquity', 'N/A'))
    co_44.metric('Return on Equity', info.get('returnOnEquity', 'N/A'))

with st.container():
    c_1, c_2, c_3, c_4 = st.columns(4)
    c_1.metric('Closing Price', millify(float(closing_price.iloc[0]) if not closing_price.empty else 0, precision=2))
    c_2.metric('Simple Moving Average', millify(float(sma_df_tail.iloc[0]) if not sma_df_tail.empty else 0, precision=2))
    c_3.metric('Exponential Moving Average', millify(float(ema_df_tail.iloc[0]) if not ema_df_tail.empty else 0, precision=2))
    c_4.metric('Exponential Moving Average over period 26', millify(float(ema26_df_tail.iloc[0]) if not ema26_df_tail.empty else 0, 2))

with st.container():
    c_11, c_22, c_33, c_44 = st.columns(4)
    c_11.metric('Relative Strength Index', float(rsi_df_tail.iloc[0]) if not rsi_df_tail.empty else 0)
    c_22.metric('Average Directional Index', float(adx_df_tail.iloc[0]) if not adx_df_tail.empty else 0)
    c_33.metric('MACD', float(macd_df_tail.iloc[0]) if not macd_df_tail.empty else 0)
    c_44.metric('Signal', float(signal_df_tail.iloc[0]) if not signal_df_tail.empty else 0)

with st.container():
    cc_11, cc_22, cc_33, cc_44 = st.columns(4)
    cc_22.metric('Breaking Out??', breaking_out)
    cc_11.metric('Consolidating??', consolidating)
    cc_33.metric('50 Day Average', millify(info.get('fiftyDayAverage', 0), 2))
    cc_44.metric('Recommendation', str(info.get('recommendationKey', 'N/A')).upper())

# Additional error information for debugging
if st.checkbox("Show debug info"):
    st.subheader("Debug Information")
    st.write(f"DataFrame shape: {df.shape if df is not None else 'No data'}")
    st.write(f"Info keys: {list(info.keys()) if info else 'No info'}")