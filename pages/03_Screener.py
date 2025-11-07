import streamlit as st
import pandas as pd
import datetime as dt
from functions import *
from millify import millify
from yahooquery import Ticker
import pytz

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

# Force naive datetime (no timezone) for all date inputs
start_input = (pd.Timestamp.utcnow() - pd.Timedelta(days=120)).tz_localize(None)
end_input = pd.Timestamp.utcnow().tz_localize(None)

# YahooQuery data fetching function with robust timezone handling
def get_yahooquery_data(symbol, start_date, end_date):
    """Get historical data using YahooQuery"""
    try:
        stock = Ticker(symbol)
        
        # Convert to string format that YahooQuery expects (YYYY-MM-DD)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        hist = stock.history(start=start_str, end=end_str)
        
        if not hist.empty:
            hist = hist.reset_index()
            hist = hist.rename(columns={
                'date': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
                'adjclose': 'Adj Close'
            })
            # Ensure required columns exist
            if 'Adj Close' not in hist.columns:
                hist['Adj Close'] = hist['Close']
            hist['symbols'] = symbol
            
            # Robust timezone handling
            hist['Date'] = pd.to_datetime(hist['Date'])
            
            # If timezone exists, convert to UTC then remove timezone
            if hist['Date'].dt.tz is not None:
                hist['Date'] = hist['Date'].dt.tz_convert('UTC').dt.tz_localize(None)
            else:
                hist['Date'] = hist['Date'].dt.tz_localize(None)
            
            return hist
    except Exception as e:
        st.error(f"YahooQuery error for {symbol}: {e}")
    return pd.DataFrame()

# Download data with error handling - USING YAHOOQUERY
try:
    df = get_yahooquery_data(ticker_input, start_input, end_input)
    
    if df.empty:
        # Try fallback period if date range fails
        stock = Ticker(ticker_input)
        hist = stock.history(period="6mo")
        if not hist.empty:
            hist = hist.reset_index()
            hist = hist.rename(columns={
                'date': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
                'adjclose': 'Adj Close'
            })
            if 'Adj Close' not in hist.columns:
                hist['Adj Close'] = hist['Close']
            hist['symbols'] = ticker_input
            
            # Apply same timezone handling to fallback data
            hist['Date'] = pd.to_datetime(hist['Date'])
            if hist['Date'].dt.tz is not None:
                hist['Date'] = hist['Date'].dt.tz_convert('UTC').dt.tz_localize(None)
            else:
                hist['Date'] = hist['Date'].dt.tz_localize(None)
                
            df = hist
        else:
            st.error(f"No data available for {ticker_input} in the selected date range.")
            st.stop()
    
    # Ensure all required columns exist
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in df.columns:
            st.error(f"Missing required column: {col}")
            st.stop()
            
except Exception as e:
    st.error(f"Error fetching data for {ticker_input}: {e}")
    st.stop()

# Get stock info using YahooQuery
def get_yahooquery_info(symbol):
    """Get stock info using YahooQuery"""
    try:
        stock = Ticker(symbol)
        info = {}
        
        # Get summary details
        summary = stock.summary_detail
        if summary and symbol in summary:
            summary_data = summary[symbol]
            if isinstance(summary_data, dict):
                for key, value in summary_data.items():
                    info[key] = value
            
        # Get company profile  
        profile = stock.asset_profile
        if profile and symbol in profile:
            profile_data = profile[symbol]
            if isinstance(profile_data, dict):
                for key, value in profile_data.items():
                    info[key] = value
            
        # Get price info
        price = stock.price
        if price and symbol in price:
            price_data = price[symbol]
            if isinstance(price_data, dict):
                for key, value in price_data.items():
                    info[key] = value
            
        # Get financial data
        try:
            key_stats = stock.key_stats
            if key_stats and symbol in key_stats:
                key_stats_data = key_stats[symbol]
                if isinstance(key_stats_data, dict):
                    for key, value in key_stats_data.items():
                        info[key] = value
        except:
            pass
            
        return info
    except Exception as e:
        st.error(f"Error fetching info for {symbol}: {e}")
        return {}

info = get_yahooquery_info(ticker_input)

# Safe data extraction with None checks and timezone verification
try:
    # Verify dataframe has no timezone info
    if df['Date'].dt.tz is not None:
        df['Date'] = df['Date'].dt.tz_convert('UTC').dt.tz_localize(None)
    
    closing_price = round((df['Close'].iloc[-1:]), 2)
    opening_price = round(df['Open'].iloc[-1:], 2).astype('str')
    
    # Import and call functions with error handling for timezone issues
    sma_df = calc_moving_average(df, 12)
    # Ensure returned dataframe also has no timezone issues
    if 'Date' in sma_df.columns and sma_df['Date'].dt.tz is not None:
        sma_df['Date'] = sma_df['Date'].dt.tz_convert('UTC').dt.tz_localize(None)
    
    sma_df_tail = round(sma_df['sma'].iloc[-1:].astype('int64'), 2)
    ema_df_tail = round(sma_df['ema'].iloc[-1:].astype('int64'), 2)

    macd_df = calc_macd(df)
    if 'Date' in macd_df.columns and macd_df['Date'].dt.tz is not None:
        macd_df['Date'] = macd_df['Date'].dt.tz_convert('UTC').dt.tz_localize(None)
    
    ema26_df_tail = round(macd_df['ema26'].iloc[-1:].astype('int64'), 2)
    macd_df_tail = round(macd_df['macd'].iloc[-1:].astype('int64'), 2)
    signal_df_tail = round(macd_df['signal'].iloc[-1:].astype('int64'), 2)

    rsi_df = RSI(df, 14)
    if 'Date' in rsi_df.columns and rsi_df['Date'].dt.tz is not None:
        rsi_df['Date'] = rsi_df['Date'].dt.tz_convert('UTC').dt.tz_localize(None)
    
    rsi_df_tail = round(rsi_df['RSI'].iloc[-1:].astype('int64'), 2)

    adx_df = ADX(df, 14)
    if hasattr(adx_df, 'dt') and adx_df.dt.tz is not None:
        adx_df = adx_df.dt.tz_convert('UTC').dt.tz_localize(None)
    
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

# Helper function to safely get info values
def safe_info_get(key, default="N/A"):
    """Safely get value from info dictionary with flexible default types"""
    try:
        value = info.get(key, default)
        return value if value is not None else default
    except:
        return default


# metrics at a glance with safe info access
with st.container():
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('52 Week Low', safe_millify(safe_info_get('fiftyTwoWeekLow', "0")))
    col2.metric('52 Week High', safe_millify(safe_info_get('fiftyTwoWeekHigh', "0")))
    col3.metric('50 Day Average', safe_millify(safe_info_get('fiftyDayAverage', "0")))
    col4.metric('200 Day Average', safe_millify(safe_info_get('twoHundredDayAverage',"0")))

with st.container():
    col_1, col_2, col_3, col_4 = st.columns(4)
    col_1.metric('Market Day Low', safe_millify(safe_info_get('regularMarketDayLow', "0")))
    col_2.metric('Market Day High', safe_millify(safe_info_get('regularMarketDayHigh', "0")))
    col_3.metric('Previous Close', safe_millify(safe_info_get('previousClose', "0")))
    col_4.metric('Open', safe_millify(safe_info_get('open', "0")))

with st.container():
    co_1, co_2, co_3, co_4 = st.columns(4)
    co_1.metric('EBITDA Margin', safe_info_get('ebitdaMargins', 'N/A'))
    co_2.metric('Profit Margin', safe_info_get('profitMargins', 'N/A'))
    co_3.metric('Gross Margin', safe_info_get('grossMargins', 'N/A'))
    co_4.metric('Operating Margin', safe_info_get('operatingMargins', 'N/A'))

with st.container():
    co_11, co_22, co_33, co_44 = st.columns(4)
    co_11.metric('Current Ratio', safe_info_get('currentRatio', 'N/A'))
    co_22.metric('Return on Assets', safe_info_get('returnOnAssets', 'N/A'))
    co_33.metric('Debt to Equity', safe_info_get('debtToEquity', 'N/A'))
    co_44.metric('Return on Equity', safe_info_get('returnOnEquity', 'N/A'))

with st.container():
    c_1, c_2, c_3, c_4 = st.columns(4)
    c_1.metric('Closing Price', safe_millify(float(closing_price.iloc[0]) if not closing_price.empty else 0))
    c_2.metric('Simple Moving Average', safe_millify(float(sma_df_tail.iloc[0]) if not sma_df_tail.empty else 0))
    c_3.metric('Exponential Moving Average', safe_millify(float(ema_df_tail.iloc[0]) if not ema_df_tail.empty else 0))
    c_4.metric('EMA 26', safe_millify(float(ema26_df_tail.iloc[0]) if not ema26_df_tail.empty else 0))

with st.container():
    c_11, c_22, c_33, c_44 = st.columns(4)
    c_11.metric('Relative Strength Index', float(rsi_df_tail.iloc[0]) if not rsi_df_tail.empty else 0)
    c_22.metric('Average Directional Index', float(adx_df_tail.iloc[0]) if not adx_df_tail.empty else 0)
    c_33.metric('MACD', float(macd_df_tail.iloc[0]) if not macd_df_tail.empty else 0)
    c_44.metric('Signal', float(signal_df_tail.iloc[0]) if not signal_df_tail.empty else 0)

with st.container():
    cc_11, cc_22, cc_33, cc_44 = st.columns(4)
    cc_11.metric('Breaking Out??', breaking_out)
    cc_22.metric('Consolidating??', consolidating)
    cc_33.metric('Volume', safe_millify(safe_info_get('volume', "0")))
    cc_44.metric('Average Volume', safe_millify(safe_info_get('averageVolume', "0")))

# Additional metrics
with st.container():
    metrics_1, metrics_2, metrics_3, metrics_4 = st.columns(4)
    metrics_1.metric('Market Cap', safe_millify(safe_info_get('marketCap', "0")))
    metrics_2.metric('Enterprise Value', safe_millify(safe_info_get('enterpriseValue', "0")))
    metrics_3.metric('P/E Ratio', safe_info_get('trailingPE', 'N/A'))
    metrics_4.metric('Forward P/E', safe_info_get('forwardPE', 'N/A'))

# # Additional error information for debugging
# if st.checkbox("Show debug info"):
#     st.subheader("Debug Information")
#     st.write(f"DataFrame shape: {df.shape if df is not None else 'No data'}")
#     st.write(f"Info keys: {list(info.keys()) if info else 'No info'}")
#     st.write(f"Sample data: {df.head(3) if df is not None else 'No data'}")