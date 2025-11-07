import streamlit as st
import pandas as pd
import datetime as dt
import plotly.graph_objects as go
from typing import Optional, Dict, Any
from yahooquery import Ticker

st.set_page_config(page_title='Fundamental Information', layout='wide')
st.title('Fundamental Information')
st.write('Explore company fundamentals (financial statements, actions, and historical prices) fetched via Yahoo Finance.')

# ----------------- Load Symbols -----------------
@st.cache_data
def load_symbols(path='symbols.csv'):
    try:
        csv = pd.read_csv(path)
        return [str(s).strip() + '.NS' for s in csv['Symbol'].tolist()]
    except Exception:
        return []

symbols = load_symbols()
default = symbols[0] if symbols else 'RELIANCE.NS'
ticker = st.selectbox('Enter or choose NSE listed stock symbol', symbols if symbols else [default])

# ----------------- YahooQuery helpers -----------------
@st.cache_data
def get_info(symbol: str) -> Dict[str, Any]:
    try:
        stock = Ticker(symbol)
        info = {}
        for attr in ['summary_detail', 'asset_profile', 'price', 'key_stats']:
            try:
                val = getattr(stock, attr)
                if isinstance(val, dict) and symbol in val:
                    info.update(val[symbol])
            except Exception:
                continue
        return info
    except Exception:
        return {}

@st.cache_data
def get_history(symbol: str, period_days: int = 365*2):
    try:
        stock = Ticker(symbol)
        hist = stock.history(period=f"{period_days}d")
        if hist is None or getattr(hist, 'empty', True):
            return pd.DataFrame()
        hist = hist.reset_index()
        # FIX 2: Corrected rename method - use dictionary directly without Literal[True]
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
        hist['Date'] = pd.to_datetime(hist['Date'], utc=True).dt.tz_convert('UTC').dt.tz_localize(None)
        return hist
    except Exception:
        return pd.DataFrame()

@st.cache_data
def get_financial(symbol: str, statement='income', frequency='a') -> pd.DataFrame:
    try:
        stock = Ticker(symbol)
        if statement=='income':
            df = stock.income_statement(frequency=frequency)
        elif statement=='balance':
            df = stock.balance_sheet(frequency=frequency)
        elif statement=='cash':
            df = stock.cash_flow(frequency=frequency)
        else:
            return pd.DataFrame()
        if df is None or getattr(df, 'empty', True):
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()

# ----------------- CORRECTED Dividends & Splits Helper -----------------
@st.cache_data
def get_corporate_actions(symbol: str):
    """Get dividends and splits from price history for NSE stocks"""
    try:
        stock = Ticker(symbol)
        hist = stock.history(period="max")
        dividends = pd.DataFrame()
        splits = pd.DataFrame()
        
        if hist is not None and not hist.empty:
            # Extract dividends
            if 'dividends' in hist.columns:
                div_data = hist[hist['dividends'] > 0]['dividends']
                if not div_data.empty:
                    dividends = div_data.reset_index()
                    dividends = dividends.rename(columns={'date': 'Date', 'dividends': 'Dividends'})
                    dividends['Date'] = pd.to_datetime(dividends['Date']).dt.tz_localize(None)
            
            # Extract splits
            if 'splits' in hist.columns:
                split_data = hist[hist['splits'] > 0]['splits']
                if not split_data.empty:
                    splits = split_data.reset_index()
                    splits = splits.rename(columns={'date': 'Date', 'splits': 'SplitRatio'})
                    splits['Date'] = pd.to_datetime(splits['Date']).dt.tz_localize(None)
        
        return dividends, splits
    except Exception as e:
        st.error(f"Error fetching corporate actions: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

# ----------------- Display Company Info -----------------
info = get_info(ticker)
if info:
    st.subheader(info.get('longName', 'N/A'))
    st.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
    st.markdown(f"**Industry:** {info.get('industry', 'N/A')}")
    st.markdown(f"**Phone:** {info.get('phone', 'N/A')}")
    address_parts = [info.get('address1'), info.get('city'), info.get('zip'), info.get('country')]
    address = ', '.join([str(x) for x in address_parts if x]) if any(address_parts) else 'N/A'
    st.markdown(f"**Address:** {address}")
    st.markdown(f"**Website:** {info.get('website', 'N/A')}")
    with st.expander('Business Summary'):
        st.write(info.get('longBusinessSummary', 'No business summary available.'))
else:
    st.warning("No company overview available")

# ----------------- Historical Prices -----------------
min_value = dt.date.today() - dt.timedelta(days=365*10)
max_value = dt.date.today()
start_input = st.date_input('Start date', value=dt.date.today()-dt.timedelta(days=90), min_value=min_value, max_value=max_value)
end_input = st.date_input('End date', value=dt.date.today(), min_value=min_value, max_value=max_value)

if end_input >= start_input:
    hist = get_history(ticker, (end_input-start_input).days+30)
    if hist.empty:
        st.warning("No historical price data available")
    else:
        hist_price = hist[(hist['Date'] >= pd.Timestamp(start_input)) & (hist['Date'] <= pd.Timestamp(end_input))]
        if hist_price.empty:
            st.warning("No price data in selected range")
        else:
            st.download_button("Download historical data", hist_price.to_csv(index=False).encode('utf-8'), file_name='historical.csv')
            chart_type = st.radio("Chart Style", ('Line','Candlestick'))
            if chart_type=='Line':
                fig = go.Figure(data=[go.Scatter(x=hist_price['Date'], y=hist_price['Close'], mode='lines')])
            else:
                fig = go.Figure(data=[go.Candlestick(
                    x=hist_price['Date'],
                    open=hist_price['Open'],
                    high=hist_price['High'],
                    low=hist_price['Low'],
                    close=hist_price['Close'],
                    increasing_line_color='green',
                    decreasing_line_color='red'
                )])
            fig.update_layout(height=600, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
else:
    st.error("End date must be after start date")

# ----------------- Financial Statements -----------------
st.markdown('---')
st.subheader("Financial Statements")
with st.spinner("Fetching financials..."):
    inc_a = get_financial(ticker, 'income', 'a')
    bal_a = get_financial(ticker, 'balance', 'a')
    cash_a = get_financial(ticker, 'cash', 'a')

def show_df(df, title):
    if df.empty:
        st.info(f"No data available for {title}")
    else:
        st.write(f"### {title}")
        st.dataframe(df, use_container_width=True)

show_df(inc_a, "Profit & Loss (Annual)")
show_df(bal_a, "Balance Sheet (Annual)")
show_df(cash_a, "Cash Flow (Annual)")

# ----------------- Dividends & Splits -----------------
st.markdown('---')
st.subheader("Dividends & Splits")

# Fetch and display - FIX 1 & 3: Using the single corrected function
dividends, splits = get_corporate_actions(ticker)

if dividends.empty and splits.empty:
    st.info("""**No dividends or splits data available for this stock.**
    
This could be because:
- The company has never paid dividends or had stock splits
- Yahoo Finance doesn't have this data for NSE stocks
- The stock is relatively new or less frequently traded
""")
else:
    if not dividends.empty:
        st.write("### 💰 Dividends")
        st.dataframe(dividends[['Date','Dividends']], use_container_width=True)
        total_dividends = len(dividends)
        latest_dividend = dividends.iloc[0]['Dividends'] if 'Dividends' in dividends.columns and len(dividends) > 0 else 'N/A'
        st.write(f"**Summary:** {total_dividends} dividend events | Latest: {latest_dividend}")
    
    if not splits.empty:
        st.write("### 📊 Stock Splits")
        st.dataframe(splits[['Date','SplitRatio']], use_container_width=True)
        total_splits = len(splits)
        st.write(f"**Summary:** {total_splits} split events")

st.caption("Data provided by Yahoo Finance via YahooQuery")
