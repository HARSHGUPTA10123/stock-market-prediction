import time
import pandas as pd
import numpy as np
import tensorflow as tf
import random as rn
import streamlit as st
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from yahooquery import Ticker
np.random.seed(1)
tf.random.set_seed(1)
rn.seed(1)
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
try:
    from st_aggrid import AgGrid
    from st_aggrid.grid_options_builder import GridOptionsBuilder
except ImportError:
    pass  # Silently ignore if not installed
from finta import TA
import ta.momentum
import ta.trend
import sqlite3
import ta

from millify import millify
try:
    from annotated_text import annotated_text  #type:ignore
except ImportError:
    annotated_text = None  # Handle missing import


# Robust YahooQuery data fetcher
def get_stock_data(ticker, start_date, end_date):
    """Robust function to fetch stock data with multiple fallback options using YahooQuery"""
    # Try different symbol formats
    symbol_formats = [
        ticker,  # Original format
        ticker.replace('.NS', '.BO'),  # BSE format
        ticker.replace('.NS', ''),  # Without exchange
        ticker + '.BO',  # Explicit BSE
    ]
    
    for symbol_format in symbol_formats:
        try:
            # UPDATED: Use YahooQuery instead of yfinance
            stock = Ticker(symbol_format)
            hist = stock.history(start=start_date, end=end_date)
            
            if not hist.empty:
                # Reset index and rename columns
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
                
                hist['symbols'] = ticker  # Keep original symbol for consistency
                hist['Date'] = pd.to_datetime(hist['Date'])
                
                # Ensure all required columns exist
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_columns = [col for col in required_columns if col not in hist.columns]
                
                if missing_columns:
                    st.warning(f"Missing columns for {symbol_format}: {missing_columns}")
                    continue
                
                st.success(f"✓ Data fetched successfully using: {symbol_format}")
                return hist
                
        except Exception as e:
            continue
    
    # If all formats fail, try period-based fallback
    try:
        stock = Ticker(ticker)
        hist = stock.history(period="1y")
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
            hist['symbols'] = ticker
            hist['Date'] = pd.to_datetime(hist['Date'])
            st.success(f"✓ Data fetched successfully using period fallback for: {ticker}")
            return hist
    except:
        pass
    
    # If all formats fail
    st.error(f"❌ Could not fetch data for {ticker} with any symbol format")
    st.info("💡 Try popular stocks like: RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS")
    return None

def get_stock_info(ticker):
    """Get stock info with robust error handling using YahooQuery"""
    symbol_formats = [ticker, ticker.replace('.NS', '.BO'), ticker.replace('.NS', '')]
    
    for symbol_format in symbol_formats:
        try:
            # UPDATED: Use YahooQuery instead of yfinance
            stock = Ticker(symbol_format)
            
            # Get comprehensive info from multiple endpoints
            info = {}
            
            # Get summary details
            summary = stock.summary_detail
            if summary and symbol_format in summary:
                info.update(summary[symbol_format])
            
            # Get price info
            price = stock.price
            if price and symbol_format in price:
                info.update(price[symbol_format])
            
            # Get key statistics
            try:
                key_stats = stock.key_stats
                if key_stats and symbol_format in key_stats:
                    info.update(key_stats[symbol_format])
            except:
                pass
            
            if info and len(info) > 10:  # Basic check if info is not empty
                return info
        except:
            continue
    
    return {}

# Streamlit page config
st.set_page_config(
    page_title="MarketEdge by Harsh", 
    layout="wide", 
    page_icon="💸",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
    header {visibility: visible !important;}
    [data-testid="stToolbar"] {visibility: visible !important;}
    footer {visibility: visible !important;}
    </style>
""", unsafe_allow_html=True)

# Show Streamlit menu + header + footer properly
show_menu_style = """
    <style>
    header {visibility: visible !important;}
    [data-testid="stToolbar"] {visibility: visible !important;}
    #MainMenu {visibility: visible !important;}
    footer {visibility: visible !important;}
    .css-1d391kg {padding-top: 0rem;}
    </style>
"""
st.markdown(show_menu_style, unsafe_allow_html=True)


# =============================================
# MAIN APPLICATION CONTENT
# =============================================

# Custom CSS for styling with dual-mode compatible colors
st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1f77b4, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #ff6b6b;
        padding-bottom: 0.5rem;
        font-weight: 600;
    }
    .feature-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #1f77b4;
        transition: transform 0.3s ease;
        border: 1px solid #e0e0e0;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }
    .highlight {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border-left: 5px solid #667eea;
        border: 1px solid #cbd5e1;
    }
    .metric-box {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border: 1px solid #e0e0e0;
    }
    .stats-container {
        display: flex;
        justify-content: space-around;
        text-align: center;
        margin: 2rem 0;
    }
    .stat-item {
        padding: 1rem;
    }
    .stat-number {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .stat-label {
        font-size: 1rem;
        color: #666;
        font-weight: 500;
    }
    .cta-section {
        text-align: center;
        padding: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        color: white;
        margin: 2rem 0;
        border: 1px solid #3730a3;
    }
    .stock-input-section {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 2rem 0;
        border: 1px solid #e0e0e0;
    }
    
    /* Feature card text styling */
    .feature-text {
        color: #2d3748;
        font-weight: 500;
        line-height: 1.6;
    }
    .feature-text strong {
        color: #e53e3e;
        font-weight: 600;
    }
    
    /* Highlight text styling */
    .highlight-text {
        color: #2c3e50;
        font-weight: 600;
        line-height: 1.6;
    }
    
    /* Metric box text */
    .metric-text {
        color: #4a5568;
        font-weight: 500;
    }
    
    /* Dark mode overrides */
    @media (prefers-color-scheme: dark) {
        .feature-card {
            background-color: #2d3748;
            border-color: #4a5568;
        }
        .feature-text {
            color: #e2e8f0;
        }
        .highlight {
            background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        }
        .highlight-text {
            color: #e2e8f0;
        }
        .metric-box {
            background-color: #2d3748;
            border-color: #4a5568;
        }
        .metric-text {
            color: #e2e8f0;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Header Section - Properly Centered with Original Styling
st.markdown("""
    <div style="text-align: center;">
        <h1 class="main-header">📈 MarketEdge by Harsh</h1>
        <h3 style="color: #FFFFFF; margin-bottom: 2rem;">Your All-in-One Financial Platform for Retail Investors</h3>
    </div>
""", unsafe_allow_html=True)
    
# Stats Section
st.markdown("""
<div class="stats-container">
    <div class="stat-item">
        <div class="stat-number">1,770+</div>
        <div class="stat-label">NSE Stocks</div>
    </div>
    <div class="stat-item">
        <div class="stat-number">10+</div>
        <div class="stat-label">Years Data</div>
    </div>
    <div class="stat-item">
        <div class="stat-number">11+</div>
        <div class="stat-label">Indicators</div>
    </div>
    <div class="stat-item">
        <div class="stat-number">AI</div>
        <div class="stat-label">Powered</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Introduction with highlight
st.markdown("""
<div class="highlight">
    <h3 style="margin-top: 0; color: #1f77b4;">🎯 One Platform, Infinite Possibilities</h3>
    <p class="highlight-text" style="font-size: 1.1rem; margin-bottom: 0;">
    StockForecast is the comprehensive financial platform designed specifically for retail investors. 
    Access <strong>Fundamental Information</strong>, <strong>Technical Indicators</strong>, <strong>Smart Screeners</strong>, 
    <strong>Pattern Recognition</strong>, and <strong>AI-Powered Forecasting</strong> for all National Stock Exchange (NSE) listed stocks.
    </p>
</div>
""", unsafe_allow_html=True)

# =============================================
# INTERACTIVE STOCK ANALYSIS SECTION
# =============================================

st.markdown("---")
st.markdown('<p class="sub-header">🔍 Live Stock Analysis</p>', unsafe_allow_html=True)

# Stock input section
with st.container():
    st.markdown("""
    <div class="stock-input-section">
        <h3 style="color: #1f77b4; margin-top: 0;">Enter Stock Details</h3>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # getting symbols/tickers - Use reliable symbols first
        try:
            csv = pd.read_csv('symbols.csv')
            symbol = csv['Symbol'].tolist()
            for i in range(0, len(symbol)):
                symbol[i] = symbol[i] + ".NS"

            # Ensure we have some reliable stocks at the top
            reliable_stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'HINDUNILVR.NS']
            for stock in reliable_stocks:
                if stock in symbol:
                    symbol.remove(stock)
            symbol = reliable_stocks + symbol
        except Exception as e:
            st.error(f"Error loading symbols: {e}")
            symbol = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS']
        
        selected_stock = st.selectbox(
            "Select or Enter Stock Symbol:",
            options=symbol,
            index=0,
            help="Choose from 1,773+ NSE listed stocks"
        )
    
    with col2:
        start_date = st.date_input(
            "Start Date",
            value=dt.date.today() - dt.timedelta(days=365),  # 1 year back from today
            help="Select analysis start date"
        )
    
    with col3:
        end_date = st.date_input(
            "End Date", 
            value=dt.date.today(),
            help="Select analysis end date"
        )
    
    st.markdown("</div>", unsafe_allow_html=True)

# Analyze button
if st.button("🚀 Analyze Stock", type="primary", use_container_width=True):
    with st.spinner("Fetching stock data and performing analysis..."):
        # Get stock data using YahooQuery
        stock_data = get_stock_data(selected_stock, start_date, end_date)
        
        if stock_data is not None:
            # Display fundamental information
            st.markdown("### 📊 Fundamental Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Selected Stock", selected_stock)
                st.metric("Data Points", len(stock_data))
                st.metric("Date Range", f"{start_date} to {end_date}")
            
            with col2:
                if len(stock_data) > 0:
                    latest = stock_data.iloc[-1]
                    prev = stock_data.iloc[-2] if len(stock_data) > 1 else latest
                    
                    price_change = latest['Close'] - prev['Close']
                    price_change_pct = (price_change / prev['Close']) * 100
                    
                    st.metric(
                        "Current Price", 
                        f"₹{latest['Close']:.2f}", 
                        f"{price_change_pct:+.2f}%"
                    )
                    st.metric("Volume", f"{latest['Volume']:,.0f}")

            
            # Display Previous data
            st.markdown("#### Previous Data")
            st.dataframe(stock_data.tail(10), use_container_width=True)
            
            # Simple price chart
            st.markdown("#### Price Chart")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=stock_data['Date'], 
                y=stock_data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#1f77b4', width=2)
            ))
            fig.update_layout(
                title=f"{selected_stock} Price Movement",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

# Key Features Overview
st.markdown('<p class="sub-header">🚀 Platform Features</p>', unsafe_allow_html=True)

# Feature Cards in columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h4 style="color: #1f77b4; margin-top: 0;">📊 Comprehensive Coverage</h4>
        <p class="feature-text">Access complete data for all <strong>1,773 companies</strong> listed on the National Stock Exchange (NSE) with 10+ years of historical data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h4 style="color: #1f77b4; margin-top: 0;">🔍 Technical Analysis</h4>
        <p class="feature-text">Utilize <strong>11+ professional technical indicators</strong> including RSI, MACD, Bollinger Bands, and Moving Averages.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h4 style="color: #1f77b4; margin-top: 0;">🎯 Pattern Recognition</h4>
        <p class="feature-text">Automated detection of <strong>bullish and bearish candlestick patterns</strong> with confidence scoring and real-time alerts.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h4 style="color: #1f77b4; margin-top: 0;">📈 Smart Screening</h4>
        <p class="feature-text">Advanced screener for <strong>breakout detection</strong>, consolidation signals, and multi-parameter filtering.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h4 style="color: #1f77b4; margin-top: 0;">🤖 AI Forecasting</h4>
        <p class="feature-text">Machine Learning-powered <strong>next-day price predictions</strong> trained on 5 years of market data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h4 style="color: #1f77b4; margin-top: 0;">📋 Fundamental Analysis</h4>
        <p class="feature-text">Complete financial analysis with <strong>quarterly and annual reports</strong>, balance sheets, and dividend history.</p>
    </div>
    """, unsafe_allow_html=True)

# Data Source Section
st.markdown('<p class="sub-header">💾 Data Source & Methodology</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="metric-box">
        <h4 style="color: #1f77b4; margin-top: 0;">📅 Time Period</h4>
        <p class="metric-text"><strong>10 years</strong> of comprehensive historical data</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="metric-box">
        <h4 style="color: #1f77b4; margin-top: 0;">⏰ Frequency</h4>
        <p class="metric-text"><strong>Daily</strong> price data with intraday insights</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="metric-box">
        <h4 style="color: #1f77b4; margin-top: 0;">📊 Source</h4>
        <p class="metric-text"><strong>Yahoo Finance API via YahooQuery</strong> with real-time updates</p>
    </div>
    """, unsafe_allow_html=True)

# Detailed Feature Breakdown
st.markdown('<p class="sub-header">🔎 Feature Deep Dive</p>', unsafe_allow_html=True)

# Fundamental Information
with st.expander("📚 Fundamental Information", expanded=True):
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        **Complete Corporate Intelligence:**
        - **Company Overview**: Detailed corporate information and business background
        - **Interactive Charts**: Candlestick and line charts with 10-year historical data
        - **Data Export**: Download historical prices in CSV format
        - **Financial Statements**: Comprehensive quarterly and annual reports:
        - Income Statements & Profit/Loss
        - Balance Sheets & Asset Management
        - Cash Flow Statements & Liquidity
        - Dividends History & Stock Splits
        """)
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #f0f8ff; border-radius: 10px; border: 1px solid #bfdbfe;">
            <span style="font-size: 3rem;">📊</span>
            <p style="font-weight: bold; margin: 0.5rem 0 0 0; color: #1e40af;">Deep Analysis</p>
        </div>
        """, unsafe_allow_html=True)

# Technical Indicators
with st.expander("📈 Technical Indicators"):
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #fff0f5; border-radius: 10px; border: 1px solid #fbcfe8;">
            <span style="font-size: 3rem;">🔍</span>
            <p style="font-weight: bold; margin: 0.5rem 0 0 0; color: #be185d;">11+ Tools</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        **Professional Trading Toolkit:**
        - **Trend Analysis**: Moving Averages (SMA, EMA), MACD
        - **Momentum Indicators**: RSI, Stochastic Oscillator, Williams %R
        - **Volatility Measures**: Bollinger Bands, ATR, Standard Deviation
        - **Volume Analysis**: OBV, Volume Weighted Average Price
        - **Support/Resistance**: Pivot Points, Fibonacci Retracement
        """)

# Technical Screener
with st.expander("🎯 Technical Screener"):
    st.markdown("""
    **Advanced Market Scanning Engine:**
    - **Breakout Detection**: Automatically identify stocks breaking out of consolidation patterns with volume confirmation
    - **Consolidation Signals**: Spot stocks in accumulation phases before major moves
    - **Multi-parameter Filtering**: Screen based on volume, price movements, technical levels, and fundamental metrics
    - **Custom Alert System**: Set up personalized screening criteria with real-time notifications
    - **Sector-wise Analysis**: Compare stocks within the same industry
    """)

# Pattern Recognition
with st.expander("🔍 Pattern Recognition"):
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        **Automated Candlestick Pattern Intelligence:**
        - **Bullish Patterns**: Hammer, Bullish Engulfing, Morning Star, Piercing Line, Three White Soldiers
        - **Bearish Patterns**: Shooting Star, Bearish Engulfing, Evening Star, Dark Cloud Cover, Three Black Crows
        - **Continuation Patterns**: Flags, Pennants, Triangles, Rectangles
        - **Real-time Scanning**: Continuous monitoring of all NSE stocks across multiple timeframes
        - **Confidence Scoring**: Probability assessment with historical accuracy metrics
        """)
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #f0fff0; border-radius: 10px; border: 1px solid #bbf7d0;">
            <span style="font-size: 3rem;">🔄</span>
            <p style="font-weight: bold; margin: 0.5rem 0 0 0; color: #166534;">Auto Detect</p>
        </div>
        """, unsafe_allow_html=True)

# Next-Day Forecasting
with st.expander("🤖 Next-Day Forecasting"):
    st.markdown("""
    **Advanced Machine Learning Prediction System:**
    
    <div class="metric-box">
    <h4 style="color: #1f77b4; margin-top: 0;">🧠 Model Architecture</h4>
    <p class="metric-text">Deep Learning LSTM (Long Short-Term Memory) model specifically designed for time-series forecasting. 
    The model processes sequential stock data through specialized memory cells that retain important historical patterns while filtering out market noise.</p>
    </div>
    
    <div class="metric-box">
    <h4 style="color: #1f77b4; margin-top: 0;">📚 Training Data</h4>
    <p class="metric-text">5 years of comprehensive historical market data with multiple feature engineering</p>
    </div>
    
    <div class="metric-box">
    <h4 style="color: #1f77b4; margin-top: 0;">🔮 Prediction Window</h4>
    <p class="metric-text">Analyzes past 60 days of price action, volume, and market sentiment to forecast next trading day</p>
    </div>
    
    <div class="metric-box">
    <h4 style="color: #1f77b4; margin-top: 0;">📊 Performance Metrics</h4>
    <p class="metric-text">Continuous model evaluation with backtesting, accuracy scores, and confidence intervals</p>
    </div>
    """, unsafe_allow_html=True)

# Call to Action
st.markdown("""
<div class="cta-section">
    <h2 style="margin-top: 0;">Ready to Transform Your Trading Strategy?</h2>
    <p style="font-size: 1.3rem; margin-bottom: 2rem;">
    Join thousands of investors making data-driven decisions with our powerful analytical tools
    </p>
    <div style="font-size: 4rem; margin: 1rem 0;">🚀</div>
    <p style="font-size: 1.1rem; font-style: italic;">
    Start your journey to smarter investing today!
    </p>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>© 2025 Harsh Gupta — Smart Stock Forecasting | Data Powered by Yahoo Finance (YahooQuery API)
 | NSE: National Stock Exchange of India</p>
</div>
""", unsafe_allow_html=True)
