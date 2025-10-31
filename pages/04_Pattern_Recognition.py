import streamlit as st
import pandas as pd
import yfinance as yf
import datetime as dt
from functions import *
import plotly.graph_objects as go
from patterns import candlestick_patterns
import talib
import numpy as np

st.title('Pattern Recognition')
st.write('A pattern is identified by a line that connects common price points, such as closing prices or highs or lows, during a specific period of time.')
st.write('Technical analysts and chartists seek to identify patterns as a way to anticipate the future direction of a security\'s price.')
st.write('We automated this thing, for a specific ticker/symbol we scan through all the candlestick patterns and generate signals.')
st.markdown('- Neutral - Not such activity or no trendline present at current moment')
st.markdown('- Bullish - The stock is in up trendline ')
st.markdown('- Bearish - The stock is in down trendline')

# Load symbols with error handling
try:
    csv = pd.read_csv('symbols.csv')
    symbol = csv['Symbol'].tolist()
    for i in range(0, len(symbol)):
        symbol[i] = symbol[i] + ".NS"
except Exception as e:
    st.error(f"Error loading symbols: {e}")
    symbol = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]

st.write('#### Select Stock ')
try:
    default_index = symbol.index('VISHWARAJ.NS') if 'VISHWARAJ.NS' in symbol else 0
except:
    default_index = 0

ticker_input = st.selectbox('Enter or Choose NSE listed stock', symbol, index=default_index)

# Plotting prices
show = st.radio(
    "Show/Hide Prices",
    ('Show', 'Hide'))
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

# Closing price chart - UPDATED SECTION
if show == 'Show':
    st.write("Enter period to check price of ", ticker_input)

    # Getting date input
    min_value = dt.datetime.today() - dt.timedelta(10 * 365)
    max_value = dt.datetime.today()

    start_input = st.date_input(
        'Enter starting date',
        value=dt.datetime.today() - dt.timedelta(90),
        min_value=min_value, max_value=max_value, 
        help='Enter the starting date from which you have to look the price'
    )

    end_input = st.date_input(
        'Enter last date',
        value=dt.datetime.today(),
        min_value=min_value, max_value=max_value, 
        help='Enter the last date till which you have to look the price'
    )

    try:
        # UPDATED: Use Ticker().history() instead of download()
        stock_data = yf.Ticker(ticker_input)
        hist_price = stock_data.history(start=start_input, end=end_input)
        
        # Check if hist_price is None or empty
        if hist_price is None:
            st.error("Failed to download data. The download function returned None.")
        elif hist_price.empty:
            st.warning("No data available for the selected date range.")
        else:
            # UPDATED: Proper data preprocessing
            hist_price = hist_price.reset_index()
            hist_price['symbols'] = ticker_input
            hist_price['Date'] = pd.to_datetime(hist_price['Date'])
            
            # UPDATED: Ensure all required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in hist_price.columns:
                    st.error(f"Missing required column: {col}")
                    break
                    
            # UPDATED: Add Adj Close if not present
            if 'Adj Close' not in hist_price.columns:
                hist_price['Adj Close'] = hist_price['Close']

            # Radio button to switch between style
            chart = st.radio(
                "Choose Style",
                ('Candlestick', 'Line Chart'))
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

            if chart == 'Line Chart':
                # UPDATED: Line chart plot with proper configuration
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=hist_price['Date'],
                        y=hist_price['Close'],
                        name='Closing price',
                        line=dict(color='blue', width=2)
                    )
                )
                fig.update_layout(
                    title={
                        'text': f'Stock Prices of {ticker_input}',
                        'y': 0.9,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'
                    },
                    height=600,
                    template='plotly_white',
                    xaxis_title='Date',
                    yaxis_title='Price (₹)',
                    yaxis=dict(tickprefix='₹')
                )
                st.plotly_chart(fig, use_container_width=True)

            elif chart == 'Candlestick':
                # UPDATED: Candlestick chart with proper configuration
                fig = go.Figure()
                fig.add_trace(
                    go.Candlestick(
                        x=hist_price['Date'],
                        open=hist_price['Open'],
                        high=hist_price['High'],
                        low=hist_price['Low'],
                        close=hist_price['Close'],
                        name='OHLC',
                        increasing_line_color='green',
                        decreasing_line_color='red'
                    )
                )
                fig.update_layout(
                    title={
                        'text': f'Stock Prices of {ticker_input}',
                        'y': 0.9,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'
                    },
                    height=600,
                    template='plotly_white',
                    xaxis_title='Date',
                    yaxis_title='Price (₹)',
                    yaxis=dict(tickprefix='₹'),
                    xaxis_rangeslider_visible=False  # Hide range slider for cleaner look
                )
                st.plotly_chart(fig, use_container_width=True)
                
    except Exception as e:
        st.error(f"Error downloading price data: {e}")
else:
    st.write('Select show to check prices')

# Pattern recognition section - UPDATED
try:
    # Retrieving data for pattern analysis
    start_input = dt.datetime.today() - dt.timedelta(365)
    end_input = dt.datetime.today()
    
    # UPDATED: Use Ticker().history() for consistency
    stock_data = yf.Ticker(ticker_input)
    df = stock_data.history(start=start_input, end=end_input)
    
    # Check if download returned None or empty
    if df is None:
        st.error("Failed to download data for pattern analysis. The download function returned None.")
        st.stop()
    elif df.empty:
        st.error("No data available for pattern analysis.")
        st.stop()
        
    # UPDATED: Proper data preprocessing
    df = df.reset_index()
    df['symbols'] = ticker_input
    df['Date'] = pd.to_datetime(df['Date'])
    
    # UPDATED: Ensure all required columns exist
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in df.columns:
            st.error(f"Missing required column: {col}")
            st.stop()

    # Convert DataFrame columns to proper 1D numpy arrays for TA-Lib
    open_prices = np.asarray(df['Open'], dtype=float).ravel()
    high_prices = np.asarray(df['High'], dtype=float).ravel()
    low_prices = np.asarray(df['Low'], dtype=float).ravel()
    close_prices = np.asarray(df['Close'], dtype=float).ravel()

    # Scanning for patterns
    candle_names = candlestick_patterns.keys()
    pattern_results = []

    for candle, names in candlestick_patterns.items():
        try:
            pattern_func = getattr(talib, candle, None)
            if pattern_func:
                # Pass numpy arrays instead of DataFrame columns
                result = pattern_func(open_prices, high_prices, low_prices, close_prices)
                df[candle] = result
                last_value = result[-1] if len(result) > 0 else 0
                pattern_results.append(last_value)
            else:
                pattern_results.append(0)
        except Exception as e:
            st.warning(f"Error calculating pattern {candle}: {e}")
            pattern_results.append(0)

    # Create signal dataframe
    signal_df = pd.DataFrame({
        'Pattern Names': list(candlestick_patterns.values()),
        'Signal': pattern_results
    })
    
    # Map signal values to text
    def map_signal(val):
        if val > 0:
            return 'Bullish'
        elif val < 0:
            return 'Bearish'
        else:
            return 'Neutral'
    
    signal_df['Signal'] = signal_df['Signal'].apply(map_signal)

    # Metrics
    bullish_count = len(signal_df[signal_df['Signal'] == 'Bullish'])
    bearish_count = len(signal_df[signal_df['Signal'] == 'Bearish'])
    neutral_count = len(signal_df[signal_df['Signal'] == 'Neutral'])

    with st.container():
        st.write('#### Overview of pattern recognition')
        coll_11, coll_22, coll_33 = st.columns(3)
        coll_11.metric('Patterns with bullish signals', bullish_count)
        coll_22.metric('Patterns with bearish signals', bearish_count)
        coll_33.metric('Patterns with neutral signals', neutral_count)

    # UPDATED: Enhanced styling function with better colors
    # UPDATED: Enhanced styling function with better colors
    def color_signals(val):
        if val == 'Bullish':
            return 'background-color: #d4edda; color: #155724; font-weight: bold;'
        elif val == 'Bearish':
            return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
        elif val == 'Neutral':
            return 'background-color: #e2e3e5; color: #383d41; font-weight: bold;'
        return ''

    st.write('#### All Candlestick Pattern Signals')

    # Simple approach - apply to all cells, function handles which ones to style
    styled_df = signal_df.style.map(color_signals)

    st.dataframe(styled_df, use_container_width=True, height=600)

    # # UPDATED: Display with wider container and custom height
    # st.dataframe(styled_df, use_container_width=True, height=600)

    # UPDATED: Summary with better styling
    st.markdown(f"""
    <div style="background-color: #e7f3ff; padding: 15px; border-radius: 8px; border-left: 5px solid #007bff; margin-top: 20px;">
        <h4 style="margin: 0; color: #004085;">📊 Pattern Recognition Summary</h4>
        <p style="margin: 8px 0 0 0; color: #004085;">
        Found <strong>{bullish_count} bullish</strong>, <strong>{bearish_count} bearish</strong>, 
        and <strong>{neutral_count} neutral</strong> patterns out of <strong>{len(signal_df)}</strong> total patterns.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    
    
        # Simple bullish patterns display
    bullish_patterns = signal_df[signal_df['Signal'] == 'Bullish']

    st.write("**Bullish Patterns:**")
    if not bullish_patterns.empty:
        for pattern_name in bullish_patterns['Pattern Names']:
            st.write(f"🟢 {pattern_name}")
    else:
        st.write("None")

    # Additional analysis
    if bullish_count > bearish_count:
        st.success(f"📈 **Overall Bullish Bias**: {bullish_count} bullish patterns vs {bearish_count} bearish patterns")
    elif bearish_count > bullish_count:
        st.warning(f"📉 **Overall Bearish Bias**: {bearish_count} bearish patterns vs {bullish_count} bullish patterns")
    else:
        st.info(f"⚖️ **Neutral Market**: {bullish_count} bullish and {bearish_count} bearish patterns")

except Exception as e:
    st.error(f"Error in pattern recognition: {e}")