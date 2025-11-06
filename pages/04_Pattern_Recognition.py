# import streamlit as st
# import pandas as pd
# import yfinance as yf
# import datetime as dt
# from functions import *
# import plotly.graph_objects as go
# from patterns import candlestick_patterns
# import talib
# import numpy as np

# st.title('Pattern Recognition')
# st.write('A pattern is identified by a line that connects common price points, such as closing prices or highs or lows, during a specific period of time.')
# st.write('Technical analysts and chartists seek to identify patterns as a way to anticipate the future direction of a security\'s price.')
# st.write('We automated this thing, for a specific ticker/symbol we scan through all the candlestick patterns and generate signals.')
# st.markdown('- Neutral - Not such activity or no trendline present at current moment')
# st.markdown('- Bullish - The stock is in up trendline ')
# st.markdown('- Bearish - The stock is in down trendline')

# # Load symbols with error handling
# try:
#     csv = pd.read_csv('symbols.csv')
#     symbol = csv['Symbol'].tolist()
#     for i in range(0, len(symbol)):
#         symbol[i] = symbol[i] + ".NS"
# except Exception as e:
#     st.error(f"Error loading symbols: {e}")
#     symbol = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]

# st.write('#### Select Stock ')
# try:
#     default_index = symbol.index('VISHWARAJ.NS') if 'VISHWARAJ.NS' in symbol else 0
# except:
#     default_index = 0

# ticker_input = st.selectbox('Enter or Choose NSE listed stock', symbol, index=default_index)

# # Plotting prices
# show = st.radio(
#     "Show/Hide Prices",
#     ('Show', 'Hide'))
# st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

# # Closing price chart - UPDATED SECTION
# if show == 'Show':
#     st.write("Enter period to check price of ", ticker_input)

#     # Getting date input
#     min_value = dt.datetime.today() - dt.timedelta(10 * 365)
#     max_value = dt.datetime.today()

#     start_input = st.date_input(
#         'Enter starting date',
#         value=dt.datetime.today() - dt.timedelta(90),
#         min_value=min_value, max_value=max_value, 
#         help='Enter the starting date from which you have to look the price'
#     )

#     end_input = st.date_input(
#         'Enter last date',
#         value=dt.datetime.today(),
#         min_value=min_value, max_value=max_value, 
#         help='Enter the last date till which you have to look the price'
#     )

#     try:
#         # UPDATED: Use Ticker().history() instead of download()
#         stock_data = yf.Ticker(ticker_input)
#         hist_price = stock_data.history(start=start_input, end=end_input)
        
#         # Check if hist_price is None or empty
#         if hist_price is None:
#             st.error("Failed to download data. The download function returned None.")
#         elif hist_price.empty:
#             st.warning("No data available for the selected date range.")
#         else:
#             # UPDATED: Proper data preprocessing
#             hist_price = hist_price.reset_index()
#             hist_price['symbols'] = ticker_input
#             hist_price['Date'] = pd.to_datetime(hist_price['Date'])
            
#             # UPDATED: Ensure all required columns exist
#             required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
#             for col in required_columns:
#                 if col not in hist_price.columns:
#                     st.error(f"Missing required column: {col}")
#                     break
                    
#             # UPDATED: Add Adj Close if not present
#             if 'Adj Close' not in hist_price.columns:
#                 hist_price['Adj Close'] = hist_price['Close']

#             # Radio button to switch between style
#             chart = st.radio(
#                 "Choose Style",
#                 ('Candlestick', 'Line Chart'))
#             st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

#             if chart == 'Line Chart':
#                 # UPDATED: Line chart plot with proper configuration
#                 fig = go.Figure()
#                 fig.add_trace(
#                     go.Scatter(
#                         x=hist_price['Date'],
#                         y=hist_price['Close'],
#                         name='Closing price',
#                         line=dict(color='blue', width=2)
#                     )
#                 )
#                 fig.update_layout(
#                     title={
#                         'text': f'Stock Prices of {ticker_input}',
#                         'y': 0.9,
#                         'x': 0.5,
#                         'xanchor': 'center',
#                         'yanchor': 'top'
#                     },
#                     height=600,
#                     template='plotly_white',
#                     xaxis_title='Date',
#                     yaxis_title='Price (₹)',
#                     yaxis=dict(tickprefix='₹')
#                 )
#                 st.plotly_chart(fig, use_container_width=True)

#             elif chart == 'Candlestick':
#                 # UPDATED: Candlestick chart with proper configuration
#                 fig = go.Figure()
#                 fig.add_trace(
#                     go.Candlestick(
#                         x=hist_price['Date'],
#                         open=hist_price['Open'],
#                         high=hist_price['High'],
#                         low=hist_price['Low'],
#                         close=hist_price['Close'],
#                         name='OHLC',
#                         increasing_line_color='green',
#                         decreasing_line_color='red'
#                     )
#                 )
#                 fig.update_layout(
#                     title={
#                         'text': f'Stock Prices of {ticker_input}',
#                         'y': 0.9,
#                         'x': 0.5,
#                         'xanchor': 'center',
#                         'yanchor': 'top'
#                     },
#                     height=600,
#                     template='plotly_white',
#                     xaxis_title='Date',
#                     yaxis_title='Price (₹)',
#                     yaxis=dict(tickprefix='₹'),
#                     xaxis_rangeslider_visible=False  # Hide range slider for cleaner look
#                 )
#                 st.plotly_chart(fig, use_container_width=True)
                
#     except Exception as e:
#         st.error(f"Error downloading price data: {e}")
# else:
#     st.write('Select show to check prices')

# # Pattern recognition section - UPDATED
# try:
#     # Retrieving data for pattern analysis
#     start_input = dt.datetime.today() - dt.timedelta(365)
#     end_input = dt.datetime.today()
    
#     # UPDATED: Use Ticker().history() for consistency
#     stock_data = yf.Ticker(ticker_input)
#     df = stock_data.history(start=start_input, end=end_input)
    
#     # Check if download returned None or empty
#     if df is None:
#         st.error("Failed to download data for pattern analysis. The download function returned None.")
#         st.stop()
#     elif df.empty:
#         st.error("No data available for pattern analysis.")
#         st.stop()
        
#     # UPDATED: Proper data preprocessing
#     df = df.reset_index()
#     df['symbols'] = ticker_input
#     df['Date'] = pd.to_datetime(df['Date'])
    
#     # UPDATED: Ensure all required columns exist
#     required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
#     for col in required_columns:
#         if col not in df.columns:
#             st.error(f"Missing required column: {col}")
#             st.stop()

#     # Convert DataFrame columns to proper 1D numpy arrays for TA-Lib
#     open_prices = np.asarray(df['Open'], dtype=float).ravel()
#     high_prices = np.asarray(df['High'], dtype=float).ravel()
#     low_prices = np.asarray(df['Low'], dtype=float).ravel()
#     close_prices = np.asarray(df['Close'], dtype=float).ravel()

#     # Scanning for patterns
#     candle_names = candlestick_patterns.keys()
#     pattern_results = []

#     for candle, names in candlestick_patterns.items():
#         try:
#             pattern_func = getattr(talib, candle, None)
#             if pattern_func:
#                 # Pass numpy arrays instead of DataFrame columns
#                 result = pattern_func(open_prices, high_prices, low_prices, close_prices)
#                 df[candle] = result
#                 last_value = result[-1] if len(result) > 0 else 0
#                 pattern_results.append(last_value)
#             else:
#                 pattern_results.append(0)
#         except Exception as e:
#             st.warning(f"Error calculating pattern {candle}: {e}")
#             pattern_results.append(0)

#     # Create signal dataframe
#     signal_df = pd.DataFrame({
#         'Pattern Names': list(candlestick_patterns.values()),
#         'Signal': pattern_results
#     })
    
#     # Map signal values to text
#     def map_signal(val):
#         if val > 0:
#             return 'Bullish'
#         elif val < 0:
#             return 'Bearish'
#         else:
#             return 'Neutral'
    
#     signal_df['Signal'] = signal_df['Signal'].apply(map_signal)

#     # Metrics
#     bullish_count = len(signal_df[signal_df['Signal'] == 'Bullish'])
#     bearish_count = len(signal_df[signal_df['Signal'] == 'Bearish'])
#     neutral_count = len(signal_df[signal_df['Signal'] == 'Neutral'])

#     with st.container():
#         st.write('#### Overview of pattern recognition')
#         coll_11, coll_22, coll_33 = st.columns(3)
#         coll_11.metric('Patterns with bullish signals', bullish_count)
#         coll_22.metric('Patterns with bearish signals', bearish_count)
#         coll_33.metric('Patterns with neutral signals', neutral_count)

#     # UPDATED: Enhanced styling function with better colors
#     # UPDATED: Enhanced styling function with better colors
#     def color_signals(val):
#         if val == 'Bullish':
#             return 'background-color: #d4edda; color: #155724; font-weight: bold;'
#         elif val == 'Bearish':
#             return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
#         elif val == 'Neutral':
#             return 'background-color: #e2e3e5; color: #383d41; font-weight: bold;'
#         return ''

#     st.write('#### All Candlestick Pattern Signals')

#     # Simple approach - apply to all cells, function handles which ones to style
#     styled_df = signal_df.style.map(color_signals)

#     st.dataframe(styled_df, use_container_width=True, height=600)

#     # # UPDATED: Display with wider container and custom height
#     # st.dataframe(styled_df, use_container_width=True, height=600)

#     # UPDATED: Summary with better styling
#     st.markdown(f"""
#     <div style="background-color: #e7f3ff; padding: 15px; border-radius: 8px; border-left: 5px solid #007bff; margin-top: 20px;">
#         <h4 style="margin: 0; color: #004085;">📊 Pattern Recognition Summary</h4>
#         <p style="margin: 8px 0 0 0; color: #004085;">
#         Found <strong>{bullish_count} bullish</strong>, <strong>{bearish_count} bearish</strong>, 
#         and <strong>{neutral_count} neutral</strong> patterns out of <strong>{len(signal_df)}</strong> total patterns.
#         </p>
#     </div>
#     """, unsafe_allow_html=True)
    
    
    
#         # Simple bullish patterns display
#     bullish_patterns = signal_df[signal_df['Signal'] == 'Bullish']

#     st.write("**Bullish Patterns:**")
#     if not bullish_patterns.empty:
#         for pattern_name in bullish_patterns['Pattern Names']:
#             st.write(f"🟢 {pattern_name}")
#     else:
#         st.write("None")

#     # Additional analysis
#     if bullish_count > bearish_count:
#         st.success(f"📈 **Overall Bullish Bias**: {bullish_count} bullish patterns vs {bearish_count} bearish patterns")
#     elif bearish_count > bullish_count:
#         st.warning(f"📉 **Overall Bearish Bias**: {bearish_count} bearish patterns vs {bullish_count} bullish patterns")
#     else:
#         st.info(f"⚖️ **Neutral Market**: {bullish_count} bullish and {bearish_count} bearish patterns")

# except Exception as e:
#     st.error(f"Error in pattern recognition: {e}")

# import streamlit as st
# import pandas as pd
# import datetime as dt
# from functions import *
# import plotly.graph_objects as go
# from patterns import candlestick_patterns
# import talib
# import numpy as np
# from yahooquery import Ticker

# st.title('Pattern Recognition')
# st.write('A pattern is identified by a line that connects common price points, such as closing prices or highs or lows, during a specific period of time.')
# st.write('Technical analysts and chartists seek to identify patterns as a way to anticipate the future direction of a security\'s price.')
# st.write('We automated this thing, for a specific ticker/symbol we scan through all the candlestick patterns and generate signals.')
# st.markdown('- Neutral - Not such activity or no trendline present at current moment')
# st.markdown('- Bullish - The stock is in up trendline ')
# st.markdown('- Bearish - The stock is in down trendline')

# # Load symbols with error handling
# try:
#     csv = pd.read_csv('symbols.csv')
#     symbol = csv['Symbol'].tolist()
#     for i in range(0, len(symbol)):
#         symbol[i] = symbol[i] + ".NS"
# except Exception as e:
#     st.error(f"Error loading symbols: {e}")
#     symbol = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]

# st.write('#### Select Stock ')
# try:
#     default_index = symbol.index('VISHWARAJ.NS') if 'VISHWARAJ.NS' in symbol else 0
# except:
#     default_index = 0

# ticker_input = st.selectbox('Enter or Choose NSE listed stock', symbol, index=default_index)

# # Plotting prices
# show = st.radio(
#     "Show/Hide Prices",
#     ('Show', 'Hide'))
# st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

# # YahooQuery data fetching function
# def get_yahooquery_data(symbol, start_date, end_date):
#     """Get historical data using YahooQuery"""
#     try:
#         stock = Ticker(symbol)
        
#         # Convert to string format for YahooQuery
#         start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else start_date
#         end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else end_date
        
#         hist = stock.history(start=start_str, end=end_str)
#         if not hist.empty:
#             hist = hist.reset_index()
#             hist = hist.rename(columns={
#                 'date': 'Date',
#                 'open': 'Open',
#                 'high': 'High',
#                 'low': 'Low',
#                 'close': 'Close',
#                 'volume': 'Volume',
#                 'adjclose': 'Adj Close'
#             })
#             # Ensure required columns exist
#             if 'Adj Close' not in hist.columns:
#                 hist['Adj Close'] = hist['Close']
#             hist['symbols'] = symbol
            
#             # FIX: Convert Date and remove timezone info
#             hist['Date'] = pd.to_datetime(hist['Date'])
#             if hist['Date'].dt.tz is not None:
#                 hist['Date'] = hist['Date'].dt.tz_convert('UTC').dt.tz_localize(None)
#             else:
#                 hist['Date'] = hist['Date'].dt.tz_localize(None)
            
#             return hist
#     except Exception as e:
#         st.error(f"YahooQuery error for {symbol}: {e}")
#     return pd.DataFrame()

# # Closing price chart - UPDATED TO YAHOOQUERY
# if show == 'Show':
#     st.write("Enter period to check price of ", ticker_input)

#     # Getting date input - ensure timezone-naive
#     min_value = dt.datetime.today().replace(tzinfo=None) - dt.timedelta(10 * 365)
#     max_value = dt.datetime.today().replace(tzinfo=None)

#     start_input = st.date_input(
#         'Enter starting date',
#         value=dt.datetime.today().replace(tzinfo=None) - dt.timedelta(90),
#         min_value=min_value, max_value=max_value, 
#         help='Enter the starting date from which you have to look the price'
#     )

#     end_input = st.date_input(
#         'Enter last date',
#         value=dt.datetime.today().replace(tzinfo=None),
#         min_value=min_value, max_value=max_value, 
#         help='Enter the last date till which you have to look the price'
#     )

#     try:
#         # Convert date inputs to datetime
#         start_dt = dt.datetime.combine(start_input, dt.time())
#         end_dt = dt.datetime.combine(end_input, dt.time())
        
#         # UPDATED: Use YahooQuery instead of yfinance
#         hist_price = get_yahooquery_data(ticker_input, start_dt, end_dt)
        
#         # Check if hist_price is None or empty
#         if hist_price is None:
#             st.error("Failed to fetch data. The function returned None.")
#         elif hist_price.empty:
#             st.warning("No data available for the selected date range.")
#         else:
#             # Ensure all required columns exist
#             required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
#             for col in required_columns:
#                 if col not in hist_price.columns:
#                     st.error(f"Missing required column: {col}")
#                     break

#             # Radio button to switch between style
#             chart = st.radio(
#                 "Choose Style",
#                 ('Candlestick', 'Line Chart'))
#             st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

#             if chart == 'Line Chart':
#                 # Line chart plot with proper configuration
#                 fig = go.Figure()
#                 fig.add_trace(
#                     go.Scatter(
#                         x=hist_price['Date'],
#                         y=hist_price['Close'],
#                         name='Closing price',
#                         line=dict(color='blue', width=2)
#                     )
#                 )
#                 fig.update_layout(
#                     title={
#                         'text': f'Stock Prices of {ticker_input}',
#                         'y': 0.9,
#                         'x': 0.5,
#                         'xanchor': 'center',
#                         'yanchor': 'top'
#                     },
#                     height=600,
#                     template='plotly_white',
#                     xaxis_title='Date',
#                     yaxis_title='Price (₹)',
#                     yaxis=dict(tickprefix='₹')
#                 )
#                 st.plotly_chart(fig, use_container_width=True)

#             elif chart == 'Candlestick':
#                 # Candlestick chart with proper configuration
#                 fig = go.Figure()
#                 fig.add_trace(
#                     go.Candlestick(
#                         x=hist_price['Date'],
#                         open=hist_price['Open'],
#                         high=hist_price['High'],
#                         low=hist_price['Low'],
#                         close=hist_price['Close'],
#                         name='OHLC',
#                         increasing_line_color='green',
#                         decreasing_line_color='red'
#                     )
#                 )
#                 fig.update_layout(
#                     title={
#                         'text': f'Stock Prices of {ticker_input}',
#                         'y': 0.9,
#                         'x': 0.5,
#                         'xanchor': 'center',
#                         'yanchor': 'top'
#                     },
#                     height=600,
#                     template='plotly_white',
#                     xaxis_title='Date',
#                     yaxis_title='Price (₹)',
#                     yaxis=dict(tickprefix='₹'),
#                     xaxis_rangeslider_visible=False  # Hide range slider for cleaner look
#                 )
#                 st.plotly_chart(fig, use_container_width=True)
                
#     except Exception as e:
#         st.error(f"Error fetching price data: {e}")
# else:
#     st.write('Select show to check prices')

# # Pattern recognition section - UPDATED TO YAHOOQUERY
# try:
#     # Retrieving data for pattern analysis - make timezone-naive
#     start_input = dt.datetime.today().replace(tzinfo=None) - dt.timedelta(365)
#     end_input = dt.datetime.today().replace(tzinfo=None)
    
#     # UPDATED: Use YahooQuery for pattern analysis
#     df = get_yahooquery_data(ticker_input, start_input, end_input)
    
#     # Check if data fetching returned None or empty
#     if df is None:
#         st.error("Failed to fetch data for pattern analysis. The function returned None.")
#         st.stop()
#     elif df.empty:
#         # Try fallback period
#         stock = Ticker(ticker_input)
#         hist = stock.history(period="1y")
#         if not hist.empty:
#             hist = hist.reset_index()
#             hist = hist.rename(columns={
#                 'date': 'Date',
#                 'open': 'Open',
#                 'high': 'High',
#                 'low': 'Low',
#                 'close': 'Close',
#                 'volume': 'Volume',
#                 'adjclose': 'Adj Close'
#             })
#             if 'Adj Close' not in hist.columns:
#                 hist['Adj Close'] = hist['Close']
#             hist['symbols'] = ticker_input
            
#             # Apply the same timezone fix to fallback data
#             hist['Date'] = pd.to_datetime(hist['Date'])
#             if hist['Date'].dt.tz is not None:
#                 hist['Date'] = hist['Date'].dt.tz_convert('UTC').dt.tz_localize(None)
#             else:
#                 hist['Date'] = hist['Date'].dt.tz_localize(None)
                
#             df = hist
#         else:
#             st.error("No data available for pattern analysis.")
#             st.stop()
        
#     # Ensure all required columns exist
#     required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
#     for col in required_columns:
#         if col not in df.columns:
#             st.error(f"Missing required column: {col}")
#             st.stop()

#     # Convert DataFrame columns to proper 1D numpy arrays for TA-Lib
#     open_prices = np.asarray(df['Open'], dtype=float).ravel()
#     high_prices = np.asarray(df['High'], dtype=float).ravel()
#     low_prices = np.asarray(df['Low'], dtype=float).ravel()
#     close_prices = np.asarray(df['Close'], dtype=float).ravel()

#     # Scanning for patterns
#     candle_names = candlestick_patterns.keys()
#     pattern_results = []

#     for candle, names in candlestick_patterns.items():
#         try:
#             pattern_func = getattr(talib, candle, None)
#             if pattern_func:
#                 # Pass numpy arrays instead of DataFrame columns
#                 result = pattern_func(open_prices, high_prices, low_prices, close_prices)
#                 df[candle] = result
#                 last_value = result[-1] if len(result) > 0 else 0
#                 pattern_results.append(last_value)
#             else:
#                 pattern_results.append(0)
#         except Exception as e:
#             st.warning(f"Error calculating pattern {candle}: {e}")
#             pattern_results.append(0)

#     # Create signal dataframe
#     signal_df = pd.DataFrame({
#         'Pattern Names': list(candlestick_patterns.values()),
#         'Signal': pattern_results
#     })
    
#     # Map signal values to text
#     def map_signal(val):
#         if val > 0:
#             return 'Bullish'
#         elif val < 0:
#             return 'Bearish'
#         else:
#             return 'Neutral'
    
#     signal_df['Signal'] = signal_df['Signal'].apply(map_signal)

#     # Metrics
#     bullish_count = len(signal_df[signal_df['Signal'] == 'Bullish'])
#     bearish_count = len(signal_df[signal_df['Signal'] == 'Bearish'])
#     neutral_count = len(signal_df[signal_df['Signal'] == 'Neutral'])

#     with st.container():
#         st.write('#### Overview of pattern recognition')
#         coll_11, coll_22, coll_33 = st.columns(3)
#         coll_11.metric('Patterns with bullish signals', bullish_count)
#         coll_22.metric('Patterns with bearish signals', bearish_count)
#         coll_33.metric('Patterns with neutral signals', neutral_count)

#     # Enhanced styling function with better colors
#     def color_signals(val):
#         if val == 'Bullish':
#             return 'background-color: #d4edda; color: #155724; font-weight: bold;'
#         elif val == 'Bearish':
#             return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
#         elif val == 'Neutral':
#             return 'background-color: #e2e3e5; color: #383d41; font-weight: bold;'
#         return ''

#     st.write('#### All Candlestick Pattern Signals')

#     # Simple approach - apply to all cells, function handles which ones to style
#     styled_df = signal_df.style.map(color_signals)

#     st.dataframe(styled_df, use_container_width=True, height=600)

#     # Summary with better styling
#     st.markdown(f"""
#     <div style="background-color: #e7f3ff; padding: 15px; border-radius: 8px; border-left: 5px solid #007bff; margin-top: 20px;">
#         <h4 style="margin: 0; color: #004085;">📊 Pattern Recognition Summary</h4>
#         <p style="margin: 8px 0 0 0; color: #004085;">
#         Found <strong>{bullish_count} bullish</strong>, <strong>{bearish_count} bearish</strong>, 
#         and <strong>{neutral_count} neutral</strong> patterns out of <strong>{len(signal_df)}</strong> total patterns.
#         </p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Simple bullish patterns display
#     bullish_patterns = signal_df[signal_df['Signal'] == 'Bullish']

#     st.write("**Bullish Patterns:**")
#     if not bullish_patterns.empty:
#         for pattern_name in bullish_patterns['Pattern Names']:
#             st.write(f"🟢 {pattern_name}")
#     else:
#         st.write("None")

#     # Additional analysis
#     if bullish_count > bearish_count:
#         st.success(f"📈 **Overall Bullish Bias**: {bullish_count} bullish patterns vs {bearish_count} bearish patterns")
#     elif bearish_count > bullish_count:
#         st.warning(f"📉 **Overall Bearish Bias**: {bearish_count} bearish patterns vs {bullish_count} bullish patterns")
#     else:
#         st.info(f"⚖️ **Neutral Market**: {bullish_count} bullish and {bearish_count} bearish patterns")

# except Exception as e:
#     st.error(f"Error in pattern recognition: {e}")




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
