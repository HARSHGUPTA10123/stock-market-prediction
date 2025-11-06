# import streamlit as st
# import yfinance as yf
# import datetime as dt
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from functions import *
# from finta import TA
# import ta.momentum
# import ta.trend

# st.title('Technical Indicators')

# # getting symbols/tickers - Use reliable symbols first
# csv = pd.read_csv('symbols.csv')
# symbol = csv['Symbol'].tolist()
# for i in range(0, len(symbol)):
#     symbol[i] = symbol[i] + ".NS"

# # Ensure we have some reliable stocks at the top
# reliable_stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'HINDUNILVR.NS']
# for stock in reliable_stocks:
#     if stock in symbol:
#         symbol.remove(stock)
# symbol = reliable_stocks + symbol

# ticker = st.selectbox('Enter or Choose', symbol, index=0)

# # Date inputs and data fetching
# min_value = dt.datetime.today() - dt.timedelta(10 * 365)
# max_value = dt.datetime.today()

# start_input = st.date_input(
#     'Enter starting date',
#     value=dt.datetime.today() - dt.timedelta(180),
#     min_value=min_value, max_value=max_value
# )

# end_input = st.date_input(
#     'Enter last date',
#     value=dt.datetime.today(),
#     min_value=min_value, max_value=max_value
# )

# # Robust data fetcher function
# def get_stock_data(ticker, start_date, end_date):
#     """Robust function to fetch stock data with multiple fallback options"""
#     symbol_formats = [
#         ticker,
#         ticker.replace('.NS', '.BO'),
#         ticker.replace('.NS', ''),
#         ticker + '.BO',
#     ]
    
#     for symbol_format in symbol_formats:
#         try:
#             stock_data = yf.Ticker(symbol_format)
#             df = stock_data.history(start=start_date, end=end_date, auto_adjust=False)
            
#             if df.empty:
#                 days_diff = (end_date - start_date).days
#                 df = stock_data.history(period=f"{max(60, days_diff)}d", auto_adjust=False)
            
#             if not df.empty:
#                 df = df.reset_index()
#                 df['symbols'] = ticker
                
#                 if 'Date' in df.columns:
#                     df['Date'] = pd.to_datetime(df['Date'])
                
#                 required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
#                 missing_columns = [col for col in required_columns if col not in df.columns]
                
#                 if missing_columns:
#                     st.warning(f"Missing columns for {symbol_format}: {missing_columns}")
#                     continue
                
#                 if 'Adj Close' not in df.columns:
#                     df['Adj Close'] = df['Close']
                
#                 st.success(f"✓ Data fetched successfully using: {symbol_format}")
#                 return df
                
#         except Exception as e:
#             continue
    
#     st.error(f"❌ Could not fetch data for {ticker} with any symbol format")
#     st.info("💡 Try popular stocks like: RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS")
#     return None

# # Use robust data fetcher
# try:
#     df = get_stock_data(ticker, start_input, end_input)
    
#     if df is None or df.empty:
#         st.error("Cannot proceed without stock data. Please try a different stock.")
#         st.stop()
        
#     df['Date'] = pd.to_datetime(df['Date'])
        
# except Exception as e:
#     st.error(f"Error fetching data for {ticker}: {e}")
#     st.stop()

# # graph template
# temp_style = st.radio(
#     "Choose Template Style",
#     ('ggplot2', 'seaborn', 'plotly_white', 'plotly_dark', 'gridon'))
# st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

# # OVERLAP STUDIES INDICATOR
# st.header('OVERLAY INDICATORS')
# st.write("Technical indicators that use the same scale as prices are plotted over the top of the prices on a stock chart. ")

# st.write('#### Moving Average')
# st.write(
#     "Moving averages (MA) are one of the most popular and often-used technical indicators in the financial markets. In simple word, a moving average is an indicator that shows the average value of a stock's price over a period (i.e. 10 days, 50 days, 200 days, etc) and is usually plotted along with the closing price.")

# df_ma = calc_moving_average(df, 14)
# df_ma = df_ma.reset_index()

# figMA = go.Figure()
# figMA.add_trace(
#     go.Scatter(
#         x=df_ma['Date'],
#         y=df_ma['Close'],
#         name="Prices"
#     )
# )

# figMA.add_trace(
#     go.Scatter(
#         x=df_ma['Date'],
#         y=df_ma['sma'],
#         name='SMA '
#     )
# )

# figMA.add_trace(
#     go.Scatter(
#         x=df_ma['Date'],
#         y=df_ma['ema'],
#         name='EMA '
#     )
# )

# figMA.update_layout(legend=dict(
#     orientation='h',
#     yanchor='bottom',
#     y=1,
#     xanchor='left',
#     x=0
# ))

# figMA.update_layout(height=600, width=1000, title_text='Closing Price of Stock & Moving Average',
#                     template=temp_style)

# st.plotly_chart(figMA, use_container_width=True)

# # hma
# st.subheader('Hull Moving Average (HMA)')
# st.write(
#     'The Hull Moving Average (HMA) is a directional trend indicator. It captures the current state of the market and uses recent price action to determine if conditions are bullish or bearish relative to historical data.')

# df['HMA'] = TA.HMA(df, 14)

# fig_apo = go.Figure()
# fig_apo.add_trace(
#     go.Scatter(
#         x=df['Date'],
#         y=df['HMA'],
#         name='HMA'
#     )
# )
# fig_apo.add_trace(
#     go.Scatter(
#         x=df['Date'],
#         y=df['Close'],
#         name='Close'
#     )
# )

# fig_apo.update_layout(legend=dict(
#     orientation='h',
#     yanchor='bottom',
#     y=1,
#     xanchor='left',
#     x=0)
#     , height=600, title_text='HMA', template=temp_style
# )

# st.plotly_chart(fig_apo, use_container_width=True)

# # bollinger bands
# st.subheader('Bollinger Bands')
# st.write(
#     "Bollinger Bands are envelopes plotted at a standard deviation level above and below a simple moving average of the price. Because the distance of the bands is based on standard deviation, they adjust to volatility swings in the underlying price.")

# # calculating bollinger bands
# df_boll = calc_bollinger(df, 20)
# df_boll = df_boll.reset_index()

# figBoll = go.Figure()
# figBoll.add_trace(
#     go.Scatter(
#         x=df_boll['Date'],
#         y=df_boll['bolu'],
#         name='Upper Band'
#     )
# )

# figBoll.add_trace(
#     go.Scatter(
#         x=df_boll['Date'],
#         y=df_boll['sma'],
#         name='SMA'
#     )
# )

# figBoll.add_trace(
#     go.Scatter(
#         x=df_boll['Date'],
#         y=df_boll['bold'],
#         name="Lower Band"
#     )
# )

# figBoll.add_trace(
#     go.Scatter(
#         x=df_boll['Date'],
#         y=df_boll['Close'],
#         name="closing price"
#     )
# )

# figBoll.update_layout(legend=dict(
#     orientation='h',
#     yanchor='bottom',
#     y=1,
#     xanchor='left',
#     x=0)
#     , height=600, title_text='Closing Price of Stock & Bollinger Band', template=temp_style
# )

# st.plotly_chart(figBoll, use_container_width=True)

# # kama
# st.subheader('KAMA Indicator')
# st.write(" Kaufman's Adaptive Moving Average (KAMA) is to identify the general trend of current market price action. Basically, when the KAMA indicator line is moving lower, it indicates the existence of a downtrend. On the other hand, when the KAMA line is moving higher, it shows an uptrend. ")

# try:
#     kama = ta.momentum.KAMAIndicator(df['Close'], 20, 2, 20)
#     df['kama'] = kama.kama()
# except Exception:
#     df['kama'] = 0

# df = df.reset_index()
# fig_kama = go.Figure()
# fig_kama.add_trace(
#     go.Scatter(
#         x=df['Date'],
#         y=df['kama'],
#         name="KAMA"
#     )
# )
# fig_kama.add_trace(
#     go.Scatter(
#         x=df['Date'],
#         y=df['Close'],
#         name="Close"
#     )
# )

# fig_kama.update_layout(legend=dict(
#     orientation='h',
#     yanchor='bottom',
#     y=1,
#     xanchor='left',
#     x=0)
#     , height=600, title_text='KAMA Indicator', template=temp_style
# )
# st.plotly_chart(fig_kama, use_container_width=True)

# ## Momentum indicators
# st.header('Momentum Indicators')
# st.write('Technical indicator which shows the trend direction and measures the pace of the price fluctuation by comparing current and past values.')

# st.subheader('Average Directional Index (ADX)')
# st.write(
#     "ADX stands for Average Directional Movement Index and can be used to help measure the overall strength of a trend. Indicator suggests that a strong trend is present when ADX is above 25 and no trend is present when below 20.")

# df['ADX'] = ADX(df, 14)

# fig_ADX = go.Figure()
# fig_ADX.add_trace(
#     go.Scatter(
#         x=df['Date'],
#         y=df['ADX'],
#         name='Average Directional Index'
#     )
# )

# fig_ADX.update_layout(legend=dict(
#     orientation='h',
#     yanchor='bottom',
#     y=1,
#     xanchor='left',
#     x=0)
#     , height=600, title_text='Average Directional Index', template=temp_style
# )

# st.plotly_chart(fig_ADX, use_container_width=True)

# # aroon
# st.subheader('Aroon Indicator')
# st.write(
#     "The Aroon indicator is a technical indicator that is used to identify trend changes in the price of an asset, as well as the strength of that trend. In essence, the indicator measures the time between highs and the time between lows over a time period.The indicator consists of the 'Aroon up' line, which measures the strength of the uptrend, and the 'Aroon down' line, which measures the strength of the downtrend.")

# try:
#     aroon = ta.trend.AroonIndicator(high=df['High'], low=df['Low'], window=14)
#     df['aroon_down'] = aroon.aroon_down()
#     df['aroon_indicator'] = aroon.aroon_indicator()
#     df['aroon_up'] = aroon.aroon_up()
# except Exception:
#     df['aroon_down'] = 0
#     df['aroon_indicator'] = 0
#     df['aroon_up'] = 0

# data_aroon = df.reset_index()
# fig_aroon = go.Figure()
# fig_aroon.add_trace(
#     go.Scatter(
#         x=data_aroon['Date'],
#         y=data_aroon['aroon_down'],
#         name='Aroon Down'
#     )
# )

# fig_aroon.add_trace(
#     go.Scatter(
#         x=data_aroon['Date'],
#         y=data_aroon['aroon_up'],
#         name="Aroon up"
#     )
# )

# fig_aroon.update_layout(legend=dict(
#     orientation='h',
#     yanchor='bottom',
#     y=1,
#     xanchor='left',
#     x=0)
#     , height=600, title_text='Aroon Indicator', template=temp_style
# )
# st.plotly_chart(fig_aroon, use_container_width=True)

# # plotting MACD
# st.subheader("Moving Average Convergance Divergence (MACD) ")
# st.write("Moving average convergence divergence (MACD) is a trend-following momentum indicator that shows the relationship between two moving averages of a stock. The MACD is calculated by subtracting the 26-period exponential moving average (EMA) from the 12-period EMA.")

# df_macd = calc_macd(df)
# df_macd = df_macd.reset_index()

# figMACD = make_subplots(rows=2, cols=1,
#                         shared_xaxes=True,
#                         vertical_spacing=0.01)

# figMACD.add_trace(
#     go.Scatter(
#         x=df_macd['Date'],
#         y=df_macd['Close'],
#         name="Prices"
#     )
# )

# figMACD.add_trace(
#     go.Scatter(
#         x=df_macd['Date'],
#         y=df_macd['ema12'],
#         name='EMA12 '
#     ),
#     row=1, col=1
# )

# figMACD.add_trace(
#     go.Scatter(
#         x=df_macd['Date'],
#         y=df_macd['ema26'],
#         name='EMA26'
#     ),
#     row=1, col=1
# )

# figMACD.add_trace(
#     go.Scatter(
#         x=df_macd['Date'],
#         y=df_macd['macd'],
#         name='MACD Line'
#     ),
#     row=2, col=1
# )

# figMACD.add_trace(
#     go.Scatter(
#         x=df_macd['Date'],
#         y=df_macd['signal'],
#         name='Signal Line'
#     ),
#     row=2, col=1
# )

# figMACD.update_layout(legend=dict(
#     orientation='h',
#     yanchor='bottom',
#     y=1,
#     xanchor='left',
#     x=0
# ), height=800, template=temp_style, title_text='Closing Price of Stock & MACD'
# )

# st.plotly_chart(figMACD, use_container_width=True)

# # plotting RSI
# st.subheader('Relative Strength Index (RSI)')
# st.write("The relative strength index (RSI) is a momentum indicator used in technical analysis that measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the price of a stock or other asset.Traditionally the RSI is considered overbought when above 70 and oversold when below 30.")

# df_RSI = RSI(df, 14)
# df_RSI = df_RSI.reset_index()

# fig_RSI = make_subplots(rows=2, cols=1,
#                         shared_xaxes=True,
#                         vertical_spacing=0.01)
# fig_RSI.add_trace(
#     go.Scatter(
#         x=df_RSI['Date'],
#         y=df_RSI['Adj Close'],
#         name='Closing Prices'
#     ),
#     row=1, col=1
# )

# fig_RSI.add_trace(
#     go.Scatter(
#         x=df_RSI['Date'],
#         y=df_RSI['RSI'],
#         name='RSI'
#     ),
#     row=2, col=1
# )

# fig_RSI.update_layout(legend=dict(
#     orientation='h',
#     yanchor='bottom',
#     y=1,
#     xanchor='left',
#     x=0
# ),
#     height=800, width=1000, title_text="Closing Price of Stock & RSI", template=temp_style)
# st.plotly_chart(fig_RSI, use_container_width=True)

# # TRIX
# st.subheader('TRIX Indicator')
# st.write(
#     "The triple exponential average (TRIX) is a momentum indicator used by technical traders that shows the percentage change in a moving average that has been smoothed exponentially three times. "
# )

# try:
#     trix = ta.trend.TRIXIndicator(df['Close'], 14)
#     df['trix'] = trix.trix()
# except Exception:
#     df['trix'] = 0

# data_trix = df.reset_index()
# fig_trix = go.Figure()
# fig_trix.add_trace(
#     go.Scatter(
#         x=data_trix['Date'],
#         y=data_trix['trix'],
#         name='TRIX'
#     )
# )

# fig_trix.update_layout(legend=dict(
#     orientation='h',
#     yanchor='bottom',
#     y=1,
#     xanchor='left',
#     x=0)
#     , height=600, title_text='TRIX Indicator', template=temp_style
# )
# st.plotly_chart(fig_trix, use_container_width=True)

# # stc
# st.subheader('Schaff Trend Cycle (STC)')
# st.write("The Schaff trend cycle indicator is popular for a general trading strategy. The strategy suggests buying when it surges above 25 level and sell when the signal lines go below the 75 leve")

# stc = TA.STC(df, 14)

# fig_stc = go.Figure()
# fig_stc.add_trace(
#     go.Scatter(
#         x=df['Date'],
#         y=stc,
#         name="Schaff Trend Cycle"
#     )
# )

# fig_stc.update_layout(legend=dict(
#     orientation='h',
#     yanchor='bottom',
#     y=1,
#     xanchor='left',
#     x=0)
#     , height=600, title_text='Schaff Trend Cycle', template=temp_style
# )

# st.plotly_chart(fig_stc, use_container_width=True)

# # Volume indicators
# st.header('Volume Indicators')
# st.write('Trading volume is a measure of how much a given financial asset has traded in a period of time. For stocks, volume is measured in the number of shares traded.Volume indicators are mathematical formulas that are visually represented in the most commonly used charting platforms.')

# # OBV
# # plotting OBV
# st.subheader('On Balance Volume (OBV)')
# st.write(
#     "On-balance volume (OBV) is a technical trading momentum indicator that uses volume flow to predict changes in stock price.")

# df['obv'] = OBV(df)

# fig_OBV = go.Figure()
# fig_OBV.add_trace(
#     go.Scatter(
#         x=df['Date'],
#         y=df['obv'],
#         name='On Balance Volume'
#     )
# )

# fig_OBV.update_layout(legend=dict(
#     orientation='h',
#     yanchor='bottom',
#     y=1,
#     xanchor='left',
#     x=0)
#     , height=600, title_text='On Balance Volume', template=temp_style
# )

# st.plotly_chart(fig_OBV, use_container_width=True)

# # volatility indicators
# st.header('Volatility Indicators')
# st.write('The volatility indicator is a technical tool that measures how far security stretches away from its mean price, higher and lower. ')

# # plotting ATR
# st.subheader('Average True Range (ATR)')
# st.write(
#     "Average True Range (ATR) is the average of true ranges over the specified period. ATR measures volatility, taking into account any gaps in the price movement.")

# df_ATR = ATR(df, 20)

# fig_ATR = go.Figure()
# fig_ATR.add_trace(
#     go.Scatter(
#         x=df_ATR['Date'],
#         y=df_ATR['ATR'],
#         name='Average True Range'
#     )
# )

# fig_ATR.update_layout(legend=dict(
#     orientation='h',
#     yanchor='bottom',
#     y=1,
#     xanchor='left',
#     x=0)
#     , height=600, title_text='Average True Range', template=temp_style
# )

# st.plotly_chart(fig_ATR, use_container_width=True)

# st.success("🎉 All Technical Indicators loaded successfully!")



import streamlit as st
import datetime as dt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from functions import *
from finta import TA
import ta.momentum
import ta.trend
from yahooquery import Ticker

st.title('Technical Indicators')

# getting symbols/tickers - Use reliable symbols first
try:
    csv = pd.read_csv('symbols.csv')
    symbol = csv['Symbol'].tolist()
    for i in range(0, len(symbol)):
        symbol[i] = symbol[i] + ".NS"
except Exception as e:
    st.error(f"Error loading symbols: {e}")
    symbol = []

# Ensure we have some reliable stocks at the top
reliable_stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'HINDUNILVR.NS']
for stock in reliable_stocks:
    if stock in symbol:
        symbol.remove(stock)
symbol = reliable_stocks + symbol

ticker = st.selectbox('Enter or Choose', symbol, index=0)

# Date inputs and data fetching
min_value = dt.datetime.today() - dt.timedelta(10 * 365)
max_value = dt.datetime.today()

start_input = st.date_input(
    'Enter starting date',
    value=dt.datetime.today() - dt.timedelta(180),
    min_value=min_value, max_value=max_value
)

end_input = st.date_input(
    'Enter last date',
    value=dt.datetime.today(),
    min_value=min_value, max_value=max_value
)

# YahooQuery data fetching function
def get_yahooquery_data(symbol, start_date, end_date):
    """Get historical data using YahooQuery (with tz-safe date conversion)."""
    try:
        stock = Ticker(symbol)
        hist = stock.history(start=start_date, end=end_date)

        if hist is not None and not hist.empty:
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

            hist['symbols'] = symbol

            # ✅ FIX: Consistent timezone handling (prevents tz-aware vs tz-naive errors)
            hist['Date'] = pd.to_datetime(hist['Date'], utc=True).dt.tz_convert('UTC').dt.tz_localize(None)

            st.success(f"✓ Data fetched successfully using: {symbol}")
            return hist

    except Exception as e:
        st.warning(f"YahooQuery error for {symbol}: {e}")

    return pd.DataFrame()


# Robust data fetcher function with YahooQuery
def get_stock_data(ticker, start_date, end_date):
    """Robust function to fetch stock data with multiple fallback options using YahooQuery"""
    symbol_formats = [
        ticker,
        ticker.replace('.NS', '.BO'),
        ticker.replace('.NS', ''),
        ticker + '.BO',
    ]
    
    for symbol_format in symbol_formats:
        try:
            df = get_yahooquery_data(symbol_format, start_date, end_date)
            
            if df is not None and not df.empty:
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if not missing_columns:
                    return df
                else:
                    st.warning(f"Missing columns for {symbol_format}: {missing_columns}")
                    continue
                
        except Exception as e:
            continue
    
    st.error(f"❌ Could not fetch data for {ticker} with any symbol format")
    st.info("💡 Try popular stocks like: RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS")
    return None

# Use robust data fetcher
try:
    df = get_stock_data(ticker, start_input, end_input)
    
    if df is None or df.empty:
        st.error("Cannot proceed without stock data. Please try a different stock.")
        st.stop()
        
    # Ensure Date column is properly formatted
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        
except Exception as e:
    st.error(f"Error fetching data for {ticker}: {e}")
    st.stop()

# graph template
temp_style = st.radio(
    "Choose Template Style",
    ('ggplot2', 'seaborn', 'plotly_white', 'plotly_dark', 'gridon'))
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

# OVERLAP STUDIES INDICATOR
st.header('OVERLAY INDICATORS')
st.write("Technical indicators that use the same scale as prices are plotted over the top of the prices on a stock chart. ")

st.write('#### Moving Average')
st.write(
    "Moving averages (MA) are one of the most popular and often-used technical indicators in the financial markets. In simple word, a moving average is an indicator that shows the average value of a stock's price over a period (i.e. 10 days, 50 days, 200 days, etc) and is usually plotted along with the closing price.")

df_ma = calc_moving_average(df, 14)
df_ma = df_ma.reset_index()

figMA = go.Figure()
figMA.add_trace(
    go.Scatter(
        x=df_ma['Date'],
        y=df_ma['Close'],
        name="Prices"
    )
)

figMA.add_trace(
    go.Scatter(
        x=df_ma['Date'],
        y=df_ma['sma'],
        name='SMA '
    )
)

figMA.add_trace(
    go.Scatter(
        x=df_ma['Date'],
        y=df_ma['ema'],
        name='EMA '
    )
)

figMA.update_layout(legend=dict(
    orientation='h',
    yanchor='bottom',
    y=1,
    xanchor='left',
    x=0
))

figMA.update_layout(height=600, width=1000, title_text='Closing Price of Stock & Moving Average',
                    template=temp_style)

st.plotly_chart(figMA, use_container_width=True)

# hma
st.subheader('Hull Moving Average (HMA)')
st.write(
    'The Hull Moving Average (HMA) is a directional trend indicator. It captures the current state of the market and uses recent price action to determine if conditions are bullish or bearish relative to historical data.')

df['HMA'] = TA.HMA(df, 14)

fig_apo = go.Figure()
fig_apo.add_trace(
    go.Scatter(
        x=df['Date'],
        y=df['HMA'],
        name='HMA'
    )
)
fig_apo.add_trace(
    go.Scatter(
        x=df['Date'],
        y=df['Close'],
        name='Close'
    )
)

fig_apo.update_layout(legend=dict(
    orientation='h',
    yanchor='bottom',
    y=1,
    xanchor='left',
    x=0)
    , height=600, title_text='HMA', template=temp_style
)

st.plotly_chart(fig_apo, use_container_width=True)

# bollinger bands
st.subheader('Bollinger Bands')
st.write(
    "Bollinger Bands are envelopes plotted at a standard deviation level above and below a simple moving average of the price. Because the distance of the bands is based on standard deviation, they adjust to volatility swings in the underlying price.")

# calculating bollinger bands
df_boll = calc_bollinger(df, 20)
df_boll = df_boll.reset_index()

figBoll = go.Figure()
figBoll.add_trace(
    go.Scatter(
        x=df_boll['Date'],
        y=df_boll['bolu'],
        name='Upper Band'
    )
)

figBoll.add_trace(
    go.Scatter(
        x=df_boll['Date'],
        y=df_boll['sma'],
        name='SMA'
    )
)

figBoll.add_trace(
    go.Scatter(
        x=df_boll['Date'],
        y=df_boll['bold'],
        name="Lower Band"
    )
)

figBoll.add_trace(
    go.Scatter(
        x=df_boll['Date'],
        y=df_boll['Close'],
        name="closing price"
    )
)

figBoll.update_layout(legend=dict(
    orientation='h',
    yanchor='bottom',
    y=1,
    xanchor='left',
    x=0)
    , height=600, title_text='Closing Price of Stock & Bollinger Band', template=temp_style
)

st.plotly_chart(figBoll, use_container_width=True)

# kama
st.subheader('KAMA Indicator')
st.write(" Kaufman's Adaptive Moving Average (KAMA) is to identify the general trend of current market price action. Basically, when the KAMA indicator line is moving lower, it indicates the existence of a downtrend. On the other hand, when the KAMA line is moving higher, it shows an uptrend. ")

try:
    kama = ta.momentum.KAMAIndicator(df['Close'], 20, 2, 20)
    df['kama'] = kama.kama()
except Exception:
    df['kama'] = 0

df = df.reset_index()
fig_kama = go.Figure()
fig_kama.add_trace(
    go.Scatter(
        x=df['Date'],
        y=df['kama'],
        name="KAMA"
    )
)
fig_kama.add_trace(
    go.Scatter(
        x=df['Date'],
        y=df['Close'],
        name="Close"
    )
)

fig_kama.update_layout(legend=dict(
    orientation='h',
    yanchor='bottom',
    y=1,
    xanchor='left',
    x=0)
    , height=600, title_text='KAMA Indicator', template=temp_style
)
st.plotly_chart(fig_kama, use_container_width=True)

## Momentum indicators
st.header('Momentum Indicators')
st.write('Technical indicator which shows the trend direction and measures the pace of the price fluctuation by comparing current and past values.')

st.subheader('Average Directional Index (ADX)')
st.write(
    "ADX stands for Average Directional Movement Index and can be used to help measure the overall strength of a trend. Indicator suggests that a strong trend is present when ADX is above 25 and no trend is present when below 20.")

df['ADX'] = ADX(df, 14)

fig_ADX = go.Figure()
fig_ADX.add_trace(
    go.Scatter(
        x=df['Date'],
        y=df['ADX'],
        name='Average Directional Index'
    )
)

fig_ADX.update_layout(legend=dict(
    orientation='h',
    yanchor='bottom',
    y=1,
    xanchor='left',
    x=0)
    , height=600, title_text='Average Directional Index', template=temp_style
)

st.plotly_chart(fig_ADX, use_container_width=True)

# aroon
st.subheader('Aroon Indicator')
st.write(
    "The Aroon indicator is a technical indicator that is used to identify trend changes in the price of an asset, as well as the strength of that trend. In essence, the indicator measures the time between highs and the time between lows over a time period.The indicator consists of the 'Aroon up' line, which measures the strength of the uptrend, and the 'Aroon down' line, which measures the strength of the downtrend.")

try:
    aroon = ta.trend.AroonIndicator(high=df['High'], low=df['Low'], window=14)
    df['aroon_down'] = aroon.aroon_down()
    df['aroon_indicator'] = aroon.aroon_indicator()
    df['aroon_up'] = aroon.aroon_up()
except Exception:
    df['aroon_down'] = 0
    df['aroon_indicator'] = 0
    df['aroon_up'] = 0

data_aroon = df.reset_index()
fig_aroon = go.Figure()
fig_aroon.add_trace(
    go.Scatter(
        x=data_aroon['Date'],
        y=data_aroon['aroon_down'],
        name='Aroon Down'
    )
)

fig_aroon.add_trace(
    go.Scatter(
        x=data_aroon['Date'],
        y=data_aroon['aroon_up'],
        name="Aroon up"
    )
)

fig_aroon.update_layout(legend=dict(
    orientation='h',
    yanchor='bottom',
    y=1,
    xanchor='left',
    x=0)
    , height=600, title_text='Aroon Indicator', template=temp_style
)
st.plotly_chart(fig_aroon, use_container_width=True)

# plotting MACD
st.subheader("Moving Average Convergance Divergence (MACD) ")
st.write("Moving average convergence divergence (MACD) is a trend-following momentum indicator that shows the relationship between two moving averages of a stock. The MACD is calculated by subtracting the 26-period exponential moving average (EMA) from the 12-period EMA.")

df_macd = calc_macd(df)
df_macd = df_macd.reset_index()

figMACD = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.01)

figMACD.add_trace(
    go.Scatter(
        x=df_macd['Date'],
        y=df_macd['Close'],
        name="Prices"
    )
)

figMACD.add_trace(
    go.Scatter(
        x=df_macd['Date'],
        y=df_macd['ema12'],
        name='EMA12 '
    ),
    row=1, col=1
)

figMACD.add_trace(
    go.Scatter(
        x=df_macd['Date'],
        y=df_macd['ema26'],
        name='EMA26'
    ),
    row=1, col=1
)

figMACD.add_trace(
    go.Scatter(
        x=df_macd['Date'],
        y=df_macd['macd'],
        name='MACD Line'
    ),
    row=2, col=1
)

figMACD.add_trace(
    go.Scatter(
        x=df_macd['Date'],
        y=df_macd['signal'],
        name='Signal Line'
    ),
    row=2, col=1
)

figMACD.update_layout(legend=dict(
    orientation='h',
    yanchor='bottom',
    y=1,
    xanchor='left',
    x=0
), height=800, template=temp_style, title_text='Closing Price of Stock & MACD'
)

st.plotly_chart(figMACD, use_container_width=True)

# plotting RSI
st.subheader('Relative Strength Index (RSI)')
st.write("The relative strength index (RSI) is a momentum indicator used in technical analysis that measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the price of a stock or other asset.Traditionally the RSI is considered overbought when above 70 and oversold when below 30.")

df_RSI = RSI(df, 14)
df_RSI = df_RSI.reset_index()

fig_RSI = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.01)
fig_RSI.add_trace(
    go.Scatter(
        x=df_RSI['Date'],
        y=df_RSI['Adj Close'],
        name='Closing Prices'
    ),
    row=1, col=1
)

fig_RSI.add_trace(
    go.Scatter(
        x=df_RSI['Date'],
        y=df_RSI['RSI'],
        name='RSI'
    ),
    row=2, col=1
)

fig_RSI.update_layout(legend=dict(
    orientation='h',
    yanchor='bottom',
    y=1,
    xanchor='left',
    x=0
),
    height=800, width=1000, title_text="Closing Price of Stock & RSI", template=temp_style)
st.plotly_chart(fig_RSI, use_container_width=True)

# TRIX
st.subheader('TRIX Indicator')
st.write(
    "The triple exponential average (TRIX) is a momentum indicator used by technical traders that shows the percentage change in a moving average that has been smoothed exponentially three times. "
)

try:
    trix = ta.trend.TRIXIndicator(df['Close'], 14)
    df['trix'] = trix.trix()
except Exception:
    df['trix'] = 0

data_trix = df.reset_index()
fig_trix = go.Figure()
fig_trix.add_trace(
    go.Scatter(
        x=data_trix['Date'],
        y=data_trix['trix'],
        name='TRIX'
    )
)

fig_trix.update_layout(legend=dict(
    orientation='h',
    yanchor='bottom',
    y=1,
    xanchor='left',
    x=0)
    , height=600, title_text='TRIX Indicator', template=temp_style
)
st.plotly_chart(fig_trix, use_container_width=True)

# stc
st.subheader('Schaff Trend Cycle (STC)')
st.write("The Schaff trend cycle indicator is popular for a general trading strategy. The strategy suggests buying when it surges above 25 level and sell when the signal lines go below the 75 leve")

stc = TA.STC(df, 14)

fig_stc = go.Figure()
fig_stc.add_trace(
    go.Scatter(
        x=df['Date'],
        y=stc,
        name="Schaff Trend Cycle"
    )
)

fig_stc.update_layout(legend=dict(
    orientation='h',
    yanchor='bottom',
    y=1,
    xanchor='left',
    x=0)
    , height=600, title_text='Schaff Trend Cycle', template=temp_style
)

st.plotly_chart(fig_stc, use_container_width=True)

# Volume indicators
st.header('Volume Indicators')
st.write('Trading volume is a measure of how much a given financial asset has traded in a period of time. For stocks, volume is measured in the number of shares traded.Volume indicators are mathematical formulas that are visually represented in the most commonly used charting platforms.')

# OBV
# plotting OBV
st.subheader('On Balance Volume (OBV)')
st.write(
    "On-balance volume (OBV) is a technical trading momentum indicator that uses volume flow to predict changes in stock price.")

df['obv'] = OBV(df)

fig_OBV = go.Figure()
fig_OBV.add_trace(
    go.Scatter(
        x=df['Date'],
        y=df['obv'],
        name='On Balance Volume'
    )
)

fig_OBV.update_layout(legend=dict(
    orientation='h',
    yanchor='bottom',
    y=1,
    xanchor='left',
    x=0)
    , height=600, title_text='On Balance Volume', template=temp_style
)

st.plotly_chart(fig_OBV, use_container_width=True)

# volatility indicators
st.header('Volatility Indicators')
st.write('The volatility indicator is a technical tool that measures how far security stretches away from its mean price, higher and lower. ')

# plotting ATR
st.subheader('Average True Range (ATR)')
st.write(
    "Average True Range (ATR) is the average of true ranges over the specified period. ATR measures volatility, taking into account any gaps in the price movement.")

df_ATR = ATR(df, 20)

fig_ATR = go.Figure()
fig_ATR.add_trace(
    go.Scatter(
        x=df_ATR['Date'],
        y=df_ATR['ATR'],
        name='Average True Range'
    )
)

fig_ATR.update_layout(legend=dict(
    orientation='h',
    yanchor='bottom',
    y=1,
    xanchor='left',
    x=0)
    , height=600, title_text='Average True Range', template=temp_style
)

st.plotly_chart(fig_ATR, use_container_width=True)

st.success("🎉 All Technical Indicators loaded successfully!")





