import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from yahooquery_wrapper import yq

# TensorFlow / Keras
try:
    from tensorflow.keras.models import Sequential          #type:ignore
    from tensorflow.keras.layers import LSTM, Dense, Dropout  #type:ignore
except ImportError:
    st.error("Please install TensorFlow: pip install tensorflow")
    st.stop()

st.set_page_config(page_title="LSTM Stock Forecast", layout="wide")
st.title("📈 LSTM Stock Price Future Prediction")

# ----------------------
# Load Stock List
# ----------------------
@st.cache_data
def load_symbols():
    return [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS",
        "BHARTIARTL.NS", "KOTAKBANK.NS", "ITC.NS", "SUZLON.NS", "ADANIGREEN.NS",
        "TATAMOTORS.NS", "ADANIPORTS.NS", "POWERGRID.NS", "NTPC.NS", "ONGC.NS",
        "BAJFINANCE.NS", "HCLTECH.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "WIPRO.NS",
        "AXISBANK.NS", "SUNPHARMA.NS", "ZOMATO.NS", "VEDL.NS"
    ]

symbols = load_symbols()

selected_stock = st.selectbox("Select Stock:", symbols, index=symbols.index("SUZLON.NS"))
start_prediction = st.button("🚀 Start Forecast", use_container_width=True)

# ----------------------
# DATA FETCHING WITH YAHOOQUERY WRAPPER
# ----------------------
def get_stock_data(symbol):
    """Get stock data using your YahooQuery wrapper"""
    try:
        # Get historical data (2 years)
        st.write("📡 Fetching data from Yahoo Finance...")
        df = yq.download(symbol, period="2y")
        
        if df.empty:
            st.warning("No data returned. Trying with 1 year period...")
            df = yq.download(symbol, period="1y")
        
        if df.empty:
            st.warning("Trying with 6 months period...")
            df = yq.download(symbol, period="6mo")
        
        if not df.empty:
            # Get current price from info
            info = yq.get_info(symbol)
            live_price = info.get('currentPrice', 
                         info.get('regularMarketPrice', 
                         info.get('previousClose', None)))
            
            # Ensure we have the Close column
            if 'Close' in df.columns:
                df = df[['Close']].dropna()
                df.rename(columns={'Close': 'close'}, inplace=True)
                return df, live_price, True
            else:
                st.error("No 'Close' price data found in historical data")
                return None, None, False
        else:
            return None, None, False
            
    except Exception as e:
        st.error(f"Error in get_stock_data: {e}")
        return None, None, False

# ----------------------
# MAIN APPLICATION
# ----------------------
if start_prediction:
    with st.spinner("🔄 Fetching stock data using YahooQuery..."):
        df, live_price, success = get_stock_data(selected_stock)
    
    if not success or df is None or df.empty:
        st.error(f"""
        ❌ Could not fetch data for {selected_stock}
        
        **Troubleshooting steps:**
        1. Check your internet connection
        2. Verify the stock symbol is correct
        3. Try a different stock symbol
        4. The stock might be delisted or data temporarily unavailable
        
        **Working symbols to try:**
        - RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS
        """)
        
        # Debug information
        with st.expander("🔧 Debug Information"):
            try:
                ticker = yq.Ticker(selected_stock)
                st.write("Available methods:", [x for x in dir(ticker) if not x.startswith('_')][:15])
                
                # Test different data endpoints
                st.write("Testing data endpoints:")
                
                # Test summary_detail
                try:
                    summary = ticker.summary_detail
                    if summary and selected_stock in summary:
                        st.success("✅ summary_detail: Available")
                    else:
                        st.warning("❌ summary_detail: Not available")
                except Exception as e:
                    st.error(f"❌ summary_detail: Error - {e}")
                
                # Test price
                try:
                    price_data = ticker.price
                    if price_data and selected_stock in price_data:
                        st.success("✅ price: Available")
                    else:
                        st.warning("❌ price: Not available")
                except Exception as e:
                    st.error(f"❌ price: Error - {e}")
                
                # Test history with different periods
                periods = ['1mo', '3mo', '6mo', '1y', '2y']
                for period in periods:
                    try:
                        test_data = ticker.history(period=period)
                        if not test_data.empty:
                            st.success(f"✅ history({period}): {len(test_data)} records")
                        else:
                            st.warning(f"❌ history({period}): No data")
                    except Exception as e:
                        st.error(f"❌ history({period}): Error - {e}")
                        
            except Exception as e:
                st.write(f"Debug error: {e}")
        
        st.stop()
    
    # Display success message and data info
    st.success(f"✅ Successfully loaded data for {selected_stock}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        # FIXED: Use only the last close price from historical data for consistency
        current_price = df['close'].iloc[-1]
        st.metric("💰 Current Price", f"₹{current_price:.2f}")
    with col2:
        st.metric("Data Points", len(df))
    with col3:
        if not df.empty:
            date_range = f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}"
            st.metric("Period", date_range)
        else:
            st.metric("Period", "N/A")
    
    # Show data preview
    st.subheader("📊 Historical Data Preview")
    st.dataframe(df.tail(10))
    
    # Plot price history
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode="lines", name="Close Price", line=dict(color='blue', width=2)))
    fig.update_layout(
        title=f"{selected_stock} Price History",
        xaxis_title="Date", 
        yaxis_title="Price (₹)",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # ----------------------
    # LSTM PROCESSING
    # ----------------------
    st.info("🧠 Preparing data for LSTM training...")
    
    # Check if we have enough data
    if len(df) < 100:
        st.warning(f"⚠️ Limited data available ({len(df)} points). LSTM works better with more data.")
    
    if len(df) < 60:
        st.error("❌ Not enough data for LSTM training. Need at least 60 data points.")
        st.stop()
    
    data = df.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    seq_len = 60
    X, y = [], []
    
    # Create sequences
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i-seq_len:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # ----------------------
    # BUILD AND TRAIN MODEL
    # ----------------------
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer="adam", loss="mse")
    
    with st.spinner("🔄 Training LSTM Model (30-45 seconds)..."):
        history = model.fit(X, y, epochs=10, batch_size=32, verbose=0, validation_split=0.1)
    
    # ----------------------
    # NEXT DAY PREDICTION
    # ----------------------
    last_seq = scaled_data[-seq_len:]
    next_scaled = model.predict(last_seq.reshape(1, seq_len, 1), verbose=0)
    next_price = scaler.inverse_transform(next_scaled)[0][0]
    
    # FIXED: Use the same current_price variable that we displayed earlier
    change = next_price - current_price
    pct_change = (change / current_price) * 100
    
    st.subheader("🔮 Next Day Prediction")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", f"₹{current_price:.2f}")
    with col2:
        st.metric("Predicted Price", f"₹{next_price:.2f}")
    with col3:
        st.metric("Change (₹)", f"₹{change:+.2f}")
    with col4:
        st.metric("Change (%)", f"{pct_change:+.2f}%")
    
    # Signal
    signal_col1, signal_col2 = st.columns([1, 3])
    with signal_col1:
        if change > 0:
            st.success("📈 BULLISH")
        else:
            st.error("📉 BEARISH")
    with signal_col2:
        if change > 0:
            st.info("Model predicts upward movement tomorrow")
        else:
            st.info("Model predicts downward movement tomorrow")
    
    # ----------------------
    # 30-DAY FORECAST
    # ----------------------
    st.info("🔭 Generating 30-day forecast...")
    
    future_predictions = []
    last_sequence = scaled_data[-seq_len:].copy()
    
    for i in range(30):
        next_pred = model.predict(last_sequence.reshape(1, seq_len, 1), verbose=0)[0][0]
        future_predictions.append(next_pred)
        # Update sequence: remove first element, add prediction to end
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[-1] = next_pred
    
    future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    
    # Generate future dates
    last_date = df.index[-1]
    if hasattr(last_date, 'tz'):  # Handle timezone info
        last_date = last_date.tz_localize(None)
    
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=30)
    
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted_Price": future_prices.flatten()
    })
    
    # Plot forecast
    fig2 = go.Figure()
    
    # Historical data (last 90 days)
    fig2.add_trace(go.Scatter(
        x=df.index[-90:], 
        y=df['close'].tail(90), 
        mode="lines", 
        name="Historical Price",
        line=dict(color='blue', width=2)
    ))
    
    # Forecast data
    fig2.add_trace(go.Scatter(
        x=forecast_df["Date"], 
        y=forecast_df["Predicted_Price"], 
        mode="lines+markers", 
        name="30-Day Forecast",
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig2.update_layout(
        title=f"{selected_stock} - 30-Day Price Forecast",
        xaxis_title="Date", 
        yaxis_title="Price (₹)",
        height=500
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Show forecast table
    st.subheader("📋 Forecast Details")
    display_forecast = forecast_df.copy()
    display_forecast['Date'] = display_forecast['Date'].dt.strftime('%Y-%m-%d')
    display_forecast['Predicted_Price'] = display_forecast['Predicted_Price'].round(2)
    
    # Add day counter and changes
    display_forecast['Day'] = range(1, 31)
    display_forecast['Change'] = display_forecast['Predicted_Price'].diff()
    display_forecast['Change_Pct'] = (display_forecast['Change'] / display_forecast['Predicted_Price'].shift(1)) * 100
    
    st.dataframe(display_forecast[['Day', 'Date', 'Predicted_Price', 'Change', 'Change_Pct']].round(2))
    
    # Performance metrics
    st.subheader("📈 Model Performance")
    train_loss = history.history['loss'][-1]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Loss", f"{train_loss:.6f}")
    with col2:
        if len(df) >= 100:
            st.success("✅ Good data quality")
        else:
            st.warning("⚠️ Limited data")
    
    # Trading insights
    st.subheader("💡 Trading Insights")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.write("**Next Day Outlook:**")
        if pct_change > 2:
            st.success("Strong bullish momentum expected")
        elif pct_change > 0.5:
            st.info("Moderate upward movement expected")
        elif pct_change < -2:
            st.error("Strong bearish momentum expected")
        elif pct_change < -0.5:
            st.warning("Moderate downward movement expected")
        else:
            st.info("Sideways movement expected")
    
    with insight_col2:
        st.write("**30-Day Trend:**")
        overall_change = forecast_df['Predicted_Price'].iloc[-1] - forecast_df['Predicted_Price'].iloc[0]
        if overall_change > 0:
            st.success(f"Bullish trend: ₹{overall_change:+.2f}")
        else:
            st.error(f"Bearish trend: ₹{overall_change:+.2f}")
    
    st.info("""
    **💡 Disclaimer:** 
    This LSTM model predicts based on historical patterns. Stock markets are influenced by many factors 
    including news, economic data, and market sentiment. Use this as one of many tools for analysis.
    """)

else:
    st.info("👆 Select a stock and click 'Start Forecast' to begin prediction")
    
    # Show available symbols
    with st.expander("📋 Available Stocks"):
        st.write("Popular NSE stocks available for analysis:")
        cols = 3
        stocks_per_col = len(symbols) // cols + 1
        
        col1, col2, col3 = st.columns(3)
        for i, symbol in enumerate(symbols):
            with col1 if i < stocks_per_col else col2 if i < 2*stocks_per_col else col3:
                st.write(f"• {symbol}")