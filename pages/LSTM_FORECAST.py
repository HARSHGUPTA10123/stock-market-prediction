import pandas as pd
import numpy as np
import tensorflow as tf
import random as rn
import yfinance as yf
import streamlit as st
import datetime as dt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------
# Streamlit Title
# -----------------------------
st.subheader("Next-Day Forecasting with Long-Short Term Memory (LSTM)")

# For reproducibility
np.random.seed(1)
tf.random.set_seed(1)
rn.seed(1)

# -----------------------------
# Load NSE Symbols
# -----------------------------
csv = pd.read_csv("symbols.csv")
symbol = [s + ".NS" for s in csv["Symbol"].tolist()]
ticker = st.selectbox("Select NSE Stock Symbol", symbol, index=0)

# -----------------------------
# LSTM Function
# -----------------------------
def my_LSTM(ticker):
    try:
        start = dt.datetime.today() - dt.timedelta(5*365)
        end = dt.datetime.today()

        df = yf.download(ticker, start=start, end=end, progress=False)
        if df is None or df.empty:
            st.error(f"No data available for {ticker}")
            return
        df = df.reset_index()




        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.dropna(subset=["Open", "High", "Low", "Close"])  # remove bad rows

        if len(df) < 200:
            st.error("Not enough data to train the model. Try another stock.")
            return

        st.info("⏳ Training the LSTM model... please wait...")

        # -----------------------------
        # Prepare Data
        # -----------------------------
        close_data = df[["Date", "Close"]].copy()
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_close = scaler.fit_transform(close_data[["Close"]])

        train_size = int(len(scaled_close) * 0.8)
        train_data = scaled_close[:train_size]
        test_data = scaled_close[train_size - 60:]

        x_train, y_train = [], []
        for i in range(60, len(train_data)):
            x_train.append(train_data[i - 60:i, 0])
            y_train.append(train_data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # -----------------------------
        # LSTM Model
        # -----------------------------
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mean_squared_error")

        model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)       #type:ignore
        st.success("✅ Model trained successfully!")

        # -----------------------------
        # Predictions
        # -----------------------------
        X_test = []
        for i in range(60, len(test_data)):
            X_test.append(test_data[i - 60:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)

        valid = df.iloc[train_size:].copy().reset_index(drop=True)
        valid["Predictions"] = predictions.flatten()[:len(valid)]

        # -----------------------------
        # ✅ Professional Candlestick Chart
        # -----------------------------
        st.markdown("### 📊 LSTM Forecast — Candlestick + Predicted Close")

        # Ensure Date is datetime (for Plotly)
        df["Date"] = pd.to_datetime(df["Date"])

        # Use last 250 days for visual clarity
        recent_data = df.tail(250).copy()

        fig = go.Figure()

        # Actual candlestick (this will show now)
        fig.add_trace(
            go.Candlestick(
                x=recent_data["Date"],
                open=recent_data["Open"],
                high=recent_data["High"],
                low=recent_data["Low"],
                close=recent_data["Close"],
                name="Actual Price",
                increasing_line_color="lime",
                decreasing_line_color="red",
                showlegend=True
            )
        )

        # Predicted trend line
        fig.add_trace(
            go.Scatter(
                x=valid["Date"],
                y=valid["Predictions"],
                mode="lines",
                name="Predicted Close",
                line=dict(color="deepskyblue", width=2.3, dash="dot")
            )
        )

        fig.update_layout(
            template="plotly_dark",
            title=dict(
                text=f"{ticker} — LSTM Predicted vs Actual Prices",
                x=0.5,
                font=dict(size=18)
            ),
            xaxis=dict(
                title="Date",
                rangeslider_visible=False,
                showgrid=False
            ),
            yaxis=dict(
                title="Price (₹)",
                showgrid=True,
                gridcolor="rgba(128,128,128,0.2)"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            hovermode="x unified",
            height=700,
            plot_bgcolor="rgba(10,10,10,1)",
            paper_bgcolor="rgba(10,10,10,1)"
        )

        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # Model Metrics
        # -----------------------------
        # --- Fix type issue for sklearn metrics ---
        actual = valid["Close"].to_numpy(dtype=float)
        predicted = valid["Predictions"].to_numpy(dtype=float)

        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)


        st.markdown("### 📈 Model Performance Metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"₹{mae:.2f}")
        c2.metric("RMSE", f"₹{rmse:.2f}")
        c3.metric("R²", f"{r2:.4f}")

        # -----------------------------
        # Next Day Forecast
        # -----------------------------
        last_60 = scaled_close[-60:]
        X_forecast = np.reshape(last_60, (1, 60, 1))
        next_day_scaled = model.predict(X_forecast)
        next_day_price = scaler.inverse_transform(next_day_scaled)[0][0]

        current_price = df["Close"].iloc[-1]
        change = next_day_price - current_price
        change_pct = (change / current_price) * 100

        st.markdown("### 🔮 Next-Day Forecast")
        c1, c2, c3 = st.columns(3)
        c1.metric("Current Price", f"₹{current_price:.2f}")
        c2.metric("Predicted Next Day", f"₹{next_day_price:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
        c3.metric("Signal", "📈 Bullish" if change > 0 else "📉 Bearish")

        st.caption(f"Data Period: {df['Date'].min().date()} → {df['Date'].max().date()}")

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Try another symbol (like RELIANCE.NS or TCS.NS).")

# -----------------------------
# Run App
# -----------------------------
my_LSTM(ticker)
