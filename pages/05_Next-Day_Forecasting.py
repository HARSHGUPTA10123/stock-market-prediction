import pandas as pd
import numpy as np
import tensorflow as tf
import random as rn
import yfinance as yf
import streamlit as st
import datetime as dt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
np.random.seed(1)
tf.random.set_seed(1)
rn.seed(1)
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.subheader('Next-Day Forecasting with Long-Short Term Memory (LSTM)')

csv = pd.read_csv('symbols.csv')
symbol = csv['Symbol'].tolist()
for i in range(0, len(symbol)):
    symbol[i] = symbol[i] + ".NS"

# creating sidebar
ticker = st.selectbox(
    'Enter or Choose NSE listed Stock Symbol',
    symbol, index=0)

def my_LSTM(ticker):
    try:
        start = dt.datetime.today() - dt.timedelta(5*365)
        end = dt.datetime.today()

        df = yf.download(ticker, start=start, end=end, progress=False)
        
        # Validate data
        if df is None or df.empty:
            st.error(f"No data available for {ticker}")
            return
            
        df = df.reset_index()
        
        # Check if Date column exists and process it
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.date
        else:
            # If no Date column, use the index
            df['Date'] = df.index
            df['Date'] = pd.to_datetime(df['Date']).dt.date
        
        # Check if we have enough data
        if len(df) < 100:
            st.error(f"Not enough historical data for {ticker}. Need at least 100 days.")
            return
            
        st.write('It will take some seconds to fit the model....')
        
        # Use Close price for LSTM
        data = df.sort_index(ascending=True, axis=0)
        new_data = pd.DataFrame(index=range(0, len(df)), columns=['Close'])
        new_data['Close'] = data['Close'].values
        
        # Use dynamic split instead of hardcoded 987
        split_ratio = 0.8
        split_index = int(len(new_data) * split_ratio)
        
        # creating train and test sets
        dataset = new_data.values
        train = dataset[0:split_index, :]
        valid = dataset[split_index:, :]

        # converting dataset into x_train and y_train
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        x_train, y_train = [], []
        for i in range(60, len(train)):
            x_train.append(scaled_data[i-60:i, 0])
            y_train.append(scaled_data[i, 0])
            
        if len(x_train) == 0:
            st.error("Not enough data for training")
            return
            
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        # create and fit the LSTM network with unique names
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1), name='lstm_1'))
        model.add(Dropout(0.2, name='dropout_1'))
        model.add(LSTM(units=50, return_sequences=False, name='lstm_2'))
        model.add(Dropout(0.2, name='dropout_2'))
        model.add(Dense(1, name='dense_output'))

        model.compile(loss='mean_squared_error', optimizer='adam')
        
        with st.spinner('Training LSTM model...'):
            model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0, validation_split=0.1) #type:ignore
            
        st.success('Model Fitted')
        
        # predicting values using past 60 days
        inputs = new_data[len(new_data) - len(valid) - 60:].values
        inputs = inputs.reshape(-1, 1)
        inputs = scaler.transform(inputs)

        X_test = []
        for i in range(60, inputs.shape[0]):
            X_test.append(inputs[i-60:i, 0])
        X_test = np.array(X_test)

        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        closing_price = model.predict(X_test)
        closing_price = scaler.inverse_transform(closing_price)

        # for plotting
        train_data = data[:split_index]
        valid_data = data[split_index:].copy()
        
        # Ensure lengths match
        min_len = min(len(valid_data), len(closing_price))
        valid_data = valid_data.iloc[:min_len]
        closing_price = closing_price[:min_len]
        
        # FIX: Ensure predictions is 1D array
        valid_data['Predictions'] = closing_price.flatten()

        st.write('#### Actual VS Predicted Prices')

        fig_preds = go.Figure()
        fig_preds.add_trace(
            go.Scatter(
                x=train_data['Date'],
                y=train_data['Close'],
                name='Training data Closing price',
                line=dict(color='blue')
            )
        )

        fig_preds.add_trace(
            go.Scatter(
                x=valid_data['Date'],
                y=valid_data['Close'],
                name='Validation data Closing price',
                line=dict(color='green')
            )
        )

        fig_preds.add_trace(
            go.Scatter(
                x=valid_data['Date'],
                y=valid_data['Predictions'],
                name='Predicted Closing price',
                line=dict(color='red', dash='dash')
            )
        )

        fig_preds.update_layout(
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ), 
            height=600, 
            title_text=f'Predictions for {ticker}',
            template='plotly_white',
            xaxis_title='Date',
            yaxis_title='Price (₹)'
        )

        st.plotly_chart(fig_preds, use_container_width=True)

        # FIXED: metrics calculation with proper type conversion
        if len(closing_price) > 0 and len(valid_data) > 0:
            # Convert to numpy arrays explicitly and ensure proper shapes
            actual_values = np.array(valid_data['Close'].iloc[:len(closing_price)], dtype=float)
            predicted_values = np.array(closing_price, dtype=float).flatten()
            
            # Ensure both arrays have the same length
            min_length = min(len(actual_values), len(predicted_values))
            actual_values = actual_values[:min_length]
            predicted_values = predicted_values[:min_length]
            
            # Calculate metrics - convert to native Python floats
            mae_val = mean_absolute_error(actual_values, predicted_values)
            rmse_val = np.sqrt(mean_squared_error(actual_values, predicted_values))
            r2_val = r2_score(actual_values, predicted_values)
            
            # Convert to simple floats to avoid type issues
            mae = float(mae_val)
            rmse = float(rmse_val)
            r2 = float(r2_val)

            with st.container():
                st.write('#### Model Performance Metrics')
                col_11, col_22, col_33 = st.columns(3)
                col_11.metric('Mean Absolute Error', f'₹ {mae:.2f}')
                col_22.metric('Root Mean Squared Error', f'₹ {rmse:.2f}')
                col_33.metric('R² Score', f'{r2:.4f}')

        # FIXED: Next-day forecasting
        last_60_days = scaled_data[-60:].flatten()  # Ensure 1D array
        real_data = last_60_days.reshape(1, 60, 1)
        
        prediction = model.predict(real_data)
        prediction = scaler.inverse_transform(prediction)
        next_day_price = float(prediction[0][0])
        
        current_price = float(df['Close'].iloc[-1])
        price_change = next_day_price - current_price
        change_percent = (price_change / current_price) * 100

        st.write('#### Next-Day Forecasting')
        
        with st.container():
            col_111, col_222, col_333 = st.columns(3)
            col_111.metric('Current Price', f'₹ {current_price:.2f}')
            col_222.metric(
                'Predicted Next Day Price', 
                f'₹ {next_day_price:.2f}',
                delta=f"{price_change:+.2f} ({change_percent:+.2f}%)",
                delta_color="normal" if price_change < 0 else "normal"
            )
            col_333.metric(
                'Prediction Direction',
                '📈 Bullish' if price_change > 0 else '📉 Bearish'
            )

    except Exception as e:
        st.error(f"Error in LSTM forecasting: {str(e)}")
        st.info("Try selecting a stock with more historical data like RELIANCE.NS or TCS.NS")

my_LSTM(ticker)