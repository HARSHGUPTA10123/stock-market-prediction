
# 📈 Stock Market Prediction & Analysis Platform

A comprehensive, AI-powered web application for stock market analysis, technical screening, and price prediction. Built specifically for the National Stock Exchange (NSE) of India, this platform provides retail investors with professional-grade tools for data-driven decision making.

## ✨ Features

### 🔍 Fundamental Analysis
- **Company Profiles**: Detailed corporate information and business overview
- **Historical Data**: 10+ years of OHLC (Open, High, Low, Close) data with interactive charts
- **Financial Metrics**: Key performance indicators and financial ratios
- **Data Export**: Download historical prices in CSV and Excel formats

### 📊 Technical Analysis
- **11+ Technical Indicators**:
  - Moving Averages (SMA, EMA)
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
  - Bollinger Bands
  - Stochastic Oscillator
  - Average True Range (ATR)
  - Williams %R
  - And more...

### 🎯 Technical Screener
- **Breakout Detection**: Identify stocks breaking out of consolidation patterns
- **Multi-Parameter Filtering**: Screen based on volume, price movements, and technical levels
- **Sector Analysis**: Compare stocks within the same industry
- **Custom Alerts**: Personalized screening criteria

### 🔍 Pattern Recognition
- **60+ Candlestick Patterns**:
  - Bullish Patterns: Hammer, Bullish Engulfing, Morning Star
  - Bearish Patterns: Shooting Star, Bearish Engulfing, Evening Star
  - Continuation Patterns: Flags, Pennants, Triangles
- **Real-time Scanning**: Automated pattern detection across all NSE stocks
- **Confidence Scoring**: Probability assessment with historical accuracy

### 🤖 AI Price Forecasting
- **LSTM Neural Network**: Advanced deep learning model for time-series prediction
- **Next-Day Prediction**: Forecasts tomorrow's closing price
- **60-Day Lookback**: Analyzes past 60 days of market data
- **Proven Accuracy**: Lowest RMSE among tested models

## 🛠 Tech Stack

### Backend & Machine Learning
- **Python 3.9+**: Core programming language
- **TensorFlow/Keras**: Deep learning framework for LSTM model
- **Scikit-learn**: Traditional machine learning algorithms
- **Pandas & NumPy**: Data manipulation and numerical computing

### Data & Analysis
- **yFinance**: Real-time and historical stock data
- **TA-Lib**: Technical analysis library
- **FinTA**: Financial technical analysis indicators
- **Plotly**: Interactive data visualization

### Frontend & Deployment
- **Streamlit**: Web application framework
- **Custom CSS**: Responsive and modern UI design
- **Plotly Charts**: Interactive and dynamic visualizations

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)

### Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/HARSHGUPTA10123/stock-market-prediction.git
   cd stock-market-prediction
   ```

2. **Create Virtual Environment (Recommended)**
   ```bash
   python -m venv stock_env
   source stock_env/bin/activate  # On Windows: stock_env\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   streamlit run webapp.py
   ```

5. **Access the Application**
   - Open your web browser
   - Navigate to `http://localhost:8501`
   - Start analyzing stocks!

## 📈 Model Performance

Our LSTM model outperformed traditional machine learning approaches:

| Model | Large-Cap (TCS) | Mid-Cap (Tata Motors) | Small-Cap (Trident) |
|-------|-----------------|----------------------|---------------------|
| Moving Average | 971.40 | 234.64 | 23.10 |
| K-Nearest Neighbors | 1174.90 | 232.54 | 23.02 |
| Linear Regression | 680.51 | 400.30 | 24.51 |
| **LSTM (Our Model)** | **117.49** | **24.47** | **2.88** |

*Lower RMSE values indicate better prediction accuracy*

## 💡 Usage Guide

### Getting Started
1. **Launch the application** using `streamlit run webapp.py`
2. **Select a stock** from 1,770+ NSE listed companies
3. **Choose analysis period** (default: 1 year historical data)
4. **Click "Analyze Stock"** to load data and generate insights

### Feature Navigation
- **Home**: Overview and stock selection
- **Fundamental Analysis**: Company data and financials
- **Technical Indicators**: Chart-based technical analysis
- **Pattern Recognition**: Candlestick pattern detection
- **Next-Day Forecasting**: AI-powered price predictions

## 📁 Project Structure

```
stock-market-prediction/
│
├── webapp.py                 # Main application entry point
├── requirements.txt          # Python dependencies
├── symbols.csv              # NSE stock symbols database
│
├── pages/                   # Streamlit multi-page modules
│   ├── 01_Fundamental_Analysis.py
│   ├── 02_Technical_Indicators.py
│   ├── 03_Pattern_Recognition.py
│   ├── 04_Technical_Screener.py
│   └── 05_Next-Day_Forecasting.py
│
├── functions.py             # Utility functions and helpers
├── patterns.py              # Candlestick pattern definitions
│
└── media/                   # Images and documentation assets
    ├── model_performance/
    └── screenshots/
```

## 🔧 Configuration

### Stock Data Source
- **Primary**: Yahoo Finance API
- **Coverage**: 1,770+ NSE listed companies
- **Historical Data**: 10+ years of daily OHLC prices
- **Update Frequency**: Real-time during market hours

### Model Settings
- **Training Period**: 5 years of historical data
- **Lookback Window**: 60 days for prediction
- **Update Schedule**: Model retrained periodically
- **Confidence Intervals**: Provided with all predictions

## 🐛 Troubleshooting

### Common Issues

1. **TA-Lib Installation Error**
   ```bash
   # Windows users may need to download pre-compiled wheel
   pip install TA_Lib-0.4.24-cp39-cp39-win_amd64.whl
   ```

2. **Memory Issues with Large Datasets**
   - Reduce the analysis period
   - Clear browser cache
   - Restart the application

3. **Data Fetching Errors**
   - Check internet connection
   - Verify stock symbol format (e.g., RELIANCE.NS)
   - Try alternative stock symbols

### Getting Help
- Check the console for error messages
- Verify all dependencies are installed
- Ensure Python version compatibility

## 🚀 Future Enhancements

- [ ] Real-time intraday data integration
- [ ] Additional timeframe analysis (weekly, monthly)
- [ ] Portfolio management features
- [ ] Social sentiment analysis
- [ ] Options chain analysis
- [ ] Backtesting framework
- [ ] Mobile application version

## 📊 Data Sources & Disclaimer

### Data Providers
- **Primary**: Yahoo Finance API
- **Exchange**: National Stock Exchange (NSE) India
- **Coverage**: All NSE equity segments

### Important Disclaimer
⚠️ **This application is for educational and research purposes only.** Stock market investments are subject to market risks. The AI predictions and technical analysis should not be considered as financial advice. Always consult with qualified financial advisors before making investment decisions.

## 👨‍💻 Developer

**Harsh Gupta**
- GitHub: [@HARSHGUPTA10123](https://github.com/HARSHGUPTA10123)
- Project: Stock Market Prediction & Analysis Platform

---

**Ready to transform your trading strategy? Clone the repository and start analyzing today! 🚀**
```

