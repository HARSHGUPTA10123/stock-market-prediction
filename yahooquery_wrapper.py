from yahooquery import Ticker
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st

class YahooQueryWrapper:
    def __init__(self):
        self.cache = {}
    
    def Ticker(self, symbol):
        """Replace yf.Ticker() with this"""
        if symbol not in self.cache:
            self.cache[symbol] = Ticker(symbol)
        return self.cache[symbol]
    
    def download(self, symbol, start=None, end=None, period=None):
        """Replace yf.download() with this - returns same format"""
        ticker = self.Ticker(symbol)
        
        try:
            # Use period if provided, otherwise use start/end
            if period:
                history = ticker.history(period=period)
            else:
                history = ticker.history(start=start, end=end)
            
            if not history.empty:
                # Convert to same format as yfinance
                history = history.reset_index()
                
                # Rename columns to match yfinance format
                column_map = {
                    'date': 'Date',
                    'open': 'Open', 
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume',
                    'adjclose': 'Adj Close'
                }
                history = history.rename(columns=column_map)
                
                # Set Date as index to match yfinance
                if 'Date' in history.columns:
                    history = history.set_index('Date')
                
                return history
                
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")
        
        return pd.DataFrame()
    
    def get_info(self, symbol):
        """Get stock info similar to yfinance .info"""
        ticker = self.Ticker(symbol)
        try:
            # Combine multiple data sources
            info = {}
            
            # Summary details (current price data)
            summary = ticker.summary_detail
            if summary and symbol in summary:
                info.update(summary[symbol])
            
            # Company profile
            profile = ticker.asset_profile
            if profile and symbol in profile:
                info.update(profile[symbol])
            
            # Key statistics
            stats = ticker.key_stats
            if stats and symbol in stats:
                info.update(stats[symbol])
                
            return info
        except:
            return {}
    
    def get_financials(self, symbol, frequency='a'):
        """Get financial statements - returns same as yfinance"""
        ticker = self.Ticker(symbol)
        try:
            return ticker.income_statement(frequency=frequency)
        except:
            return pd.DataFrame()
    
    def get_balance_sheet(self, symbol, frequency='a'):
        """Get balance sheet - returns same as yfinance"""
        ticker = self.Ticker(symbol)
        try:
            return ticker.balance_sheet(frequency=frequency)
        except:
            return pd.DataFrame()
    
    def get_cashflow(self, symbol, frequency='a'):
        """Get cash flow - returns same as yfinance"""
        ticker = self.Ticker(symbol)
        try:
            return ticker.cash_flow(frequency=frequency)
        except:
            return pd.DataFrame()
    
    def get_quarterly_financials(self, symbol):
        """Get quarterly financials"""
        return self.get_financials(symbol, frequency='q')
    
    def get_annual_financials(self, symbol):
        """Get annual financials"""
        return self.get_financials(symbol, frequency='a')
    
    def get_actions(self, symbol):
        """Get dividends and splits"""
        ticker = self.Ticker(symbol)
        try:
            # Get dividend history
            dividends = ticker.dividend_history(start='2010-01-01')
            if not dividends.empty:
                return dividends
        except:
            pass
        return pd.DataFrame()

# Create global instance
yq = YahooQueryWrapper()