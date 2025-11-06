# import streamlit as st
# import yfinance as yf
# import datetime as dt
# import pandas as pd
# import plotly.graph_objects as go
# from typing import Optional, Dict, Any

# st.title('Fundamental Information')

# # Load symbols with error handling
# try:
#     csv = pd.read_csv('symbols.csv')
#     symbol = csv['Symbol'].tolist()
#     for i in range(0, len(symbol)):
#         symbol[i] = symbol[i] + ".NS"
# except Exception as e:
#     st.error(f"Error loading symbols: {e}")
#     symbol = []

# ticker = st.selectbox(
#     'Enter or Choose NSE listed Stock Symbol',
#     symbol if symbol else ["RELIANCE.NS"])

# stock = yf.Ticker(ticker)

# # Safe info access with fallbacks
# info: Dict[str, Any] = {}
# try:
#     stock_info = stock.info
#     if stock_info is not None and isinstance(stock_info, dict):
#         info = stock_info
# except Exception:
#     info = {}

# # Display company info with explicit checks
# if info:
#     st.subheader(info.get('longName', 'N/A'))
#     st.markdown(f"**Sector**: {info.get('sector', 'N/A')}")
#     st.markdown(f"**Industry**: {info.get('industry', 'N/A')}")
#     st.markdown(f"**Phone**: {info.get('phone', 'N/A')}")

#     # Safe address construction
#     address_parts = []
#     address1 = info.get('address1')
#     city = info.get('city')
#     zip_code = info.get('zip')
#     country = info.get('country')
    
#     if address1:
#         address_parts.append(str(address1))
#     if city:
#         address_parts.append(str(city))
#     if zip_code:
#         address_parts.append(str(zip_code))
#     if country:
#         address_parts.append(str(country))
        
#     address = ', '.join(address_parts) if address_parts else 'N/A'
#     st.markdown(f"**Address**: {address}")
#     st.markdown(f"**Website**: {info.get('website', 'N/A')}")

#     with st.expander('See detailed business summary'):
#         business_summary = info.get('longBusinessSummary', 'No business summary available.')
#         st.write(business_summary)
# else:
#     st.warning("No company information available for the selected stock.")

# # Getting data
# min_value = dt.datetime.today() - dt.timedelta(10 * 365)
# max_value = dt.datetime.today()

# start_input = st.date_input(
#     'Enter starting date',
#     value=dt.datetime.today() - dt.timedelta(90),
#     min_value=min_value, max_value=max_value, 
#     help='Enter the starting date from which you have to look the price'
# )

# end_input = st.date_input(
#     'Enter last date',
#     value=dt.datetime.today(),
#     min_value=min_value, max_value=max_value, 
#     help='Enter the last date till which you have to look the price'
# )

# hist_price: pd.DataFrame = pd.DataFrame()
# try:
#     # UPDATED: Use Ticker().history() instead of download()
#     price_data = stock.history(start=start_input, end=end_input)
#     if price_data is not None and not price_data.empty:
#         hist_price = price_data.reset_index()
#         # UPDATED: Keep as datetime for better plotting
#         if 'Date' in hist_price.columns:
#             hist_price['Date'] = pd.to_datetime(hist_price['Date'])
#     else:
#         st.warning("No data available for the selected date range.")
# except Exception as e:
#     st.error(f"Error downloading data: {e}")

# @st.cache_data
# def convert_df(df: pd.DataFrame) -> bytes:
#     return df.to_csv().encode('utf-8')

# if not hist_price.empty:
#     historical_csv = convert_df(hist_price)
#     st.download_button(
#         label="Download historical data as CSV",
#         data=historical_csv,
#         file_name='historical_df.csv',
#         mime='text/csv',
#     )

#     # Radio button to switch between style
#     chart = st.radio(
#         "Choose Style",
#         ('Candlestick', 'Line Chart'))
#     st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

#     if chart == 'Line Chart':
#         # UPDATED: Line chart with proper configuration
#         fig = go.Figure()
#         fig.add_trace(
#             go.Scatter(
#                 x=hist_price['Date'],
#                 y=hist_price['Close'],  # Use Close instead of Adj Close for consistency
#                 name='Closing price',
#                 line=dict(color='blue', width=2)
#             )
#         )
#         fig.update_layout(
#             title={
#                 'text': f'Stock Prices of {ticker}',
#                 'y': 0.9,
#                 'x': 0.5,
#                 'xanchor': 'center',
#                 'yanchor': 'top'
#             }, 
#             height=600, 
#             template='plotly_white',
#             xaxis_title='Date',
#             yaxis_title='Price (₹)',
#             yaxis=dict(tickprefix='₹')
#         )
#         st.plotly_chart(fig, use_container_width=True)

#     elif chart == 'Candlestick':
#         # UPDATED: Candlestick chart with proper configuration
#         fig = go.Figure()
#         fig.add_trace(
#             go.Candlestick(
#                 x=hist_price['Date'],
#                 open=hist_price['Open'],
#                 high=hist_price['High'],
#                 low=hist_price['Low'],
#                 close=hist_price['Close'],
#                 name='OHLC',
#                 increasing_line_color='green',
#                 decreasing_line_color='red'
#             )
#         )
#         fig.update_layout(
#             title={
#                 'text': f'Stock Prices of {ticker}',
#                 'y': 0.9,
#                 'x': 0.5,
#                 'xanchor': 'center',
#                 'yanchor': 'top'
#             }, 
#             height=600, 
#             template='plotly_white',
#             xaxis_title='Date',
#             yaxis_title='Price (₹)',
#             yaxis=dict(tickprefix='₹'),
#             xaxis_rangeslider_visible=False  # Hide range slider for cleaner look
#         )
#         st.plotly_chart(fig, use_container_width=True)

# # Helper function to process financial data with proper type handling
# def process_financial_data(data: Optional[pd.DataFrame], data_type: str) -> Optional[pd.DataFrame]:
#     """Process financial data with proper type checking"""
#     if data is None or not isinstance(data, pd.DataFrame) or data.empty:
#         return None
    
#     try:
#         # Create a copy to avoid modifying original
#         processed_data = data.copy()
        
#         # FIXED: Safe column date conversion without .date errors
#         try:
#             # Check if any column appears to be datetime-like
#             datetime_like_columns = False
#             for col in processed_data.columns:
#                 # Multiple checks to identify datetime objects
#                 if (hasattr(col, 'year') and hasattr(col, 'month') and hasattr(col, 'day')):
#                     datetime_like_columns = True
#                     break
            
#             if datetime_like_columns:
#                 new_columns = []
#                 for col in processed_data.columns:
#                     # Safe datetime detection and conversion
#                     if isinstance(col, (pd.Timestamp, dt.datetime)):
#                         try:
#                             # Convert to string date format instead of using .date()
#                             new_columns.append(col.strftime('%Y-%m-%d'))
#                         except (AttributeError, TypeError):
#                             new_columns.append(col)
#                     else:
#                         new_columns.append(col)
#                 processed_data.columns = new_columns
#         except Exception:
#             # If column conversion fails, continue with original columns
#             pass
        
#         # Remove rows with all NaN values
#         processed_data = processed_data.dropna(how='all')
        
#         if processed_data.empty:
#             return None
            
#         # Convert to numeric where possible
#         for col in processed_data.columns:
#             processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
        
#         # Remove columns that became all NaN after conversion
#         processed_data = processed_data.dropna(axis=1, how='all')
        
#         if processed_data.empty:
#             return None
            
#         # Format numbers with commas
#         for col in processed_data.columns:
#             processed_data[col] = processed_data[col].apply(
#                 lambda x: f"{x:,.0f}" if pd.notna(x) and isinstance(x, (int, float)) else str(x)
#             )
            
#         return processed_data
        
#     except Exception as e:
#         st.error(f"Error processing {data_type}: {e}")
#         return None

# # Quarterly results
# st.subheader('Quarterly Result')
# st.write('A quarterly result is a summary or collection of unaudited financial statements, such as balance sheets, income statements, and cash flow statements, issued by companies every quarter (three months).')

# try:
#     quarterly_data = stock.quarterly_financials
#     # FIX: Ensure we're passing a DataFrame or None
#     quarterly_df = quarterly_data if isinstance(quarterly_data, pd.DataFrame) else None
#     processed_quarterly = process_financial_data(quarterly_df, "quarterly results")
#     if processed_quarterly is not None and not processed_quarterly.empty:
#         st.dataframe(processed_quarterly.style.highlight_max(axis=1, color='lightgreen'))
#     else:
#         st.info("No quarterly financial data available.")
# except Exception as e:
#     st.error(f"Error loading quarterly results: {e}")

# # Profit and loss
# st.subheader('Profit & Loss')
# st.write("A profit and loss (P&L) statement is an annual financial report that provides a summary of a company's revenue, expenses and profit.")

# try:
#     financials_data = stock.financials
#     # FIX: Ensure we're passing a DataFrame or None
#     financials_df = financials_data if isinstance(financials_data, pd.DataFrame) else None
#     processed_financials = process_financial_data(financials_df, "financials")
#     if processed_financials is not None and not processed_financials.empty:
#         st.dataframe(processed_financials.style.highlight_max(axis=1, color='lightgreen'))
#     else:
#         st.info("No financial data available.")
# except Exception as e:
#     st.error(f"Error loading financials: {e}")

# # Balance sheet
# st.subheader('Balance Sheet')
# st.write("A balance sheet is a financial statement that reports a company's assets, liabilities, and shareholder equity.")

# try:
#     balance_data = stock.balance_sheet
#     # FIX: Ensure we're passing a DataFrame or None
#     balance_df = balance_data if isinstance(balance_data, pd.DataFrame) else None
#     processed_balance = process_financial_data(balance_df, "balance sheet")
#     if processed_balance is not None and not processed_balance.empty:
#         st.dataframe(processed_balance.style.highlight_max(axis=1, color='lightgreen'))
#     else:
#         st.info("No balance sheet data available.")
# except Exception as e:
#     st.error(f"Error loading balance sheet: {e}")

# # Cash flow
# st.subheader('Cash Flows')
# st.write("The term cash flow refers to the net amount of cash and cash equivalents being transferred in and out of a company.")

# try:
#     cashflow_data = stock.cashflow
#     # FIX: Ensure we're passing a DataFrame or None
#     cashflow_df = cashflow_data if isinstance(cashflow_data, pd.DataFrame) else None
#     processed_cashflow = process_financial_data(cashflow_df, "cash flow")
#     if processed_cashflow is not None and not processed_cashflow.empty:
#         st.dataframe(processed_cashflow.style.highlight_max(axis=1, color='lightgreen'))
#     else:
#         st.info("No cash flow data available.")
# except Exception as e:
#     st.error(f"Error loading cash flow: {e}")

# # Actions with error handling
# st.subheader('Splits & Dividends')
# st.write('Historical stock splits and dividend information.')

# try:
#     actions_data = stock.actions
#     # FIX 1: Check if actions_data is not empty using len() instead of .empty
#     if (actions_data is not None and 
#         isinstance(actions_data, pd.DataFrame) and 
#         len(actions_data) > 0):  # Fixed: Use len() instead of .empty for list checking
        
#         # Create a copy to avoid modifying original
#         actions_display = actions_data.copy()
        
#         # FIX 2: Safe index date conversion - check if index is datetime type
#         if hasattr(actions_display.index, 'strftime'):
#             try:
#                 # Convert datetime index to date objects using proper pandas method
#                 if pd.api.types.is_datetime64_any_dtype(actions_display.index):
#                     # FIX 3: Use proper pandas index assignment instead of list
#                     actions_display.index = pd.Index(pd.to_datetime(actions_display.index).date)
#             except (AttributeError, TypeError):
#                 pass  # If date conversion fails, keep original index
        
#         st.dataframe(actions_display, width=1000)
#     else:
#         st.info("No splits or dividend data available.")
# except Exception as e:
#     st.error(f"Error loading splits and dividends: {e}")

# # Add some spacing at the bottom
# st.markdown("---")
# st.caption("Data provided by Yahoo Finance")




# import streamlit as st
# import datetime as dt
# import pandas as pd
# import plotly.graph_objects as go
# from typing import Optional, Dict, Any
# from yahooquery import Ticker

# st.title('Fundamental Information')

# # Load symbols with error handling
# try:
#     csv = pd.read_csv('symbols.csv')
#     symbol = csv['Symbol'].tolist()
#     for i in range(0, len(symbol)):
#         symbol[i] = symbol[i] + ".NS"
# except Exception as e:
#     st.error(f"Error loading symbols: {e}")
#     symbol = []

# ticker = st.selectbox(
#     'Enter or Choose NSE listed Stock Symbol',
#     symbol if symbol else ["RELIANCE.NS"])

# # YahooQuery data fetching functions
# def get_yahooquery_data(symbol, start_date, end_date):
#     """Get historical data using YahooQuery"""
#     try:
#         stock = Ticker(symbol)
        
#         # Convert to string format for YahooQuery - use UTC timezone to avoid mixing
#         start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
#         end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
        
#         # Use period instead of start/end to avoid timezone issues
#         # Calculate days between dates for period
#         if hasattr(start_date, 'strftime') and hasattr(end_date, 'strftime'):
#             days_diff = (end_date - start_date).days
#             if days_diff > 0:
#                 # Use period-based approach to avoid timezone issues
#                 hist = stock.history(period=f"{days_diff}d")
#             else:
#                 hist = stock.history(period="3mo")  # fallback
#         else:
#             hist = stock.history(period="3mo")  # fallback
            
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
            
#             # FIX: Convert Date and remove timezone info
#             hist['Date'] = pd.to_datetime(hist['Date'])
#             if hist['Date'].dt.tz is not None:
#                 hist['Date'] = hist['Date'].dt.tz_convert('UTC').dt.tz_localize(None)
#             else:
#                 hist['Date'] = hist['Date'].dt.tz_localize(None)
                
#             # Filter by date range if needed
#             if hasattr(start_date, 'strftime') and hasattr(end_date, 'strftime'):
#                 start_dt = pd.Timestamp(start_date).tz_localize(None)
#                 end_dt = pd.Timestamp(end_date).tz_localize(None)
#                 hist = hist[(hist['Date'] >= start_dt) & (hist['Date'] <= end_dt)]
                
#             return hist
#     except Exception as e:
#         st.error(f"YahooQuery error: {e}")
#     return pd.DataFrame()

# def get_yahooquery_info(symbol):
#     """Get stock info using YahooQuery"""
#     try:
#         stock = Ticker(symbol)
#         info = {}
        
#         # Get summary details
#         summary = stock.summary_detail
#         if summary and symbol in summary:
#             summary_data = summary[symbol]
#             if isinstance(summary_data, dict):
#                 for key, value in summary_data.items():
#                     info[key] = value
            
#         # Get company profile  
#         profile = stock.asset_profile
#         if profile and symbol in profile:
#             profile_data = profile[symbol]
#             if isinstance(profile_data, dict):
#                 for key, value in profile_data.items():
#                     info[key] = value
            
#         # Get price info
#         price = stock.price
#         if price and symbol in price:
#             price_data = price[symbol]
#             if isinstance(price_data, dict):
#                 for key, value in price_data.items():
#                     info[key] = value
            
#         return info
#     except:
#         return {}

# def get_yahooquery_financials(symbol, financial_type):
#     """Get financial statements using YahooQuery"""
#     try:
#         stock = Ticker(symbol)
#         if financial_type == 'quarterly_financials':
#             return stock.income_statement(frequency='q')
#         elif financial_type == 'financials':
#             return stock.income_statement(frequency='a')
#         elif financial_type == 'balance_sheet':
#             return stock.balance_sheet(frequency='a')
#         elif financial_type == 'cashflow':
#             return stock.cash_flow(frequency='a')
#         elif financial_type == 'actions':
#             return stock.dividend_history(start='2010-01-01')
#     except:
#         return None

# # Safe info access with fallbacks
# info: Dict[str, Any] = {}
# try:
#     stock_info = get_yahooquery_info(ticker)
#     if stock_info is not None and isinstance(stock_info, dict):
#         info = stock_info
# except Exception:
#     info = {}

# # Display company info with explicit checks
# if info:
#     st.subheader(info.get('longName', 'N/A'))
#     st.markdown(f"**Sector**: {info.get('sector', 'N/A')}")
#     st.markdown(f"**Industry**: {info.get('industry', 'N/A')}")
#     st.markdown(f"**Phone**: {info.get('phone', 'N/A')}")

#     # Safe address construction
#     address_parts = []
#     address1 = info.get('address1')
#     city = info.get('city')
#     zip_code = info.get('zip')
#     country = info.get('country')
    
#     if address1:
#         address_parts.append(str(address1))
#     if city:
#         address_parts.append(str(city))
#     if zip_code:
#         address_parts.append(str(zip_code))
#     if country:
#         address_parts.append(str(country))
        
#     address = ', '.join(address_parts) if address_parts else 'N/A'
#     st.markdown(f"**Address**: {address}")
#     st.markdown(f"**Website**: {info.get('website', 'N/A')}")

#     with st.expander('See detailed business summary'):
#         business_summary = info.get('longBusinessSummary', 'No business summary available.')
#         st.write(business_summary)
# else:
#     st.warning("No company information available for the selected stock.")

# # Getting data - use date objects directly (no timezone issues)
# min_value = dt.date.today() - dt.timedelta(10 * 365)
# max_value = dt.date.today()

# start_input = st.date_input(
#     'Enter starting date',
#     value=dt.date.today() - dt.timedelta(90),
#     min_value=min_value, max_value=max_value, 
#     help='Enter the starting date from which you have to look the price'
# )

# end_input = st.date_input(
#     'Enter last date',
#     value=dt.date.today(),
#     min_value=min_value, max_value=max_value, 
#     help='Enter the last date till which you have to look the price'
# )

# hist_price: pd.DataFrame = pd.DataFrame()
# try:
#     # FIX: Use a simpler approach - get more data and filter locally
#     # Calculate days difference for period
#     days_diff = (end_input - start_input).days
#     if days_diff <= 0:
#         st.error("End date must be after start date")
#     else:
#         # Add buffer to ensure we get enough data
#         period_days = min(days_diff + 30, 365 * 5)  # Max 5 years
        
#         # Get data using period instead of specific dates
#         stock = Ticker(ticker)
#         hist = stock.history(period=f"{period_days}d")
        
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
            
#             # Convert Date and remove timezone info
#             hist['Date'] = pd.to_datetime(hist['Date'])
#             if hist['Date'].dt.tz is not None:
#                 hist['Date'] = hist['Date'].dt.tz_convert('UTC').dt.tz_localize(None)
#             else:
#                 hist['Date'] = hist['Date'].dt.tz_localize(None)
            
#             # Convert input dates to timezone-naive datetime for filtering
#             start_dt = pd.Timestamp(start_input).tz_localize(None)
#             end_dt = pd.Timestamp(end_input).tz_localize(None)
            
#             # Filter to requested date range
#             hist_price = hist[(hist['Date'] >= start_dt) & (hist['Date'] <= end_dt)]
            
#             if hist_price.empty:
#                 st.warning("No data available for the selected date range.")
#         else:
#             st.warning("No data available for the selected stock.")
            
# except Exception as e:
#     st.error(f"Error downloading data: {e}")

# @st.cache_data
# def convert_df(df: pd.DataFrame) -> bytes:
#     return df.to_csv().encode('utf-8')

# if not hist_price.empty:
#     historical_csv = convert_df(hist_price)
#     st.download_button(
#         label="Download historical data as CSV",
#         data=historical_csv,
#         file_name='historical_df.csv',
#         mime='text/csv',
#     )

#     # Radio button to switch between style
#     chart = st.radio(
#         "Choose Style",
#         ('Candlestick', 'Line Chart'))
#     st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

#     if chart == 'Line Chart':
#         # UPDATED: Line chart with proper configuration
#         fig = go.Figure()
#         fig.add_trace(
#             go.Scatter(
#                 x=hist_price['Date'],
#                 y=hist_price['Close'],  # Use Close instead of Adj Close for consistency
#                 name='Closing price',
#                 line=dict(color='blue', width=2)
#             )
#         )
#         fig.update_layout(
#             title={
#                 'text': f'Stock Prices of {ticker}',
#                 'y': 0.9,
#                 'x': 0.5,
#                 'xanchor': 'center',
#                 'yanchor': 'top'
#             }, 
#             height=600, 
#             template='plotly_white',
#             xaxis_title='Date',
#             yaxis_title='Price (₹)',
#             yaxis=dict(tickprefix='₹')
#         )
#         st.plotly_chart(fig, use_container_width=True)

#     elif chart == 'Candlestick':
#         # UPDATED: Candlestick chart with proper configuration
#         fig = go.Figure()
#         fig.add_trace(
#             go.Candlestick(
#                 x=hist_price['Date'],
#                 open=hist_price['Open'],
#                 high=hist_price['High'],
#                 low=hist_price['Low'],
#                 close=hist_price['Close'],
#                 name='OHLC',
#                 increasing_line_color='green',
#                 decreasing_line_color='red'
#             )
#         )
#         fig.update_layout(
#             title={
#                 'text': f'Stock Prices of {ticker}',
#                 'y': 0.9,
#                 'x': 0.5,
#                 'xanchor': 'center',
#                 'yanchor': 'top'
#             }, 
#             height=600, 
#             template='plotly_white',
#             xaxis_title='Date',
#             yaxis_title='Price (₹)',
#             yaxis=dict(tickprefix='₹'),
#             xaxis_rangeslider_visible=False  # Hide range slider for cleaner look
#         )
#         st.plotly_chart(fig, use_container_width=True)

# # [Rest of your code remains the same - quarterly results, financials, etc.]

# # Quarterly results
# st.subheader('Quarterly Result')
# st.write('A quarterly result is a summary or collection of unaudited financial statements, such as balance sheets, income statements, and cash flow statements, issued by companies every quarter (three months).')

# try:
#     # UPDATED: Use YahooQuery for quarterly data
#     quarterly_data = get_yahooquery_financials(ticker, 'quarterly_financials')
#     # FIX: Ensure we're passing a DataFrame or None
#     quarterly_df = quarterly_data if isinstance(quarterly_data, pd.DataFrame) else None
#     processed_quarterly = process_financial_data(quarterly_df, "quarterly results")
#     if processed_quarterly is not None and not processed_quarterly.empty:
#         st.dataframe(processed_quarterly)
#     else:
#         st.info("No quarterly financial data available.")
# except Exception as e:
#     st.error(f"Error loading quarterly results: {e}")

# # Profit and loss
# st.subheader('Profit & Loss')
# st.write("A profit and loss (P&L) statement is an annual financial report that provides a summary of a company's revenue, expenses and profit.")

# try:
#     # UPDATED: Use YahooQuery for financials
#     financials_data = get_yahooquery_financials(ticker, 'financials')
#     # FIX: Ensure we're passing a DataFrame or None
#     financials_df = financials_data if isinstance(financials_data, pd.DataFrame) else None
#     processed_financials = process_financial_data(financials_df, "financials")
#     if processed_financials is not None and not processed_financials.empty:
#         st.dataframe(processed_financials)
#     else:
#         st.info("No financial data available.")
# except Exception as e:
#     st.error(f"Error loading financials: {e}")

# # Balance sheet
# st.subheader('Balance Sheet')
# st.write("A balance sheet is a financial statement that reports a company's assets, liabilities, and shareholder equity.")

# try:
#     # UPDATED: Use YahooQuery for balance sheet
#     balance_data = get_yahooquery_financials(ticker, 'balance_sheet')
#     # FIX: Ensure we're passing a DataFrame or None
#     balance_df = balance_data if isinstance(balance_data, pd.DataFrame) else None
#     processed_balance = process_financial_data(balance_df, "balance sheet")
#     if processed_balance is not None and not processed_balance.empty:
#         st.dataframe(processed_balance)
#     else:
#         st.info("No balance sheet data available.")
# except Exception as e:
#     st.error(f"Error loading balance sheet: {e}")

# # Cash flow
# st.subheader('Cash Flows')
# st.write("The term cash flow refers to the net amount of cash and cash equivalents being transferred in and out of a company.")

# try:
#     # UPDATED: Use YahooQuery for cash flow
#     cashflow_data = get_yahooquery_financials(ticker, 'cashflow')
#     # FIX: Ensure we're passing a DataFrame or None
#     cashflow_df = cashflow_data if isinstance(cashflow_data, pd.DataFrame) else None
#     processed_cashflow = process_financial_data(cashflow_df, "cash flow")
#     if processed_cashflow is not None and not processed_cashflow.empty:
#         st.dataframe(processed_cashflow)
#     else:
#         st.info("No cash flow data available.")
# except Exception as e:
#     st.error(f"Error loading cash flow: {e}")

# # Actions with error handling
# st.subheader('Splits & Dividends')
# st.write('Historical stock splits and dividend information.')

# try:
#     # UPDATED: Use YahooQuery for actions
#     actions_data = get_yahooquery_financials(ticker, 'actions')
    
#     if actions_data is not None and isinstance(actions_data, pd.DataFrame) and not actions_data.empty:
#         # Create a copy to avoid modifying original
#         actions_display = actions_data.copy()
        
#         # FIX: Safe timezone handling for actions data
#         # Check if index is datetime and handle timezone
#         if pd.api.types.is_datetime64_any_dtype(actions_display.index):
#             # Convert to timezone-naive dates
#             actions_display.index = pd.to_datetime(actions_display.index).tz_localize(None)
        
#         st.dataframe(actions_display, width=1000)
#     else:
#         st.info("No splits or dividend data available.")
# except Exception as e:
#     st.error(f"Error loading splits and dividends: {e}")

# # Add some spacing at the bottom
# st.markdown("---")
# st.caption("Data provided by Yahoo Finance via YahooQuery")



import streamlit as st
import pandas as pd
import datetime as dt
import numpy as np
import plotly.graph_objects as go
from typing import Optional, Dict, Any
from yahooquery import Ticker

st.set_page_config(page_title='Fundamental Information', layout='wide')

st.title('Fundamental Information')
st.write('Explore company fundamentals (financial statements, actions, and historical prices) fetched via YahooQuery.')

# ----------------- Helpers & Data loaders -----------------

@st.cache_data
def load_symbols(path: str = 'symbols.csv') -> list:
    try:
        csv = pd.read_csv(path)
        return [str(s).strip() + '.NS' for s in csv['Symbol'].tolist()]
    except Exception:
        return []


symbols = load_symbols()

# Select ticker
default = symbols[0] if symbols else 'RELIANCE.NS'

ticker = st.selectbox('Enter or choose NSE listed stock symbol', symbols if symbols else [default], index=0 if symbols else 0)

# YahooQuery wrappers
@st.cache_data
def get_yahoo_info(symbol: str) -> Dict[str, Any]:
    try:
        stock = Ticker(symbol)
        info: Dict[str, Any] = {}

        # summary_detail, asset_profile, price, key_stats
        for attr in ['summary_detail', 'asset_profile', 'price', 'key_stats']:
            try:
                val = getattr(stock, attr)
                if isinstance(val, dict) and symbol in val and isinstance(val[symbol], dict):
                    info.update(val[symbol])
            except Exception:
                continue
        return info
    except Exception:
        return {}

@st.cache_data
def get_history_by_period(symbol: str, period_days: int = 365*2) -> pd.DataFrame:
    """Fetch historical prices using a period (avoids tz start/end conflicts), return tz-naive Date column."""
    try:
        stock = Ticker(symbol)
        hist = stock.history(period=f"{period_days}d")
        if hist is None or (hasattr(hist, 'empty') and hist.empty):
            return pd.DataFrame()
        hist = hist.reset_index()
        hist.rename(columns={
            'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume', 'adjclose': 'Adj Close'
        }, inplace=True)
        if 'Adj Close' not in hist.columns and 'Close' in hist.columns:
            hist['Adj Close'] = hist['Close']

        # Force tz-aware -> convert -> drop tz info (robust)
        hist['Date'] = pd.to_datetime(hist['Date'], utc=True).dt.tz_convert('UTC').dt.tz_localize(None)
        return hist
    except Exception:
        return pd.DataFrame()

@st.cache_data
def get_actions(symbol: str) -> Optional[pd.DataFrame]:
    try:
        stock = Ticker(symbol)
        df = stock.dividend_history(start='2010-01-01')
        if df is None or (hasattr(df, 'empty') and df.empty):
            return pd.DataFrame()
        # If index is datetime, reset index
        df = df.copy()
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df.rename(columns={'index': 'Date'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_convert('UTC').dt.tz_localize(None)
        # Ensure output is always DataFrame
        if isinstance(df, pd.Series):
            df = df.to_frame().reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data
@st.cache_data
def get_financial_statement(symbol: str, statement: str = 'income', frequency: str = 'a') -> Optional[pd.DataFrame]:
    """statement: 'income'|'balance'|'cash' ; frequency: 'a' (annual) or 'q' (quarterly)"""
    try:
        stock = Ticker(symbol)

        if statement == 'income':
            df = stock.income_statement(frequency=frequency)
        elif statement == 'balance':
            df = stock.balance_sheet(frequency=frequency)
        elif statement == 'cash':
            df = stock.cash_flow(frequency=frequency)
        else:
            return pd.DataFrame()

        if df is None or (hasattr(df, 'empty') and df.empty):
            return pd.DataFrame()

        df = df.copy()

        # Normalize date columns if present
        for col in ['asOfDate', 'period', 'endDate', 'date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True).dt.tz_convert('UTC').dt.tz_localize(None)

        return df

    except Exception:
        return pd.DataFrame()


# ----- Utility to convert financials to long (tidy) format -----

def process_financial_data(fin_df: Optional[pd.DataFrame], label: str) -> Optional[pd.DataFrame]:
    """Convert various yahooquery financial outputs to a tidy long DataFrame.
    Returns columns: ['statement_date','item','value'] where statement_date is datetime or string.
    If input is None or empty -> return None.
    """
    if fin_df is None or (isinstance(fin_df, pd.DataFrame) and fin_df.empty):
        return None

    df = fin_df.copy()

    # If wide table with periods as columns (common), transpose
    # We try several heuristics to find numeric data and dates
    # If there's a column named 'endDate' or 'date' or 'period', use it
    lower_cols = [c.lower() for c in df.columns]
    date_col = None
    for c in ['enddate', 'date', 'period', 'reporteddate']:
        if c in lower_cols:
            date_col = df.columns[lower_cols.index(c)]
            break

    # If date_col is present, melt others
    if date_col is not None:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            id_vars = [date_col]
            value_vars = [c for c in df.columns if c != date_col]
            long = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='item', value_name='value')
            long.rename(columns={date_col: 'statement_date'}, inplace=True)
            long = long.dropna(subset=['value'])
            return long[['statement_date', 'item', 'value']]
        except Exception:
            pass

    # If index looks like dates (e.g., periods are index), try to transpose
    try:
        if isinstance(df.index, pd.Index) and any(isinstance(i, (str,)) for i in df.index):
            # try transpose
            t = df.T
            t = t.reset_index()
            t.rename(columns={'index': 'item'}, inplace=True)
            # find columns that look like dates
            date_cols = [c for c in t.columns if pd.api.types.is_datetime64_any_dtype(t[c]) or str(c).startswith('20')]
            if date_cols:
                # melt all others
                value_cols = [c for c in t.columns if c not in ['item']]
                long = t.melt(id_vars=['item'], value_vars=value_cols, var_name='statement_date', value_name='value')
                # coerce statement_date
                long['statement_date'] = pd.to_datetime(long['statement_date'], errors='coerce')
                return long[['statement_date', 'item', 'value']]
    except Exception:
        pass

    # As a final fallback: try to coerce numeric columns and melt them
    try:
        # keep only columns that have numeric values
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            long = df[numeric_cols].reset_index().melt(id_vars=['index'] if 'index' in df.reset_index().columns else [], value_vars=numeric_cols, var_name='item', value_name='value')
            # try to rename index to statement_date
            if 'index' in long.columns:
                long.rename(columns={'index': 'statement_date'}, inplace=True)
            return long[['statement_date', 'item', 'value']]
    except Exception:
        pass

    return None

# ----------------- Display company info -----------------
info = get_yahoo_info(ticker)

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
    st.warning('No company overview available')

# ----------------- Price download & chart -----------------
min_value = dt.date.today() - dt.timedelta(days=10*365)
max_value = dt.date.today()

start_input = st.date_input('Start date', value=dt.date.today() - dt.timedelta(days=90), min_value=min_value, max_value=max_value)
end_input = st.date_input('End date', value=dt.date.today(), min_value=min_value, max_value=max_value)

if end_input < start_input:
    st.error('End date must be after start date')
else:
    # Fetch a bit more history and filter locally
    days_diff = (end_input - start_input).days
    period_days = min(days_diff + 30, 365*10)
    hist = get_history_by_period(ticker, period_days)

    if hist.empty:
        st.error('No historical price data available')
    else:
        # filter
        start_dt = pd.Timestamp(start_input).tz_localize(None)
        end_dt = pd.Timestamp(end_input).tz_localize(None)
        hist_price = hist[(hist['Date'] >= start_dt) & (hist['Date'] <= end_dt)]

        if hist_price.empty:
            st.warning('No data in selected range')
        else:
            csv_bytes = hist_price.to_csv(index=False).encode('utf-8')
            st.download_button('Download historical data (CSV)', csv_bytes, file_name='historical.csv')

        chart = st.radio('Chart Style', ('Line','Candlestick'))
        if chart == 'Line':
            fig = go.Figure(data=[go.Scatter(x=hist_price['Date'], y=hist_price['Close'], mode='lines')])
        else:
            fig = go.Figure(data=[go.Candlestick(
                x=hist_price['Date'],
                open=hist_price['Open'],
                high=hist_price['High'],
                low=hist_price['Low'],
                close=hist_price['Close'],
                increasing_line_color='green',
                decreasing_line_color='red',
                increasing_fillcolor='green',
                decreasing_fillcolor='red'
            )])
        fig.update_layout(height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)


# ----------------- Financial statements (tidy/long format) -----------------
st.markdown('---')
st.subheader('Financial Statements (tidy/long format)')

with st.spinner('Fetching financial statements...'):
    inc_q = get_financial_statement(ticker, 'income', frequency='q')
    inc_a = get_financial_statement(ticker, 'income', frequency='a')
    bal_a = get_financial_statement(ticker, 'balance', frequency='a')
    cash_a = get_financial_statement(ticker, 'cash', frequency='a')

# Process into long format
inc_q_long = process_financial_data(inc_q, 'quarterly_income')
inc_a_long = process_financial_data(inc_a, 'annual_income')
bal_a_long = process_financial_data(bal_a, 'annual_balance')
cash_a_long = process_financial_data(cash_a, 'annual_cash')

def show_long(df_long, title):
    if df_long is None or df_long.empty:
        st.info(f'No data available for {title}')
        return
    # Ensure columns
    df_long = df_long.copy()
    st.write(f'### {title}')
    # Convert statement_date to string for better display if datetime
    if 'statement_date' in df_long.columns:
        df_long['statement_date'] = df_long['statement_date'].astype(str)
    st.dataframe(df_long, use_container_width=True)

show_long(inc_q_long, 'Quarterly Income (long)')
show_long(inc_a_long, 'Annual Income (long)')
show_long(bal_a_long, 'Annual Balance Sheet (long)')
show_long(cash_a_long, 'Annual Cash Flow (long)')

# ----------------- Actions (dividends / splits) -----------------
st.markdown('---')
st.subheader('Splits & Dividends')
acts = get_actions(ticker)
if acts is None or acts.empty:
    st.info('No splits/dividends available')
else:
    # Normalize Date column if present
    if 'Date' in acts.columns:
        acts['Date'] = pd.to_datetime(acts['Date'], errors='coerce').dt.tz_localize(None)
    st.dataframe(acts, use_container_width=True)

st.caption('Data provided by Yahoo Finance via YahooQuery')
