import streamlit as st
import yfinance as yf
import datetime as dt
import pandas as pd
import plotly.graph_objects as go
from typing import Optional, Dict, Any

st.title('Fundamental Information')

# Load symbols with error handling
try:
    csv = pd.read_csv('symbols.csv')
    symbol = csv['Symbol'].tolist()
    for i in range(0, len(symbol)):
        symbol[i] = symbol[i] + ".NS"
except Exception as e:
    st.error(f"Error loading symbols: {e}")
    symbol = []

ticker = st.selectbox(
    'Enter or Choose NSE listed Stock Symbol',
    symbol if symbol else ["RELIANCE.NS"])

stock = yf.Ticker(ticker)

# Safe info access with fallbacks
info: Dict[str, Any] = {}
try:
    stock_info = stock.info
    if stock_info is not None and isinstance(stock_info, dict):
        info = stock_info
except Exception:
    info = {}

# Display company info with explicit checks
if info:
    st.subheader(info.get('longName', 'N/A'))
    st.markdown(f"**Sector**: {info.get('sector', 'N/A')}")
    st.markdown(f"**Industry**: {info.get('industry', 'N/A')}")
    st.markdown(f"**Phone**: {info.get('phone', 'N/A')}")

    # Safe address construction
    address_parts = []
    address1 = info.get('address1')
    city = info.get('city')
    zip_code = info.get('zip')
    country = info.get('country')
    
    if address1:
        address_parts.append(str(address1))
    if city:
        address_parts.append(str(city))
    if zip_code:
        address_parts.append(str(zip_code))
    if country:
        address_parts.append(str(country))
        
    address = ', '.join(address_parts) if address_parts else 'N/A'
    st.markdown(f"**Address**: {address}")
    st.markdown(f"**Website**: {info.get('website', 'N/A')}")

    with st.expander('See detailed business summary'):
        business_summary = info.get('longBusinessSummary', 'No business summary available.')
        st.write(business_summary)
else:
    st.warning("No company information available for the selected stock.")

# Getting data
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

hist_price: pd.DataFrame = pd.DataFrame()
try:
    # UPDATED: Use Ticker().history() instead of download()
    price_data = stock.history(start=start_input, end=end_input)
    if price_data is not None and not price_data.empty:
        hist_price = price_data.reset_index()
        # UPDATED: Keep as datetime for better plotting
        if 'Date' in hist_price.columns:
            hist_price['Date'] = pd.to_datetime(hist_price['Date'])
    else:
        st.warning("No data available for the selected date range.")
except Exception as e:
    st.error(f"Error downloading data: {e}")

@st.cache_data
def convert_df(df: pd.DataFrame) -> bytes:
    return df.to_csv().encode('utf-8')

if not hist_price.empty:
    historical_csv = convert_df(hist_price)
    st.download_button(
        label="Download historical data as CSV",
        data=historical_csv,
        file_name='historical_df.csv',
        mime='text/csv',
    )

    # Radio button to switch between style
    chart = st.radio(
        "Choose Style",
        ('Candlestick', 'Line Chart'))
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    if chart == 'Line Chart':
        # UPDATED: Line chart with proper configuration
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=hist_price['Date'],
                y=hist_price['Close'],  # Use Close instead of Adj Close for consistency
                name='Closing price',
                line=dict(color='blue', width=2)
            )
        )
        fig.update_layout(
            title={
                'text': f'Stock Prices of {ticker}',
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
                'text': f'Stock Prices of {ticker}',
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

# Helper function to process financial data with proper type handling
def process_financial_data(data: Optional[pd.DataFrame], data_type: str) -> Optional[pd.DataFrame]:
    """Process financial data with proper type checking"""
    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        return None
    
    try:
        # Create a copy to avoid modifying original
        processed_data = data.copy()
        
        # FIXED: Safe column date conversion without .date errors
        try:
            # Check if any column appears to be datetime-like
            datetime_like_columns = False
            for col in processed_data.columns:
                # Multiple checks to identify datetime objects
                if (hasattr(col, 'year') and hasattr(col, 'month') and hasattr(col, 'day')):
                    datetime_like_columns = True
                    break
            
            if datetime_like_columns:
                new_columns = []
                for col in processed_data.columns:
                    # Safe datetime detection and conversion
                    if isinstance(col, (pd.Timestamp, dt.datetime)):
                        try:
                            # Convert to string date format instead of using .date()
                            new_columns.append(col.strftime('%Y-%m-%d'))
                        except (AttributeError, TypeError):
                            new_columns.append(col)
                    else:
                        new_columns.append(col)
                processed_data.columns = new_columns
        except Exception:
            # If column conversion fails, continue with original columns
            pass
        
        # Remove rows with all NaN values
        processed_data = processed_data.dropna(how='all')
        
        if processed_data.empty:
            return None
            
        # Convert to numeric where possible
        for col in processed_data.columns:
            processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
        
        # Remove columns that became all NaN after conversion
        processed_data = processed_data.dropna(axis=1, how='all')
        
        if processed_data.empty:
            return None
            
        # Format numbers with commas
        for col in processed_data.columns:
            processed_data[col] = processed_data[col].apply(
                lambda x: f"{x:,.0f}" if pd.notna(x) and isinstance(x, (int, float)) else str(x)
            )
            
        return processed_data
        
    except Exception as e:
        st.error(f"Error processing {data_type}: {e}")
        return None

# Quarterly results
st.subheader('Quarterly Result')
st.write('A quarterly result is a summary or collection of unaudited financial statements, such as balance sheets, income statements, and cash flow statements, issued by companies every quarter (three months).')

try:
    quarterly_data = stock.quarterly_financials
    # FIX: Ensure we're passing a DataFrame or None
    quarterly_df = quarterly_data if isinstance(quarterly_data, pd.DataFrame) else None
    processed_quarterly = process_financial_data(quarterly_df, "quarterly results")
    if processed_quarterly is not None and not processed_quarterly.empty:
        st.dataframe(processed_quarterly.style.highlight_max(axis=1, color='lightgreen'))
    else:
        st.info("No quarterly financial data available.")
except Exception as e:
    st.error(f"Error loading quarterly results: {e}")

# Profit and loss
st.subheader('Profit & Loss')
st.write("A profit and loss (P&L) statement is an annual financial report that provides a summary of a company's revenue, expenses and profit.")

try:
    financials_data = stock.financials
    # FIX: Ensure we're passing a DataFrame or None
    financials_df = financials_data if isinstance(financials_data, pd.DataFrame) else None
    processed_financials = process_financial_data(financials_df, "financials")
    if processed_financials is not None and not processed_financials.empty:
        st.dataframe(processed_financials.style.highlight_max(axis=1, color='lightgreen'))
    else:
        st.info("No financial data available.")
except Exception as e:
    st.error(f"Error loading financials: {e}")

# Balance sheet
st.subheader('Balance Sheet')
st.write("A balance sheet is a financial statement that reports a company's assets, liabilities, and shareholder equity.")

try:
    balance_data = stock.balance_sheet
    # FIX: Ensure we're passing a DataFrame or None
    balance_df = balance_data if isinstance(balance_data, pd.DataFrame) else None
    processed_balance = process_financial_data(balance_df, "balance sheet")
    if processed_balance is not None and not processed_balance.empty:
        st.dataframe(processed_balance.style.highlight_max(axis=1, color='lightgreen'))
    else:
        st.info("No balance sheet data available.")
except Exception as e:
    st.error(f"Error loading balance sheet: {e}")

# Cash flow
st.subheader('Cash Flows')
st.write("The term cash flow refers to the net amount of cash and cash equivalents being transferred in and out of a company.")

try:
    cashflow_data = stock.cashflow
    # FIX: Ensure we're passing a DataFrame or None
    cashflow_df = cashflow_data if isinstance(cashflow_data, pd.DataFrame) else None
    processed_cashflow = process_financial_data(cashflow_df, "cash flow")
    if processed_cashflow is not None and not processed_cashflow.empty:
        st.dataframe(processed_cashflow.style.highlight_max(axis=1, color='lightgreen'))
    else:
        st.info("No cash flow data available.")
except Exception as e:
    st.error(f"Error loading cash flow: {e}")

# Actions with error handling
st.subheader('Splits & Dividends')
st.write('Historical stock splits and dividend information.')

try:
    actions_data = stock.actions
    # FIX 1: Check if actions_data is not empty using len() instead of .empty
    if (actions_data is not None and 
        isinstance(actions_data, pd.DataFrame) and 
        len(actions_data) > 0):  # Fixed: Use len() instead of .empty for list checking
        
        # Create a copy to avoid modifying original
        actions_display = actions_data.copy()
        
        # FIX 2: Safe index date conversion - check if index is datetime type
        if hasattr(actions_display.index, 'strftime'):
            try:
                # Convert datetime index to date objects using proper pandas method
                if pd.api.types.is_datetime64_any_dtype(actions_display.index):
                    # FIX 3: Use proper pandas index assignment instead of list
                    actions_display.index = pd.Index(pd.to_datetime(actions_display.index).date)
            except (AttributeError, TypeError):
                pass  # If date conversion fails, keep original index
        
        st.dataframe(actions_display, width=1000)
    else:
        st.info("No splits or dividend data available.")
except Exception as e:
    st.error(f"Error loading splits and dividends: {e}")

# Add some spacing at the bottom
st.markdown("---")
st.caption("Data provided by Yahoo Finance")