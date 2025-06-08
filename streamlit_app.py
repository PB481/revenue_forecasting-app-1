import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import plotly.express as px
import os # Import the os module to check for file existence

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Fund Income & P&L Forecast")

# --- Dummy Data Generation Functions (to prevent FileNotFoundError) ---
# These functions create sample CSVs if they don't exist, making the app runnable.
def create_dummy_portfolio_data(file_path="sample_portfolio.csv"):
    if not os.path.exists(file_path):
        st.warning(f"'{file_path}' not found. Creating dummy data.")
        data = {
            'Ticker': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'IBM', 'XOM', 'CVX'],
            'Type': ['Equity', 'Equity', 'Equity', 'Equity', 'Equity', 'Equity', 'Equity', 'Equity'],
            'Quantity': [100, 50, 75, 30, 20, 120, 80, 60],
            'Purchase_Price': [150.00, 100.00, 250.00, 100.00, 200.00, 130.00, 90.00, 160.00]
        }
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)

def create_dummy_historical_prices(file_path="sample_historical_prices.csv"):
    if not os.path.exists(file_path):
        st.warning(f"'{file_path}' not found. Creating dummy data.")
        today = date.today()
        dates = [today - timedelta(days=i) for i in range(30)] # Last 30 days
        tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'IBM', 'XOM', 'CVX']
        
        data = []
        for ticker in tickers:
            base_price = np.random.uniform(50, 300)
            for d in dates:
                price = base_price * (1 + np.random.uniform(-0.02, 0.02)) # Add some variance
                data.append({'Date': d.strftime('%Y-%m-%d'), 'Ticker': ticker, 'Close': price})
        
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)

def create_dummy_bond_coupons(file_path="sample_bond_coupons.csv"):
    if not os.path.exists(file_path):
        st.warning(f"'{file_path}' not found. Creating dummy data.")
        data = {
            'Ticker': ['GOV1', 'CORP1', 'MUN1'],
            'Payment_Date': [(date.today() + timedelta(days=30)).strftime('%Y-%m-%d'), 
                             (date.today() + timedelta(days=90)).strftime('%Y-%m-%d'), 
                             (date.today() + timedelta(days=180)).strftime('%Y-%m-%d')],
            'Coupon_Amount': [5.00, 7.50, 3.25] # Coupon per bond, not total
        }
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)

def create_dummy_equity_dividends(file_path="sample_equity_dividends.csv"):
    if not os.path.exists(file_path):
        st.warning(f"'{file_path}' not found. Creating dummy data.")
        data = {
            'Ticker': ['AAPL', 'MSFT', 'IBM'],
            'Payment_Date': [(date.today() + timedelta(days=45)).strftime('%Y-%m-%d'), 
                             (date.today() + timedelta(days=100)).strftime('%Y-%m-%d'), 
                             (date.today() + timedelta(days=200)).strftime('%Y-%m-%d')],
            'Dividend_Amount': [0.25, 0.62, 1.65] # Dividend per share
        }
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)

def create_dummy_corporate_actions(file_path="sample_corporate_actions.csv"):
    if not os.path.exists(file_path):
        st.warning(f"'{file_path}' not found. Creating dummy data.")
        data = {
            'Ticker': ['TSLA', 'GOOGL'],
            'Action_Type': ['Stock Split', 'Bond Call'], # Assuming 'Bond Call' for GOOGL for diversity
            'Effective_Date': [(date.today() + timedelta(days=60)).strftime('%Y-%m-%d'), 
                               (date.today() + timedelta(days=150)).strftime('%Y-%m-%d')],
            'Details': ['3-for-1', '101.50'] # Example: 3-for-1 split, Bond called at 101.50
        }
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)

# --- Data Loading Functions (with caching) ---
@st.cache_data
def load_portfolio_data(file_path="sample_portfolio.csv"):
    """
    Loads portfolio holdings data.
    Assumes no specific date column needs conversion across the whole DataFrame.
    """
    df = pd.read_csv(file_path)
    # If there was a 'Purchase_Date' column, you would convert it like:
    # df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'])
    return df

@st.cache_data
def load_historical_prices(file_path="sample_historical_prices.csv"):
    """
    Loads historical prices data.
    Converts the 'Date' column to datetime objects.
    """
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date']) # Convert 'Date' column to datetime
    return df

@st.cache_data
def load_bond_coupons(file_path="sample_bond_coupons.csv"):
    """
    Loads future bond coupon payments.
    Converts the 'Payment_Date' column to datetime objects.
    """
    df = pd.read_csv(file_path)
    df['Payment_Date'] = pd.to_datetime(df['Payment_Date']) # Convert 'Payment_Date' column to datetime
    return df

@st.cache_data
def load_equity_dividends(file_path="sample_equity_dividends.csv"):
    """
    Loads future equity dividend payments.
    Converts the 'Payment_Date' column to datetime objects.
    """
    df = pd.read_csv(file_path)
    df['Payment_Date'] = pd.to_datetime(df['Payment_Date']) # Convert 'Payment_Date' column to datetime
    return df

@st.cache_data
def load_corporate_actions(file_path="sample_corporate_actions.csv"):
    """
    Loads future corporate actions data.
    Converts the 'Effective_Date' column to datetime objects.
    """
    df = pd.read_csv(file_path)
    df['Effective_Date'] = pd.to_datetime(df['Effective_Date']) # Convert 'Effective_Date' column to datetime
    return df

# --- Financial Calculation Functions ---

def get_latest_prices(historical_prices_df):
    """Gets the latest closing price for each ticker."""
    if historical_prices_df.empty:
        # Correctly initialize an empty DataFrame with expected columns
        return pd.DataFrame(columns=['Ticker', 'Close'])
    
    # Get the latest date from the 'Date' column
    latest_date = historical_prices_df['Date'].max()
    
    # Filter for the latest date and set 'Ticker' as index to get a Series of prices
    latest_prices = historical_prices_df[historical_prices_df['Date'] == latest_date].set_index('Ticker')['Close']
    return latest_prices.to_dict()

def calculate_current_pnl(portfolio_df, latest_prices):
    """Calculates current (realized and unrealized) P&L."""
    pnl_data = [] # Correctly initialize an empty list
    total_unrealized_pnl = 0.0
    total_market_value = 0.0
    total_cost_basis = 0.0

    for index, row in portfolio_df.iterrows():
        # Access columns using dictionary-like keys
        ticker = row['Ticker'] 
        quantity = row['Quantity']
        purchase_price = row['Purchase_Price']
        
        current_price = latest_prices.get(ticker)
        
        if current_price is not None:
            market_value = current_price * quantity
            cost_basis = purchase_price * quantity
            unrealized_pnl = market_value - cost_basis
            
            total_unrealized_pnl += unrealized_pnl
            total_market_value += market_value
            total_cost_basis += cost_basis

            pnl_data.append({
                'Ticker': ticker,
                'Type': row['Type'], # Access 'Type' column
                'Quantity': quantity,
                'Purchase Price': f"{purchase_price:,.2f}",
                'Current Price': f"{current_price:,.2f}",
                'Cost Basis': f"{cost_basis:,.2f}",
                'Market Value': f"{market_value:,.2f}",
                'Unrealized P&L': f"{unrealized_pnl:,.2f}"
            })
        else:
            pnl_data.append({
                'Ticker': ticker,
                'Type': row['Type'], # Access 'Type' column
                'Quantity': quantity,
                'Purchase Price': f"{purchase_price:,.2f}",
                'Current Price': "N/A",
                'Cost Basis': f"{purchase_price * quantity:,.2f}",
                'Market Value': "N/A",
                'Unrealized P&L': "N/A"
            })
            st.warning(f"No current price found for {ticker}. P&L for this asset may be incomplete.")

    current_pnl_df = pd.DataFrame(pnl_data)
    return current_pnl_df, total_unrealized_pnl, total_market_value, total_cost_basis

def apply_corporate_actions(portfolio_state, corporate_actions_df, current_date, forecast_end_date):
    """
    Applies corporate actions to the portfolio state and generates adjusted income events.
    This is a simplified simulation.
    """
    adjusted_portfolio = portfolio_state.copy()
    future_income_events = [] # Correctly initialize an empty list
    
    # Filter corporate actions relevant for the forecast period
    # FIX: Corrected the SyntaxError here by properly chaining boolean conditions
    # Assumes 'Effective_Date' is the date column in corporate_actions_df
    relevant_actions = corporate_actions_df[
        (corporate_actions_df['Effective_Date'] >= current_date) &
        (corporate_actions_df['Effective_Date'] <= forecast_end_date)
    ].sort_values(by='Effective_Date')

    for _, action in relevant_actions.iterrows():
        # Access columns using dictionary-like keys from the 'action' Series
        ticker = action['Ticker'] 
        action_type = action['Action_Type'] 
        effective_date = action['Effective_Date'] 
        details = action['Details'] 

        # Find the asset in the current portfolio state
        # FIX: Corrected DataFrame filtering for 'Ticker'
        asset_idx = adjusted_portfolio[adjusted_portfolio['Ticker'] == ticker].index
        if asset_idx.empty:
            continue # Asset not in portfolio

        # Get scalar values from Series using .iloc[0] or .item()
        current_quantity = adjusted_portfolio.loc[asset_idx, 'Quantity'].iloc[0]
        current_purchase_price = adjusted_portfolio.loc[asset_idx, 'Purchase_Price'].iloc[0]

        if action_type == 'Stock Split':
            # Example: 2-for-1 split. Parsing 'Details' like '2-for-1'
            split_parts = details.split('-for-')
            if len(split_parts) == 2 and split_parts[1].replace('.', '', 1).isdigit():
                try:
                    # FIX: Corrected float conversion for split ratio (e.g., 2/1)
                    split_ratio = float(split_parts[0]) / float(split_parts[1])
                except ValueError:
                    st.warning(f"Invalid split ratio format '{details}' for {ticker}. Skipping split.")
                    continue
            else:
                st.warning(f"Unexpected 'Stock Split' details format: '{details}' for {ticker}. Skipping split.")
                continue
            
            new_quantity = current_quantity * split_ratio
            new_purchase_price = current_purchase_price / split_ratio
            
            # Update the portfolio state using .loc
            adjusted_portfolio.loc[asset_idx, 'Quantity'] = new_quantity
            adjusted_portfolio.loc[asset_idx, 'Purchase_Price'] = new_purchase_price
            
            future_income_events.append({
                'Date': effective_date,
                'Ticker': ticker,
                'Event': f"Stock Split ({details})",
                'Impact': f"Quantity changed from {current_quantity:.0f} to {new_quantity:.0f}, Purchase Price from {current_purchase_price:.2f} to {new_purchase_price:.2f}"
            })
            st.info(f"Applied {ticker} Stock Split ({details}) on {effective_date.strftime('%Y-%m-%d')}")

        elif action_type == 'Bond Call':
            # Example: Bond is called, remove from portfolio and record cash inflow
            # Assumes details like 'Call_Price_102.50' or simply '102.50'
            try:
                call_price = float(details.split('_')[-1]) # Attempt to parse last part as float
            except (ValueError, IndexError):
                try: # If splitting by '_' fails, try direct conversion if 'details' is just the price
                    call_price = float(details)
                except ValueError:
                    st.warning(f"Could not parse call price from details: '{details}' for {ticker}. Skipping bond call.")
                    continue

            cash_inflow = current_quantity * call_price
            
            # Remove the bond from the adjusted portfolio
            adjusted_portfolio = adjusted_portfolio.drop(asset_idx).reset_index(drop=True)
            
            future_income_events.append({
                'Date': effective_date,
                'Ticker': ticker,
                'Event': f"Bond Called (at {call_price:,.2f})",
                'Impact': f"Received {cash_inflow:,.2f} cash, bond removed from portfolio."
            })
            st.info(f"Applied {ticker} Bond Call on {effective_date.strftime('%Y-%m-%d')}")
    
    return adjusted_portfolio, pd.DataFrame(future_income_events)

def forecast_income_and_pnl(portfolio_df, historical_prices_df, bond_coupons_df, equity_dividends_df, corporate_actions_df, forecast_horizon_days):
    """
    Forecasts income and future P&L based on current portfolio, historical data,
    and corporate actions.
    This is a simplified projection.
    """
    # Get the latest date from historical prices for current_date
    current_date = historical_prices_df['Date'].max()
    forecast_end_date = current_date + timedelta(days=forecast_horizon_days)

    # Apply corporate actions to get an adjusted portfolio for future projections
    # We pass a copy of the initial portfolio to avoid modifying the original in session state
    adjusted_portfolio_for_forecast, ca_impact_events = apply_corporate_actions(
        portfolio_df.copy(), corporate_actions_df, current_date, forecast_end_date
    )

    # --- Income Forecasting ---
    projected_income = [] # Correctly initialize an empty list

    # Bond Coupons
    # FIX: Corrected DataFrame filtering for 'Payment_Date'
    future_coupons = bond_coupons_df[
        (bond_coupons_df['Payment_Date'] > current_date) &
        (bond_coupons_df['Payment_Date'] <= forecast_end_date)
    ]
    for _, coupon_row in future_coupons.iterrows():
        # Access columns using dictionary-like keys from 'coupon_row' Series
        bond_id = coupon_row['Ticker'] 
        
        # Check if bond is still in the adjusted portfolio (not called)
        # FIX: Corrected check using .any() for boolean Series
        if (adjusted_portfolio_for_forecast['Ticker'] == bond_id).any():
            # Get the quantity from the adjusted portfolio for the specific bond
            # FIX: Used .iloc[0] to get the scalar quantity
            quantity = adjusted_portfolio_for_forecast[adjusted_portfolio_for_forecast['Ticker'] == bond_id]['Quantity'].iloc[0]
            total_coupon_amount = coupon_row['Coupon_Amount'] * quantity # Access 'Coupon_Amount'
            projected_income.append({
                'Date': coupon_row['Payment_Date'], # Use 'Payment_Date'
                'Ticker': bond_id,
                'Type': 'Bond Coupon',
                'Amount': total_coupon_amount
            })

    # Equity Dividends
    # FIX: Corrected DataFrame filtering for 'Payment_Date'
    future_dividends = equity_dividends_df[
        (equity_dividends_df['Payment_Date'] > current_date) &
        (equity_dividends_df['Payment_Date'] <= forecast_end_date)
    ]
    for _, dividend_row in future_dividends.iterrows():
        # Access columns using dictionary-like keys from 'dividend_row' Series
        ticker = dividend_row['Ticker']
        
        # Check if equity is still in the adjusted portfolio (not acquired/merged out)
        # FIX: Corrected check using .any() for boolean Series
        if (adjusted_portfolio_for_forecast['Ticker'] == ticker).any():
            # Get the quantity from the adjusted portfolio for the specific equity
            # FIX: Used .iloc[0] to get the scalar quantity
            quantity = adjusted_portfolio_for_forecast[adjusted_portfolio_for_forecast['Ticker'] == ticker]['Quantity'].iloc[0]
            
            # Adjust quantity for any splits that occurred before this dividend's ex-date
            # This is a simplification; a real system would track share history more robustly
            
            # For simplicity, we'll assume the quantity in adjusted_portfolio_for_forecast already reflects splits
            # that occurred *before* the dividend's ex-date.
            
            total_dividend_amount = dividend_row['Dividend_Amount'] * quantity # Access 'Dividend_Amount'
            projected_income.append({
                'Date': dividend_row['Payment_Date'], # Use payment date for income receipt
                'Ticker': ticker,
                'Type': 'Equity Dividend',
                'Amount': total_dividend_amount
            })

    projected_income_df = pd.DataFrame(projected_income).sort_values(by='Date').reset_index(drop=True)
    
    # --- Future P&L Calculation (Simplified) ---
    # For simplicity, we'll assume future prices are the latest current prices,
    # adjusted for any stock splits. This is a very basic forecast.
    # A real model would use time series forecasting (e.g., ARIMA, LSTM) for price prediction.
    
    latest_prices = get_latest_prices(historical_prices_df)
    
    # Project future portfolio value
    projected_market_value = 0.0
    projected_cost_basis = 0.0
    
    for _, row in adjusted_portfolio_for_forecast.iterrows():
        # Access columns using dictionary-like keys from the 'row' Series
        ticker = row['Ticker'] 
        quantity = row['Quantity']
        purchase_price = row['Purchase_Price'] # This purchase price is already adjusted for past splits in apply_corporate_actions
        
        # Use the latest price as the forecasted price for simplicity
        forecasted_price = latest_prices.get(ticker)
        
        if forecasted_price is not None:
            projected_market_value += forecasted_price * quantity
            projected_cost_basis += purchase_price * quantity
        else:
            st.warning(f"No latest price available for {ticker} for future P&L projection.")

    total_projected_income = projected_income_df['Amount'].sum() if not projected_income_df.empty else 0.0
    
    # Future P&L = (Projected Market Value - Projected Cost Basis) + Total Projected Income
    # This is a simplified view. A full P&L would include operating expenses, etc.
    future_unrealized_pnl = projected_market_value - projected_cost_basis
    total_future_pnl = future_unrealized_pnl + total_projected_income

    return projected_income_df, total_future_pnl, adjusted_portfolio_for_forecast, ca_impact_events

# --- Streamlit UI ---
st.title("Fund Income Forecasting & P&L Analysis")

# Sidebar for navigation and global settings
st.sidebar.header("Navigation & Settings")
page = st.sidebar.radio("Go to", ("Portfolio Overview", "Current P&L", "Future Projections & Scenarios", "Source Code")) # Corrected initialization of tuple

# --- Ensure dummy data exists before attempting to load ---
create_dummy_portfolio_data()
create_dummy_historical_prices()
create_dummy_bond_coupons()
create_dummy_equity_dividends()
create_dummy_corporate_actions()

# Load all data once
portfolio_df_initial = load_portfolio_data()
historical_prices_df = load_historical_prices()
bond_coupons_df = load_bond_coupons()
equity_dividends_df = load_equity_dividends()
corporate_actions_df = load_corporate_actions()

# Get latest prices for current P&L
# Get the latest date from the 'Date' column of historical_prices_df
latest_prices = get_latest_prices(historical_prices_df)
current_date = historical_prices_df['Date'].max()

if page == "Portfolio Overview":
    st.header("Current Portfolio Holdings")
    st.write(f"Data as of: **{current_date.strftime('%Y-%m-%d')}**")
    st.dataframe(portfolio_df_initial.style.format({
        'Purchase_Price': "{:,.2f}",
        'Quantity': "{:,.0f}"
    }), use_container_width=True)

    st.subheader("Latest Market Prices")
    # Correctly initialize DataFrame for latest prices
    latest_prices_df = pd.DataFrame(latest_prices.items(), columns=['Ticker', 'Latest Price'])
    st.dataframe(latest_prices_df.style.format({'Latest Price': "{:,.2f}"}), use_container_width=True)

elif page == "Current P&L":
    st.header("Current Fund P&L")
    st.write(f"Calculated as of: **{current_date.strftime('%Y-%m-%d')}**")

    current_pnl_df, total_unrealized_pnl, total_market_value, total_cost_basis = calculate_current_pnl(portfolio_df_initial, latest_prices)
    
    st.subheader("Individual Position P&L")
    st.dataframe(current_pnl_df, use_container_width=True)

    st.subheader("Summary P&L")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Cost Basis", f"${total_cost_basis:,.2f}")
    col2.metric("Total Market Value", f"${total_market_value:,.2f}")
    col3.metric("Total Unrealized P&L", f"${total_unrealized_pnl:,.2f}", 
                delta=f"{total_unrealized_pnl/total_cost_basis:.2%}" if total_cost_basis else "0.00%")

elif page == "Future Projections & Scenarios":
    st.header("Future Income & P&L Projections")
    st.write(f"Current Date: **{current_date.strftime('%Y-%m-%d')}**")

    st.subheader("Projection Settings")
    forecast_horizon_months = st.slider("Forecast Horizon (Months)", 1, 24, 12)
    forecast_horizon_days = forecast_horizon_months * 30 # Approximate days

    st.subheader("Corporate Actions for Scenario Testing")
    st.write("You can edit the corporate actions below to test different scenarios (e.g., remove an action to simulate it not occurring).")
    
    # Use st.session_state to manage editable corporate actions
    if 'editable_corporate_actions' not in st.session_state:
        st.session_state.editable_corporate_actions = corporate_actions_df.copy()

    edited_ca_df = st.data_editor(
        st.session_state.editable_corporate_actions,
        num_rows="dynamic",
        use_container_width=True,
        key="corporate_actions_editor"
    )
    st.session_state.editable_corporate_actions = edited_ca_df

    if st.button("Run Projection with Scenarios"):
        st.subheader(f"Forecasted Income & P&L for next {forecast_horizon_months} months")
        
        # Ensure dates are datetime objects after editing by the user in data_editor
        # This assumes the user inputs dates in a format pandas can parse.
        # It's better to explicitly map specific columns if there's a mix of data types
        # For example: edited_ca_df['Effective_Date'] = pd.to_datetime(edited_ca_df['Effective_Date'], errors='coerce')
        for col in edited_ca_df.select_dtypes(include=['object']).columns:
             try:
                 edited_ca_df[col] = pd.to_datetime(edited_ca_df[col], errors='coerce')
             except Exception:
                 pass # Not a date column, ignore

        projected_income_df, total_future_pnl, adjusted_portfolio_after_ca, ca_impact_events = \
            forecast_income_and_pnl(
                portfolio_df_initial, 
                historical_prices_df, 
                bond_coupons_df, 
                equity_dividends_df, 
                edited_ca_df, # Use the edited corporate actions
                forecast_horizon_days
            )
        
        st.write("### Corporate Action Impacts During Forecast Period")
        if not ca_impact_events.empty:
            st.dataframe(ca_impact_events.style.format({'Date': lambda x: x.strftime('%Y-%m-%d')}), use_container_width=True)
        else:
            st.info("No corporate actions impacting the portfolio found within the forecast horizon.")

        st.write("### Projected Income Events")
        if not projected_income_df.empty:
            st.dataframe(projected_income_df.style.format({
                'Date': lambda x: x.strftime('%Y-%m-%d'),
                'Amount': "${:,.2f}"
            }), use_container_width=True)
            st.metric("Total Projected Income", f"${projected_income_df['Amount'].sum():,.2f}")
        else:
            st.info("No projected income events (coupons, dividends) found within the forecast horizon.")

        st.write("### Adjusted Portfolio After Corporate Actions (for future P&L)")
        st.dataframe(adjusted_portfolio_after_ca.style.format({
            'Purchase_Price': "{:,.2f}",
            'Quantity': "{:,.0f}"
        }), use_container_width=True)

        st.write("### Total Future P&L (Simplified)")
        st.info(f"This is a simplified projection. Future prices are assumed to be the latest current prices, adjusted for corporate actions. A full model would include price forecasting and fund expenses.")
        st.metric("Total Future P&L", f"${total_future_pnl:,.2f}")

elif page == "Source Code":
    st.header("Application Source Code")
    st.write("Here you can view the Python source code for this Streamlit application.")
    
    # FIX: Corrected the file path to match the common Streamlit app naming
    with open("streamlit_app.py", "r") as f: 
        code = f.read()
    st.code(code, language="python")

