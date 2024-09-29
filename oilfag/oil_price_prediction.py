# oil_price_prediction.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import requests
import json
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# EIA API Key (Replace with your actual API key)
EIA_API_KEY = 'cxKicy0icuGitsUWHFoL92cHgsRP2i9eYcx8fGPw'
BASE_URL = 'https://api.eia.gov/v2/'

# -----------------------------
# 1. Data Collection Functions
# -----------------------------

def fetch_eia_data_v2(category, start_period, end_period, data_cols=None, facets=None):
    url = f"{BASE_URL}{category}/data/"
    params = {
        'api_key': EIA_API_KEY,
        'data': ','.join(data_cols) if data_cols else '',
        'start': start_period,
        'end': end_period,
        'offset': 0,
        'length': 5000  # Adjust as needed
    }

    # Convert facets to the required format
    if facets:
        for key, values in facets.items():
            for i, value in enumerate(values):
                params[f'facets[{key}][{i}]'] = value

    response = requests.get(url, params=params)
    data = response.json()

    if 'response' not in data or 'data' not in data['response']:
        print(f"Error fetching data: {data.get('error', 'Unknown error')}")
        print(f"Full API Response:")
        print(json.dumps(data, indent=2))
        return None

    df = pd.DataFrame(data['response']['data'])
    return df

def fetch_us_crude_oil_production(start_date, end_date):
    category = 'petroleum/crd/crdprod'
    data_cols = ['value']
    facets = {
        'product': ['CR'],       # Crude Oil
        'process': ['PAP'],      # Production
        'area': ['NUSA'],        # U.S. total
        'frequency': ['monthly'] # Specify frequency as a facet
    }
    df = fetch_eia_data_v2(category, start_date, end_date, data_cols=data_cols, facets=facets)
    if df is not None:
        df.rename(columns={'period': 'Date', 'value': 'Production'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[['Date', 'Production']]
    return df

def fetch_us_crude_oil_consumption(start_date, end_date):
    category = 'petroleum/pet/pet_cons_psup_dc_nus_mbbl_m'
    data_cols = ['value']
    facets = {
        'product': ['EPC0'],     # All Petroleum Products
        'process': ['SUP'],      # Product Supplied
        'area': ['NUSA'],        # U.S. total
        'frequency': ['monthly'] # Specify frequency as a facet
    }
    df = fetch_eia_data_v2(category, start_date, end_date, data_cols=data_cols, facets=facets)
    if df is not None:
        df.rename(columns={'period': 'Date', 'value': 'Consumption'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[['Date', 'Consumption']]
    return df

def fetch_us_crude_oil_inventories(start_date, end_date):
    category = 'petroleum/stoc/wstk'
    data_cols = ['value']
    facets = {
        'product': ['EPC0'],     # All Petroleum Products
        'area': ['NUSA'],        # U.S. total
        'frequency': ['weekly']  # Specify frequency as a facet
    }
    df = fetch_eia_data_v2(category, start_date, end_date, data_cols=data_cols, facets=facets)
    if df is not None:
        df.rename(columns={'period': 'Date', 'value': 'Inventory'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[['Date', 'Inventory']]
    return df

# Function to fetch historical WTI prices
def fetch_wti_prices(period='10y'):
    wti = yf.Ticker("CL=F")
    hist = wti.history(period=period)
    hist.reset_index(inplace=True)
    hist = hist[['Date', 'Close']]
    hist.rename(columns={'Close': 'WTI_Price'}, inplace=True)
    return hist

# -----------------------------
# 2. Data Collection
# -----------------------------

if __name__ == "__main__":
    print("Collecting data...")

    # For monthly data
    start_date_monthly = '2010-01'
    end_date_monthly = datetime.now().strftime('%Y-%m')

    # For weekly data
    start_date_weekly = '2010-01-01'
    end_date_weekly = datetime.now().strftime('%Y-%m-%d')

    # Fetching U.S. Crude Oil Production Data
    production_df = fetch_us_crude_oil_production(start_date_monthly, end_date_monthly)
    if production_df is None:
        raise Exception("Failed to fetch production data.")

    # Fetching U.S. Crude Oil Consumption Data
    consumption_df = fetch_us_crude_oil_consumption(start_date_monthly, end_date_monthly)
    if consumption_df is None:
        raise Exception("Failed to fetch consumption data.")

    # Fetching U.S. Crude Oil Inventory Levels
    inventory_df = fetch_us_crude_oil_inventories(start_date_weekly, end_date_weekly)
    if inventory_df is None:
        raise Exception("Failed to fetch inventory data.")

    # Resample inventory data to monthly frequency
    inventory_df.set_index('Date', inplace=True)
    inventory_df = inventory_df.resample('M').mean().reset_index()

    # Fetching WTI Crude Oil Prices
    wti_prices = fetch_wti_prices()

    # Merging DataFrames
    data_frames = [production_df, consumption_df, inventory_df, wti_prices]
    df = production_df
    for temp_df in data_frames[1:]:
        df = pd.merge(df, temp_df, on='Date', how='outer')

    # -----------------------------
    # 3. Data Preprocessing
    # -----------------------------

    print("Preprocessing data...")

    # Sorting by Date
    df.sort_values('Date', inplace=True)

    # Handling Missing Values
    df.interpolate(method='time', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)

    # Feature Engineering
    df['Supply_Demand_Diff'] = df['Production'] - df['Consumption']
    df['Inventory_Change'] = df['Inventory'].diff()
    df['Month'] = df['Date'].dt.month

    # Dropping Rows with NaN Values
    df.dropna(inplace=True)

    # -----------------------------
    # 4. Model Development
    # -----------------------------

    print("Developing prediction model...")

    # Features and Target Variable
    features = ['Production', 'Consumption', 'Inventory', 'Supply_Demand_Diff', 'Inventory_Change', 'Month']
    X = df[features]
    y = df['WTI_Price']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Model Initialization
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)

    # Model Training
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Model Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model Mean Absolute Error on Test Set: ${mae:.2f}")

    # -----------------------------
    # 5. Fetching Real-Time Oil Prices
    # -----------------------------

    print("Fetching real-time oil prices...")

    # Fetching Current WTI Crude Oil Price
    wti_live = yf.Ticker("CL=F")
    wti_live_price = wti_live.history(period="1d")['Close'][0]
    print(f"Current WTI Crude Oil Price: ${wti_live_price:.2f}")

    # -----------------------------
    # 6. Predicting Ideal Price
    # -----------------------------

    print("Predicting ideal price based on latest data...")

    # Preparing Latest Data for Prediction
    latest_data = df[features].iloc[-1:].copy()

    # Predicting Ideal Price
    latest_predicted_price = model.predict(latest_data)[0]
    print(f"Predicted Ideal Price: ${latest_predicted_price:.2f}")

    # -----------------------------
    # 7. Comparing Prices and Analyzing Discrepancies
    # -----------------------------

    print("Analyzing discrepancies...")

    # Price Difference
    price_difference = wti_live_price - latest_predicted_price
    percentage_difference = (price_difference / latest_predicted_price) * 100

    print(f"Price Difference: ${price_difference:.2f}")
    print(f"Percentage Difference: {percentage_difference:.2f}%")

    # Analysis of Discrepancies
    if abs(percentage_difference) > 5:
        print("\nSignificant discrepancy detected between predicted and actual prices.")
        print("Possible reasons could include:")
        print("- Geopolitical events not accounted for in the model.")
        print("- Sudden market sentiment shifts.")
        print("- Unusual supply disruptions or demand spikes.")
    else:
        print("\nPredicted price is close to the actual price. Model is performing well.")

    # Suggestions for Improvement
    print("\nSuggestions for Model Improvement:")
    print("- Incorporate real-time news sentiment analysis.")
    print("- Add geopolitical risk indices as features.")
    print("- Include currency exchange rate fluctuations.")

    # -----------------------------
    # 8. Visualization
    # -----------------------------

    print("Generating visualizations...")

    # Adding Predictions to DataFrame
    df_results = df.iloc[-len(y_test):].copy()
    df_results['Predicted_Price'] = y_pred

    # Plotting Actual vs Predicted Prices
    plt.figure(figsize=(14,7))
    plt.plot(df_results['Date'], df_results['WTI_Price'], label='Actual Price')
    plt.plot(df_results['Date'], df_results['Predicted_Price'], label='Predicted Price', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('WTI Crude Oil Price ($)')
    plt.title('Actual vs Predicted WTI Crude Oil Prices')
    plt.legend()
    plt.grid(True)
    plt.show()
