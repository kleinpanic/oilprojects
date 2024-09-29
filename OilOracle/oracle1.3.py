import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
from datetime import datetime

# Function to calculate RSI (Relative Strength Index)
def calculate_rsi(data, period=14):
    delta = data['Adj Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(data, period=20):
    sma = data['Adj Close'].rolling(window=period).mean()
    std = data['Adj Close'].rolling(window=period).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return sma, upper_band, lower_band

# Function to calculate SMA (Simple Moving Average)
def calculate_sma(data, period=50):
    sma = data['Adj Close'].rolling(window=period).mean()
    return sma

# Enhanced Oracle function with multiple technical indicators and error calculation
def enhanced_oracle(ticker, start_date, end_date):
    print(f"\nAnalyzing Stock with Enhanced Oracle: {ticker}")
    
    # Fetch the stock data
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        print(f"No data found for {ticker}. Please check the ticker symbol and date range.")
        return
    
    # Calculate daily returns
    data['Return'] = data['Adj Close'].pct_change()
    returns = data['Return'].dropna()

    # Calculate key statistics
    current_price = data['Adj Close'][-1]
    expected_return = returns.mean()
    risk = returns.std()

    # Calculate RSI, Bollinger Bands, and SMA
    data['RSI'] = calculate_rsi(data)
    data['SMA'] = calculate_sma(data)
    data['SMA_Boll'], data['Upper_Boll'], data['Lower_Boll'] = calculate_bollinger_bands(data)

    # Initialize error calculations
    total_error = 0
    error_count = 0
    errors = []

    # Loop through historical data for predictions and error calculation
    for i in range(len(data) - 1):  # Loop over all days except the last one
        current_day_price = data['Adj Close'].iloc[i]
        rsi_latest = data['RSI'].iloc[i]
        sma_latest = data['SMA'].iloc[i]
        upper_boll_latest = data['Upper_Boll'].iloc[i]
        lower_boll_latest = data['Lower_Boll'].iloc[i]

        # Use Bollinger Bands and RSI for price prediction logic
        if rsi_latest < 30 and current_day_price < lower_boll_latest:
            predicted_price_next_day = current_day_price * (1 + 0.02)  # Predict 2% rise
        elif rsi_latest > 70 and current_day_price > upper_boll_latest:
            predicted_price_next_day = current_day_price * (1 - 0.02)  # Predict 2% fall
        else:
            if current_day_price > sma_latest:
                predicted_price_next_day = current_day_price * (1 + expected_return)
            else:
                predicted_price_next_day = current_day_price * (1 - expected_return)

        actual_price_next_day = data['Adj Close'].iloc[i + 1]
        percentage_error = ((predicted_price_next_day - actual_price_next_day) / actual_price_next_day) * 100
        total_error += abs(percentage_error)
        errors.append(abs(percentage_error))
        error_count += 1

    # Prediction for the most recent day (for tomorrow's price)
    rsi_latest = data['RSI'].iloc[-1]
    sma_latest = data['SMA'].iloc[-1]
    upper_boll_latest = data['Upper_Boll'].iloc[-1]
    lower_boll_latest = data['Lower_Boll'].iloc[-1]

    if rsi_latest < 30 and current_price < lower_boll_latest:
        predicted_price_tomorrow = current_price * (1 + 0.02)  # Predict 2% rise
    elif rsi_latest > 70 and current_price > upper_boll_latest:
        predicted_price_tomorrow = current_price * (1 - 0.02)  # Predict 2% fall
    else:
        if current_price > sma_latest:
            predicted_price_tomorrow = current_price * (1 + expected_return)
        else:
            predicted_price_tomorrow = current_price * (1 - expected_return)

    # Calculate average percentage error and standard error
    if error_count > 0:
        avg_percentage_error = total_error / error_count
        std_error = np.std(errors)
    else:
        avg_percentage_error = None
        std_error = None

    # Define confidence levels
    confidence_levels = [0.90, 0.95, 0.99, 0.999]
    confidence_intervals = {}

    # Calculate price ranges for each confidence level
    for conf in confidence_levels:
        z_score = norm.ppf(1 - (1 - conf) / 2)  # Calculate z-score for each confidence level
        margin_of_error = z_score * risk
        lower_price = predicted_price_tomorrow * (1 - margin_of_error)
        upper_price = predicted_price_tomorrow * (1 + margin_of_error)
        confidence_intervals[conf] = (lower_price, upper_price)

    # Print the results with neat formatting
    print("\nOracle Analysis")
    print("=" * 40)
    print(f"Stock Ticker:        {ticker}")
    print(f"Current Price:       ${current_price:.2f}")
    print(f"Expected Return:     {expected_return * 100:.2f}%")
    print(f"Risk (Std Dev):      {risk * 100:.2f}%")
    print(f"Predicted Price Tomorrow: ${predicted_price_tomorrow:.2f}")
    print(f"Average Percentage Error: {avg_percentage_error:.2f}%")
    print(f"Standard Error of Prediction: {std_error:.2f}%")
    print("=" * 40)

    # Print the top 4 confidence intervals
    print("\nPrice Ranges for Confidence Intervals:")
    for conf in confidence_levels:
        lower_price, upper_price = confidence_intervals[conf]
        print(f"{int(conf * 100)}% Confidence Interval:")
        print(f"  Lower: ${lower_price:.2f}")
        print(f"  Upper: ${upper_price:.2f}")
        print("-" * 40)

# Main code to call the enhanced oracle function
if __name__ == "__main__":
    # Get stock ticker and date range from the user
    ticker = input("Enter the stock ticker symbol (e.g., 'AAPL'): ").strip().upper()
    start_date = input("Enter the start date (YYYY-MM-DD): ").strip()
    end_date = input("Enter the end date (YYYY-MM-DD): ").strip()

    # Call the enhanced oracle function with the user-provided inputs
    enhanced_oracle(ticker, start_date, end_date)


def save_prediction_to_file(ticker, current_price, expected_return, risk, predicted_price_tomorrow, avg_percentage_error, std_error, confidence_intervals):
    # Get current date and time
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Open a file in append mode
    with open("stock_predictions.txt", "a") as file:
        # Write the header with date and time
        file.write(f"\nPrediction Analysis for {ticker} on {current_datetime}\n")
        file.write("=" * 50 + "\n")
        
        # Write the stock information and analysis
        file.write(f"Stock Ticker:        {ticker}\n")
        file.write(f"Current Price:       ${current_price:.2f}\n")
        file.write(f"Expected Return:     {expected_return * 100:.2f}%\n")
        file.write(f"Risk (Std Dev):      {risk * 100:.2f}%\n")
        file.write(f"Predicted Price Tomorrow: ${predicted_price_tomorrow:.2f}\n")
        file.write(f"Average Percentage Error: {avg_percentage_error:.2f}%\n")
        file.write(f"Standard Error of Prediction: {std_error:.2f}%\n")
        file.write("=" * 50 + "\n")
        
        # Write the confidence intervals
        file.write("Confidence Intervals:\n")
        for conf, (lower_price, upper_price) in confidence_intervals.items():
            file.write(f"{int(conf * 100)}% Confidence Interval:\n")
            file.write(f"  Lower: ${lower_price:.2f}\n")
            file.write(f"  Upper: ${upper_price:.2f}\n")
            file.write("-" * 50 + "\n")
        
        # End of entry
        file.write("\n")
save_prediction_to_file(ticker, current_price, expected_return, risk, predicted_price_tomorrow, avg_percentage_error, std_error, confidence_intervals)

