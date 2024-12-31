import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor  # Regressor for predicting actual prices
from sklearn.metrics import mean_absolute_error  # Using MAE for regression evaluation
import matplotlib.pyplot as plt

# Function to load and process stock data
def load_and_process_stock_data(file_path):
    # Load stock data from CSV
    stock_data = pd.read_csv(file_path, parse_dates=["Business Date"], index_col="Business Date")
    
    # Check if the dataset has enough rows before proceeding
    if stock_data.shape[0] < 5:  # Check if data has at least 5 rows
        print(f"Not enough data to process for {file_path}.")
        return pd.DataFrame()  # Return an empty dataframe if not enough data

    # Select only the necessary columns for prediction
    stock_data = stock_data[["Open Price", "High Price", "Low Price", "Close Price", "Total Traded Quantity"]]

    # Feature Engineering
    stock_data["Tomorrow_Open"] = stock_data["Open Price"].shift(-1)
    stock_data["Tomorrow_High"] = stock_data["High Price"].shift(-1)
    stock_data["Tomorrow_Low"] = stock_data["Low Price"].shift(-1)
    stock_data["Tomorrow_Close"] = stock_data["Close Price"].shift(-1)

    # Creating Targets for prediction
    stock_data["Target_Open"] = stock_data["Tomorrow_Open"]
    stock_data["Target_High"] = stock_data["Tomorrow_High"]
    stock_data["Target_Low"] = stock_data["Tomorrow_Low"]
    stock_data["Target_Close"] = stock_data["Tomorrow_Close"]

    # Adding rolling averages and trend indicators (only apply rolling features that are sensible for the dataset size)
    horizons = [2, 5]  # Limited horizons for small dataset
    for horizon in horizons:
        rolling = stock_data.rolling(horizon, min_periods=1)  # Min period to avoid dropping data
        stock_data[f"Close_Ratio_{horizon}"] = stock_data["Close Price"] / rolling.mean()["Close Price"]
        stock_data[f"Trend_{horizon}"] = rolling.sum()["Target_Close"]
    
    # Adding lagged features
    lags = [1, 2, 3, 5]
    for lag in lags:
        stock_data[f"Lag_{lag}"] = stock_data["Close Price"].shift(lag)
    
    # Clean data by dropping NaN values (only drop rows where necessary)
    stock_data = stock_data.dropna()
    
    # Print shape to check the data
    print(f"Data after feature engineering: {stock_data.shape}")
    
    return stock_data

# Function to train and evaluate the model for each target
def train_and_evaluate_model(stock_data):
    if stock_data.empty:
        print("No data to process.")
        return
    
    # Define predictors (use only the relevant columns)
    predictors = [col for col in stock_data.columns if "Close" in col or "Trend" in col or "Lag" in col]

    # Automatically adjust the train-test split based on available data
    train_size = int(0.8 * len(stock_data))  # 80% of data for training
    test_size = len(stock_data) - train_size  # Remaining 20% for testing
    
    # Split data
    train = stock_data.iloc[:train_size]
    test = stock_data.iloc[train_size:]

    # Check if there is enough data in the train and test sets
    print(f"Training data size: {train.shape[0]}")
    print(f"Testing data size: {test.shape[0]}")

    # Model training (using RandomForestRegressor for predicting actual prices)
    model_open = RandomForestRegressor(random_state=1)
    model_high = RandomForestRegressor(random_state=1)
    model_low = RandomForestRegressor(random_state=1)
    model_close = RandomForestRegressor(random_state=1)

    # Train models for Open, High, Low, and Close
    model_open.fit(train[predictors], train["Target_Open"])
    model_high.fit(train[predictors], train["Target_High"])
    model_low.fit(train[predictors], train["Target_Low"])
    model_close.fit(train[predictors], train["Target_Close"])

    # Predicting on the test set
    preds_open = model_open.predict(test[predictors])
    preds_high = model_high.predict(test[predictors])
    preds_low = model_low.predict(test[predictors])
    preds_close = model_close.predict(test[predictors])

    # Check if predictions are made correctly
    print(f"Predictions for Open: {preds_open[:5]}")
    print(f"Predictions for High: {preds_high[:5]}")
    print(f"Predictions for Low: {preds_low[:5]}")
    print(f"Predictions for Close: {preds_close[:5]}")

    # Calculate Mean Absolute Error (MAE)
    mae_open = mean_absolute_error(test["Target_Open"], preds_open)
    mae_high = mean_absolute_error(test["Target_High"], preds_high)
    mae_low = mean_absolute_error(test["Target_Low"], preds_low)
    mae_close = mean_absolute_error(test["Target_Close"], preds_close)

    print(f"Mean Absolute Error for Open: {mae_open}")
    print(f"Mean Absolute Error for High: {mae_high}")
    print(f"Mean Absolute Error for Low: {mae_low}")
    print(f"Mean Absolute Error for Close: {mae_close}")

    # Filter the last 2-3 months' data (for example, if you have 23 days, you can take the last 15)
    plot_data = test[-10:]  # Adjust the number of points to match the last 2-3 months of data
    
    # Ensure there is data for plotting
    if plot_data.empty:
        print("No data available for plotting.")
        return

    # Plot actual vs predicted values for Open, High, Low, and Close
    plt.figure(figsize=(12, 8))
    plt.plot(plot_data.index, plot_data["Target_Open"], label="Actual Open Price", color='blue', linestyle='-', marker='o')
    plt.plot(plot_data.index, preds_open[-10:], label="Predicted Open Price", color='red', linestyle='--')

    plt.plot(plot_data.index, plot_data["Target_High"], label="Actual High Price", color='green', linestyle='-', marker='o')
    plt.plot(plot_data.index, preds_high[-10:], label="Predicted High Price", color='orange', linestyle='--')

    plt.plot(plot_data.index, plot_data["Target_Low"], label="Actual Low Price", color='purple', linestyle='-', marker='o')
    plt.plot(plot_data.index, preds_low[-10:], label="Predicted Low Price", color='yellow', linestyle='--')

    plt.plot(plot_data.index, plot_data["Target_Close"], label="Actual Close Price", color='cyan', linestyle='-', marker='o')
    plt.plot(plot_data.index, preds_close[-10:], label="Predicted Close Price", color='magenta', linestyle='--')

    plt.title("Actual vs Predicted Prices (Open, High, Low, Close)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

    # Predict the prices for the next 3 days based on the last row of the data
    last_data_point = stock_data.iloc[-1][predictors].values.reshape(1, -1)

    future_predictions = {
        'Open': model_open.predict(last_data_point),
        'High': model_high.predict(last_data_point),
        'Low': model_low.predict(last_data_point),
        'Close': model_close.predict(last_data_point)
    }

    # Print the predicted prices for the next 3 days
    print("\nPredicted prices for the next 3 days:")
    for day in range(1, 4):
        print(f"Day {day}:")
        future_predictions['Open'] = model_open.predict(last_data_point)
        future_predictions['High'] = model_high.predict(last_data_point)
        future_predictions['Low'] = model_low.predict(last_data_point)
        future_predictions['Close'] = model_close.predict(last_data_point)
        last_data_point = future_predictions  # Update the data point with predictions for the next day

        print(f"Predicted Open: {future_predictions['Open'][0]:.2f}")
        print(f"Predicted High: {future_predictions['High'][0]:.2f}")
        print(f"Predicted Low: {future_predictions['Low'][0]:.2f}")
        print(f"Predicted Close: {future_predictions['Close'][0]:.2f}")
        print("")

# Function to process the specific stock file based on the stock_name
def process_specific_stock(stock_name, stock_folder):
    # Convert stock_name to uppercase to handle case insensitivity
    stock_name = stock_name.upper()
    
    # Construct the file path for the specific stock file
    file_path = os.path.join(stock_folder, f"{stock_name}.csv")
    
    # Check if the file exists
    if os.path.exists(file_path):
        print(f"\nProcessing {stock_name}.csv...")
        stock_data = load_and_process_stock_data(file_path)
        if not stock_data.empty:
            train_and_evaluate_model(stock_data)
    else:
        print(f"{stock_name}.csv not found in the Stock folder.")

# Set the path for your Stock folder
stock_folder = "Stock"  # Assuming Stock folder is inside the paisapro directory

# Ask for the stock symbol input from the user (case-insensitive)
stock_name = input("Enter the stock name (e.g., 'ADBL'): ")

# Process the specific stock you want by passing the stock_name
process_specific_stock(stock_name, stock_folder)
