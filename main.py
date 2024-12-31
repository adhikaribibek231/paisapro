import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt

# Set the stock symbol you want to analyze (e.g., "ADBL")
stock_name = "ADBL"  # Change this to the stock you want to process (e.g., "AAPL", "TSLA")

# Function to load and process stock data
def load_and_process_stock_data(file_path):
    # Load stock data from CSV
    stock_data = pd.read_csv(file_path, parse_dates=["Business Date"], index_col="Business Date")
    
    # Select only the necessary columns for prediction
    stock_data = stock_data[["Open Price", "High Price", "Low Price", "Close Price", "Total Traded Quantity"]]
    
    # Feature Engineering
    stock_data["Tomorrow"] = stock_data["Close Price"].shift(-1)
    stock_data["Target"] = (stock_data["Tomorrow"] > stock_data["Close Price"]).astype(int)
    
    # Adding rolling averages and trend indicators
    horizons = [2, 5, 60, 250, 1000]
    for horizon in horizons:
        rolling = stock_data.rolling(horizon)
        stock_data[f"Close_Ratio_{horizon}"] = stock_data["Close Price"] / rolling.mean()["Close Price"]
        stock_data[f"Trend_{horizon}"] = rolling.sum()["Target"]
    
    # Adding lagged features
    lags = [1, 2, 3, 5, 10]
    for lag in lags:
        stock_data[f"Lag_{lag}"] = stock_data["Close Price"].shift(lag)
    
    # Clean data by dropping NaN values
    stock_data = stock_data.dropna()
    
    return stock_data

# Function to train and evaluate the model
def train_and_evaluate_model(stock_data):
    # Define predictors (use only the relevant columns)
    predictors = [col for col in stock_data.columns if "Close" in col or "Trend" in col or "Lag" in col]
    
    # If there are less than 10 samples, we'll use all available data for training/testing
    if len(stock_data) < 10:
        train = stock_data
        test = stock_data
    else:
        # Train with the first 18 days (as there are only 23 days of data)
        train = stock_data.iloc[:18]
        test = stock_data.iloc[18:]
    
    # Model training
    model = RandomForestClassifier(random_state=1)
    model.fit(train[predictors], train["Target"])

    # Predicting on the test set
    preds = model.predict_proba(test[predictors])[:, 1]
    preds = (preds >= 0.6).astype(int)  # Threshold for classification
    preds = pd.Series(preds, index=test.index, name="Predictions")
    
    # Precision score
    precision = precision_score(test["Target"], preds)
    print(f"Precision: {precision}")
    
    # Plot predictions vs actual
    combined = pd.concat([test["Target"], preds], axis=1)
    combined.plot(title="Stock Predicted vs Actual", figsize=(10, 6))
    plt.show()

# Function to process the specific stock file based on the stock_name
def process_specific_stock(stock_name, stock_folder):
    # Construct the file path for the specific stock file
    file_path = os.path.join(stock_folder, f"{stock_name}.csv")
    
    # Check if the file exists
    if os.path.exists(file_path):
        print(f"\nProcessing {stock_name}.csv...")
        stock_data = load_and_process_stock_data(file_path)
        train_and_evaluate_model(stock_data)
    else:
        print(f"{stock_name}.csv not found in the Stock folder.")

# Set the path for your Stock folder
stock_folder = "Stock"  # Assuming Stock folder is inside the paisapro directory

# Process the specific stock you want by passing the stock_name
process_specific_stock(stock_name, stock_folder)
