import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
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
    stock_data["Tomorrow"] = stock_data["Close Price"].shift(-1)
    stock_data["Target"] = (stock_data["Tomorrow"] > stock_data["Close Price"]).astype(int)

    # Adding rolling averages and trend indicators (only apply rolling features that are sensible for the dataset size)
    horizons = [2, 5]  # Limited horizons for small dataset
    for horizon in horizons:
        rolling = stock_data.rolling(horizon, min_periods=1)  # Min period to avoid dropping data
        stock_data[f"Close_Ratio_{horizon}"] = stock_data["Close Price"] / rolling.mean()["Close Price"]
        stock_data[f"Trend_{horizon}"] = rolling.sum()["Target"]
    
    # Adding lagged features
    lags = [1, 2, 3, 5]
    for lag in lags:
        stock_data[f"Lag_{lag}"] = stock_data["Close Price"].shift(lag)
    
    # Clean data by dropping NaN values (only drop rows where necessary)
    stock_data = stock_data.dropna()
    
    # Print shape to check the data
    print(f"Data after feature engineering: {stock_data.shape}")
    
    return stock_data

# Function to train and evaluate the model
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

    # Model training
    model = RandomForestClassifier(random_state=1)
    model.fit(train[predictors], train["Target"])

    # Predicting on the test set
    preds = model.predict_proba(test[predictors])[:, 1]
    preds = (preds >= 0.6).astype(int)  # Threshold for classification
    preds = pd.Series(preds, index=test.index, name="Predictions")
    
    # Check if predictions are made correctly
    print(f"Predictions: {preds.head()}")
    
    # Precision score
    precision = precision_score(test["Target"], preds)
    print(f"Precision: {precision}")
    
    # Filter the last 2-3 months' data (for example, if you have 23 days, you can take the last 15)
    plot_data = test[-10:]  # Adjust the number of points to match the last 2-3 months of data
    
    # Ensure there is data for plotting
    if plot_data.empty:
        print("No data available for plotting.")
        return

    # Plot predictions vs actual for the filtered period
    combined = pd.concat([plot_data["Target"], preds], axis=1)
    ax = combined.plot(title="Stock Predicted vs Actual", figsize=(10, 6))
    
    # Set xlim to focus on the specific range (last 2-3 months of data)
    ax.set_xlim([plot_data.index.min(), plot_data.index.max()])  # Focus on the limited range
    plt.show()

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
