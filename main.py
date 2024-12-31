import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import matplotlib.pyplot as plt

# Function to load and process ADBL stock data
def load_and_process_adbl_data(file_path):
    # Load ADBL stock data
    adbl_data = pd.read_csv(file_path, parse_dates=["Business Date"], index_col="Business Date")
    
    # Select only the necessary columns for prediction
    adbl_data = adbl_data[["Open Price", "High Price", "Low Price", "Close Price", "Total Traded Quantity"]]
    
    # Feature Engineering
    adbl_data["Tomorrow"] = adbl_data["Close Price"].shift(-1)
    adbl_data["Target"] = (adbl_data["Tomorrow"] > adbl_data["Close Price"]).astype(int)
    
    # Adding rolling averages and trend indicators
    horizons = [2, 5, 60, 250, 1000]
    for horizon in horizons:
        rolling = adbl_data.rolling(horizon)
        adbl_data[f"Close_Ratio_{horizon}"] = adbl_data["Close Price"] / rolling.mean()["Close Price"]
        adbl_data[f"Trend_{horizon}"] = rolling.sum()["Target"]
    
    # Adding lagged features
    lags = [1, 2, 3, 5, 10]
    for lag in lags:
        adbl_data[f"Lag_{lag}"] = adbl_data["Close Price"].shift(lag)
    
    # Clean data by dropping NaN values
    adbl_data = adbl_data.dropna()
    
    return adbl_data

# File path for ADBL stock CSV file
adbl_file_path = "Stock/ADBL.csv"  # Change this to the correct path if needed

# Load and process ADBL stock data
adbl_data = load_and_process_adbl_data(adbl_file_path)

# Model training and evaluation for ADBL stock
def train_and_evaluate_adbl_model(adbl_data):
    # Define predictors (use only the relevant columns)
    predictors = [col for col in adbl_data.columns if "Close" in col or "Trend" in col or "Lag" in col]
    
    # Splitting the data into training and testing sets (last 100 data points for testing)
    train = adbl_data.iloc[:-100]
    test = adbl_data.iloc[-100:]
    
    # Hyperparameter tuning using GridSearchCV
    model = RandomForestClassifier(random_state=1)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'min_samples_split': [50, 100, 200],
        'max_depth': [10, 20, None]
    }
    tscv = TimeSeriesSplit(n_splits=5)  # Time-series split for cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='precision', n_jobs=-1)
    grid_search.fit(train[predictors], train["Target"])

    # Best model
    best_model = grid_search.best_estimator_

    # Evaluate model
    preds = best_model.predict_proba(test[predictors])[:, 1]
    preds = (preds >= 0.6).astype(int)  # Threshold for classification
    preds = pd.Series(preds, index=test.index, name="Predictions")
    
    # Precision score
    precision = precision_score(test["Target"], preds)
    print(f"Precision for ADBL: {precision}")
    
    # Plot predictions vs actual
    combined = pd.concat([test["Target"], preds], axis=1)
    combined.plot(title="ADBL Predicted vs Actual", figsize=(10, 6))
    plt.show()

# Train and evaluate the model for ADBL stock
train_and_evaluate_adbl_model(adbl_data)
