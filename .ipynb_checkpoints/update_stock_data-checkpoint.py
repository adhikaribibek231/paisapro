import os
import pandas as pd

# Define folder paths
base_folder = "paisapro"
daily_summary_folder = os.path.join(base_folder, "Daily Stock Summary")
stock_wise_folder = os.path.join(base_folder, "Stock Wise")

# Create the stock-wise folder if it doesn't exist
os.makedirs(stock_wise_folder, exist_ok=True)

def process_daily_summaries():
    # Get all daily summary files
    daily_files = [f for f in os.listdir(daily_summary_folder) if f.endswith('.csv')]

    for daily_file in daily_files:
        # Read the daily summary file
        daily_file_path = os.path.join(daily_summary_folder, daily_file)
        daily_data = pd.read_csv(daily_file_path)
        
        # Ensure the daily data has a 'Business Date' column
        if 'Business Date' not in daily_data.columns:
            print(f"Skipping file {daily_file}: 'Business Date' column not found.")
            continue
        
        daily_data['Business Date'] = pd.to_datetime(daily_data['Business Date'])  # Convert to datetime
        daily_date = daily_file.split(".")[0]  # Extract the date from filename

        # Process each symbol in the daily data
        for symbol in daily_data['Symbol'].unique():
            symbol_data = daily_data[daily_data['Symbol'] == symbol]
            symbol_file = os.path.join(stock_wise_folder, f"{symbol}.csv")
            
            # If the symbol file exists, append new data
            if os.path.exists(symbol_file):
                existing_data = pd.read_csv(symbol_file, parse_dates=['Business Date'])
                updated_data = pd.concat([existing_data, symbol_data]).drop_duplicates(subset=['Business Date']).sort_values('Business Date')
            else:
                updated_data = symbol_data
            
            # Save the updated symbol data back to the file
            updated_data.to_csv(symbol_file, index=False)
            print(f"Updated {symbol}.csv")

# Run the process
process_daily_summaries()