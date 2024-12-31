import os
import pandas as pd
import csv

# Define folder paths
base_folder = "E:/projects/paisapro"
daily_summary_folder = os.path.join(base_folder, "Daily Stock Summary")
stock_folder = os.path.join(base_folder, "Stock")

# Create the stock folder if it doesn't exist
os.makedirs(stock_folder, exist_ok=True)

def clean_file(file_path, temp_path):
    """
    Cleans a CSV file by removing rows with mismatched column counts.
    """
    with open(file_path, 'r', encoding='utf-8') as infile, open(temp_path, 'w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        header = next(reader)
        expected_columns = len(header)
        writer.writerow(header)  # Write the header to the cleaned file

        for i, row in enumerate(reader, start=2):
            if len(row) == expected_columns:
                writer.writerow(row)
            else:
                print(f"Skipped malformed row in {os.path.basename(file_path)}, Line {i}: {row}")

def process_daily_summaries():
    for file_name in os.listdir(daily_summary_folder):
        if not file_name.endswith('.csv'):
            continue

        daily_file_path = os.path.join(daily_summary_folder, file_name)
        temp_file_path = os.path.join(daily_summary_folder, f"temp_{file_name}")

        print(f"Processing file: {file_name}")

        # Step 1: Clean the file
        try:
            clean_file(daily_file_path, temp_file_path)
        except Exception as e:
            print(f"Error cleaning file: {file_name}\n{e}")
            continue

        # Step 2: Load cleaned data
        try:
            daily_data = pd.read_csv(temp_file_path)
        except Exception as e:
            print(f"Error reading cleaned file: {file_name}\n{e}")
            os.remove(temp_file_path)  # Clean up temporary file
            continue

        os.remove(temp_file_path)  # Remove temporary cleaned file

        # Step 3: Process data
        if 'Business Date' not in daily_data.columns or 'Symbol' not in daily_data.columns:
            print(f"Skipping file {file_name}: Missing required columns ('Business Date' or 'Symbol').")
            continue

        daily_data['Business Date'] = pd.to_datetime(daily_data['Business Date'], errors='coerce')
        if daily_data['Business Date'].isnull().any():
            print(f"File {file_name} contains invalid dates. Skipping rows with invalid dates.")
            daily_data = daily_data.dropna(subset=['Business Date'])

        for symbol in daily_data['Symbol'].unique():
            # Filter data for the current symbol
            symbol_data = daily_data[daily_data['Symbol'] == symbol]

            # Normalize symbol to prevent directory issues
            sanitized_symbol = symbol.replace("/", "_").replace("\\", "_").replace(":", "_")
            symbol_file = os.path.join(stock_folder, f"{sanitized_symbol}.csv")

            # Ensure the parent directory exists
            os.makedirs(os.path.dirname(symbol_file), exist_ok=True)

            # Append to or create the symbol-specific file
            if os.path.exists(symbol_file):
                existing_data = pd.read_csv(symbol_file, parse_dates=['Business Date'])
                combined_data = pd.concat([existing_data, symbol_data]).drop_duplicates(subset=['Business Date']).sort_values('Business Date')
            else:
                combined_data = symbol_data

            combined_data.to_csv(symbol_file, index=False)
            print(f"Updated {sanitized_symbol}.csv")

# Run the full process
process_daily_summaries()
