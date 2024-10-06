import yfinance as yf
import pandas as pd

# Set the ticker symbol for Bitcoin
ticker_symbol = "BTC-USD"

# Fetch all available historical data
btc_data = yf.Ticker(ticker_symbol)
df = btc_data.history(period="max")

# Reset index to make Date a column
df = df.reset_index()

# Keep relevant columns
df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

# Rename columns
df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

# Convert Date to string format
df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

# Display the first few rows and basic info
print(df.head())
print("\nDataset Info:")
print(f"Start Date: {df['Date'].iloc[0]}")
print(f"End Date: {df['Date'].iloc[-1]}")
print(f"Total Days: {len(df)}")

# Get additional information
info = btc_data.info
print("\nAdditional Information:")
print(f"Name: {info.get('longName', 'N/A')}")
print(f"Current Price: ${info.get('regularMarketPrice', 'N/A')}")
print(f"Market Cap: ${info.get('marketCap', 'N/A'):,}")
print(f"24h Volume: ${info.get('volume24h', 'N/A')}")
print(f"Circulating Supply: {info.get('circulatingSupply', 'N/A'):,} BTC")

# Optionally, save to CSV
df.to_csv('bitcoin_all_time_daily_prices.csv', index=False)
print("\nData saved to 'bitcoin_all_time_daily_prices.csv'")
