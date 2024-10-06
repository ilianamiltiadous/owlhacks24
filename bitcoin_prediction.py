import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Load and preprocess the data
df = pd.read_csv('bitcoin_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Feature engineering without look-ahead bias
def add_features(df):
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_30'] = df['Close'].rolling(window=30).mean()
    df['Volatility'] = df['Returns'].rolling(window=30).std()

    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df

# Apply feature engineering to the entire dataset
df = add_features(df)

# Remove NaN values
df = df.dropna()

# Prepare features and target
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'Log_Returns', 'MA_7', 'MA_30', 'Volatility', 'RSI']

# Split the data into training, validation, and test sets
test_size = 30  # Last 30 days for testing
val_size = int(0.1 * (len(df) - test_size))  # 10% of remaining data for validation
train_size = len(df) - test_size - val_size

train_df = df[:train_size]
val_df = df[train_size:-test_size]
test_df = df[-test_size:]

# Prepare features and target for each set
X_train, y_train = train_df[features], train_df['Close']
X_val, y_val = val_df[features], val_df['Close']
X_test, y_test = test_df[features], test_df['Close']

# Scale the features and target
scaler_X = RobustScaler()
scaler_y = RobustScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# Reshape the input data for LSTM
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_val_reshaped = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Build and train the LSTM model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(1, X_train.shape[1])),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train_reshaped, y_train_scaled, epochs=100, batch_size=32, validation_data=(X_val_reshaped, y_val_scaled), verbose=1)

# Make predictions for training, validation, and test data
y_train_pred_scaled = model.predict(X_train_reshaped)
y_val_pred_scaled = model.predict(X_val_reshaped)
y_test_pred_scaled = model.predict(X_test_reshaped)

# Inverse transform the predictions and actual values
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled)
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)

# Plot actual prices, trained data, validation data, and test data (last month)
plt.figure(figsize=(20, 10))

# Actual Prices
plt.plot(df['Date'], df['Close'], label='Actual Prices', color='blue', linewidth=2)

# Training Predictions
plt.plot(df['Date'][:train_size], y_train_pred, label='Training Predictions', 
         color='green', linestyle='--', alpha=0.7)

# Validation Predictions
plt.plot(df['Date'][train_size:-test_size], y_val_pred, label='Validation Predictions', 
         color='orange', linestyle='-.', alpha=0.7)

# Test Predictions (Last Month)
plt.plot(df['Date'][-test_size:], y_test_pred, label='Test Predictions (Last Month)', 
         color='red', linestyle=':', linewidth=3, marker='o', markersize=6)

plt.title('Bitcoin Price: Actual vs Predicted', fontsize=20, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price (USD)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Highlight the test period
plt.axvspan(df['Date'].iloc[-test_size], df['Date'].iloc[-1], 
            color='gray', alpha=0.2, label='Test Period (Last Month)')

# Add text annotations
plt.text(df['Date'].iloc[int(len(df)*0.1)], plt.ylim()[1], 'Training Data', 
         fontsize=12, color='green', verticalalignment='top')
plt.text(df['Date'].iloc[int(len(df)*0.85)], plt.ylim()[1], 'Validation', 
         fontsize=12, color='orange', verticalalignment='top')
plt.text(df['Date'].iloc[-int(test_size/2)], plt.ylim()[1], 'Test', 
         fontsize=12, color='red', verticalalignment='top')

plt.show()

# Calculate accuracy for the last month (test data)
last_month_accuracy = 100 - (np.mean(np.abs((df['Close'].values[-test_size:] - y_test_pred.flatten()) / df['Close'].values[-test_size:])) * 100)
print(f"\nModel accuracy for the last month (unseen data): {last_month_accuracy:.2f}%")

# Calculate the trend for the last month
actual_trend_last_month = ((df['Close'].iloc[-1] - df['Close'].iloc[-test_size]) / df['Close'].iloc[-test_size]) * 100
predicted_trend_last_month = ((y_test_pred[-1][0] - y_test_pred[0][0]) / y_test_pred[0][0]) * 100

print(f"\nActual trend over the last month: {actual_trend_last_month:.2f}%")
print(f"Predicted trend over the last month: {predicted_trend_last_month:.2f}%")

# Plot actual vs predicted for the test period
plt.figure(figsize=(15, 7))
plt.plot(df['Date'][-test_size:], df['Close'][-test_size:], label='Actual', color='blue')
plt.plot(df['Date'][-test_size:], y_test_pred, label='Predicted', color='red')
plt.title('Actual vs Predicted Bitcoin Prices (Test Period)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calculate additional metrics
mae = mean_absolute_error(df['Close'][-test_size:], y_test_pred)
rmse = np.sqrt(mean_squared_error(df['Close'][-test_size:], y_test_pred))
mape = np.mean(np.abs((df['Close'][-test_size:] - y_test_pred.flatten()) / df['Close'][-test_size:])) * 100

print(f"Mean Absolute Error: ${mae:.2f}")
print(f"Root Mean Squared Error: ${rmse:.2f}")
print(f"Mean Absolute Percentage Error: {mape:.2f}%")

# Print actual and predicted values
print("\nActual vs Predicted Prices:")
for i in range(test_size):
    print(f"Date: {df['Date'].iloc[-test_size+i].date()}, Actual: ${df['Close'].iloc[-test_size+i]:.2f}, Predicted: ${y_test_pred[i][0]:.2f}")