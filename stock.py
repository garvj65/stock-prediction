import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# GET STOCK TICKER FROM USER
# ============================================================================
print("\n" + "="*70)
print("STOCK PRICE PREDICTION & ANALYSIS TOOL")
print("="*70)
print("\nPopular stocks: AAPL (Apple), MSFT (Microsoft), GOOGL (Google),")
print("   TSLA (Tesla), AMZN (Amazon), META (Meta), NVDA (Nvidia)\n")

stock_ticker = input("🔍 Enter stock ticker (e.g., AAPL): ").upper().strip()

if not stock_ticker:
    stock_ticker = "AAPL"
    print(f" No input provided, using default: {stock_ticker}\n")
else:
    print(f" Selected: {stock_ticker}\n")

# ============================================================================
# STEP 1: FETCH REAL STOCK DATA
# ============================================================================
print(" FETCHING STOCK DATA...")
print("=" * 70)

# Download stock prices from 2018 to 2024
stock = yf.download(stock_ticker, start="2018-01-01", end="2026-01-10", progress=False)
stock = stock.dropna()

print(f" Successfully loaded {len(stock)} trading days of data")
print(f" Price range: ${float(stock['Close'].min()):.2f} - ${float(stock['Close'].max()):.2f}")
print(f" Date range: {stock.index[0].date()} to {stock.index[-1].date()}\n")

print("First few days of data:")
print(stock.head(3))
print()

# ============================================================================
# STEP 2: VISUALIZE HISTORICAL PRICES
# ============================================================================
print(" VISUALIZING HISTORICAL PRICES...")
print("=" * 70)

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(stock.index, stock['Close'], linewidth=2, color='#1f77b4', label='Closing Price')
ax.set_title(f'{stock_ticker} Stock Closing Price Over Time (2018-2026)', fontsize=14, fontweight='bold')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Price ($)', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================================
# STEP 3: CREATE TECHNICAL INDICATORS
# ============================================================================
print(" CREATING TECHNICAL INDICATORS...")
print("=" * 70)

# Moving Average (10-day): Average price over last 10 days - smooths out daily noise
stock['MA_10'] = stock['Close'].rolling(window=10).mean()
print("✓ 10-day Moving Average: Smooths out daily price fluctuations")

# Moving Average (50-day): Average price over last 50 days - shows medium-term trend
stock['MA_50'] = stock['Close'].rolling(window=50).mean()
print("✓ 50-day Moving Average: Shows the medium-term trend\n")

# Daily Return: How much the price changed from yesterday
# E.g., if yesterday was $100 and today is $105, return = 5%
stock['Return'] = stock['Close'].pct_change()
print("✓ Daily Return: Percentage change from previous day\n")

# Remove rows with NaN values created by rolling averages
stock = stock.dropna()
print(f" After processing: {len(stock)} valid trading days\n")

# ============================================================================
# STEP 4: VISUALIZE MOVING AVERAGES (TRADING STRATEGIES)
# ============================================================================
print(" VISUALIZING MOVING AVERAGES...")
print("=" * 70)

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(stock.index, stock['Close'], label='Actual Price', linewidth=1.5, alpha=0.7, color='gray')
ax.plot(stock.index, stock['MA_10'], label='10-Day Average (Short-term Trend)', 
        linewidth=2, color='orange')
ax.plot(stock.index, stock['MA_50'], label='50-Day Average (Medium-term Trend)', 
        linewidth=2, color='red')

ax.set_title(f'{stock_ticker} Stock: Price with Moving Averages', fontsize=14, fontweight='bold')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Price ($)', fontsize=12)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)

# Fun insight: When short-term MA crosses above long-term MA, it's often a buy signal!
ax.text(stock.index[len(stock)//2], stock['Close'].max() * 0.95, 
        ' When orange line crosses red: Consider buying!\nWhen orange dips below red: Caution!',
        fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.show()

# ============================================================================
# STEP 5: PREPARE DATA FOR MACHINE LEARNING MODEL
# ============================================================================
print(" PREPARING DATA FOR PREDICTION MODEL...")
print("=" * 70)

# Use moving averages and daily returns to predict future price
# Think of it like: "Based on recent trends, what will price be tomorrow?"
X = stock[['MA_10', 'MA_50', 'Return']]
y = stock['Close']

print(f" Features used: 10-day MA, 50-day MA, Daily Return")
print(f" Target: Stock closing price")
print(f" Total samples: {len(X)}\n")

# Split data: 80% for training (teaching the model), 20% for testing (evaluating)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # shuffle=False keeps time order intact
)

print(f" Training data: {len(X_train)} days (2018-2023)")
print(f" Testing data: {len(X_test)} days (2023-2024)\n")

# ============================================================================
# STEP 6: TRAIN PREDICTION MODEL
# ============================================================================
print(" TRAINING LINEAR REGRESSION MODEL...")
print("=" * 70)

model = LinearRegression()
model.fit(X_train, y_train)

# Flatten coefficients in case the model returns a 2D array
coefs = np.ravel(model.coef_)

print(" Model trained successfully!")
print(f"   - Coefficient for 10-day MA: {coefs[0]:.4f}")
print(f"   - Coefficient for 50-day MA: {coefs[1]:.4f}")
print(f"   - Coefficient for Daily Return: {coefs[2]:.4f}\n")

# ============================================================================
# STEP 7: MAKE PREDICTIONS & EVALUATE
# ============================================================================
print("  MAKING PREDICTIONS & EVALUATING MODEL...")
print("=" * 70)

# Make predictions on test data
predictions = model.predict(X_test)

# Calculate accuracy metrics
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)
mean_actual = y_test.mean()
mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100  # Mean Absolute Percentage Error

print(" MODEL PERFORMANCE METRICS:")
print(f"   • RMSE (Root Mean Squared Error): ${rmse:.2f}")
print(f"     └─ Average prediction error: ±${rmse:.2f}")
print(f"   • R² Score: {r2:.4f}")
print(f"     └─ Explains {r2*100:.2f}% of price variations")
print(f"   • MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
print(f"     └─ Average prediction accuracy: ±{mape:.2f}%\n")

# Show some sample predictions
print(" SAMPLE PREDICTIONS:")
print(f"{'Actual Price':<15} {'Predicted Price':<20} {'Error':<15}")
print("-" * 50)
for i in range(min(10, len(y_test))):
   actual = float(np.ravel(y_test)[i])
   pred = float(np.ravel(predictions)[i])
   error = abs(actual - pred)
   print(f"${actual:<14.2f} ${pred:<19.2f} ±${error:<14.2f}")
print()

# ============================================================================
# STEP 8: VISUALIZE PREDICTIONS VS ACTUAL
# ============================================================================
print(" VISUALIZING PREDICTIONS...")
print("=" * 70)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Full comparison
ax1.plot(y_test.index, y_test.values, label='Actual Price', linewidth=2.5, 
         marker='o', markersize=3, color='#2ca02c')
ax1.plot(y_test.index, predictions, label='Predicted Price', linewidth=2.5, 
         marker='s', markersize=3, color='#d62728', alpha=0.8)
ax1.set_title('Actual vs Predicted Apple Stock Price (2023-2024)', 
              fontsize=14, fontweight='bold')
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Price ($)', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: Prediction errors
errors = np.ravel(y_test.values - predictions)  # flatten to 1D for plotting
indices = np.arange(len(errors))
colors = ['red' if e < 0 else 'green' for e in errors]
ax2.bar(indices, errors, color=colors, alpha=0.7, edgecolor='black', linewidth=0.8)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax2.set_title('Prediction Errors (Negative = Underestimated, Positive = Overestimated)', 
              fontsize=14, fontweight='bold')
ax2.set_xlabel('Days in Test Set', fontsize=12)
ax2.set_ylabel('Error ($)', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# ============================================================================
# STEP 9: CREATE PERFORMANCE DASHBOARD
# ============================================================================
print(" CREATING PERFORMANCE DASHBOARD...")
print("=" * 70)

# Create a comprehensive dashboard
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

# Color scheme
header_color = '#1f77b4'
metric_color = '#d62728'
success_color = '#2ca02c'
warning_color = '#ff7f0e'

# ============================================================================
# Dashboard Title & Stock Info (Top)
# ============================================================================
ax_title = fig.add_subplot(gs[0, :])
ax_title.axis('off')
ax_title.text(0.5, 0.8, f' {stock_ticker} STOCK ANALYSIS DASHBOARD', 
              ha='center', va='center', fontsize=24, fontweight='bold', color=header_color)
ax_title.text(0.5, 0.4, f'Analysis Period: {stock.index[0].date()} to {stock.index[-1].date()} | Total Days: {len(stock):,}', 
              ha='center', va='center', fontsize=12, color='gray')

# ============================================================================
# Metric Cards (Left Column)
# ============================================================================
# Card 1: Current Price
ax_price = fig.add_subplot(gs[1, 0])
ax_price.axis('off')
current_price = float(stock['Close'].iloc[-1])
ax_price.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor='#e8f4f8', edgecolor=header_color, linewidth=2))
ax_price.text(0.5, 0.7, 'Current Price', ha='center', fontsize=11, fontweight='bold')
ax_price.text(0.5, 0.35, f'${current_price:.2f}', ha='center', fontsize=18, fontweight='bold', color=header_color)
ax_price.set_xlim(0, 1)
ax_price.set_ylim(0, 1)

# Card 2: Price Range
ax_range = fig.add_subplot(gs[1, 1])
ax_range.axis('off')
min_price = float(stock['Close'].min())
max_price = float(stock['Close'].max())
ax_range.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor='#f0e8f8', edgecolor=metric_color, linewidth=2))
ax_range.text(0.5, 0.75, 'Price Range', ha='center', fontsize=11, fontweight='bold')
ax_range.text(0.5, 0.5, f'${min_price:.2f}', ha='center', fontsize=10, color='red')
ax_range.text(0.5, 0.25, f'${max_price:.2f}', ha='center', fontsize=10, color='green')
ax_range.set_xlim(0, 1)
ax_range.set_ylim(0, 1)

# Card 3: Volatility
ax_vol = fig.add_subplot(gs[1, 2])
ax_vol.axis('off')
volatility = float(stock['Return'].std() * 100)
ax_vol.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor='#f8f4e8', edgecolor=warning_color, linewidth=2))
ax_vol.text(0.5, 0.7, 'Daily Volatility', ha='center', fontsize=11, fontweight='bold')
ax_vol.text(0.5, 0.35, f'{volatility:.2f}%', ha='center', fontsize=18, fontweight='bold', color=warning_color)
ax_vol.set_xlim(0, 1)
ax_vol.set_ylim(0, 1)

# ============================================================================
# Model Performance Metrics (Middle Row)
# ============================================================================
# Card 4: RMSE
ax_rmse = fig.add_subplot(gs[2, 0])
ax_rmse.axis('off')
ax_rmse.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor='#e8f8e8', edgecolor=success_color, linewidth=2))
ax_rmse.text(0.5, 0.7, 'RMSE', ha='center', fontsize=11, fontweight='bold')
ax_rmse.text(0.5, 0.35, f'${rmse:.2f}', ha='center', fontsize=16, fontweight='bold', color=success_color)
ax_rmse.set_xlim(0, 1)
ax_rmse.set_ylim(0, 1)

# Card 5: R² Score
ax_r2 = fig.add_subplot(gs[2, 1])
ax_r2.axis('off')
ax_r2.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor='#e8f8f8', edgecolor=header_color, linewidth=2))
ax_r2.text(0.5, 0.7, 'R² Score', ha='center', fontsize=11, fontweight='bold')
ax_r2.text(0.5, 0.35, f'{r2*100:.2f}%', ha='center', fontsize=16, fontweight='bold', color=header_color)
ax_r2.set_xlim(0, 1)
ax_r2.set_ylim(0, 1)

# Card 6: Accuracy (MAPE)
ax_mape = fig.add_subplot(gs[2, 2])
ax_mape.axis('off')
ax_mape.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor='#f8e8f8', edgecolor=metric_color, linewidth=2))
ax_mape.text(0.5, 0.7, 'Accuracy (MAPE)', ha='center', fontsize=11, fontweight='bold')
ax_mape.text(0.5, 0.35, f'{100-mape:.2f}%', ha='center', fontsize=16, fontweight='bold', color=metric_color)
ax_mape.set_xlim(0, 1)
ax_mape.set_ylim(0, 1)

plt.suptitle('')  # Add spacing
plt.tight_layout()
plt.show()

print("=" * 70)
