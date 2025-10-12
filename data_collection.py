import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*60)
print("TESLA STOCK PRICE PREDICTION - DATA COLLECTION")
print("="*60)

# Download Tesla stock data
# We'll get 5 years of data for good training
stock_symbol = 'TSLA'
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)  # 5 years

print(f"\n📥 Downloading {stock_symbol} data from {start_date.date()} to {end_date.date()}...")
df = yf.download(stock_symbol, start=start_date, end=end_date, progress=False)

# Fix: Flatten multi-level columns if they exist
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

print(f"\n✅ Data downloaded successfully!")
print(f"📊 Total records: {len(df)}")
print(f"📅 Date range: {df.index[0].date()} to {df.index[-1].date()}")

# Display first few rows
print("\n" + "="*60)
print("FIRST 5 ROWS OF DATA:")
print("="*60)
print(df.head())

# Display basic statistics
print("\n" + "="*60)
print("STATISTICAL SUMMARY:")
print("="*60)
print(df.describe())

# Check for missing values
print("\n" + "="*60)
print("MISSING VALUES CHECK:")
print("="*60)
missing = df.isnull().sum()
print(missing)
if missing.sum() == 0:
    print("✅ No missing values found!")
else:
    print("⚠️ Missing values detected. We'll handle them in preprocessing.")

# Data info
print("\n" + "="*60)
print("DATA INFORMATION:")
print("="*60)
print(df.info())

# Save the data
df.to_csv('tesla_stock_data.csv')
print("\n💾 Data saved to 'tesla_stock_data.csv'")

# ============================================
# VISUALIZATIONS
# ============================================

print("\n" + "="*60)
print("CREATING VISUALIZATIONS...")
print("="*60)

fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle('Tesla Stock Data - Exploratory Analysis', fontsize=16, fontweight='bold')

# 1. Closing Price Over Time
axes[0, 0].plot(df.index, df['Close'], color='blue', linewidth=1.5)
axes[0, 0].set_title('Closing Price Over Time', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Price ($)')
axes[0, 0].grid(True, alpha=0.3)

# 2. Volume Traded
axes[0, 1].bar(df.index, df['Volume'], color='green', alpha=0.6, width=1)
axes[0, 1].set_title('Trading Volume Over Time', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Volume')
axes[0, 1].grid(True, alpha=0.3)

# 3. Daily Price Range (High - Low)
daily_range = df['High'] - df['Low']
axes[1, 0].fill_between(df.index, daily_range, color='orange', alpha=0.6)
axes[1, 0].set_title('Daily Price Range (High - Low)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('Price Range ($)')
axes[1, 0].grid(True, alpha=0.3)

# 4. Distribution of Closing Prices
axes[1, 1].hist(df['Close'], bins=50, color='purple', alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Distribution of Closing Prices', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Price ($)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid(True, alpha=0.3)

# 5. Daily Returns
df['Daily_Return'] = df['Close'].pct_change() * 100
axes[2, 0].plot(df.index, df['Daily_Return'], color='red', alpha=0.7, linewidth=0.8)
axes[2, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[2, 0].set_title('Daily Returns (%)', fontsize=12, fontweight='bold')
axes[2, 0].set_xlabel('Date')
axes[2, 0].set_ylabel('Return (%)')
axes[2, 0].grid(True, alpha=0.3)

# 6. Candlestick-style visualization (simplified)
recent_data = df.tail(60)  # Last 60 days
colors = ['green' if close > open_ else 'red' 
          for close, open_ in zip(recent_data['Close'], recent_data['Open'])]
axes[2, 1].bar(range(len(recent_data)), 
               recent_data['Close'] - recent_data['Open'],
               bottom=recent_data['Open'],
               color=colors, alpha=0.7, width=0.8)
axes[2, 1].set_title('Price Movement (Last 60 Days)', fontsize=12, fontweight='bold')
axes[2, 1].set_xlabel('Days')
axes[2, 1].set_ylabel('Price ($)')
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tesla_exploratory_analysis.png', dpi=300, bbox_inches='tight')
print("📊 Visualizations saved as 'tesla_exploratory_analysis.png'")
plt.show()

# ============================================
# KEY INSIGHTS
# ============================================

print("\n" + "="*60)
print("KEY INSIGHTS:")
print("="*60)

avg_price = df['Close'].mean()
min_price = df['Close'].min()
max_price = df['Close'].max()
current_price = df['Close'].iloc[-1]
volatility = df['Daily_Return'].std()
avg_volume = df['Volume'].mean()

print(f"💰 Average Closing Price: ${avg_price:.2f}")
print(f"📉 Minimum Price: ${min_price:.2f} (on {df['Close'].idxmin().date()})")
print(f"📈 Maximum Price: ${max_price:.2f} (on {df['Close'].idxmax().date()})")
print(f"🎯 Current Price: ${current_price:.2f}")
print(f"📊 Daily Volatility (Std Dev): {volatility:.2f}%")
print(f"📦 Average Daily Volume: {avg_volume:,.0f} shares")

price_change = ((current_price - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
print(f"📈 Total Price Change: {price_change:+.2f}% over {len(df)} days")

print("\n" + "="*60)
print("✅ STEP 1 COMPLETE - Data Collection & Exploration Done!")
print("="*60)
print("\n📌 Next Steps:")
print("   1. Feature Engineering (Creating technical indicators)")
print("   2. Data Preprocessing (Normalization, sequence creation)")
print("   3. Model Building (LSTM architecture)")
print("   4. Training & Evaluation")