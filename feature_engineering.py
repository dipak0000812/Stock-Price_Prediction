import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("STEP 2: FEATURE ENGINEERING - TECHNICAL INDICATORS")
print("="*70)

# ============================================
# LOAD DATA
# ============================================
print("\n📂 Loading Tesla stock data...")
df = pd.read_csv('tesla_stock_data.csv', index_col='Date', parse_dates=True)

# If multi-level columns exist, flatten them
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

print(f"✅ Data loaded: {len(df)} records")
print(f"📅 Date range: {df.index[0].date()} to {df.index[-1].date()}")

# Display original data
print("\n" + "="*70)
print("ORIGINAL DATA (First 5 rows):")
print("="*70)
print(df.head())

# ============================================
# PART 1: SIMPLE MOVING AVERAGES (SMA)
# ============================================
print("\n" + "="*70)
print("CREATING SIMPLE MOVING AVERAGES (SMA)...")
print("="*70)

# SMA Formula: Average of last N closing prices
# Example: SMA(5) = (Price₁ + Price₂ + Price₃ + Price₄ + Price₅) / 5

# SMA 20 days (Short-term trend - ~1 month)
df['SMA_20'] = df['Close'].rolling(window=20).mean()
# rolling(window=20) = take last 20 values
# .mean() = calculate average

# SMA 50 days (Medium-term trend - ~2.5 months)
df['SMA_50'] = df['Close'].rolling(window=50).mean()

# SMA 200 days (Long-term trend - ~10 months)
df['SMA_200'] = df['Close'].rolling(window=200).mean()

print("✅ Created SMA_20, SMA_50, SMA_200")
print(f"   Example (latest): Close=${df['Close'].iloc[-1]:.2f}, SMA_20=${df['SMA_20'].iloc[-1]:.2f}")

# ============================================
# PART 2: EXPONENTIAL MOVING AVERAGES (EMA)
# ============================================
print("\n" + "="*70)
print("CREATING EXPONENTIAL MOVING AVERAGES (EMA)...")
print("="*70)

# EMA Formula: Gives more weight to recent prices
# EMA = (Price × multiplier) + (Previous EMA × (1 - multiplier))
# multiplier = 2 / (span + 1)

# EMA 12 days (Fast EMA - for MACD)
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
# ewm() = exponentially weighted moving average
# span=12 = use 12 periods
# adjust=False = standard EMA formula

# EMA 26 days (Slow EMA - for MACD)
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

# EMA 9 days (Signal line - for MACD)
df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()

print("✅ Created EMA_12, EMA_26, EMA_9")
print(f"   EMA reacts faster than SMA to price changes")

# ============================================
# PART 3: MACD (Moving Average Convergence Divergence)
# ============================================
print("\n" + "="*70)
print("CREATING MACD INDICATOR...")
print("="*70)

# MACD Line = EMA(12) - EMA(26)
# Shows momentum by comparing fast and slow EMAs
df['MACD'] = df['EMA_12'] - df['EMA_26']

# Signal Line = EMA(9) of MACD
# Smoothed version of MACD for comparison
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# MACD Histogram = MACD - Signal
# Visual representation of difference
# Positive histogram = bullish, Negative = bearish
df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

print("✅ Created MACD, MACD_Signal, MACD_Histogram")
print(f"   Latest MACD: {df['MACD'].iloc[-1]:.2f}")
print(f"   Latest Signal: {df['MACD_Signal'].iloc[-1]:.2f}")

# Trading signal
if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
    print("   📈 MACD Signal: BULLISH (MACD above Signal)")
else:
    print("   📉 MACD Signal: BEARISH (MACD below Signal)")

# ============================================
# PART 4: RSI (Relative Strength Index)
# ============================================
print("\n" + "="*70)
print("CREATING RSI INDICATOR...")
print("="*70)

# RSI measures momentum on 0-100 scale
# RSI > 70 = Overbought (might go down)
# RSI < 30 = Oversold (might go up)

# Step 1: Calculate price changes
delta = df['Close'].diff()  # Today's price - Yesterday's price
# diff() calculates difference between consecutive rows

# Step 2: Separate gains and losses
gain = delta.where(delta > 0, 0)  # Keep positive changes, zero out negative
loss = -delta.where(delta < 0, 0)  # Keep negative changes (as positive), zero out positive
# where(condition, value_if_true, value_if_false)

# Step 3: Calculate average gain and loss (14 periods standard)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()

# Step 4: Calculate Relative Strength (RS)
rs = avg_gain / avg_loss
# If avg_gain = 3 and avg_loss = 1, then RS = 3

# Step 5: Calculate RSI
# RSI Formula: 100 - (100 / (1 + RS))
df['RSI'] = 100 - (100 / (1 + rs))

print("✅ Created RSI (Relative Strength Index)")
print(f"   Latest RSI: {df['RSI'].iloc[-1]:.2f}")

# Interpret RSI
latest_rsi = df['RSI'].iloc[-1]
if latest_rsi > 70:
    print("   ⚠️  RSI Signal: OVERBOUGHT (RSI > 70)")
elif latest_rsi < 30:
    print("   ⚠️  RSI Signal: OVERSOLD (RSI < 30)")
else:
    print("   ✅ RSI Signal: NEUTRAL (30 < RSI < 70)")

# ============================================
# PART 5: BOLLINGER BANDS
# ============================================
print("\n" + "="*70)
print("CREATING BOLLINGER BANDS...")
print("="*70)

# Bollinger Bands show volatility
# Middle Band = SMA(20)
# Upper Band = SMA(20) + (2 × Standard Deviation)
# Lower Band = SMA(20) - (2 × Standard Deviation)

# Middle Band (already have SMA_20)
df['BB_Middle'] = df['SMA_20']

# Calculate standard deviation (volatility measure)
df['BB_Std'] = df['Close'].rolling(window=20).std()
# std() = standard deviation (how spread out prices are)

# Upper Band
df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])

# Lower Band
df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])

# BB Width (measures volatility)
df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
# Wide bands = high volatility, Narrow bands = low volatility

# Price position in BB (0 to 1 scale)
# 0 = at lower band, 0.5 = at middle, 1 = at upper band
df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

print("✅ Created Bollinger Bands (Upper, Middle, Lower, Width, Position)")
print(f"   Current price position in BB: {df['BB_Position'].iloc[-1]:.2f}")

# Interpret BB position
if df['BB_Position'].iloc[-1] > 0.8:
    print("   📈 BB Signal: Near UPPER band (might reverse down)")
elif df['BB_Position'].iloc[-1] < 0.2:
    print("   📉 BB Signal: Near LOWER band (might reverse up)")
else:
    print("   ✅ BB Signal: In MIDDLE range (normal)")

# ============================================
# PART 6: ADDITIONAL MOMENTUM INDICATORS
# ============================================
print("\n" + "="*70)
print("CREATING ADDITIONAL INDICATORS...")
print("="*70)

# 1. Rate of Change (ROC) - Percentage change over N days
df['ROC_10'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
# shift(10) = get value from 10 days ago
# Formula: ((Today - 10 days ago) / 10 days ago) × 100

# 2. Average True Range (ATR) - Volatility measure
high_low = df['High'] - df['Low']
high_close = np.abs(df['High'] - df['Close'].shift())
low_close = np.abs(df['Low'] - df['Close'].shift())
true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df['ATR'] = true_range.rolling(window=14).mean()

# 3. On-Balance Volume (OBV) - Cumulative volume indicator
obv = []
obv_value = 0
for i in range(len(df)):
    if i == 0:
        obv.append(df['Volume'].iloc[i])
        obv_value = df['Volume'].iloc[i]
    else:
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv_value += df['Volume'].iloc[i]  # Price up = add volume
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv_value -= df['Volume'].iloc[i]  # Price down = subtract volume
        obv.append(obv_value)
df['OBV'] = obv

# 4. Volume Moving Average
df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()

print("✅ Created ROC_10, ATR, OBV, Volume_MA_20")

# ============================================
# PART 7: DERIVED FEATURES
# ============================================
print("\n" + "="*70)
print("CREATING DERIVED FEATURES...")
print("="*70)

# 1. Daily Return (percentage change)
df['Daily_Return'] = df['Close'].pct_change() * 100

# 2. Daily High-Low Range
df['Daily_Range'] = df['High'] - df['Low']

# 3. Daily Range Percentage
df['Daily_Range_Pct'] = (df['Daily_Range'] / df['Close']) * 100

# 4. Distance from SMA (how far price is from moving average)
df['Distance_SMA_20'] = ((df['Close'] - df['SMA_20']) / df['SMA_20']) * 100
df['Distance_SMA_50'] = ((df['Close'] - df['SMA_50']) / df['SMA_50']) * 100

# 5. Volume Ratio (today's volume vs average)
df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']

# 6. Volatility (20-day rolling standard deviation)
df['Volatility'] = df['Daily_Return'].rolling(window=20).std()

# 7. Golden/Death Cross Signal
# Golden Cross: When SMA_50 crosses above SMA_200 (bullish)
# Death Cross: When SMA_50 crosses below SMA_200 (bearish)
df['SMA_Cross'] = df['SMA_50'] - df['SMA_200']

print("✅ Created derived features:")
print("   - Daily_Return, Daily_Range, Daily_Range_Pct")
print("   - Distance_SMA_20, Distance_SMA_50")
print("   - Volume_Ratio, Volatility, SMA_Cross")

# ============================================
# PART 8: CLEAN DATA
# ============================================
print("\n" + "="*70)
print("CLEANING DATA...")
print("="*70)

# Check for NaN values (will exist in first rows due to rolling calculations)
print(f"Missing values before cleaning:")
print(df.isnull().sum())

# Drop rows with NaN values
# (First 200 rows will have NaN due to SMA_200 calculation)
df_clean = df.dropna()

print(f"\n✅ Cleaned data: {len(df_clean)} records (dropped {len(df) - len(df_clean)} rows with NaN)")

# Save engineered features
df_clean.to_csv('tesla_features_engineered.csv')
print("💾 Saved to 'tesla_features_engineered.csv'")

# ============================================
# PART 9: DISPLAY SUMMARY
# ============================================
print("\n" + "="*70)
print("FEATURE ENGINEERING SUMMARY:")
print("="*70)

all_features = [
    'Close', 'Volume',
    'SMA_20', 'SMA_50', 'SMA_200',
    'EMA_12', 'EMA_26', 'EMA_9',
    'MACD', 'MACD_Signal', 'MACD_Histogram',
    'RSI',
    'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width', 'BB_Position',
    'ROC_10', 'ATR', 'OBV', 'Volume_MA_20',
    'Daily_Return', 'Daily_Range', 'Daily_Range_Pct',
    'Distance_SMA_20', 'Distance_SMA_50',
    'Volume_Ratio', 'Volatility', 'SMA_Cross'
]

print(f"\n📊 Total Features Created: {len(all_features)}")
print("\nFeature Categories:")
print("  • Original: Close, Volume")
print("  • Moving Averages: SMA_20, SMA_50, SMA_200, EMA_12, EMA_26, EMA_9")
print("  • Momentum: MACD, MACD_Signal, MACD_Histogram, RSI, ROC_10")
print("  • Volatility: BB_Upper, BB_Middle, BB_Lower, BB_Width, ATR, Volatility")
print("  • Volume: OBV, Volume_MA_20, Volume_Ratio")
print("  • Derived: Daily_Return, Daily_Range, Distance_SMA_20/50, SMA_Cross")

print("\n📈 Latest Values:")
print(f"  Close:        ${df_clean['Close'].iloc[-1]:.2f}")
print(f"  SMA_20:       ${df_clean['SMA_20'].iloc[-1]:.2f}")
print(f"  RSI:          {df_clean['RSI'].iloc[-1]:.2f}")
print(f"  MACD:         {df_clean['MACD'].iloc[-1]:.2f}")
print(f"  BB_Position:  {df_clean['BB_Position'].iloc[-1]:.2f}")

# ============================================
# PART 10: VISUALIZATIONS
# ============================================
print("\n" + "="*70)
print("CREATING VISUALIZATIONS...")
print("="*70)

fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

# Plot 1: Price with Moving Averages
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df_clean.index, df_clean['Close'], label='Close Price', linewidth=2, color='black')
ax1.plot(df_clean.index, df_clean['SMA_20'], label='SMA 20', linewidth=1.5, alpha=0.7)
ax1.plot(df_clean.index, df_clean['SMA_50'], label='SMA 50', linewidth=1.5, alpha=0.7)
ax1.plot(df_clean.index, df_clean['SMA_200'], label='SMA 200', linewidth=1.5, alpha=0.7)
ax1.set_title('Tesla Stock Price with Moving Averages', fontsize=14, fontweight='bold')
ax1.set_ylabel('Price ($)')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# Plot 2: Bollinger Bands
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(df_clean.index, df_clean['Close'], label='Close', color='black', linewidth=1.5)
ax2.plot(df_clean.index, df_clean['BB_Upper'], label='Upper Band', linestyle='--', alpha=0.7, color='red')
ax2.plot(df_clean.index, df_clean['BB_Middle'], label='Middle', linestyle='--', alpha=0.7, color='blue')
ax2.plot(df_clean.index, df_clean['BB_Lower'], label='Lower Band', linestyle='--', alpha=0.7, color='green')
ax2.fill_between(df_clean.index, df_clean['BB_Upper'], df_clean['BB_Lower'], alpha=0.1)
ax2.set_title('Bollinger Bands', fontsize=12, fontweight='bold')
ax2.set_ylabel('Price ($)')
ax2.legend(loc='best', fontsize=8)
ax2.grid(True, alpha=0.3)

# Plot 3: RSI
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(df_clean.index, df_clean['RSI'], color='purple', linewidth=1.5)
ax3.axhline(y=70, color='r', linestyle='--', label='Overbought (70)', alpha=0.7)
ax3.axhline(y=30, color='g', linestyle='--', label='Oversold (30)', alpha=0.7)
ax3.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
ax3.fill_between(df_clean.index, 30, 70, alpha=0.1, color='blue')
ax3.set_title('RSI (Relative Strength Index)', fontsize=12, fontweight='bold')
ax3.set_ylabel('RSI')
ax3.set_ylim(0, 100)
ax3.legend(loc='best', fontsize=8)
ax3.grid(True, alpha=0.3)

# Plot 4: MACD
ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(df_clean.index, df_clean['MACD'], label='MACD', linewidth=1.5, color='blue')
ax4.plot(df_clean.index, df_clean['MACD_Signal'], label='Signal', linewidth=1.5, color='red')
ax4.bar(df_clean.index, df_clean['MACD_Histogram'], label='Histogram', alpha=0.3, color='gray')
ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax4.set_title('MACD (Moving Average Convergence Divergence)', fontsize=12, fontweight='bold')
ax4.set_ylabel('MACD')
ax4.legend(loc='best', fontsize=8)
ax4.grid(True, alpha=0.3)

# Plot 5: Volume & OBV
ax5 = fig.add_subplot(gs[2, 1])
ax5_twin = ax5.twinx()
ax5.bar(df_clean.index, df_clean['Volume'], alpha=0.3, color='blue', label='Volume')
ax5_twin.plot(df_clean.index, df_clean['OBV'], color='orange', linewidth=2, label='OBV')
ax5.set_title('Volume & On-Balance Volume (OBV)', fontsize=12, fontweight='bold')
ax5.set_ylabel('Volume', color='blue')
ax5_twin.set_ylabel('OBV', color='orange')
ax5.legend(loc='upper left', fontsize=8)
ax5_twin.legend(loc='upper right', fontsize=8)
ax5.grid(True, alpha=0.3)

# Plot 6: Volatility & ATR
ax6 = fig.add_subplot(gs[3, 0])
ax6.plot(df_clean.index, df_clean['Volatility'], label='Volatility', color='red', linewidth=1.5)
ax6_twin = ax6.twinx()
ax6_twin.plot(df_clean.index, df_clean['ATR'], label='ATR', color='purple', linewidth=1.5)
ax6.set_title('Volatility & ATR (Average True Range)', fontsize=12, fontweight='bold')
ax6.set_ylabel('Volatility (%)', color='red')
ax6_twin.set_ylabel('ATR', color='purple')
ax6.legend(loc='upper left', fontsize=8)
ax6_twin.legend(loc='upper right', fontsize=8)
ax6.grid(True, alpha=0.3)

# Plot 7: Feature Correlation Heatmap
ax7 = fig.add_subplot(gs[3, 1])
# Select key features for correlation
corr_features = ['Close', 'SMA_20', 'RSI', 'MACD', 'BB_Position', 'Volume', 'ATR']
corr_matrix = df_clean[corr_features].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, ax=ax7, cbar_kws={'shrink': 0.8})
ax7.set_title('Feature Correlation Heatmap', fontsize=12, fontweight='bold')

plt.suptitle('Technical Indicators Analysis - Tesla Stock', fontsize=16, fontweight='bold', y=0.995)
plt.savefig('tesla_technical_indicators.png', dpi=300, bbox_inches='tight')
print("📊 Visualizations saved as 'tesla_technical_indicators.png'")
plt.show()

print("\n" + "="*70)
print("✅ STEP 2 COMPLETE - FEATURE ENGINEERING DONE!")
print("="*70)
print("\n📌 Next Steps:")
print("   3. Data Preprocessing (Normalization, Sequence Creation)")
print("   4. LSTM Model Building")
print("   5. Training & Evaluation")
print("   6. Predictions")