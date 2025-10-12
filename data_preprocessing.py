import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("STEP 3: DATA PREPROCESSING & SEQUENCE CREATION")
print("="*70)

# ============================================
# PART 1: LOAD ENGINEERED FEATURES
# ============================================
print("\n📂 Loading engineered features...")
df = pd.read_csv('tesla_features_engineered.csv', index_col='Date', parse_dates=True)

print(f"✅ Data loaded: {len(df)} records")
print(f"📅 Date range: {df.index[0].date()} to {df.index[-1].date()}")
print(f"📊 Total features: {len(df.columns)}")

# Display first few rows
print("\n" + "="*70)
print("FIRST 3 ROWS OF ENGINEERED DATA:")
print("="*70)
print(df.head(3))

# ============================================
# PART 2: SELECT FEATURES FOR TRAINING
# ============================================
print("\n" + "="*70)
print("SELECTING FEATURES FOR MODEL...")
print("="*70)

# Select relevant features (exclude some derived ones to avoid multicollinearity)
# We want features that provide unique information

selected_features = [
    'Close',           # Target variable (we'll predict this)
    'Open',            # Opening price
    'High',            # Daily high
    'Low',             # Daily low
    'Volume',          # Trading volume
    'SMA_20',          # Short-term trend
    'SMA_50',          # Medium-term trend
    'SMA_200',         # Long-term trend
    'EMA_12',          # Fast EMA
    'EMA_26',          # Slow EMA
    'RSI',             # Momentum oscillator
    'MACD',            # Momentum indicator
    'MACD_Signal',     # MACD signal line
    'MACD_Histogram',  # MACD histogram
    'BB_Upper',        # Bollinger upper band
    'BB_Lower',        # Bollinger lower band
    'BB_Width',        # Volatility measure
    'BB_Position',     # Price position in BB
    'ROC_10',          # Rate of change
    'ATR',             # Average True Range (volatility)
    'OBV',             # On-Balance Volume
    'Volume_MA_20',    # Volume moving average
    'Daily_Return',    # Daily percentage change
    'Daily_Range',     # High - Low
    'Distance_SMA_20', # Distance from SMA
    'Volatility',      # Rolling volatility
]

# Create feature matrix
data = df[selected_features].values  # Convert to numpy array
# .values converts pandas DataFrame to numpy array
# This is what ML models work with

print(f"✅ Selected {len(selected_features)} features")
print(f"📊 Data shape: {data.shape}")
print(f"   - {data.shape[0]} samples (days)")
print(f"   - {data.shape[1]} features per day")

# Display feature list
print("\n📋 Selected Features:")
for i, feature in enumerate(selected_features, 1):
    print(f"   {i:2d}. {feature}")

# ============================================
# PART 3: TRAIN-TEST SPLIT (TIME SERIES)
# ============================================
print("\n" + "="*70)
print("SPLITTING DATA - TRAIN/TEST...")
print("="*70)

# CRITICAL: For time series, we MUST split chronologically
# Can't shuffle! Past must predict future, not other way around

# Calculate split index (80% train, 20% test)
split_ratio = 0.8
split_index = int(len(data) * split_ratio)

# Split the data
train_data = data[:split_index]      # First 80% (older data)
test_data = data[split_index:]       # Last 20% (recent data)

# Also split the dates for visualization later
train_dates = df.index[:split_index]
test_dates = df.index[split_index:]

print(f"✅ Data split completed:")
print(f"   📊 Total samples: {len(data)}")
print(f"   📚 Training samples: {len(train_data)} ({split_ratio*100:.0f}%)")
print(f"   📝 Testing samples: {len(test_data)} ({(1-split_ratio)*100:.0f}%)")
print(f"\n   📅 Training period: {train_dates[0].date()} to {train_dates[-1].date()}")
print(f"   📅 Testing period: {test_dates[0].date()} to {test_dates[-1].date()}")

# Visualize the split
plt.figure(figsize=(14, 5))
plt.plot(train_dates, train_data[:, 0], label='Training Data', linewidth=1.5)
plt.plot(test_dates, test_data[:, 0], label='Testing Data', linewidth=1.5, color='orange')
plt.axvline(x=train_dates[-1], color='red', linestyle='--', linewidth=2, label='Train-Test Split')
plt.title('Train-Test Split Visualization (Close Price)', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Close Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('train_test_split.png', dpi=300, bbox_inches='tight')
print("\n📊 Train-test split visualization saved as 'train_test_split.png'")
plt.close()

# ============================================
# PART 4: NORMALIZATION (FEATURE SCALING)
# ============================================
print("\n" + "="*70)
print("NORMALIZING DATA...")
print("="*70)

# MinMaxScaler: Scales data to [0, 1] range
# Formula: (X - X_min) / (X_max - X_min)

scaler = MinMaxScaler(feature_range=(0, 1))
# feature_range=(0, 1) means scale to 0-1

# CRITICAL: Fit ONLY on training data!
# This prevents data leakage (test data influencing training)
scaler.fit(train_data)
print("✅ Scaler fitted on training data")

# Show what the scaler learned
print("\n📊 Scaler Statistics (from training data):")
print(f"   Min values: {scaler.data_min_[:3]}... (first 3 features)")
print(f"   Max values: {scaler.data_max_[:3]}... (first 3 features)")

# Transform both train and test using the SAME scaler
train_scaled = scaler.transform(train_data)
test_scaled = scaler.transform(test_data)

print(f"\n✅ Data normalized:")
print(f"   Training data shape: {train_scaled.shape}")
print(f"   Testing data shape: {test_scaled.shape}")

# Show before/after normalization
print("\n📊 Sample values BEFORE normalization:")
print(f"   Close price: ${train_data[0, 0]:.2f}")
print(f"   Volume: {train_data[0, 4]:,.0f}")
print(f"   RSI: {train_data[0, 10]:.2f}")

print("\n📊 Sample values AFTER normalization:")
print(f"   Close price: {train_scaled[0, 0]:.4f}")
print(f"   Volume: {train_scaled[0, 4]:.4f}")
print(f"   RSI: {train_scaled[0, 10]:.4f}")

# Save the scaler for later use (needed to inverse transform predictions)
joblib.dump(scaler, 'scaler.pkl')
print("\n💾 Scaler saved as 'scaler.pkl'")

# ============================================
# PART 5: CREATE SEQUENCES (SLIDING WINDOW)
# ============================================
print("\n" + "="*70)
print("CREATING SEQUENCES FOR LSTM...")
print("="*70)

def create_sequences(data, target_col_index=0, sequence_length=60):
    """
    Create sequences for LSTM training
    
    Parameters:
    -----------
    data : numpy array
        Normalized data (samples, features)
    target_col_index : int
        Index of target variable (Close price = 0)
    sequence_length : int
        Number of time steps to look back
        
    Returns:
    --------
    X : numpy array
        Input sequences (samples, sequence_length, features)
    y : numpy array
        Target values (samples,)
        
    How it works:
    ------------
    If sequence_length = 60:
    - Use days 1-60 to predict day 61
    - Use days 2-61 to predict day 62
    - Use days 3-62 to predict day 63
    ...and so on (sliding window)
    """
    X, y = [], []
    
    # Loop through data creating sequences
    for i in range(len(data) - sequence_length):
        # Extract sequence of 'sequence_length' days
        # This is our input (X)
        sequence = data[i:i + sequence_length]  # Shape: (60, 26)
        X.append(sequence)
        
        # Extract the target value (next day's Close price)
        # This is what we want to predict (y)
        target = data[i + sequence_length, target_col_index]  # Close price
        y.append(target)
    
    # Convert lists to numpy arrays
    X = np.array(X)  # Shape: (samples, 60, 26)
    y = np.array(y)  # Shape: (samples,)
    
    return X, y

# Set sequence length (lookback period)
SEQUENCE_LENGTH = 60  # Use 60 days to predict next day
print(f"📊 Sequence length (lookback): {SEQUENCE_LENGTH} days")
print(f"   This means: Use past {SEQUENCE_LENGTH} days to predict day {SEQUENCE_LENGTH + 1}")

# Create sequences for training data
print("\n⚙️  Creating training sequences...")
X_train, y_train = create_sequences(train_scaled, 
                                     target_col_index=0,  # Close price
                                     sequence_length=SEQUENCE_LENGTH)

# Create sequences for testing data
print("⚙️  Creating testing sequences...")
X_test, y_test = create_sequences(test_scaled, 
                                   target_col_index=0,
                                   sequence_length=SEQUENCE_LENGTH)

print(f"\n✅ Sequences created successfully!")
print(f"\n📊 Final Data Shapes:")
print(f"   X_train: {X_train.shape}  (samples, timesteps, features)")
print(f"   y_train: {y_train.shape}  (samples,)")
print(f"   X_test:  {X_test.shape}   (samples, timesteps, features)")
print(f"   y_test:  {y_test.shape}   (samples,)")

# Explain the shapes
print(f"\n📚 Shape Explanation:")
print(f"   X_train shape: ({X_train.shape[0]}, {X_train.shape[1]}, {X_train.shape[2]})")
print(f"   - {X_train.shape[0]} sequences (training samples)")
print(f"   - {X_train.shape[1]} time steps (days to look back)")
print(f"   - {X_train.shape[2]} features (indicators per day)")

# ============================================
# PART 6: VISUALIZE A SAMPLE SEQUENCE
# ============================================
print("\n" + "="*70)
print("VISUALIZING SAMPLE SEQUENCE...")
print("="*70)

# Take first sequence from training data
sample_sequence = X_train[0]  # Shape: (60, 26)
sample_target = y_train[0]    # Single value

print(f"📊 Sample Sequence Details:")
print(f"   Shape: {sample_sequence.shape}")
print(f"   - {sample_sequence.shape[0]} days")
print(f"   - {sample_sequence.shape[1]} features per day")
print(f"   Target (Day 61 Close): {sample_target:.4f} (normalized)")

# Visualize Close price in the sequence
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Sample Sequence Visualization', fontsize=16, fontweight='bold')

# Plot 1: Close price sequence
axes[0, 0].plot(range(1, SEQUENCE_LENGTH + 1), sample_sequence[:, 0], 
                marker='o', markersize=4, linewidth=2)
axes[0, 0].axhline(y=sample_target, color='r', linestyle='--', 
                    linewidth=2, label=f'Target (Day 61): {sample_target:.4f}')
axes[0, 0].set_title('Close Price Sequence (Normalized)', fontweight='bold')
axes[0, 0].set_xlabel('Day')
axes[0, 0].set_ylabel('Normalized Price')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: RSI sequence
axes[0, 1].plot(range(1, SEQUENCE_LENGTH + 1), sample_sequence[:, 10], 
                marker='o', markersize=4, linewidth=2, color='purple')
axes[0, 1].set_title('RSI Sequence (Normalized)', fontweight='bold')
axes[0, 1].set_xlabel('Day')
axes[0, 1].set_ylabel('Normalized RSI')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Volume sequence
axes[1, 0].bar(range(1, SEQUENCE_LENGTH + 1), sample_sequence[:, 4], 
               color='green', alpha=0.6)
axes[1, 0].set_title('Volume Sequence (Normalized)', fontweight='bold')
axes[1, 0].set_xlabel('Day')
axes[1, 0].set_ylabel('Normalized Volume')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Multiple features
axes[1, 1].plot(range(1, SEQUENCE_LENGTH + 1), sample_sequence[:, 0], 
                label='Close', linewidth=2)
axes[1, 1].plot(range(1, SEQUENCE_LENGTH + 1), sample_sequence[:, 5], 
                label='SMA_20', linewidth=2)
axes[1, 1].plot(range(1, SEQUENCE_LENGTH + 1), sample_sequence[:, 11], 
                label='MACD', linewidth=2)
axes[1, 1].set_title('Multiple Features (Normalized)', fontweight='bold')
axes[1, 1].set_xlabel('Day')
axes[1, 1].set_ylabel('Normalized Values')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sample_sequence_visualization.png', dpi=300, bbox_inches='tight')
print("📊 Sample sequence visualization saved as 'sample_sequence_visualization.png'")
plt.close()

# ============================================
# PART 7: SAVE PREPROCESSED DATA
# ============================================
print("\n" + "="*70)
print("SAVING PREPROCESSED DATA...")
print("="*70)

# Save as numpy files (efficient for large arrays)
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

# Save feature names for reference
with open('feature_names.txt', 'w') as f:
    f.write('\n'.join(selected_features))

print("✅ Preprocessed data saved:")
print("   📁 X_train.npy")
print("   📁 y_train.npy")
print("   📁 X_test.npy")
print("   📁 y_test.npy")
print("   📁 scaler.pkl")
print("   📁 feature_names.txt")

# ============================================
# PART 8: DATA SUMMARY & STATISTICS
# ============================================
print("\n" + "="*70)
print("DATA PREPROCESSING SUMMARY:")
print("="*70)

print(f"\n📊 Original Data:")
print(f"   Total records: {len(df)}")
print(f"   Total features: {len(df.columns)}")
print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")

print(f"\n📊 After Preprocessing:")
print(f"   Selected features: {len(selected_features)}")
print(f"   Sequence length: {SEQUENCE_LENGTH} days")
print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples: {len(X_test)}")

print(f"\n📊 Data Shapes:")
print(f"   X_train: {X_train.shape}")
print(f"   y_train: {y_train.shape}")
print(f"   X_test:  {X_test.shape}")
print(f"   y_test:  {y_test.shape}")

print(f"\n📊 Memory Usage:")
print(f"   X_train: {X_train.nbytes / 1024 / 1024:.2f} MB")
print(f"   X_test:  {X_test.nbytes / 1024 / 1024:.2f} MB")

print(f"\n📊 Target Statistics (y_train - normalized):")
print(f"   Mean: {y_train.mean():.4f}")
print(f"   Std:  {y_train.std():.4f}")
print(f"   Min:  {y_train.min():.4f}")
print(f"   Max:  {y_train.max():.4f}")

# ============================================
# PART 9: VERIFICATION CHECKS
# ============================================
print("\n" + "="*70)
print("VERIFICATION CHECKS:")
print("="*70)

# Check 1: No NaN values
print(f"\n✓ Check 1: NaN values")
print(f"   X_train contains NaN: {np.isnan(X_train).any()}")
print(f"   X_test contains NaN:  {np.isnan(X_test).any()}")

# Check 2: Correct normalization (values between 0 and 1)
print(f"\n✓ Check 2: Normalization range")
print(f"   X_train min: {X_train.min():.4f} (should be ≈ 0)")
print(f"   X_train max: {X_train.max():.4f} (should be ≈ 1)")
print(f"   X_test min:  {X_test.min():.4f}")
print(f"   X_test max:  {X_test.max():.4f} (can exceed 1 if test has new highs)")

# Check 3: Correct shapes
print(f"\n✓ Check 3: Correct shapes")
print(f"   X_train is 3D: {len(X_train.shape) == 3}")
print(f"   y_train is 1D: {len(y_train.shape) == 1}")
print(f"   Sequences match: {len(X_train) == len(y_train)}")

# Check 4: No data leakage (test dates after train dates)
print(f"\n✓ Check 4: Time series integrity")
print(f"   Last train date: {train_dates[-1].date()}")
print(f"   First test date: {test_dates[0].date()}")
print(f"   Test after train: {test_dates[0] > train_dates[-1]}")

print("\n" + "="*70)
print("✅ STEP 3 COMPLETE - DATA PREPROCESSING DONE!")
print("="*70)

print("\n📌 Files Created:")
print("   1. X_train.npy, y_train.npy - Training data")
print("   2. X_test.npy, y_test.npy - Testing data")
print("   3. scaler.pkl - For inverse transforming predictions")
print("   4. feature_names.txt - List of features used")
print("   5. train_test_split.png - Split visualization")
print("   6. sample_sequence_visualization.png - Sequence example")

print("\n📌 Next Steps:")
print("   4. Build LSTM Model Architecture")
print("   5. Train the Model")
print("   6. Evaluate Performance")
print("   7. Make Predictions")

print("\n💡 Key Takeaways:")
print("   ✓ Data normalized to [0, 1] range")
print("   ✓ Sequences created with 60-day lookback")
print("   ✓ Train-test split maintains time order")
print("   ✓ No data leakage (scaler fit on train only)")
print("   ✓ Ready for LSTM training!")