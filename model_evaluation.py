import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import keras
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("STEP 5: MODEL EVALUATION & PREDICTIONS")
print("="*70)

# ============================================
# PART 1: LOAD MODEL AND DATA
# ============================================
print("\n" + "="*70)
print("LOADING MODEL AND DATA...")
print("="*70)

# Load the best trained model
model = keras.models.load_model('best_model.keras')
print("✅ Model loaded: best_model.keras")

# Load preprocessed data
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')
print("✅ Data loaded")

# Load scaler (needed to inverse transform predictions)
scaler = joblib.load('scaler.pkl')
print("✅ Scaler loaded")

# Load feature names
with open('feature_names.txt', 'r') as f:
    feature_names = f.read().splitlines()

n_features = len(feature_names)
print(f"\n📊 Data Summary:")
print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")
print(f"   Features: {n_features}")

# ============================================
# PART 2: MAKE PREDICTIONS
# ============================================
print("\n" + "="*70)
print("MAKING PREDICTIONS...")
print("="*70)

# Predict on training set
print("⚙️  Predicting on training set...")
train_predictions_scaled = model.predict(X_train, verbose=0)

# Predict on test set
print("⚙️  Predicting on test set...")
test_predictions_scaled = model.predict(X_test, verbose=0)

print("✅ Predictions complete!")

# ============================================
# PART 3: INVERSE TRANSFORM TO ACTUAL PRICES
# ============================================
print("\n" + "="*70)
print("CONVERTING TO ACTUAL PRICES...")
print("="*70)

# Create dummy arrays for inverse transform
# Scaler was fitted on all features, so we need to provide all features
# We only care about Close price (index 0)

def inverse_transform_predictions(predictions, scaler, n_features):
    """
    Inverse transform predictions back to actual price scale
    
    predictions: (samples, 1) - normalized predictions
    scaler: MinMaxScaler object
    n_features: number of features used in training
    """
    # Create array with predictions in first column, zeros for other features
    dummy = np.zeros((len(predictions), n_features))
    dummy[:, 0] = predictions.flatten()
    
    # Inverse transform
    actual = scaler.inverse_transform(dummy)
    
    # Return only Close price column
    return actual[:, 0]

# Inverse transform predictions
train_predictions = inverse_transform_predictions(train_predictions_scaled, scaler, n_features)
test_predictions = inverse_transform_predictions(test_predictions_scaled, scaler, n_features)

# Inverse transform actual values
y_train_actual = inverse_transform_predictions(y_train.reshape(-1, 1), scaler, n_features)
y_test_actual = inverse_transform_predictions(y_test.reshape(-1, 1), scaler, n_features)

print("✅ Converted to actual price scale")
print(f"\n📊 Sample Predictions (Test Set):")
print(f"{'Actual':<12} {'Predicted':<12} {'Error':<12}")
print("-" * 40)
for i in range(min(10, len(y_test_actual))):
    actual = y_test_actual[i]
    pred = test_predictions[i]
    error = pred - actual
    print(f"${actual:10.2f} ${pred:10.2f} ${error:+10.2f}")

# ============================================
# PART 4: CALCULATE METRICS
# ============================================
print("\n" + "="*70)
print("CALCULATING PERFORMANCE METRICS...")
print("="*70)

# Training set metrics
train_mae = mean_absolute_error(y_train_actual, train_predictions)
train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predictions))
train_r2 = r2_score(y_train_actual, train_predictions)
train_mape = np.mean(np.abs((y_train_actual - train_predictions) / y_train_actual)) * 100

# Test set metrics
test_mae = mean_absolute_error(y_test_actual, test_predictions)
test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predictions))
test_r2 = r2_score(y_test_actual, test_predictions)
test_mape = np.mean(np.abs((y_test_actual - test_predictions) / y_test_actual)) * 100

print(f"\n📊 TRAINING SET PERFORMANCE:")
print(f"   MAE (Mean Absolute Error):   ${train_mae:.2f}")
print(f"   RMSE (Root Mean Squared):    ${train_rmse:.2f}")
print(f"   R² Score:                    {train_r2:.4f}")
print(f"   MAPE (Mean Abs % Error):     {train_mape:.2f}%")
print(f"   Accuracy:                    {100 - train_mape:.2f}%")

print(f"\n📊 TEST SET PERFORMANCE:")
print(f"   MAE (Mean Absolute Error):   ${test_mae:.2f}")
print(f"   RMSE (Root Mean Squared):    ${test_rmse:.2f}")
print(f"   R² Score:                    {test_r2:.4f}")
print(f"   MAPE (Mean Abs % Error):     {test_mape:.2f}%")
print(f"   Accuracy:                    {100 - test_mape:.2f}%")

# Interpretation
print(f"\n💡 Interpretation:")
if test_r2 > 0.8:
    print(f"   ✅ Excellent! R²={test_r2:.4f} - Model explains {test_r2*100:.1f}% of variance")
elif test_r2 > 0.6:
    print(f"   ✅ Good! R²={test_r2:.4f} - Model explains {test_r2*100:.1f}% of variance")
elif test_r2 > 0.4:
    print(f"   ⚠️  Moderate! R²={test_r2:.4f} - Model explains {test_r2*100:.1f}% of variance")
else:
    print(f"   ❌ Poor! R²={test_r2:.4f} - Model needs improvement")

if test_mape < 5:
    print(f"   ✅ Excellent accuracy! MAPE={test_mape:.2f}% (predictions within {test_mape:.1f}%)")
elif test_mape < 10:
    print(f"   ✅ Good accuracy! MAPE={test_mape:.2f}% (predictions within {test_mape:.1f}%)")
elif test_mape < 20:
    print(f"   ⚠️  Moderate accuracy! MAPE={test_mape:.2f}% (predictions within {test_mape:.1f}%)")
else:
    print(f"   ❌ Low accuracy! MAPE={test_mape:.2f}% - Model needs improvement")

# ============================================
# PART 5: DETAILED ERROR ANALYSIS
# ============================================
print("\n" + "="*70)
print("ERROR ANALYSIS...")
print("="*70)

# Calculate errors
train_errors = train_predictions - y_train_actual
test_errors = test_predictions - y_test_actual

# Error statistics
print(f"\n📊 Test Set Errors:")
print(f"   Mean Error:      ${np.mean(test_errors):+.2f}")
print(f"   Std Dev:         ${np.std(test_errors):.2f}")
print(f"   Min Error:       ${np.min(test_errors):+.2f}")
print(f"   Max Error:       ${np.max(test_errors):+.2f}")
print(f"   Median Error:    ${np.median(test_errors):+.2f}")

# Percentage of predictions within certain thresholds
within_5_pct = np.sum(np.abs(test_errors / y_test_actual) < 0.05) / len(y_test_actual) * 100
within_10_pct = np.sum(np.abs(test_errors / y_test_actual) < 0.10) / len(y_test_actual) * 100
within_15_pct = np.sum(np.abs(test_errors / y_test_actual) < 0.15) / len(y_test_actual) * 100

print(f"\n📊 Prediction Accuracy Breakdown:")
print(f"   Within  5%: {within_5_pct:.1f}% of predictions")
print(f"   Within 10%: {within_10_pct:.1f}% of predictions")
print(f"   Within 15%: {within_15_pct:.1f}% of predictions")

# ============================================
# PART 6: COMPREHENSIVE VISUALIZATIONS
# ============================================
print("\n" + "="*70)
print("CREATING VISUALIZATIONS...")
print("="*70)

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

# Plot 1: Training Set - Actual vs Predicted
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(y_train_actual, label='Actual Price', linewidth=2, alpha=0.7)
ax1.plot(train_predictions, label='Predicted Price', linewidth=2, alpha=0.7)
ax1.set_title(f'Training Set: Actual vs Predicted (R²={train_r2:.4f})', 
              fontsize=14, fontweight='bold')
ax1.set_xlabel('Sample')
ax1.set_ylabel('Price ($)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Training Error Distribution
ax2 = fig.add_subplot(gs[0, 2])
ax2.hist(train_errors, bins=30, color='blue', alpha=0.7, edgecolor='black')
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax2.set_title('Training Error Distribution', fontweight='bold')
ax2.set_xlabel('Error ($)')
ax2.set_ylabel('Frequency')
ax2.grid(True, alpha=0.3)

# Plot 3: Test Set - Actual vs Predicted (MOST IMPORTANT!)
ax3 = fig.add_subplot(gs[1, :2])
ax3.plot(y_test_actual, label='Actual Price', linewidth=2.5, color='blue')
ax3.plot(test_predictions, label='Predicted Price', linewidth=2.5, color='red', alpha=0.7)
ax3.fill_between(range(len(y_test_actual)), 
                  y_test_actual - test_mae, 
                  y_test_actual + test_mae, 
                  alpha=0.2, color='gray', label=f'±MAE (${test_mae:.2f})')
ax3.set_title(f'Test Set: Actual vs Predicted (R²={test_r2:.4f}, MAPE={test_mape:.2f}%)', 
              fontsize=14, fontweight='bold')
ax3.set_xlabel('Sample')
ax3.set_ylabel('Price ($)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Test Error Distribution
ax4 = fig.add_subplot(gs[1, 2])
ax4.hist(test_errors, bins=30, color='purple', alpha=0.7, edgecolor='black')
ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax4.set_title('Test Error Distribution', fontweight='bold')
ax4.set_xlabel('Error ($)')
ax4.set_ylabel('Frequency')
ax4.grid(True, alpha=0.3)

# Plot 5: Scatter - Actual vs Predicted (Training)
ax5 = fig.add_subplot(gs[2, 0])
ax5.scatter(y_train_actual, train_predictions, alpha=0.5, s=20)
min_val = min(y_train_actual.min(), train_predictions.min())
max_val = max(y_train_actual.max(), train_predictions.max())
ax5.plot([min_val, max_val], [min_val, max_val], 
         'r--', linewidth=2, label='Perfect Prediction')
ax5.set_title('Training: Actual vs Predicted', fontweight='bold')
ax5.set_xlabel('Actual Price ($)')
ax5.set_ylabel('Predicted Price ($)')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Scatter - Actual vs Predicted (Test)
ax6 = fig.add_subplot(gs[2, 1])
ax6.scatter(y_test_actual, test_predictions, alpha=0.6, s=30, color='purple')
min_val = min(y_test_actual.min(), test_predictions.min())
max_val = max(y_test_actual.max(), test_predictions.max())
ax6.plot([min_val, max_val], [min_val, max_val], 
         'r--', linewidth=2, label='Perfect Prediction')
ax6.set_title('Test: Actual vs Predicted', fontweight='bold')
ax6.set_xlabel('Actual Price ($)')
ax6.set_ylabel('Predicted Price ($)')
ax6.legend()
ax6.grid(True, alpha=0.3)

# Plot 7: Residual Plot (Test Set)
ax7 = fig.add_subplot(gs[2, 2])
ax7.scatter(test_predictions, test_errors, alpha=0.6, s=30, color='orange')
ax7.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax7.set_title('Residual Plot (Test Set)', fontweight='bold')
ax7.set_xlabel('Predicted Price ($)')
ax7.set_ylabel('Error ($)')
ax7.grid(True, alpha=0.3)

# Plot 8: Cumulative Error (Test Set)
ax8 = fig.add_subplot(gs[3, 0])
cumulative_error = np.cumsum(np.abs(test_errors))
ax8.plot(cumulative_error, linewidth=2, color='red')
ax8.set_title('Cumulative Absolute Error', fontweight='bold')
ax8.set_xlabel('Sample')
ax8.set_ylabel('Cumulative Error ($)')
ax8.grid(True, alpha=0.3)

# Plot 9: Error by Price Range
ax9 = fig.add_subplot(gs[3, 1])
price_bins = pd.cut(y_test_actual, bins=5)
error_by_bin = pd.DataFrame({
    'price_bin': price_bins,
    'abs_error': np.abs(test_errors)
}).groupby('price_bin')['abs_error'].mean()
error_by_bin.plot(kind='bar', ax=ax9, color='teal', alpha=0.7)
ax9.set_title('Avg Error by Price Range', fontweight='bold')
ax9.set_xlabel('Price Range')
ax9.set_ylabel('Mean Absolute Error ($)')
ax9.tick_params(axis='x', rotation=45)
ax9.grid(True, alpha=0.3)

# Plot 10: Metrics Comparison
ax10 = fig.add_subplot(gs[3, 2])
metrics = ['MAE', 'RMSE', 'R²', 'MAPE']
train_vals = [train_mae, train_rmse, train_r2*100, train_mape]
test_vals = [test_mae, test_rmse, test_r2*100, test_mape]

x = np.arange(len(metrics))
width = 0.35
ax10.bar(x - width/2, train_vals, width, label='Training', alpha=0.8)
ax10.bar(x + width/2, test_vals, width, label='Test', alpha=0.8)
ax10.set_title('Metrics Comparison', fontweight='bold')
ax10.set_ylabel('Value')
ax10.set_xticks(x)
ax10.set_xticklabels(metrics)
ax10.legend()
ax10.grid(True, alpha=0.3, axis='y')

plt.suptitle('Complete Model Evaluation & Predictions Analysis', 
             fontsize=18, fontweight='bold', y=0.995)
plt.savefig('model_evaluation_results.png', dpi=300, bbox_inches='tight')
print("📊 Evaluation results saved as 'model_evaluation_results.png'")
plt.show()

# ============================================
# PART 7: SAVE RESULTS
# ============================================
print("\n" + "="*70)
print("SAVING RESULTS...")
print("="*70)

# Save predictions to CSV
results_df = pd.DataFrame({
    'Actual_Price': y_test_actual,
    'Predicted_Price': test_predictions,
    'Error': test_errors,
    'Abs_Error': np.abs(test_errors),
    'Pct_Error': (test_errors / y_test_actual) * 100
})
results_df.to_csv('test_predictions.csv', index=False)
print("✅ Predictions saved as 'test_predictions.csv'")

# Save metrics
metrics_df = pd.DataFrame({
    'Metric': ['MAE', 'RMSE', 'R²', 'MAPE', 'Accuracy'],
    'Training': [train_mae, train_rmse, train_r2, train_mape, 100-train_mape],
    'Test': [test_mae, test_rmse, test_r2, test_mape, 100-test_mape]
})
metrics_df.to_csv('model_metrics.csv', index=False)
print("✅ Metrics saved as 'model_metrics.csv'")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "="*70)
print("🎉 EVALUATION COMPLETE!")
print("="*70)

print(f"\n📊 Final Model Performance:")
print(f"   Test R² Score:   {test_r2:.4f}")
print(f"   Test MAE:        ${test_mae:.2f}")
print(f"   Test MAPE:       {test_mape:.2f}%")
print(f"   Test Accuracy:   {100-test_mape:.2f}%")

print(f"\n✅ Model Performance Rating:")
if test_r2 > 0.7 and test_mape < 10:
    print("   ⭐⭐⭐⭐⭐ EXCELLENT! Model ready for use!")
elif test_r2 > 0.5 and test_mape < 15:
    print("   ⭐⭐⭐⭐ GOOD! Model performs well!")
elif test_r2 > 0.3 and test_mape < 20:
    print("   ⭐⭐⭐ MODERATE! Model needs improvement!")
else:
    print("   ⭐⭐ NEEDS WORK! Consider more data or features!")

print(f"\n📁 Files Created:")
print(f"   1. model_evaluation_results.png - Complete visualization")
print(f"   2. test_predictions.csv - All predictions")
print(f"   3. model_metrics.csv - Performance metrics")

print(f"\n💡 You can now:")
print(f"   - Use the model for future predictions")
print(f"   - Analyze prediction errors")
print(f"   - Present results with visualizations")
print(f"   - Explain model performance with metrics")

print("\n" + "="*70)
print("PROJECT COMPLETE! 🎊")
print("="*70)