import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# TensorFlow and Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("STEP 4: LSTM MODEL BUILDING & TRAINING")
print("="*70)

# Check TensorFlow/GPU availability
print(f"\n🔧 TensorFlow Version: {tf.__version__}")
print(f"🔧 GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
if len(tf.config.list_physical_devices('GPU')) > 0:
    print(f"   GPU Device: {tf.config.list_physical_devices('GPU')}")
else:
    print("   Running on CPU (this is fine, just slower)")

# ============================================
# PART 1: LOAD PREPROCESSED DATA
# ============================================
print("\n" + "="*70)
print("LOADING PREPROCESSED DATA...")
print("="*70)

# Load the numpy arrays we created in previous step
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

print(f"✅ Data loaded successfully!")
print(f"\n📊 Data Shapes:")
print(f"   X_train: {X_train.shape}  (samples, timesteps, features)")
print(f"   y_train: {y_train.shape}  (samples,)")
print(f"   X_test:  {X_test.shape}   (samples, timesteps, features)")
print(f"   y_test:  {y_test.shape}   (samples,)")

# Extract dimensions
n_samples, n_timesteps, n_features = X_train.shape

print(f"\n📊 Model Input Parameters:")
print(f"   Timesteps (lookback): {n_timesteps} days")
print(f"   Features per day: {n_features}")
print(f"   Training samples: {n_samples}")

# ============================================
# PART 2: BUILD LSTM MODEL ARCHITECTURE
# ============================================
print("\n" + "="*70)
print("BUILDING LSTM MODEL ARCHITECTURE...")
print("="*70)

# Create Sequential model (layers stacked one after another)
model = Sequential(name='Tesla_Stock_Predictor')

# Layer 1: First LSTM layer
# - 50 units: Number of LSTM cells (neurons)
# - return_sequences=True: Return output for each timestep (needed for next LSTM layer)
# - input_shape: (timesteps, features) - only needed for first layer
model.add(LSTM(units=50, 
               return_sequences=True, 
               input_shape=(n_timesteps, n_features),
               name='LSTM_Layer_1'))
print("✅ Added LSTM Layer 1: 50 units, return_sequences=True")

# Dropout 1: Prevents overfitting
# - 0.2 = 20% of connections randomly dropped during training
model.add(Dropout(0.2, name='Dropout_1'))
print("✅ Added Dropout 1: 20% dropout rate")

# Layer 2: Second LSTM layer
# - Still returns sequences for the third LSTM layer
model.add(LSTM(units=50, 
               return_sequences=True,
               name='LSTM_Layer_2'))
print("✅ Added LSTM Layer 2: 50 units, return_sequences=True")

# Dropout 2
model.add(Dropout(0.2, name='Dropout_2'))
print("✅ Added Dropout 2: 20% dropout rate")

# Layer 3: Third LSTM layer
# - return_sequences=False: Only return last timestep output (not needed anymore)
# - Output goes to Dense layers
model.add(LSTM(units=50, 
               return_sequences=False,
               name='LSTM_Layer_3'))
print("✅ Added LSTM Layer 3: 50 units, return_sequences=False")

# Dropout 3
model.add(Dropout(0.2, name='Dropout_3'))
print("✅ Added Dropout 3: 20% dropout rate")

# Layer 4: Dense (Fully Connected) layer
# - 25 units: Intermediate layer to combine LSTM outputs
# - ReLU activation: f(x) = max(0, x) - introduces non-linearity
model.add(Dense(units=25, 
                activation='relu',
                name='Dense_Layer'))
print("✅ Added Dense Layer: 25 units, ReLU activation")

# Layer 5: Output layer
# - 1 unit: Single output (predicted price)
# - Linear activation: No transformation, can output any real number
model.add(Dense(units=1, 
                activation='linear',
                name='Output_Layer'))
print("✅ Added Output Layer: 1 unit, Linear activation")

print("\n" + "="*70)
print("MODEL ARCHITECTURE SUMMARY:")
print("="*70)
model.summary()

# ============================================
# PART 3: COMPILE MODEL
# ============================================
print("\n" + "="*70)
print("COMPILING MODEL...")
print("="*70)

# Compile the model
# - Optimizer: Adam (adaptive learning rate)
# - Loss: MSE (Mean Squared Error) - good for regression
# - Metrics: MAE, RMSE for monitoring

model.compile(
    optimizer=Adam(learning_rate=0.001),  # 0.001 = standard learning rate
    loss='mean_squared_error',            # MSE loss function
    metrics=[
        'mae',                             # Mean Absolute Error
        tf.keras.metrics.RootMeanSquaredError(name='rmse')  # RMSE
    ]
)

print("✅ Model compiled successfully!")
print(f"   Optimizer: Adam (learning_rate=0.001)")
print(f"   Loss Function: Mean Squared Error (MSE)")
print(f"   Metrics: MAE, RMSE")

# ============================================
# PART 4: SETUP CALLBACKS
# ============================================
print("\n" + "="*70)
print("SETTING UP TRAINING CALLBACKS...")
print("="*70)

# Callbacks are functions called during training

# 1. Early Stopping: Stop training if validation loss doesn't improve
# - monitor='val_loss': Track validation loss
# - patience=15: Wait 15 epochs before stopping
# - restore_best_weights=True: Use weights from best epoch
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)
print("✅ Early Stopping: patience=15, monitor='val_loss'")

# 2. Model Checkpoint: Save best model during training
# - save_best_only=True: Only save when improvement happens
model_checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)
print("✅ Model Checkpoint: Saving best model to 'best_model.keras'")

# 3. Reduce Learning Rate: Reduce LR when validation loss plateaus
# - factor=0.5: Multiply LR by 0.5 when plateau detected
# - patience=10: Wait 10 epochs before reducing
# - min_lr=0.00001: Don't go below this learning rate
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=0.00001,
    verbose=1
)
print("✅ Reduce LR on Plateau: factor=0.5, patience=10")

callbacks = [early_stopping, model_checkpoint, reduce_lr]

# ============================================
# PART 5: TRAIN THE MODEL
# ============================================
print("\n" + "="*70)
print("TRAINING MODEL...")
print("="*70)

# Training parameters
EPOCHS = 100          # Maximum epochs (early stopping may stop earlier)
BATCH_SIZE = 32       # Process 32 sequences at a time
VALIDATION_SPLIT = 0.2  # Use 20% of training data for validation

print(f"📊 Training Parameters:")
print(f"   Epochs: {EPOCHS}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Validation Split: {VALIDATION_SPLIT * 100}%")
print(f"\n🚀 Starting training... (this may take a few minutes)")
print("="*70)

# Train the model
# fit() returns a History object containing training metrics
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    callbacks=callbacks,
    verbose=1  # 1 = progress bar, 2 = one line per epoch, 0 = silent
)

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)

# Get training history
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_mae = history.history['mae']
val_mae = history.history['val_mae']

# Find best epoch
best_epoch = np.argmin(val_loss) + 1
best_val_loss = np.min(val_loss)

print(f"\n📊 Training Results:")
print(f"   Total Epochs Trained: {len(train_loss)}")
print(f"   Best Epoch: {best_epoch}")
print(f"   Best Validation Loss: {best_val_loss:.6f}")
print(f"   Final Training Loss: {train_loss[-1]:.6f}")
print(f"   Final Validation Loss: {val_loss[-1]:.6f}")

# ============================================
# PART 6: VISUALIZE TRAINING HISTORY
# ============================================
print("\n" + "="*70)
print("CREATING TRAINING VISUALIZATIONS...")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('LSTM Model Training History', fontsize=16, fontweight='bold')

# Plot 1: Loss over epochs
axes[0, 0].plot(train_loss, label='Training Loss', linewidth=2)
axes[0, 0].plot(val_loss, label='Validation Loss', linewidth=2)
axes[0, 0].axvline(x=best_epoch-1, color='r', linestyle='--', 
                    label=f'Best Epoch ({best_epoch})', alpha=0.7)
axes[0, 0].set_title('Model Loss (MSE)', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: MAE over epochs
axes[0, 1].plot(train_mae, label='Training MAE', linewidth=2)
axes[0, 1].plot(val_mae, label='Validation MAE', linewidth=2)
axes[0, 1].axvline(x=best_epoch-1, color='r', linestyle='--', 
                    label=f'Best Epoch ({best_epoch})', alpha=0.7)
axes[0, 1].set_title('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('MAE')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Loss (log scale) - better for seeing small changes
axes[1, 0].plot(train_loss, label='Training Loss', linewidth=2)
axes[1, 0].plot(val_loss, label='Validation Loss', linewidth=2)
axes[1, 0].set_title('Model Loss (Log Scale)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss (log scale)')
axes[1, 0].set_yscale('log')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Learning rate schedule (if it changed)
if 'lr' in history.history:
    axes[1, 1].plot(history.history['lr'], linewidth=2, color='orange')
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
else:
    # If LR didn't change, show overfitting check
    axes[1, 1].plot(np.array(val_loss) - np.array(train_loss), 
                    linewidth=2, color='purple')
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].set_title('Overfitting Check (Val - Train Loss)', 
                          fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Difference')
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("📊 Training history saved as 'training_history.png'")
plt.close()

# ============================================
# PART 7: SAVE TRAINING HISTORY
# ============================================
print("\n" + "="*70)
print("SAVING TRAINING HISTORY...")
print("="*70)

# Save history as CSV for later analysis
history_df = pd.DataFrame({
    'epoch': range(1, len(train_loss) + 1),
    'train_loss': train_loss,
    'val_loss': val_loss,
    'train_mae': train_mae,
    'val_mae': val_mae
})
history_df.to_csv('training_history.csv', index=False)
print("✅ Training history saved as 'training_history.csv'")

# Save final model
model.save('final_model.keras')
print("✅ Final model saved as 'final_model.keras'")

# ============================================
# PART 8: MODEL EVALUATION SUMMARY
# ============================================
print("\n" + "="*70)
print("MODEL EVALUATION SUMMARY:")
print("="*70)

# Check for overfitting
final_train_loss = train_loss[-1]
final_val_loss = val_loss[-1]
loss_difference = final_val_loss - final_train_loss
loss_difference_pct = (loss_difference / final_train_loss) * 100

print(f"\n📊 Loss Analysis:")
print(f"   Final Training Loss:   {final_train_loss:.6f}")
print(f"   Final Validation Loss: {final_val_loss:.6f}")
print(f"   Difference:            {loss_difference:.6f} ({loss_difference_pct:+.2f}%)")

if loss_difference_pct < 10:
    print("   ✅ Status: Good generalization (low overfitting)")
elif loss_difference_pct < 30:
    print("   ⚠️  Status: Moderate overfitting")
else:
    print("   ❌ Status: High overfitting - consider more dropout or data")

print(f"\n📊 Training Efficiency:")
print(f"   Epochs Trained: {len(train_loss)}")
print(f"   Early Stopped: {'Yes' if len(train_loss) < EPOCHS else 'No'}")
if len(train_loss) < EPOCHS:
    print(f"   Epochs Saved: {EPOCHS - len(train_loss)}")

# Calculate improvement
initial_loss = train_loss[0]
final_loss = train_loss[-1]
improvement = ((initial_loss - final_loss) / initial_loss) * 100

print(f"\n📊 Learning Progress:")
print(f"   Initial Loss:  {initial_loss:.6f}")
print(f"   Final Loss:    {final_loss:.6f}")
print(f"   Improvement:   {improvement:.2f}%")

# ============================================
# PART 9: NEXT STEPS INFORMATION
# ============================================
print("\n" + "="*70)
print("✅ STEP 4 COMPLETE - MODEL TRAINING DONE!")
print("="*70)

print("\n📁 Files Created:")
print("   1. best_model.keras - Best model (lowest validation loss)")
print("   2. final_model.keras - Final model after all epochs")
print("   3. training_history.png - Training visualization")
print("   4. training_history.csv - Training metrics")

print("\n📌 Next Steps:")
print("   5. Evaluate Model on Test Set")
print("   6. Make Predictions")
print("   7. Visualize Results")
print("   8. Calculate Performance Metrics")

print("\n💡 Model Ready for Prediction!")
print("   - Load with: model = keras.models.load_model('best_model.keras')")
print("   - Predict with: predictions = model.predict(X_test)")

print("\n" + "="*70)
print("🎉 TRAINING SUCCESSFUL! Ready for evaluation and predictions!")
print("="*70)