"""
ENHANCED STOCK PREDICTION MODEL
Improvements:
1. More training data (7 years instead of 5)
2. Additional trend features
3. Increased dropout (30%)
4. More LSTM units
5. Learning rate scheduling
6. Data augmentation
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import joblib
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

class EnhancedStockPredictor:
    def __init__(self, stock_symbol='TSLA', years=7, sequence_length=60):
        self.stock_symbol = stock_symbol
        self.years = years
        self.sequence_length = sequence_length
        self.scaler = None
        self.model = None
        
    def download_data(self):
        """Download stock data"""
        print(f"\n📥 Downloading {self.stock_symbol} data for {self.years} years...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.years*365)
        
        df = yf.download(self.stock_symbol, start=start_date, end=end_date, progress=False)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        print(f"✅ Downloaded {len(df)} records")
        return df
    
    def create_features(self, df):
        """Create technical indicators with TREND features"""
        print("\n⚙️  Creating enhanced features...")
        
        # Original features
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['SMA_20']
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
        df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # NEW: Trend Features (to capture uptrends/downtrends)
        df['Trend_5'] = df['Close'].pct_change(5)    # 5-day momentum
        df['Trend_10'] = df['Close'].pct_change(10)  # 10-day momentum
        df['Trend_20'] = df['Close'].pct_change(20)  # 20-day momentum
        
        # NEW: Volatility indicators
        df['Volatility_10'] = df['Close'].rolling(10).std() / df['Close'].rolling(10).mean()
        df['Volatility_30'] = df['Close'].rolling(30).std() / df['Close'].rolling(30).mean()
        
        # NEW: Volume indicators
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['Volume_Trend'] = df['Volume'].pct_change(5)
        
        # Clean data
        df_clean = df.dropna()
        print(f"✅ Created {len(df_clean.columns)} features, {len(df_clean)} clean records")
        
        return df_clean
    
    def prepare_data(self, df):
        """Prepare data for training"""
        print("\n⚙️  Preparing data for training...")
        
        # Select features
        feature_cols = [
            'Close', 'Open', 'High', 'Low', 'Volume',
            'SMA_20', 'SMA_50', 'SMA_200',
            'EMA_12', 'EMA_26',
            'RSI', 'MACD', 'MACD_Signal',
            'BB_Upper', 'BB_Lower', 'BB_Position',
            'Trend_5', 'Trend_10', 'Trend_20',  # NEW
            'Volatility_10', 'Volatility_30',   # NEW
            'Volume_Ratio', 'Volume_Trend'      # NEW
        ]
        
        data = df[feature_cols].values
        
        # Split (70% train, 15% validation, 15% test for better generalization)
        train_size = int(len(data) * 0.7)
        val_size = int(len(data) * 0.15)
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size+val_size]
        test_data = data[train_size+val_size:]
        
        # Normalize
        self.scaler = MinMaxScaler()
        self.scaler.fit(train_data)
        
        train_scaled = self.scaler.transform(train_data)
        val_scaled = self.scaler.transform(val_data)
        test_scaled = self.scaler.transform(test_data)
        
        # Create sequences
        X_train, y_train = self._create_sequences(train_scaled)
        X_val, y_val = self._create_sequences(val_scaled)
        X_test, y_test = self._create_sequences(test_scaled)
        
        print(f"✅ Data prepared:")
        print(f"   Train: {X_train.shape}")
        print(f"   Val:   {X_val.shape}")
        print(f"   Test:  {X_test.shape}")
        
        # Save feature names
        with open(f'{self.stock_symbol}_features.txt', 'w') as f:
            f.write('\n'.join(feature_cols))
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_cols
    
    def _create_sequences(self, data):
        """Create sequences for LSTM"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length, 0])
        return np.array(X), np.array(y)
    
    def build_model(self, n_features):
        """Build enhanced LSTM model"""
        print("\n🏗️  Building enhanced model...")
        
        model = Sequential([
            # Layer 1: More units, return sequences
            LSTM(100, return_sequences=True, input_shape=(self.sequence_length, n_features)),
            BatchNormalization(),  # NEW: Helps with training stability
            Dropout(0.3),          # Increased dropout
            
            # Layer 2
            LSTM(100, return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),
            
            # Layer 3
            LSTM(100, return_sequences=False),
            BatchNormalization(),
            Dropout(0.3),
            
            # Dense layers
            Dense(50, activation='relu'),
            Dropout(0.3),
            Dense(25, activation='relu'),
            Dense(1, activation='linear')
        ], name=f'{self.stock_symbol}_Predictor')
        
        # Compile with lower initial learning rate
        model.compile(
            optimizer=Adam(learning_rate=0.0005),  # Lower LR
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        print("✅ Model built")
        model.summary()
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100):
        """Train the model"""
        print("\n🚀 Training model...")
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
            ModelCheckpoint(f'{self.stock_symbol}_best_model.keras', 
                          monitor='val_loss', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, 
                            min_lr=0.00001, verbose=1)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✅ Training complete!")
        
        # Save model and scaler
        self.model.save(f'{self.stock_symbol}_final_model.keras')
        joblib.dump(self.scaler, f'{self.stock_symbol}_scaler.pkl')
        
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model"""
        print("\n📊 Evaluating model...")
        
        predictions = self.model.predict(X_test, verbose=0)
        
        # Inverse transform
        n_features = len(self.scaler.data_min_)
        
        def inverse_transform(values):
            dummy = np.zeros((len(values), n_features))
            dummy[:, 0] = values.flatten()
            return self.scaler.inverse_transform(dummy)[:, 0]
        
        y_test_actual = inverse_transform(y_test.reshape(-1, 1))
        predictions_actual = inverse_transform(predictions)
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(y_test_actual, predictions_actual)
        rmse = np.sqrt(mean_squared_error(y_test_actual, predictions_actual))
        r2 = r2_score(y_test_actual, predictions_actual)
        mape = np.mean(np.abs((y_test_actual - predictions_actual) / y_test_actual)) * 100
        
        print(f"\n📊 Test Set Performance:")
        print(f"   MAE:  ${mae:.2f}")
        print(f"   RMSE: ${rmse:.2f}")
        print(f"   R²:   {r2:.4f}")
        print(f"   MAPE: {mape:.2f}%")
        print(f"   Accuracy: {100-mape:.2f}%")
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'accuracy': 100-mape,
            'predictions': predictions_actual,
            'actual': y_test_actual
        }

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    print("="*70)
    print("ENHANCED STOCK PRICE PREDICTION MODEL")
    print("="*70)
    
    # Choose stock (you can change this!)
    STOCK = input("\nEnter stock symbol (default: TSLA): ").upper() or 'TSLA'
    
    # Create predictor
    predictor = EnhancedStockPredictor(stock_symbol=STOCK, years=7)
    
    # Download data
    df = predictor.download_data()
    
    # Create features
    df_features = predictor.create_features(df)
    
    # Prepare data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), features = predictor.prepare_data(df_features)
    
    # Build model
    model = predictor.build_model(n_features=len(features))
    
    # Train model
    history = predictor.train(X_train, y_train, X_val, y_val)
    
    # Evaluate
    results = predictor.evaluate(X_test, y_test)
    
    print("\n" + "="*70)
    print("✅ ENHANCED MODEL TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📁 Files created:")
    print(f"   - {STOCK}_best_model.keras")
    print(f"   - {STOCK}_final_model.keras")
    print(f"   - {STOCK}_scaler.pkl")
    print(f"   - {STOCK}_features.txt")
    print(f"\n🎯 Ready for Streamlit app!")