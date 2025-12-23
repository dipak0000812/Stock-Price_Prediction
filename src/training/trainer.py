import numpy as np
import os
import json
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from src.data.loader import DataLoader
from src.features.feature_builder import FeatureBuilder
from src.features.feature_schema import CONFIG
from src.models.lstm_attention import LSTMAttentionModel
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from src.training.callbacks import JSONLogger

class Trainer:
    """
    Orchestrates the training pipeline.
    """
    
    def __init__(self, config=None):
        self.config = config if config else CONFIG
        self.output_dir = "models"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def run(self, epochs=100, batch_size=32):
        print("🚀 Starting Training Pipeline...")
        
        # ... (lines 27-89 omitted for brevity, will rely on context matching or careful MultiReplace if needed. 
        # Actually replace_file_content is single block. I need to change signature AND usage.
        # Use MultiReplace.
        
        # 1. Load Data
        loader = DataLoader(self.config.data.symbol, self.config.data.history_years)
        raw_df = loader.fetch()
        
        # 2. Build Features
        print("⚙️ Building Features...")
        builder = FeatureBuilder(self.config)
        features_df = builder.build(raw_df)
        
        # 3. Scale Features
        print("⚖️ Scaling Data...")
        scaler = StandardScaler()
        # Fit on everything for now (in a real strict setup, we fit on train only)
        # But we need to split train/val/test first.
        
        features_array = features_df.values
        feature_names = features_df.columns.tolist()
        
        # Split Data (Chronological)
        train_split = 1.0 - (self.config.data.validation_split + self.config.data.test_split)
        n = len(features_array)
        train_idx = int(n * train_split)
        val_idx = int(n * (1.0 - self.config.data.test_split))
        
        train_data = features_array[:train_idx]
        val_data = features_array[train_idx:val_idx]
        test_data = features_array[val_idx:]
        
        # Fit scaler on training data ONLY
        scaler.fit(train_data)
        
        # Transform all
        train_scaled = scaler.transform(train_data)
        val_scaled = scaler.transform(val_data)
        test_scaled = scaler.transform(test_data)
        
        # Save Scaler
        joblib.dump(scaler, os.path.join(self.output_dir, "scaler.pkl"))
        
        # 4. Create Sequences
        seq_len = self.config.sequence_length
        X_train, y_train = self._create_sequences(train_scaled, seq_len)
        X_val, y_val = self._create_sequences(val_scaled, seq_len)
        X_test, y_test = self._create_sequences(test_scaled, seq_len)
        
        print(f"   Train: {X_train.shape}")
        print(f"   Val:   {X_val.shape}")
        print(f"   Test:  {X_test.shape}")
        
        # 5. Build Model
        print("🏗️ Building Model...")
        input_shape = (seq_len, X_train.shape[2])
        model = LSTMAttentionModel.build(input_shape)
        
        # 6. Train
        print("🏋️ Training...")
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
            # ModelCheckpoint(os.path.join(self.output_dir, "model_v1.keras"), save_best_only=True), # Disabled for immutable workflow, relying on final save or need generic temp path
            JSONLogger(os.path.join(self.output_dir, "training_metrics.json"))
        ]
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # 7. Evaluate
        print("📊 Evaluating...")
        loss, mae, mape = model.evaluate(X_test, y_test, verbose=0)
        print(f"   Test MAE: {mae:.4f}")
        print(f"   Test MAPE: {mape:.4f}%")
        
        # 8. Save Metadata (Immutable Artifacts)
        # Create a unique version identifier (e.g., timestamp)
        version_id = datetime.now().strftime("v%Y%m%d_%H%M%S")
        model_version_dir = os.path.join(self.output_dir, version_id)
        os.makedirs(model_version_dir, exist_ok=True)
        
        # Save Keras Model
        model_path = os.path.join(model_version_dir, "model.keras")
        model.save(model_path)
        
        # Save Scaler
        joblib.dump(scaler, os.path.join(model_version_dir, "scaler.pkl"))

        # Save Feature Contract (Snapshot)
        # We save the features list used for this specific training run
        feature_contract = {
            "features": feature_names,
            "sequence_length": seq_len,
            "scaler": "standard",
            "symbol": self.config.data.symbol
        }
        
        metadata = {
            "model_version": version_id,
            "trained_on": datetime.now().isoformat(),
            "config": feature_contract,
            "metrics": {
                "test_mae": mae,
                "test_mape": mape
            }
        }
        
        with open(os.path.join(model_version_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
            
        print(f"✅ Training Complete. Immutable artifact saved to {model_version_dir}/")
        
        # Also update 'latest' symlink or copy for convenience if needed, 
        # but for enterprise, we usually refer to specific versions.
        # For simplicity in this demo, we can also overwrite a 'latest' folder or file 
        # but strict versioning is better. 
        # STARTUP logic in API will need to know WHICH version to load.
        # We will implement a quick 'get_latest_model' utility or update 'latest.json' pointer.
        
        self._update_latest_pointer(version_id)

    def _update_latest_pointer(self, version_id):
        """Update a pointer file to the latest model version."""
        pointer_path = os.path.join(self.output_dir, "latest.json")
        with open(pointer_path, "w") as f:
            json.dump({"version": version_id}, f)

    def _create_sequences(self, data, seq_len):
        """Standard sequence creator (target is 0-th column 'close')"""
        # Note: 'close' is the first column in our features list in yaml.
        # But we need to ensure it's index 0. 
        # FeatureBuilder returns columns in order of features.yaml.
        # Close is first there? Yes.
        
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len, 0]) # Target is Close price (index 0)
        return np.array(X), np.array(y)

if __name__ == "__main__":
    Trainer().run()
