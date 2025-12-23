import os
import json
import joblib
import numpy as np
import pandas as pd
import keras
from ..models.base import BasePredictor
from ..features.feature_builder import FeatureBuilder
from ..features.feature_schema import FeatureConfig, DataConfig

class Predictor(BasePredictor):
    """
    Production-ready predictor.
    Loads model artifacts and ensures inference features match training features exactly.
    """
    
    def __init__(self, model, scaler, config, feature_names, version="unknown"):
        self.model = model
        self.scaler = scaler
        self.config = config
        self.feature_names = feature_names
        self.version = version
        self.builder = FeatureBuilder(self.config)

    @classmethod
    def load(cls, model_dir: str):
        """
        Load all artifacts from the model directory.
        Expects:
        - model_v1.keras
        - model_v1.meta.json
        - scaler.pkl
        """
        print(f"Loading model from {model_dir}...")
        
        # 1. Load Metadata
        meta_path = os.path.join(model_dir, "metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata not found at {meta_path}")
            
        with open(meta_path, "r") as f:
            meta = json.load(f)
            
        # Reconstruct Config from Metadata to ensure exact match
        # We assume the layout of metadata matches what FeatureConfig expects or we map it.
        # current FeatureConfig takes: sequence_length, scaler, features, data
        # metadata has: sequence_length, scaler, features, symbol...
        
        # We need to reconstruct the config object so FeatureBuilder calculates the right things.
        loaded_config = FeatureConfig(
            sequence_length=meta["config"]["sequence_length"],
            scaler=meta["config"]["scaler"],
            features=meta["config"]["features"],
            data=DataConfig(
                symbol=meta["config"]["symbol"],
                history_years=5, 
                validation_split=0.0,
                test_split=0.0
            )
        )

        # STRICT VALIDATION: Ensure loaded model matches current codebase contract
        # This prevents "drift" where code changes but model is old, or vice versa.
        from src.features.feature_schema import CONFIG as CURRENT_CONFIG
        
        if loaded_config.features != CURRENT_CONFIG.features:
            raise RuntimeError(f"FATAL: Model features mismatch! \nModel expects: {loaded_config.features}\nCode expects: {CURRENT_CONFIG.features}")
            
        if loaded_config.sequence_length != CURRENT_CONFIG.sequence_length:
             raise RuntimeError(f"FATAL: Sequence length mismatch! Model: {loaded_config.sequence_length}, Code: {CURRENT_CONFIG.sequence_length}")
             
        if loaded_config.scaler != CURRENT_CONFIG.scaler:
             raise RuntimeError(f"FATAL: Scaler mismatch! Model: {loaded_config.scaler}, Code: {CURRENT_CONFIG.scaler}")
        
        print("✅ Schema Validation Passed: Model artifacts match codebase contract.")
        
        # 2. Load Model
        model_path = os.path.join(model_dir, "model.keras")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        model = keras.models.load_model(model_path)
        
        # 3. Load Scaler
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        scaler = joblib.load(scaler_path)
        
        return cls(model, scaler, loaded_config, meta["config"]["features"], version=meta["model_version"])

    def predict(self, raw_df: pd.DataFrame) -> dict:
        """
        End-to-end prediction: Raw Data -> Features -> Scale -> Sequence -> Predict
        Returns a dictionary with predictions and metadata.
        """
        # 1. Build Features
        features_df = self.builder.build(raw_df)
        
        # 2. Scale
        # Ensure we only have the columns the scaler expects
        features_df = features_df[self.feature_names] 
        features_array = features_df.values
        scaled_array = self.scaler.transform(features_array)
        
        # 3. Create Sequence (Last N days)
        seq_len = self.config.sequence_length
        if len(scaled_array) < seq_len:
            raise ValueError(f"Not enough data. Need {seq_len} rows, got {len(scaled_array)}")
            
        # We want to predict the *next* day.
        # Usually we take the last sequence from the available data.
        last_sequence = scaled_array[-seq_len:] # (seq_len, n_features)
        
        # Reshape for model (1, seq_len, n_features)
        input_seq = last_sequence.reshape(1, seq_len, last_sequence.shape[1])
        
        # 4. Predict
        prediction_scaled = self.model.predict(input_seq, verbose=0)
        
        # 5. Inverse Transform
        # The scaler was fitted on N features. The model outputs 1 value (Close price).
        # We need to construct a dummy row to inverse transform.
        # We assume 'close' is the first feature (index 0).
        
        dummy_row = np.zeros((1, len(self.feature_names)))
        dummy_row[0, 0] = prediction_scaled[0][0]
        
        prediction_actual = self.scaler.inverse_transform(dummy_row)[0, 0]
        
        return {
            "prediction": float(prediction_actual),
            "last_date": str(features_df.index[-1]),
            "model_version": self.version
        }
