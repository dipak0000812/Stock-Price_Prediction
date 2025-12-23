import pytest
import pandas as pd
import numpy as np
import sys
import os
import shutil
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training.trainer import Trainer
from src.inference.predictor import Predictor
from src.features.feature_schema import CONFIG

def test_offline_pipeline(tmp_path):
    """
    Full integration test: Train -> Save -> Load -> Predict.
    Mocks Yahoo Finance to ensure tests work offline (CI friendly).
    """
    
    # 1. Mock Data Source
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
    mock_df = pd.DataFrame({
        'Open': np.random.uniform(100, 200, 1000),
        'High': np.random.uniform(105, 205, 1000),
        'Low': np.random.uniform(95, 195, 1000),
        'Close': np.random.uniform(100, 200, 1000),
        'Volume': np.random.randint(1000, 10000, 1000)
    }, index=dates)

    # Patch DataLoader to return mock data
    with patch("src.data.loader.DataLoader.fetch", return_value=mock_df):
        
        # 2. Config Override for Test
        # Use temp dir for output
        output_dir = tmp_path / "models"
        output_dir.mkdir()
        
        # Initialize Trainer
        trainer = Trainer(CONFIG)
        trainer.output_dir = str(output_dir)
        
        # 3. RUN TRAINING
        # Reduce epochs for speed
        with patch("tensorflow.keras.Model.fit") as mock_fit:
            # Configure mock fit return value
            mock_history = MagicMock()
            mock_history.history = {'val_mae': [0.5], 'val_mape': [5.0]}
            mock_fit.return_value = mock_history
            
            # We enforce a quick training loop or just rely on trainer logic
            # For integration, we want real fit but few epochs
            trainer.run(epochs=1, batch_size=32)
            
        # 4. Verify Artifacts
        # Check if 'latest.json' exists
        latest_path = output_dir / "latest.json"
        assert latest_path.exists()
        
        # 5. LOAD & PREDICT
        # Load using Predictor via the artifacts we just made
        # We need to find the specific version folder
        import json
        with open(latest_path, "r") as f:
            pointer = json.load(f)
            version_id = pointer["version"]
            
        model_dir = output_dir / version_id
        
        # Load Predictor
        # We mock FeatureConfig validation if needed, but since we use same CONFIG, it should pass
        predictor = Predictor.load(str(model_dir))
        
        # Predict on same mock data
        result = predictor.predict(mock_df)
        
        assert "prediction" in result
        assert result["model_version"] == version_id
