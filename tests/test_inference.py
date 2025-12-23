import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference.predictor import Predictor
from src.features.feature_schema import FeatureConfig, DataConfig

@pytest.fixture
def mock_predictor():
    # Mock resources
    mock_model = MagicMock()
    # Model output: (batch, 1) -> scaled prediction
    mock_model.predict.return_value = np.array([[0.5]]) 
    
    mock_scaler = MagicMock()
    # Scaler inverse: dummy row -> actual value
    mock_scaler.inverse_transform.return_value = np.array([[150.0] + [0]*19]) # Assuming 20 features
    mock_scaler.transform.return_value = np.random.random((100, 20)) # Mock scaled data
    
    config = FeatureConfig(
        sequence_length=10, # Shorten for test
        scaler="standard",
        features=["close", "volume"], # Minimal features
        data=DataConfig("TEST", 1, 0.1, 0.1)
    )
    
    # We need to patch FeatureBuilder to return compatible shape for our mock config
    # Or just use the real builder if it works with specific config.
    # For unit testing Predictor, strictly we should mock builder too but let's do integration style.
    feature_names = ["close", "volume"]
    
    return Predictor(mock_model, mock_scaler, config, feature_names, version="1.0.0")

def test_prediction_flow(mock_predictor):
    # Dummy raw data
    dates = pd.date_range(start='2020-01-01', periods=50, freq='D')
    df = pd.DataFrame({
        'Open': np.random.uniform(100, 200, 50),
        'High': np.random.uniform(105, 205, 50),
        'Low': np.random.uniform(95, 195, 50),
        'Close': np.random.uniform(100, 200, 50),
        'Volume': np.random.randint(1000, 10000, 50)
    }, index=dates)
    
    # Mock the builder.build to return just what we need
    mock_predictor.builder = MagicMock()
    mock_predictor.builder.build.return_value = pd.DataFrame(
        np.random.random((50, 2)), 
        columns=["close", "volume"],
        index=dates
    )
    
    # Test predict
    result = mock_predictor.predict(df)
    
    assert "prediction" in result
    assert result["prediction"] == 150.0
    assert result["model_version"] == "1.0.0"
