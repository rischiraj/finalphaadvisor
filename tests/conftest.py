"""
Pytest configuration and fixtures for anomaly detection tests.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os
from unittest.mock import AsyncMock, MagicMock

from core.models import TimeSeriesData, AnomalyDetectionRequest
from core.config import Settings


@pytest.fixture
def sample_time_series_data():
    """
    Create sample time-series data for testing.
    """
    # Generate timestamps
    start_time = datetime.now() - timedelta(days=30)
    timestamps = [start_time + timedelta(hours=i) for i in range(100)]
    
    # Generate normal values with some outliers
    values = []
    for i in range(100):
        if i in [25, 50, 75]:  # Add some outliers
            values.append(100.0 + i)  # Clear outliers
        else:
            values.append(10.0 + (i % 10))  # Normal pattern
    
    return TimeSeriesData(
        timestamp=timestamps,
        values=values,
        column_name="test_values"
    )


@pytest.fixture
def sample_csv_file(tmp_path):
    """
    Create a temporary CSV file for testing.
    """
    # Create sample data
    data = {
        'timestamp': pd.date_range('2025-01-01', periods=50, freq='H'),
        'value': [10 + i + (100 if i in [10, 25, 40] else 0) for i in range(50)]
    }
    df = pd.DataFrame(data)
    
    # Save to temporary file
    file_path = tmp_path / "test_data.csv"
    df.to_csv(file_path, index=False)
    
    return str(file_path)


@pytest.fixture
def sample_excel_file(tmp_path):
    """
    Create a temporary Excel file for testing.
    """
    # Create sample data
    data = {
        'Date': pd.date_range('2025-01-01', periods=30, freq='D'),
        'Amount': [50 + i + (200 if i in [5, 15, 25] else 0) for i in range(30)]
    }
    df = pd.DataFrame(data)
    
    # Save to temporary file
    file_path = tmp_path / "test_data.xlsx"
    df.to_excel(file_path, index=False)
    
    return str(file_path)


@pytest.fixture
def test_settings():
    """
    Create test settings configuration.
    """
    return Settings(
        google_ai_api_key="test_api_key_12345",
        llm_model="gemini-2.0-flash",
        llm_temperature=0.1,
        data_dir=Path("./test_data"),
        output_dir=Path("./test_outputs"),
        plots_dir=Path("./test_plots"),
        log_level="INFO"
    )


@pytest.fixture
def mock_llm():
    """
    Mock LLM for testing without API calls.
    """
    mock = AsyncMock()
    mock.ainvoke = AsyncMock(return_value=MagicMock(content='{"summary": "Test analysis completed", "anomaly_explanations": ["Test explanation"], "recommendations": ["Test recommendation"], "root_causes": ["Test cause"], "confidence_score": 85}'))
    return mock


@pytest.fixture
def sample_anomaly_request(sample_time_series_data):
    """
    Create sample anomaly detection request.
    """
    return AnomalyDetectionRequest(
        data=sample_time_series_data,
        method="z-score",
        threshold=3.0,
        query="Test anomaly detection"
    )


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch, tmp_path):
    """
    Setup test environment with temporary directories.
    """
    # Create test directories
    test_data_dir = tmp_path / "data"
    test_output_dir = tmp_path / "outputs" 
    test_plots_dir = tmp_path / "plots"
    
    test_data_dir.mkdir()
    test_output_dir.mkdir()
    test_plots_dir.mkdir()
    
    # Set environment variables for testing
    monkeypatch.setenv("DATA_DIR", str(test_data_dir))
    monkeypatch.setenv("OUTPUT_DIR", str(test_output_dir))
    monkeypatch.setenv("PLOTS_DIR", str(test_plots_dir))
    monkeypatch.setenv("GOOGLE_AI_API_KEY", "test_key_12345")
    
    yield
    
    # Cleanup is automatic with tmp_path


@pytest.fixture
def mock_file_reader():
    """
    Mock file reader for testing.
    """
    from agents.tools.file_reader import FileReaderTool
    from unittest.mock import patch
    
    with patch.object(FileReaderTool, 'read_file') as mock_read:
        # Configure mock to return sample data
        timestamps = [datetime.now() + timedelta(hours=i) for i in range(10)]
        values = [10.0 + i + (50 if i == 5 else 0) for i in range(10)]
        
        mock_read.return_value = TimeSeriesData(
            timestamp=timestamps,
            values=values,
            column_name="mock_values"
        )
        
        yield mock_read


@pytest.fixture
def disable_llm_calls(monkeypatch):
    """
    Disable actual LLM API calls for testing.
    """
    # Mock the ChatGoogleGenerativeAI class
    mock_response = MagicMock()
    mock_response.content = '{"summary": "Mock analysis", "anomaly_explanations": ["Mock explanation"], "recommendations": ["Mock recommendation"], "root_causes": ["Mock cause"], "confidence_score": 75}'
    
    async def mock_ainvoke(*args, **kwargs):
        return mock_response
    
    monkeypatch.setattr("langchain_google_genai.ChatGoogleGenerativeAI.ainvoke", mock_ainvoke)
    
    yield


# Test data generators
def generate_normal_data(size=100, mean=10, std=2):
    """Generate normal distributed data."""
    import numpy as np
    timestamps = [datetime.now() + timedelta(hours=i) for i in range(size)]
    values = np.random.normal(mean, std, size).tolist()
    
    return TimeSeriesData(
        timestamp=timestamps,
        values=values,
        column_name="normal_data"
    )


def generate_data_with_outliers(size=100, outlier_indices=None, outlier_value=100):
    """Generate data with specific outliers."""
    if outlier_indices is None:
        outlier_indices = [size//4, size//2, 3*size//4]
    
    timestamps = [datetime.now() + timedelta(hours=i) for i in range(size)]
    values = [10.0 + (i % 10) for i in range(size)]
    
    # Add outliers
    for idx in outlier_indices:
        if 0 <= idx < size:
            values[idx] = outlier_value
    
    return TimeSeriesData(
        timestamp=timestamps,
        values=values,
        column_name="outlier_data"
    )


# Async test helper
@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()