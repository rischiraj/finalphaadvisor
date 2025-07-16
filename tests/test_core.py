"""
Tests for core functionality including models, configuration, and exceptions.
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import os

from core.models import (
    TimeSeriesData, AnomalyDetectionRequest, AnomalyResult, 
    VisualizationResult, InsightResponse, AnalysisResponse, 
    FileInfo, ProcessingStatus
)
from core.config import Settings, get_settings
from core.exceptions import (
    FileProcessingError, DataValidationError, AnomalyDetectionError,
    AnomalyDetectionMethodError, LLMError, AgentError, ConfigurationError
)


class TestTimeSeriesDataModel:
    """Test TimeSeriesData model validation."""
    
    def test_valid_time_series_data(self):
        """Test creating valid time series data."""
        timestamps = [
            datetime(2025, 1, 1, 0, 0),
            datetime(2025, 1, 1, 1, 0),
            datetime(2025, 1, 1, 2, 0)
        ]
        values = [10.0, 15.0, 12.0]
        
        data = TimeSeriesData(
            timestamp=timestamps,
            values=values,
            column_name="test_values"
        )
        
        assert len(data.timestamp) == 3
        assert len(data.values) == 3
        assert data.column_name == "test_values"
        assert data.timestamp[0] < data.timestamp[1] < data.timestamp[2]
    
    def test_invalid_values_validation(self):
        """Test validation of invalid values."""
        timestamps = [datetime(2025, 1, 1, 0, 0), datetime(2025, 1, 1, 1, 0)]
        
        # Test NaN values
        with pytest.raises(ValueError, match="All values must be finite numeric values"):
            TimeSeriesData(
                timestamp=timestamps,
                values=[10.0, float('nan')],
                column_name="test"
            )
        
        # Test inf values
        with pytest.raises(ValueError, match="All values must be finite numeric values"):
            TimeSeriesData(
                timestamp=timestamps,
                values=[10.0, float('inf')],
                column_name="test"
            )
    
    def test_timestamp_values_length_mismatch(self):
        """Test validation of mismatched timestamp and values lengths."""
        timestamps = [datetime(2025, 1, 1, 0, 0)]
        values = [10.0, 15.0]  # Different length
        
        with pytest.raises(ValueError, match="timestamp and values must have the same length"):
            TimeSeriesData(
                timestamp=timestamps,
                values=values,
                column_name="test"
            )
    
    def test_unsorted_timestamps(self):
        """Test validation of unsorted timestamps."""
        timestamps = [
            datetime(2025, 1, 1, 2, 0),  # Out of order
            datetime(2025, 1, 1, 1, 0),
            datetime(2025, 1, 1, 3, 0)
        ]
        values = [10.0, 15.0, 12.0]
        
        with pytest.raises(ValueError, match="Timestamps must be in ascending order"):
            TimeSeriesData(
                timestamp=timestamps,
                values=values,
                column_name="test"
            )
    
    def test_duplicate_timestamps(self):
        """Test validation of duplicate timestamps."""
        timestamps = [
            datetime(2025, 1, 1, 1, 0),
            datetime(2025, 1, 1, 1, 0),  # Duplicate
            datetime(2025, 1, 1, 2, 0)
        ]
        values = [10.0, 15.0, 12.0]
        
        with pytest.raises(ValueError, match="Timestamps must be in ascending order"):
            TimeSeriesData(
                timestamp=timestamps,
                values=values,
                column_name="test"
            )


class TestAnomalyDetectionRequestModel:
    """Test AnomalyDetectionRequest model validation."""
    
    def test_valid_request_with_file_path(self):
        """Test valid request with file path."""
        request = AnomalyDetectionRequest(
            file_path="./test_data.csv",
            method="z-score",
            threshold=3.0,
            query="Test analysis"
        )
        
        assert request.file_path == "./test_data.csv"
        assert request.data is None
        assert request.method == "z-score"
        assert request.threshold == 3.0
        assert request.query == "Test analysis"
    
    def test_valid_request_with_data(self):
        """Test valid request with direct data."""
        timestamps = [datetime(2025, 1, 1, 0, 0), datetime(2025, 1, 1, 1, 0)]
        values = [10.0, 15.0]
        
        data = TimeSeriesData(
            timestamp=timestamps,
            values=values,
            column_name="test"
        )
        
        request = AnomalyDetectionRequest(
            data=data,
            method="iqr",
            threshold=1.5,
            query="Test direct data analysis"
        )
        
        assert request.file_path is None
        assert request.data == data
        assert request.method == "iqr"
        assert request.threshold == 1.5
    
    def test_invalid_request_no_data_or_file(self):
        """Test invalid request with neither data nor file path."""
        with pytest.raises(ValueError, match="Either file_path or data must be provided"):
            AnomalyDetectionRequest(
                method="z-score",
                query="Test analysis"
            )
    
    def test_invalid_request_both_data_and_file(self):
        """Test invalid request with both data and file path."""
        timestamps = [datetime(2025, 1, 1, 0, 0)]
        values = [10.0]
        
        data = TimeSeriesData(
            timestamp=timestamps,
            values=values,
            column_name="test"
        )
        
        with pytest.raises(ValueError, match="Provide either file_path or data, not both"):
            AnomalyDetectionRequest(
                file_path="./test_data.csv",
                data=data,
                method="z-score",
                query="Test analysis"
            )
    
    def test_valid_methods(self):
        """Test valid detection methods."""
        valid_methods = ["z-score", "iqr", "rolling-iqr", "dbscan"]
        
        for method in valid_methods:
            request = AnomalyDetectionRequest(
                file_path="./test_data.csv",
                method=method,
                query="Test analysis"
            )
            assert request.method == method
    
    def test_invalid_method(self):
        """Test invalid detection method."""
        with pytest.raises(ValueError):
            AnomalyDetectionRequest(
                file_path="./test_data.csv",
                method="invalid_method",
                query="Test analysis"
            )
    
    def test_invalid_threshold(self):
        """Test invalid threshold values."""
        with pytest.raises(ValueError):
            AnomalyDetectionRequest(
                file_path="./test_data.csv",
                method="z-score",
                threshold=-1.0,  # Negative threshold
                query="Test analysis"
            )


class TestAnomalyResultModel:
    """Test AnomalyResult model validation."""
    
    def test_valid_anomaly_result(self):
        """Test creating valid anomaly result."""
        timestamps = [datetime(2025, 1, 1, 0, 0), datetime(2025, 1, 1, 1, 0)]
        
        result = AnomalyResult(
            anomaly_indices=[0, 1],
            method_used="z-score",
            threshold_used=3.0,
            total_points=10,
            anomaly_count=2,
            anomaly_percentage=20.0,
            anomaly_values=[100.0, 150.0],
            anomaly_timestamps=timestamps
        )
        
        assert result.anomaly_count == 2
        assert result.anomaly_percentage == 20.0
        assert len(result.anomaly_indices) == 2
        assert len(result.anomaly_values) == 2
        assert len(result.anomaly_timestamps) == 2
    
    def test_anomaly_count_validation(self):
        """Test anomaly count validation."""
        with pytest.raises(ValueError, match="anomaly_count must match length of anomaly_indices"):
            AnomalyResult(
                anomaly_indices=[0, 1, 2],  # Length 3
                method_used="z-score",
                threshold_used=3.0,
                total_points=10,
                anomaly_count=2,  # Mismatch
                anomaly_percentage=20.0,
                anomaly_values=[100.0, 150.0],
                anomaly_timestamps=[]
            )
    
    def test_percentage_calculation_validation(self):
        """Test percentage calculation validation."""
        with pytest.raises(ValueError, match="anomaly_percentage calculation is incorrect"):
            AnomalyResult(
                anomaly_indices=[0, 1],
                method_used="z-score",
                threshold_used=3.0,
                total_points=10,
                anomaly_count=2,
                anomaly_percentage=50.0,  # Should be 20.0
                anomaly_values=[100.0, 150.0],
                anomaly_timestamps=[]
            )
    
    def test_empty_anomaly_result(self):
        """Test empty anomaly result."""
        result = AnomalyResult(
            anomaly_indices=[],
            method_used="z-score",
            threshold_used=3.0,
            total_points=10,
            anomaly_count=0,
            anomaly_percentage=0.0,
            anomaly_values=[],
            anomaly_timestamps=[]
        )
        
        assert result.anomaly_count == 0
        assert result.anomaly_percentage == 0.0
        assert len(result.anomaly_indices) == 0


class TestVisualizationResultModel:
    """Test VisualizationResult model validation."""
    
    def test_valid_visualization_result(self):
        """Test creating valid visualization result."""
        result = VisualizationResult(
            plot_path="./outputs/plots/test_plot.png",
            plot_base64="base64_encoded_image_data",
            plot_description="Test anomaly detection plot",
            plot_type="matplotlib"
        )
        
        assert result.plot_path == "./outputs/plots/test_plot.png"
        assert result.plot_base64 == "base64_encoded_image_data"
        assert result.plot_description == "Test anomaly detection plot"
        assert result.plot_type == "matplotlib"
    
    def test_valid_plot_types(self):
        """Test valid plot types."""
        valid_types = ["matplotlib", "plotly"]
        
        for plot_type in valid_types:
            result = VisualizationResult(
                plot_path="./test_plot.png",
                plot_base64="base64_data",
                plot_description="Test plot",
                plot_type=plot_type
            )
            assert result.plot_type == plot_type
    
    def test_invalid_plot_type(self):
        """Test invalid plot type."""
        with pytest.raises(ValueError):
            VisualizationResult(
                plot_path="./test_plot.png",
                plot_base64="base64_data",
                plot_description="Test plot",
                plot_type="invalid_type"
            )


class TestInsightResponseModel:
    """Test InsightResponse model validation."""
    
    def test_valid_insight_response(self):
        """Test creating valid insight response."""
        response = InsightResponse(
            summary="Test analysis summary",
            anomaly_explanations=["Explanation 1", "Explanation 2"],
            recommendations=["Recommendation 1", "Recommendation 2"],
            root_causes=["Cause 1", "Cause 2"],
            confidence_score=85.0
        )
        
        assert response.summary == "Test analysis summary"
        assert len(response.anomaly_explanations) == 2
        assert len(response.recommendations) == 2
        assert len(response.root_causes) == 2
        assert response.confidence_score == 85.0
    
    def test_confidence_score_validation(self):
        """Test confidence score validation."""
        # Test valid range
        response = InsightResponse(
            summary="Test",
            anomaly_explanations=["Test"],
            recommendations=["Test"],
            root_causes=["Test"],
            confidence_score=75.0
        )
        assert response.confidence_score == 75.0
        
        # Test invalid range
        with pytest.raises(ValueError):
            InsightResponse(
                summary="Test",
                anomaly_explanations=["Test"],
                recommendations=["Test"],
                root_causes=["Test"],
                confidence_score=150.0  # > 100
            )
        
        with pytest.raises(ValueError):
            InsightResponse(
                summary="Test",
                anomaly_explanations=["Test"],
                recommendations=["Test"],
                root_causes=["Test"],
                confidence_score=-10.0  # < 0
            )
    
    def test_empty_lists_validation(self):
        """Test validation of empty lists."""
        with pytest.raises(ValueError, match="Lists cannot be empty"):
            InsightResponse(
                summary="Test",
                anomaly_explanations=[],  # Empty list
                recommendations=["Test"],
                root_causes=["Test"],
                confidence_score=75.0
            )


class TestFileInfoModel:
    """Test FileInfo model validation."""
    
    def test_valid_file_info(self):
        """Test creating valid file info."""
        info = FileInfo(
            filename="test_data.csv",
            file_size=1024,
            file_type="csv",
            columns=["timestamp", "value"],
            row_count=100
        )
        
        assert info.filename == "test_data.csv"
        assert info.file_size == 1024
        assert info.file_type == "csv"
        assert len(info.columns) == 2
        assert info.row_count == 100
    
    def test_negative_file_size(self):
        """Test validation of negative file size."""
        with pytest.raises(ValueError):
            FileInfo(
                filename="test.csv",
                file_size=-100,  # Negative size
                file_type="csv",
                columns=["timestamp", "value"],
                row_count=100
            )
    
    def test_negative_row_count(self):
        """Test validation of negative row count."""
        with pytest.raises(ValueError):
            FileInfo(
                filename="test.csv",
                file_size=1024,
                file_type="csv",
                columns=["timestamp", "value"],
                row_count=-10  # Negative count
            )


class TestProcessingStatusModel:
    """Test ProcessingStatus model validation."""
    
    def test_valid_processing_status(self):
        """Test creating valid processing status."""
        status = ProcessingStatus(
            task_id="test_task_123",
            status="running",
            progress=50.0,
            message="Processing data..."
        )
        
        assert status.task_id == "test_task_123"
        assert status.status == "running"
        assert status.progress == 50.0
        assert status.message == "Processing data..."
        assert status.started_at is not None
        assert status.completed_at is None
    
    def test_valid_statuses(self):
        """Test valid status values."""
        valid_statuses = ["pending", "running", "completed", "failed"]
        
        for status_val in valid_statuses:
            status = ProcessingStatus(
                task_id="test_task",
                status=status_val,
                progress=0.0,
                message="Test"
            )
            assert status.status == status_val
    
    def test_invalid_progress_range(self):
        """Test invalid progress range."""
        with pytest.raises(ValueError):
            ProcessingStatus(
                task_id="test_task",
                status="running",
                progress=150.0,  # > 100
                message="Test"
            )
        
        with pytest.raises(ValueError):
            ProcessingStatus(
                task_id="test_task",
                status="running",
                progress=-10.0,  # < 0
                message="Test"
            )


class TestConfigurationSettings:
    """Test configuration settings and validation."""
    
    def test_valid_settings(self):
        """Test creating valid settings."""
        settings = Settings(
            google_ai_api_key="test_api_key_12345",
            llm_model="gemini-2.0-flash",
            llm_temperature=0.1,
            api_host="localhost",
            api_port=8000
        )
        
        assert settings.google_ai_api_key == "test_api_key_12345"
        assert settings.llm_model == "gemini-2.0-flash"
        assert settings.llm_temperature == 0.1
        assert settings.api_host == "localhost"
        assert settings.api_port == 8000
    
    def test_invalid_api_key(self):
        """Test invalid API key validation."""
        with pytest.raises(ValueError, match="Google AI API key must be provided"):
            Settings(
                google_ai_api_key="your_google_ai_api_key_here",  # Placeholder
                llm_model="gemini-2.0-flash"
            )
    
    def test_invalid_temperature_range(self):
        """Test invalid temperature range."""
        with pytest.raises(ValueError):
            Settings(
                google_ai_api_key="valid_key",
                llm_temperature=3.0  # > 2.0
            )
    
    def test_invalid_port_range(self):
        """Test invalid port range."""
        with pytest.raises(ValueError):
            Settings(
                google_ai_api_key="valid_key",
                api_port=70000  # > 65535
            )
    
    def test_directory_creation(self):
        """Test directory creation validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "test_data"
            
            settings = Settings(
                google_ai_api_key="test_key",
                data_dir=test_path
            )
            
            # Directory should be created
            assert test_path.exists()
            assert test_path.is_dir()
    
    @patch.dict(os.environ, {"GOOGLE_AI_API_KEY": "env_test_key"})
    def test_environment_variable_loading(self):
        """Test loading settings from environment variables."""
        settings = Settings()
        assert settings.google_ai_api_key == "env_test_key"
    
    def test_get_settings_function(self):
        """Test get_settings function."""
        with patch.dict(os.environ, {"GOOGLE_AI_API_KEY": "test_key"}):
            settings = get_settings()
            assert isinstance(settings, Settings)
            assert settings.google_ai_api_key == "test_key"


class TestCustomExceptions:
    """Test custom exception classes."""
    
    def test_file_processing_error(self):
        """Test FileProcessingError exception."""
        error = FileProcessingError("File not found", "test_file.csv")
        
        assert str(error) == "File not found"
        assert error.filename == "test_file.csv"
        assert error.error_code == "FILE_PROCESSING_ERROR"
    
    def test_data_validation_error(self):
        """Test DataValidationError exception."""
        error = DataValidationError("Invalid data format", "value_column")
        
        assert str(error) == "Invalid data format"
        assert error.field == "value_column"
        assert error.error_code == "DATA_VALIDATION_ERROR"
    
    def test_anomaly_detection_method_error(self):
        """Test AnomalyDetectionMethodError exception."""
        error = AnomalyDetectionMethodError("Unknown method", "invalid_method")
        
        assert str(error) == "Unknown method"
        assert error.method == "invalid_method"
        assert error.error_code == "ANOMALY_DETECTION_METHOD_ERROR"
    
    def test_llm_error(self):
        """Test LLMError exception."""
        error = LLMError("API rate limit exceeded", "gemini-2.0-flash")
        
        assert str(error) == "API rate limit exceeded"
        assert error.model == "gemini-2.0-flash"
        assert error.error_code == "LLM_ERROR"
    
    def test_agent_error(self):
        """Test AgentError exception."""
        error = AgentError("Agent execution failed", "anomaly_agent")
        
        assert str(error) == "Agent execution failed"
        assert error.agent_name == "anomaly_agent"
        assert error.error_code == "AGENT_ERROR"
    
    def test_configuration_error(self):
        """Test ConfigurationError exception."""
        error = ConfigurationError("Missing required setting", "api_key")
        
        assert str(error) == "Missing required setting"
        assert error.config_key == "api_key"
        assert error.error_code == "CONFIGURATION_ERROR"
    
    def test_base_exception_inheritance(self):
        """Test that custom exceptions inherit from proper base classes."""
        # All should inherit from AnomalyDetectionError
        assert issubclass(FileProcessingError, AnomalyDetectionError)
        assert issubclass(DataValidationError, AnomalyDetectionError)
        assert issubclass(AnomalyDetectionMethodError, AnomalyDetectionError)
        assert issubclass(LLMError, AnomalyDetectionError)
        assert issubclass(AgentError, AnomalyDetectionError)
        assert issubclass(ConfigurationError, AnomalyDetectionError)
        
        # AnomalyDetectionError should inherit from Exception
        assert issubclass(AnomalyDetectionError, Exception)