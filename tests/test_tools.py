"""
Tests for anomaly detection tools.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from agents.tools.file_reader import FileReaderTool
from agents.tools.anomaly_detector import AnomalyDetectionTool
from agents.tools.visualizer import VisualizationTool
from core.exceptions import FileProcessingError, DataValidationError, AnomalyDetectionMethodError
from core.models import TimeSeriesData


class TestFileReaderTool:
    """Test file reading functionality."""
    
    def test_read_csv_file(self, sample_csv_file):
        """Test reading CSV file."""
        reader = FileReaderTool()
        data = reader.read_file(sample_csv_file)
        
        assert isinstance(data, TimeSeriesData)
        assert len(data.values) > 0
        assert len(data.timestamp) == len(data.values)
        assert data.column_name in ['value', 'Value']
    
    def test_read_excel_file(self, sample_excel_file):
        """Test reading Excel file."""
        reader = FileReaderTool()
        data = reader.read_file(sample_excel_file)
        
        assert isinstance(data, TimeSeriesData)
        assert len(data.values) > 0
        assert len(data.timestamp) == len(data.values)
        assert data.column_name in ['Amount', 'amount']
    
    def test_file_not_found(self):
        """Test handling of non-existent file."""
        reader = FileReaderTool()
        
        with pytest.raises(FileProcessingError):
            reader.read_file("non_existent_file.csv")
    
    def test_get_file_info(self, sample_csv_file):
        """Test getting file information."""
        reader = FileReaderTool()
        info = reader.get_file_info(sample_csv_file)
        
        assert info.filename == "test_data.csv"
        assert info.file_type == "csv"
        assert info.file_size > 0
        assert len(info.columns) >= 2
        assert info.row_count > 0
    
    def test_detect_timestamp_column(self, tmp_path):
        """Test timestamp column detection."""
        # Create file with various column names
        data = pd.DataFrame({
            'Date': pd.date_range('2025-01-01', periods=10),
            'Value': range(10)
        })
        file_path = tmp_path / "test_columns.csv"
        data.to_csv(file_path, index=False)
        
        reader = FileReaderTool()
        result = reader.read_file(str(file_path))
        
        assert result.column_name == "Value"
        assert len(result.timestamp) == 10


class TestAnomalyDetectionTool:
    """Test anomaly detection methods."""
    
    def test_z_score_detection(self, sample_time_series_data):
        """Test Z-score anomaly detection."""
        detector = AnomalyDetectionTool()
        result = detector.detect_anomalies(sample_time_series_data, "z-score", threshold=2.0)
        
        assert result.method_used == "z-score"
        assert result.threshold_used == 2.0
        assert result.total_points == len(sample_time_series_data.values)
        assert result.anomaly_count >= 0
        assert 0 <= result.anomaly_percentage <= 100
        assert len(result.anomaly_indices) == result.anomaly_count
    
    def test_iqr_detection(self, sample_time_series_data):
        """Test IQR anomaly detection."""
        detector = AnomalyDetectionTool()
        result = detector.detect_anomalies(sample_time_series_data, "iqr", threshold=1.5)
        
        assert result.method_used == "iqr"
        assert result.threshold_used == 1.5
        assert result.total_points == len(sample_time_series_data.values)
        assert result.anomaly_count >= 0
        assert len(result.anomaly_indices) == result.anomaly_count
    
    def test_dbscan_detection(self, sample_time_series_data):
        """Test DBSCAN anomaly detection."""
        detector = AnomalyDetectionTool()
        result = detector.detect_anomalies(
            sample_time_series_data, 
            "dbscan", 
            threshold=1.0,
            min_samples=3
        )
        
        assert result.method_used == "dbscan"
        assert result.threshold_used == 1.0
        assert result.total_points == len(sample_time_series_data.values)
        assert result.anomaly_count >= 0
        assert len(result.anomaly_indices) == result.anomaly_count
    
    def test_invalid_method(self, sample_time_series_data):
        """Test handling of invalid detection method."""
        detector = AnomalyDetectionTool()
        
        with pytest.raises(AnomalyDetectionMethodError):
            detector.detect_anomalies(sample_time_series_data, "invalid_method")
    
    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        # Create minimal data
        timestamps = [datetime.now(), datetime.now() + timedelta(hours=1)]
        values = [1.0, 2.0]
        
        data = TimeSeriesData(
            timestamp=timestamps,
            values=values,
            column_name="minimal"
        )
        
        detector = AnomalyDetectionTool()
        
        with pytest.raises(DataValidationError):
            detector.detect_anomalies(data, "z-score")
    
    def test_method_recommendation(self, sample_time_series_data):
        """Test method recommendation."""
        detector = AnomalyDetectionTool()
        recommended = detector.recommend_method(sample_time_series_data)
        
        assert recommended in ['z-score', 'iqr', 'dbscan']
    
    def test_get_method_info(self):
        """Test getting method information."""
        detector = AnomalyDetectionTool()
        
        info = detector.get_method_info('z-score')
        assert 'name' in info
        assert 'description' in info
        assert 'default_threshold' in info
        
        # Test all methods
        for method in ['z-score', 'iqr', 'dbscan']:
            info = detector.get_method_info(method)
            assert isinstance(info, dict)
            assert len(info) > 0
    
    def test_z_score_with_constant_data(self):
        """Test Z-score with constant data (zero variance)."""
        # Create constant data
        timestamps = [datetime.now() + timedelta(hours=i) for i in range(10)]
        values = [5.0] * 10  # All same value
        
        data = TimeSeriesData(
            timestamp=timestamps,
            values=values,
            column_name="constant"
        )
        
        detector = AnomalyDetectionTool()
        result = detector.detect_anomalies(data, "z-score")
        
        # Should detect no anomalies with constant data
        assert result.anomaly_count == 0
    
    def test_anomaly_result_validation(self, sample_time_series_data):
        """Test that anomaly results are properly validated."""
        detector = AnomalyDetectionTool()
        result = detector.detect_anomalies(sample_time_series_data, "z-score")
        
        # Check that percentages are calculated correctly
        expected_percentage = (result.anomaly_count / result.total_points) * 100
        assert abs(result.anomaly_percentage - expected_percentage) < 0.01
        
        # Check that anomaly values and timestamps match indices
        for i, idx in enumerate(result.anomaly_indices):
            assert result.anomaly_values[i] == sample_time_series_data.values[idx]
            assert result.anomaly_timestamps[i] == sample_time_series_data.timestamp[idx]


class TestVisualizationTool:
    """Test visualization generation."""
    
    def test_create_matplotlib_plot(self, sample_time_series_data, test_settings):
        """Test creating matplotlib plot."""
        from core.models import AnomalyResult
        
        # Create mock anomaly result
        anomaly_result = AnomalyResult(
            anomaly_indices=[25, 50, 75],
            method_used="z-score",
            threshold_used=3.0,
            total_points=100,
            anomaly_count=3,
            anomaly_percentage=3.0,
            anomaly_values=[125.0, 150.0, 175.0],
            anomaly_timestamps=[sample_time_series_data.timestamp[i] for i in [25, 50, 75]]
        )
        
        with patch('core.config.get_settings', return_value=test_settings):
            visualizer = VisualizationTool()
            result = visualizer.create_anomaly_plot(
                sample_time_series_data,
                anomaly_result,
                plot_type='matplotlib'
            )
        
        assert result.plot_type == 'matplotlib'
        assert result.plot_path != ""
        assert result.plot_base64 != ""
        assert "anomaly" in result.plot_description.lower()
    
    @patch('plotly.graph_objects.Figure.write_html')
    @patch('plotly.graph_objects.Figure.to_image')
    def test_create_plotly_plot(self, mock_to_image, mock_write_html, sample_time_series_data, test_settings):
        """Test creating plotly plot."""
        from core.models import AnomalyResult
        
        # Mock plotly methods
        mock_to_image.return_value = b'fake_image_data'
        
        anomaly_result = AnomalyResult(
            anomaly_indices=[25],
            method_used="iqr",
            threshold_used=1.5,
            total_points=100,
            anomaly_count=1,
            anomaly_percentage=1.0,
            anomaly_values=[125.0],
            anomaly_timestamps=[sample_time_series_data.timestamp[25]]
        )
        
        with patch('core.config.get_settings', return_value=test_settings):
            visualizer = VisualizationTool()
            result = visualizer.create_anomaly_plot(
                sample_time_series_data,
                anomaly_result,
                plot_type='plotly'
            )
        
        assert result.plot_type == 'plotly'
        assert mock_write_html.called or mock_to_image.called
    
    def test_plot_with_no_anomalies(self, sample_time_series_data, test_settings):
        """Test creating plot when no anomalies are detected."""
        from core.models import AnomalyResult
        
        # Create result with no anomalies
        anomaly_result = AnomalyResult(
            anomaly_indices=[],
            method_used="z-score",
            threshold_used=5.0,  # Very high threshold
            total_points=100,
            anomaly_count=0,
            anomaly_percentage=0.0,
            anomaly_values=[],
            anomaly_timestamps=[]
        )
        
        with patch('core.config.get_settings', return_value=test_settings):
            visualizer = VisualizationTool()
            result = visualizer.create_anomaly_plot(
                sample_time_series_data,
                anomaly_result
            )
        
        assert result.plot_type == 'matplotlib'
        assert "0 anomalies" in result.plot_description


# Integration tests
class TestToolIntegration:
    """Test integration between different tools."""
    
    def test_file_to_detection_pipeline(self, sample_csv_file):
        """Test complete pipeline from file to detection."""
        # Read file
        reader = FileReaderTool()
        data = reader.read_file(sample_csv_file)
        
        # Detect anomalies
        detector = AnomalyDetectionTool()
        result = detector.detect_anomalies(data, "z-score")
        
        # Verify pipeline works
        assert isinstance(result.anomaly_indices, list)
        assert result.total_points == len(data.values)
        assert result.method_used == "z-score"
    
    def test_recommendation_to_detection_pipeline(self, sample_time_series_data):
        """Test using recommendation in detection."""
        detector = AnomalyDetectionTool()
        
        # Get recommendation
        recommended = detector.recommend_method(sample_time_series_data)
        
        # Use recommendation
        result = detector.detect_anomalies(sample_time_series_data, recommended)
        
        assert result.method_used == recommended
        assert result.anomaly_count >= 0