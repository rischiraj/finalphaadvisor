"""
Tests for FastAPI endpoints and API functionality.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import json

from fastapi.testclient import TestClient
from fastapi import HTTPException

from api.main import app
from api.models import APIAnomalyDetectionRequest, APIAnalysisResponse
from api.dependencies import get_cached_settings
from core.models import TimeSeriesData, AnomalyResult, VisualizationResult, InsightResponse, AnalysisResponse
from core.config import Settings


class TestAPIEndpoints:
    """Test FastAPI endpoints."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = self.client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs_url" in data
        assert data["docs_url"] == "/docs"
    
    def test_methods_endpoint(self):
        """Test available methods endpoint."""
        response = self.client.get("/api/v1/methods")
        assert response.status_code == 200
        
        data = response.json()
        assert "available_methods" in data
        assert "default_method" in data
        assert "recommendation_available" in data
        
        # Check that all expected methods are present
        methods = data["available_methods"]
        expected_methods = ["z-score", "iqr", "rolling-iqr", "dbscan"]
        for method in expected_methods:
            assert method in methods
            assert "name" in methods[method]
            assert "description" in methods[method]
    
    @patch('agents.supervisor.AnomalyDetectionSupervisor.analyze')
    def test_analyze_endpoint_with_file_path(self, mock_analyze):
        """Test analyze endpoint with file path."""
        # Mock the analysis result
        mock_result = self._create_mock_analysis_result()
        mock_analyze.return_value = mock_result
        
        # Test request
        request_data = {
            "file_path": "./test_data.csv",
            "method": "z-score",
            "threshold": 3.0,
            "query": "Test analysis"
        }
        
        response = self.client.post("/api/v1/analyze", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "anomaly_result" in data
        assert "visualization" in data
        assert "insights" in data
        assert "processing_time" in data
    
    @patch('agents.supervisor.AnomalyDetectionSupervisor.analyze')
    def test_analyze_endpoint_with_direct_data(self, mock_analyze):
        """Test analyze endpoint with direct data."""
        # Mock the analysis result
        mock_result = self._create_mock_analysis_result()
        mock_analyze.return_value = mock_result
        
        # Test request with direct data
        request_data = {
            "data": {
                "timestamp": ["2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z", "2025-01-01T02:00:00Z"],
                "values": [10.0, 15.0, 100.0],
                "column_name": "test_values"
            },
            "method": "iqr",
            "threshold": 1.5,
            "query": "Test direct data analysis"
        }
        
        response = self.client.post("/api/v1/analyze", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "anomaly_result" in data
    
    def test_analyze_endpoint_validation_error(self):
        """Test analyze endpoint validation error."""
        # Request with neither file_path nor data
        request_data = {
            "method": "z-score",
            "query": "Test analysis"
        }
        
        response = self.client.post("/api/v1/analyze", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_analyze_endpoint_both_file_and_data_error(self):
        """Test analyze endpoint with both file and data."""
        # Request with both file_path and data
        request_data = {
            "file_path": "./test_data.csv",
            "data": {
                "timestamp": ["2025-01-01T00:00:00Z"],
                "values": [10.0],
                "column_name": "test"
            },
            "method": "z-score",
            "query": "Test analysis"
        }
        
        response = self.client.post("/api/v1/analyze", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_upload_file_endpoint(self):
        """Test file upload endpoint."""
        # Create a test CSV file
        test_data = "timestamp,value\n2025-01-01,10\n2025-01-02,20\n"
        
        response = self.client.post(
            "/api/v1/upload-file",
            files={"file": ("test.csv", test_data, "text/csv")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "file_info" in data
        assert "message" in data
    
    def test_upload_file_invalid_extension(self):
        """Test file upload with invalid extension."""
        test_data = "some text data"
        
        response = self.client.post(
            "/api/v1/upload-file",
            files={"file": ("test.txt", test_data, "text/plain")}
        )
        
        assert response.status_code == 400
    
    def test_download_plot_endpoint(self):
        """Test plot download endpoint."""
        # This test would require an actual plot file
        # For now, test the 404 case
        response = self.client.get("/api/v1/download-plot/nonexistent_plot.png")
        assert response.status_code == 404
    
    def _create_mock_analysis_result(self):
        """Create a mock analysis result for testing."""
        from datetime import datetime
        
        # Mock anomaly result
        anomaly_result = AnomalyResult(
            anomaly_indices=[2, 5, 8],
            method_used="z-score",
            threshold_used=3.0,
            total_points=10,
            anomaly_count=3,
            anomaly_percentage=30.0,
            anomaly_values=[100.0, 150.0, 200.0],
            anomaly_timestamps=[datetime.now() for _ in range(3)]
        )
        
        # Mock visualization result
        visualization_result = VisualizationResult(
            plot_path="test_plot.png",
            plot_base64="base64_encoded_image",
            plot_description="Test plot description",
            plot_type="matplotlib"
        )
        
        # Mock insight response
        insight_response = InsightResponse(
            summary="Test analysis summary",
            anomaly_explanations=["Test explanation"],
            recommendations=["Test recommendation"],
            root_causes=["Test cause"],
            confidence_score=85.0
        )
        
        # Mock analysis response
        return AnalysisResponse(
            anomaly_result=anomaly_result,
            visualization=visualization_result,
            insights=insight_response,
            processing_time=1.5,
            metadata={"test": "metadata"}
        )


class TestAPIModels:
    """Test API model validation."""
    
    def test_api_time_series_data_validation(self):
        """Test API time series data validation."""
        from api.models import APITimeSeriesData
        
        # Valid data
        valid_data = {
            "timestamp": ["2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z"],
            "values": [10.0, 20.0],
            "column_name": "test_values"
        }
        
        model = APITimeSeriesData(**valid_data)
        assert len(model.timestamp) == 2
        assert len(model.values) == 2
        assert model.column_name == "test_values"
        
        # Test conversion to core model
        core_model = model.to_core_model()
        assert isinstance(core_model, TimeSeriesData)
        assert len(core_model.timestamp) == 2
        assert len(core_model.values) == 2
    
    def test_api_anomaly_detection_request_validation(self):
        """Test API anomaly detection request validation."""
        # Valid request with file_path
        valid_request = {
            "file_path": "./test_data.csv",
            "method": "z-score",
            "threshold": 3.0,
            "query": "Test analysis"
        }
        
        model = APIAnomalyDetectionRequest(**valid_request)
        assert model.file_path == "./test_data.csv"
        assert model.method == "z-score"
        assert model.threshold == 3.0
        assert model.query == "Test analysis"
        
        # Test conversion to core model
        core_model = model.to_core_model()
        assert isinstance(core_model, AnomalyDetectionRequest)
        assert core_model.file_path == "./test_data.csv"
        assert core_model.method == "z-score"
    
    def test_api_request_validation_errors(self):
        """Test API request validation errors."""
        # Test request with neither file_path nor data
        with pytest.raises(ValueError, match="Either file_path or data must be provided"):
            APIAnomalyDetectionRequest(
                method="z-score",
                query="Test analysis"
            )
        
        # Test request with both file_path and data
        with pytest.raises(ValueError, match="Provide either file_path or data, not both"):
            APIAnomalyDetectionRequest(
                file_path="./test_data.csv",
                data={
                    "timestamp": ["2025-01-01T00:00:00Z"],
                    "values": [10.0],
                    "column_name": "test"
                },
                method="z-score",
                query="Test analysis"
            )


class TestAPIErrorHandling:
    """Test API error handling."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_404_endpoint(self):
        """Test 404 error handling."""
        response = self.client.get("/nonexistent-endpoint")
        assert response.status_code == 404
    
    def test_method_not_allowed(self):
        """Test method not allowed error."""
        response = self.client.put("/api/v1/analyze")
        assert response.status_code == 405
    
    @patch('agents.supervisor.AnomalyDetectionSupervisor.analyze')
    def test_internal_server_error(self, mock_analyze):
        """Test internal server error handling."""
        # Mock an exception
        mock_analyze.side_effect = Exception("Test error")
        
        request_data = {
            "file_path": "./test_data.csv",
            "method": "z-score",
            "query": "Test analysis"
        }
        
        response = self.client.post("/api/v1/analyze", json=request_data)
        assert response.status_code == 500
        
        data = response.json()
        assert "error_code" in data
        assert "error_message" in data


class TestAPIMiddleware:
    """Test API middleware functionality."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_cors_headers(self):
        """Test CORS headers are present."""
        response = self.client.options("/api/v1/analyze")
        # CORS headers should be present
        # Note: TestClient might not handle CORS exactly like a real browser
        assert response.status_code in [200, 405]  # Either OK or Method Not Allowed
    
    def test_trusted_host_middleware(self):
        """Test trusted host middleware."""
        # This would test host validation
        # For now, just ensure the middleware doesn't break normal requests
        response = self.client.get("/health")
        assert response.status_code == 200


class TestAPIConfiguration:
    """Test API configuration and settings."""
    
    def test_settings_dependency(self):
        """Test settings dependency injection."""
        from api.dependencies import get_cached_settings
        
        settings = get_cached_settings()
        assert isinstance(settings, Settings)
        assert hasattr(settings, 'llm_model')
        assert hasattr(settings, 'api_host')
        assert hasattr(settings, 'api_port')
    
    def test_openapi_schema(self):
        """Test OpenAPI schema generation."""
        client = TestClient(app)
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "info" in schema
        assert "paths" in schema
        assert "components" in schema
        
        # Check that our main endpoints are documented
        assert "/api/v1/analyze" in schema["paths"]
        assert "/health" in schema["paths"]
    
    def test_docs_endpoint(self):
        """Test documentation endpoint."""
        client = TestClient(app)
        response = client.get("/docs")
        assert response.status_code == 200
        # Should return HTML content for Swagger UI
        assert "text/html" in response.headers.get("content-type", "")


@pytest.mark.asyncio
class TestAsyncAPIFunctionality:
    """Test async functionality in API."""
    
    async def test_async_analyze_endpoint(self):
        """Test async analyze endpoint functionality."""
        from api.endpoints import analyze_time_series
        from api.dependencies import get_cached_supervisor, get_cached_settings
        from unittest.mock import AsyncMock
        
        # Mock dependencies
        mock_supervisor = AsyncMock()
        mock_settings = MagicMock()
        mock_logger = MagicMock()
        
        # Mock the analysis result
        mock_result = MagicMock()
        mock_result.anomaly_result = MagicMock()
        mock_result.visualization = MagicMock()
        mock_result.insights = MagicMock()
        mock_result.processing_time = 1.5
        mock_result.metadata = {}
        
        mock_supervisor.analyze.return_value = mock_result
        
        # Create test request
        request = APIAnomalyDetectionRequest(
            file_path="./test_data.csv",
            method="z-score",
            query="Test async analysis"
        )
        
        # Call the endpoint function directly
        with patch('api.endpoints.SupervisorDep', return_value=mock_supervisor), \
             patch('api.endpoints.ValidatedSettingsDep', return_value=mock_settings), \
             patch('api.endpoints.LoggerDep', return_value=mock_logger):
            
            result = await analyze_time_series(request, mock_supervisor, mock_settings, mock_logger)
            
            assert isinstance(result, APIAnalysisResponse)
            assert result.success is True
            assert mock_supervisor.analyze.called