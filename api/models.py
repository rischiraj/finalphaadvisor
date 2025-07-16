"""
Pydantic models for FastAPI endpoints and request/response validation.
"""

from datetime import datetime
from typing import List, Optional, Union, Any, Dict
from pydantic import BaseModel, Field, model_validator

# Import core models
from core.models import (
    TimeSeriesData,
    AnomalyDetectionRequest,
    AnomalyResult,
    VisualizationResult,
    InsightResponse,
    AnalysisResponse,
    FileInfo,
    ProcessingStatus
)


class APITimeSeriesData(BaseModel):
    """
    API version of TimeSeriesData with simplified validation.
    """
    timestamp: List[str] = Field(..., description="ISO format timestamp strings")
    values: List[float] = Field(..., description="Numeric values")
    column_name: str = Field(..., description="Name of the value column")
    
    def to_core_model(self) -> TimeSeriesData:
        """Convert to core TimeSeriesData model."""
        timestamps = [datetime.fromisoformat(ts.replace('Z', '+00:00')) for ts in self.timestamp]
        return TimeSeriesData(
            timestamp=timestamps,
            values=self.values,
            column_name=self.column_name
        )


class APIAnomalyDetectionRequest(BaseModel):
    """
    API request model for anomaly detection.
    """
    file_path: Optional[str] = Field(
        None,
        description="Path to existing data file on server. Required if 'data' is not provided."
    )
    data: Optional[APITimeSeriesData] = Field(
        None,
        description="Direct time-series data. Required if 'file_path' is not provided."
    )
    method: str = Field('z-score', description="Detection method")
    threshold: Optional[float] = Field(None, description="Threshold parameter")
    query: str = Field(..., description="User query about the analysis")
    
    class Config:
        json_schema_extra = {
            "example": {
                "file_path": "./data/NVDA_1Y_FROM_PERPLEXITY.csv",
                "method": "rolling-iqr",
                "threshold": 1.5,
                "query": "Analyze NVIDIA stock price anomalies for mid-term trading opportunities. Looking for news-driven events and fundamental analysis for 3-6 month holding period.",
                "plot_type": "matplotlib",
                "include_visualization": True,
                "include_insights": True
            }
        }
    
    # Additional API-specific parameters
    plot_type: str = Field('matplotlib', description="Visualization type", pattern='^(matplotlib|plotly)$')
    include_visualization: bool = Field(True, description="Whether to generate visualization")
    include_insights: bool = Field(True, description="Whether to generate insights")
    
    @model_validator(mode='after')
    def validate_data_or_file(self):
        """Validate that either file_path or data is provided."""
        if not self.file_path and not self.data:
            raise ValueError('Either file_path or data must be provided')
        if self.file_path and self.data:
            raise ValueError('Provide either file_path or data, not both')
        return self

    
    def to_core_model(self) -> AnomalyDetectionRequest:
        """Convert to core AnomalyDetectionRequest model."""
        return AnomalyDetectionRequest(
            file_path=self.file_path,
            data=self.data.to_core_model() if self.data else None,
            method=self.method,
            threshold=self.threshold,
            query=self.query
        )


class APIAnalysisResponse(BaseModel):
    """
    API response model for analysis results.
    """
    success: bool = Field(..., description="Whether the analysis was successful")
    anomaly_result: AnomalyResult = Field(..., description="Anomaly detection results")
    visualization: Optional[VisualizationResult] = Field(None, description="Visualization results")
    insights: Optional[InsightResponse] = Field(None, description="Generated insights")
    processing_time: float = Field(..., description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    
    @classmethod
    def from_core_model(cls, response: AnalysisResponse, success: bool = True, error: str = None):
        """Create from core AnalysisResponse model."""
        return cls(
            success=success,
            anomaly_result=response.anomaly_result,
            visualization=response.visualization,
            insights=response.insights,
            processing_time=response.processing_time,
            metadata=response.metadata,
            error_message=error
        )


class FileUploadResponse(BaseModel):
    """
    Response model for file upload operations.
    """
    success: bool = Field(..., description="Whether upload was successful")
    file_info: Optional[FileInfo] = Field(None, description="Information about uploaded file")
    message: str = Field(..., description="Status message")
    file_id: Optional[str] = Field(None, description="Unique file identifier")


class MethodRecommendationRequest(BaseModel):
    """
    Request model for method recommendation.
    """
    file_path: Optional[str] = Field(None, description="Path to data file")
    data: Optional[APITimeSeriesData] = Field(None, description="Direct time-series data")
    description: Optional[str] = Field(None, description="Data description")


class MethodRecommendationResponse(BaseModel):
    """
    Response model for method recommendation.
    """
    recommended_method: str = Field(..., description="Recommended detection method")
    confidence: float = Field(..., ge=0, le=100, description="Confidence in recommendation")
    explanation: str = Field(..., description="Explanation for recommendation")
    all_methods: Dict[str, Dict[str, Any]] = Field(..., description="Information about all methods")


class QuickAnalysisRequest(BaseModel):
    """
    Request model for quick analysis without full workflow.
    """
    file_path: Optional[str] = Field(None, description="Path to data file")
    data: Optional[APITimeSeriesData] = Field(None, description="Direct time-series data")
    method: str = Field('z-score', description="Detection method")
    threshold: Optional[float] = Field(None, description="Threshold parameter")


class QuickAnalysisResponse(BaseModel):
    """
    Response model for quick analysis.
    """
    success: bool = Field(..., description="Whether analysis was successful")
    anomaly_count: int = Field(..., description="Number of anomalies found")
    anomaly_percentage: float = Field(..., description="Percentage of anomalies")
    method_used: str = Field(..., description="Method used for detection")
    summary: str = Field(..., description="Brief summary of results")
    processing_time: float = Field(..., description="Processing time in seconds")


class StatusResponse(BaseModel):
    """
    Response model for status endpoints.
    """
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.now, description="Current timestamp")
    capabilities: List[str] = Field(..., description="Available capabilities")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Configuration info")


class ErrorResponse(BaseModel):
    """
    Standard error response model.
    """
    success: bool = Field(False, description="Always false for errors")
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")


class BatchAnalysisRequest(BaseModel):
    """
    Request model for batch analysis of multiple datasets.
    """
    requests: List[APIAnomalyDetectionRequest] = Field(
        ..., 
        min_items=1, 
        max_items=10,
        description="List of analysis requests"
    )
    parallel: bool = Field(True, description="Whether to process requests in parallel")


class BatchAnalysisResponse(BaseModel):
    """
    Response model for batch analysis.
    """
    success: bool = Field(..., description="Whether batch processing was successful")
    results: List[APIAnalysisResponse] = Field(..., description="Individual analysis results")
    total_processing_time: float = Field(..., description="Total processing time")
    successful_analyses: int = Field(..., description="Number of successful analyses")
    failed_analyses: int = Field(..., description="Number of failed analyses")


class ValidationRequest(BaseModel):
    """
    Request model for data validation.
    """
    file_path: Optional[str] = Field(None, description="Path to data file")
    data: Optional[APITimeSeriesData] = Field(None, description="Direct time-series data")


class ValidationResponse(BaseModel):
    """
    Response model for data validation.
    """
    valid: bool = Field(..., description="Whether data is valid")
    issues: List[str] = Field(default_factory=list, description="Validation issues found")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions for fixes")
    data_info: Optional[Dict[str, Any]] = Field(None, description="Data characteristics")


# Unused response models commented out - these were never used in the API
# class StreamingResponse(BaseModel):
#     """Response model for streaming analysis updates."""
#     step: str = Field(..., description="Current processing step")
#     status: str = Field(..., description="Step status")
#     progress: float = Field(..., ge=0, le=100, description="Progress percentage")
#     message: str = Field(..., description="Status message")
#     timestamp: datetime = Field(default_factory=datetime.now, description="Update timestamp")
#     data: Optional[Dict[str, Any]] = Field(None, description="Step-specific data")

# # Response models for different HTTP status codes - unused
# class SuccessResponse(APIAnalysisResponse):
#     """Success response (200)."""
#     pass

# class CreatedResponse(BaseModel):
#     """Created response (201) for resource creation."""
#     success: bool = Field(True, description="Resource created successfully")
#     resource_id: str = Field(..., description="ID of created resource")
#     message: str = Field(..., description="Success message")

# class AcceptedResponse(BaseModel):
#     """Accepted response (202) for async operations."""
#     success: bool = Field(True, description="Request accepted for processing")
#     task_id: str = Field(..., description="Task ID for tracking")
#     status_url: str = Field(..., description="URL to check task status")
#     estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")

# class BadRequestResponse(ErrorResponse):
#     """Bad request response (400)."""
#     error_code: str = Field("BAD_REQUEST", description="Error code")

# class UnauthorizedResponse(ErrorResponse):
#     """Unauthorized response (401)."""
#     error_code: str = Field("UNAUTHORIZED", description="Error code")

# class NotFoundResponse(ErrorResponse):
#     """Not found response (404)."""
#     error_code: str = Field("NOT_FOUND", description="Error code")

# class UnprocessableEntityResponse(ErrorResponse):
#     """Unprocessable entity response (422)."""
#     error_code: str = Field("VALIDATION_ERROR", description="Error code")

# class InternalServerErrorResponse(ErrorResponse):
#     """Internal server error response (500)."""
#     error_code: str = Field("INTERNAL_ERROR", description="Error code")