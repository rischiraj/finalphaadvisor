"""
Core data structures and Pydantic models for time-series anomaly detection.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, validator, model_validator


# Unused StructuredAnomalyPoint class - commented out
# class StructuredAnomalyPoint(BaseModel):
#     """
#     Structured anomaly point with enhanced metadata.
#     
#     Args:
#         timestamp: ISO format timestamp string
#         value: Anomaly value
#         severity: Severity level (high/medium/low)
#         deviation_score: Numerical deviation score
#         trend: Trend direction (spike/dip/increasing/decreasing/stable)
#     """
#     timestamp: str = Field(..., description="ISO format timestamp")
#     value: float = Field(..., description="Anomaly value")
#     severity: Literal['high', 'medium', 'low'] = Field(..., description="Severity level")
#     deviation_score: float = Field(..., ge=0, description="Deviation score")
#     trend: Literal['spike', 'dip', 'increasing', 'decreasing', 'stable', 'unknown'] = Field(
#         ..., description="Trend direction"
#     )


class TimeSeriesData(BaseModel):
    """
    Time series data model with validation.
    
    Args:
        timestamp: List of timestamp values
        values: List of numeric values
        column_name: Name of the value column
    """
    timestamp: List[datetime] = Field(..., description="Timestamp values")
    values: List[float] = Field(..., description="Numeric values")
    column_name: str = Field(..., description="Name of the value column")
    
    @validator('values', each_item=True)
    def validate_values_are_numeric(cls, v):
        """Validate that all values are numeric and not NaN."""
        if not isinstance(v, (int, float)) or str(v).lower() in ['nan', 'inf', '-inf']:
            raise ValueError('All values must be finite numeric values')
        return float(v)
    
    @validator('timestamp')
    def validate_timestamp_length(cls, v, values):
        """Validate that timestamp and values have the same length."""
        if 'values' in values and len(v) != len(values['values']):
            raise ValueError('timestamp and values must have the same length')
        return v
    
    @validator('timestamp')
    def validate_timestamps_sorted(cls, v):
        """Validate that timestamps are in ascending order."""
        if len(v) > 1:
            for i in range(1, len(v)):
                if v[i] <= v[i-1]:
                    raise ValueError('Timestamps must be in ascending order')
        return v


class AnomalyDetectionRequest(BaseModel):
    """
    Request model for anomaly detection.
    
    Args:
        file_path: Path to time-series data file
        data: Direct time-series data
        method: Detection method to use
        threshold: Threshold for z-score/IQR methods
        query: User query about the analysis
    """
    file_path: Optional[str] = Field(None, description="Path to data file")
    data: Optional[TimeSeriesData] = Field(None, description="Direct time-series data")
    method: Literal['z-score', 'iqr', 'rolling-iqr', 'dbscan'] = Field(
        'z-score', 
        description="Anomaly detection method"
    )
    threshold: Optional[float] = Field(
        None, 
        gt=0, 
        description="Threshold for z-score/IQR methods"
    )
    query: str = Field(..., description="User query about the analysis")
    
    @model_validator(mode='after')
    def validate_data_or_file(self):
        """Validate that either file_path or data is provided."""
        file_path = self.file_path
        data = self.data
        
        if not file_path and not data:
            raise ValueError('Either file_path or data must be provided')
        
        if file_path and data:
            raise ValueError('Provide either file_path or data, not both')
        
        return self


class AnomalyResult(BaseModel):
    """
    Anomaly detection result model.
    
    Args:
        anomaly_indices: Indices of anomalous points
        method_used: Method used for detection
        threshold_used: Threshold used
        total_points: Total data points
        anomaly_count: Number of anomalies found
        anomaly_percentage: Percentage of anomalies
        anomaly_values: Actual values at anomaly indices
        anomaly_timestamps: Timestamps at anomaly indices
    """
    anomaly_indices: List[int] = Field(..., description="Indices of anomalous points")
    method_used: str = Field(..., description="Method used for detection")
    threshold_used: float = Field(..., description="Threshold used")
    total_points: int = Field(..., ge=0, description="Total data points")
    anomaly_count: int = Field(..., ge=0, description="Number of anomalies found")
    anomaly_percentage: float = Field(..., ge=0, le=100, description="Percentage of anomalies")
    anomaly_values: List[float] = Field(..., description="Values at anomaly indices")
    anomaly_timestamps: List[datetime] = Field(..., description="Timestamps at anomaly indices")
    # structured_anomaly_points: Optional[List[StructuredAnomalyPoint]] = Field(
    #     None, description="Structured anomaly points with severity, deviation scores, and trends"
    # )
    
    @validator('anomaly_count')
    def validate_anomaly_count_matches_indices(cls, v, values):
        """Validate anomaly count matches indices length."""
        if 'anomaly_indices' in values and v != len(values['anomaly_indices']):
            raise ValueError('anomaly_count must match length of anomaly_indices')
        return v
    
    @validator('anomaly_percentage')
    def validate_percentage_calculation(cls, v, values):
        """Validate percentage calculation."""
        if 'anomaly_count' in values and 'total_points' in values:
            expected = (values['anomaly_count'] / values['total_points']) * 100
            if abs(v - expected) > 0.01:  # Allow small floating point errors
                raise ValueError('anomaly_percentage calculation is incorrect')
        return v


class VisualizationResult(BaseModel):
    """
    Visualization result model.
    
    Args:
        plot_path: Path to saved plot file
        plot_base64: Base64 encoded plot image
        plot_description: Description of the plot
        plot_type: Type of plot generated
    """
    plot_path: str = Field(..., description="Path to saved plot")
    plot_base64: str = Field(..., description="Base64 encoded plot")
    plot_description: str = Field(..., description="Description of the plot")
    plot_type: Literal['matplotlib', 'plotly'] = Field(
        'matplotlib', 
        description="Type of plot generated"
    )


class InsightResponse(BaseModel):
    """
    LLM-generated insight response.
    
    Args:
        summary: Summary of the analysis
        anomaly_explanations: Explanations for anomalies
        recommendations: Actionable recommendations
        root_causes: Potential root causes
        confidence_score: Confidence in the analysis (0-100)
    """
    summary: str = Field(..., description="Summary of the analysis")
    anomaly_explanations: List[str] = Field(..., description="Explanations for anomalies")
    recommendations: List[str] = Field(..., description="Actionable recommendations")
    root_causes: List[str] = Field(..., description="Potential root causes")
    confidence_score: float = Field(
        ..., 
        ge=0, 
        le=100, 
        description="Confidence in the analysis (0-100)"
    )
    
    @validator('anomaly_explanations', 'recommendations', 'root_causes')
    def validate_non_empty_lists(cls, v):
        """Validate that lists are not empty."""
        if not v:
            raise ValueError('Lists cannot be empty')
        return v


class AnalysisResponse(BaseModel):
    """
    Complete analysis response combining all results.
    
    Args:
        anomaly_result: Anomaly detection results
        visualization: Visualization results
        insights: LLM-generated insights
        processing_time: Processing time in seconds
        metadata: Additional metadata about the analysis
    """
    anomaly_result: AnomalyResult
    visualization: VisualizationResult
    insights: InsightResponse
    processing_time: float = Field(..., ge=0, description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional metadata"
    )


class AgentState(BaseModel):
    """
    State model for LangGraph multi-agent workflow.
    
    Args:
        request: Original analysis request
        data: Processed time-series data
        anomaly_result: Results from anomaly detection
        visualization: Visualization results
        insights: Generated insights
        errors: Any errors that occurred
        status: Current workflow status
    """
    request: AnomalyDetectionRequest
    data: Optional[TimeSeriesData] = None
    anomaly_result: Optional[AnomalyResult] = None
    visualization: Optional[VisualizationResult] = None
    insights: Optional[InsightResponse] = None
    errors: List[str] = Field(default_factory=list)
    status: Literal['pending', 'processing', 'completed', 'failed'] = Field(
        'pending', 
        description="Current workflow status"
    )


class FileInfo(BaseModel):
    """
    Information about uploaded or processed files.
    
    Args:
        filename: Name of the file
        file_size: Size of file in bytes
        file_type: Type of file (csv, xlsx, etc.)
        columns: Available columns in the file
        row_count: Number of rows in the file
    """
    filename: str = Field(..., description="Name of the file")
    file_size: int = Field(..., ge=0, description="Size of file in bytes")
    file_type: str = Field(..., description="Type of file")
    columns: List[str] = Field(..., description="Available columns")
    row_count: int = Field(..., ge=0, description="Number of rows")


class ProcessingStatus(BaseModel):
    """
    Status model for long-running operations.
    
    Args:
        task_id: Unique identifier for the task
        status: Current status
        progress: Progress percentage (0-100)
        message: Status message
        started_at: When the task started
        completed_at: When the task completed (if applicable)
    """
    task_id: str = Field(..., description="Unique task identifier")
    status: Literal['pending', 'running', 'completed', 'failed'] = Field(
        'pending',
        description="Current status"
    )
    progress: float = Field(0, ge=0, le=100, description="Progress percentage")
    message: str = Field("", description="Status message")
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None