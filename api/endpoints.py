"""
FastAPI endpoints for anomaly detection API.
"""

import logging
import time
import asyncio
from typing import Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from fastapi.responses import FileResponse

from api.models import (
    APIAnomalyDetectionRequest,
    APIAnalysisResponse,
    FileUploadResponse,
    MethodRecommendationRequest,
    MethodRecommendationResponse,
    QuickAnalysisRequest,
    QuickAnalysisResponse,
    StatusResponse,
    ErrorResponse,
    ValidationRequest,
    ValidationResponse,
    BatchAnalysisRequest,
    BatchAnalysisResponse
)
from api.dependencies import (
    SupervisorDep,
    AnomalyAgentDep,
    SuggestionAgentDep,
    SettingsDep,
    LoggerDep,
    ValidatedSettingsDep
)
from core.exceptions import (
    AnomalyDetectionError,
    FileProcessingError,
    DataValidationError,
    LLMError,
    AgentError
)
from core.models import TimeSeriesData
from agents.tools.file_reader import FileReaderTool
from agents.tools.anomaly_detector import AnomalyDetectionTool


# Create router
router = APIRouter(tags=["Anomaly Detection"])
logger = logging.getLogger(__name__)


@router.post(
    "/analyze",
    response_model=APIAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze time-series data for anomalies",
    description="""
        Perform comprehensive anomaly detection analysis using multi-agent system.

        **Data Input (Choose ONE):**
        - **file_path**: Path to existing CSV/Excel file on server
        - **data**: Time-series data sent directly in request body
        - **uploaded_file**: Upload new file (use multipart/form-data)

        **Required**: You MUST provide exactly one data source.
        """
    ,
        responses={
        200: {"description": "Analysis completed successfully", "model": APIAnalysisResponse},
        400: {"description": "Invalid request", "model": ErrorResponse},
        422: {"description": "Validation error", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse}
    }
)
async def analyze_time_series(
    request: APIAnomalyDetectionRequest,
    supervisor: SupervisorDep,
    settings: ValidatedSettingsDep,
    logger: LoggerDep
) -> APIAnalysisResponse:
    """
    Analyze time-series data for anomalies using the multi-agent system.
    
    Args:
        request: Analysis request containing method, threshold, query, and optional file path
        supervisor: Anomaly detection supervisor
        settings: Application settings
        logger: Logger instance
        
    Returns:
        APIAnalysisResponse: Complete analysis results
        
    Raises:
        HTTPException: If analysis fails
    """
    try:
        start_time = time.time()
        
        logger.info(f"Starting analysis request: method={request.method}")
        
        # Use the request directly (Pydantic has already validated it)
        api_request = request
        
        # Validate that we have either file_path or data
        if not api_request.file_path and not api_request.data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either 'file_path' or 'data' must be provided, or upload a file"
            )
        
        # Convert to core model
        core_request = api_request.to_core_model()
        
        # Run analysis
        result = await supervisor.analyze(core_request)
        
        # Convert to API response
        api_response = APIAnalysisResponse.from_core_model(result, success=True)
        
        logger.info(f"Analysis completed in {time.time() - start_time:.2f} seconds")
        return api_response
        
    except (FileProcessingError, DataValidationError) as e:
        logger.error(f"Data processing error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Data processing error: {str(e)}"
        )
    except (LLMError, AgentError) as e:
        logger.error(f"Agent processing error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis processing error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error during analysis: {str(e)}"
        )


@router.post(
    "/quick-analyze",
    response_model=QuickAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Quick anomaly detection without full insights",
    description="Perform fast anomaly detection without visualization and insights generation"
)
async def quick_analyze(
    request: QuickAnalysisRequest,
    anomaly_agent: AnomalyAgentDep,
    logger: LoggerDep
) -> QuickAnalysisResponse:
    """
    Perform quick anomaly detection without full multi-agent workflow.
    
    Args:
        request: Quick analysis request
        anomaly_agent: Anomaly detection agent
        logger: Logger instance
        
    Returns:
        QuickAnalysisResponse: Basic analysis results
    """
    try:
        start_time = time.time()
        
        # Convert data if provided
        data = request.data.to_core_model() if request.data else None
        
        # Run standalone analysis
        result = await anomaly_agent.run_standalone(
            file_path=request.file_path,
            data=data,
            method=request.method,
            threshold=request.threshold,
            query="Quick anomaly detection"
        )
        
        # Extract basic results
        anomaly_result = result.get("anomaly_result")
        if not anomaly_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get anomaly detection results"
            )
        
        processing_time = time.time() - start_time
        
        return QuickAnalysisResponse(
            success=True,
            anomaly_count=anomaly_result.anomaly_count,
            anomaly_percentage=anomaly_result.anomaly_percentage,
            method_used=anomaly_result.method_used,
            summary=f"Found {anomaly_result.anomaly_count} anomalies using {anomaly_result.method_used} method",
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Quick analysis error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quick analysis failed: {str(e)}"
        )


@router.post(
    "/recommend-method",
    response_model=MethodRecommendationResponse,
    summary="Get method recommendation for data",
    description="Analyze data characteristics and recommend the best anomaly detection method"
)
async def recommend_method(
    request: MethodRecommendationRequest,
    logger: LoggerDep
) -> MethodRecommendationResponse:
    """
    Recommend the best anomaly detection method for given data.
    
    Args:
        request: Method recommendation request
        logger: Logger instance
        
    Returns:
        MethodRecommendationResponse: Method recommendation
    """
    try:
        detector = AnomalyDetectionTool()
        
        # Load or use provided data
        if request.file_path:
            file_reader = FileReaderTool()
            data = file_reader.read_file(request.file_path)
        elif request.data:
            data = request.data.to_core_model()
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either file_path or data must be provided"
            )
        
        # Get recommendation
        recommended = detector.recommend_method(data)
        
        # Get information about all methods
        all_methods = {
            method: detector.get_method_info(method)
            for method in ['z-score', 'iqr', 'rolling-iqr', 'dbscan']
        }
        
        # Calculate confidence based on data characteristics
        import pandas as pd
        series = pd.Series(data.values)
        skewness = abs(series.skew())
        
        if recommended == 'z-score' and skewness < 1:
            confidence = 90
        elif recommended == 'iqr' and skewness > 2:
            confidence = 85
        elif recommended == 'dbscan':
            confidence = 75
        else:
            confidence = 70
        
        method_info = detector.get_method_info(recommended)
        explanation = f"Recommended based on data characteristics: {method_info.get('best_for', 'general analysis')}"
        
        return MethodRecommendationResponse(
            recommended_method=recommended,
            confidence=confidence,
            explanation=explanation,
            all_methods=all_methods
        )
        
    except Exception as e:
        logger.error(f"Method recommendation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Method recommendation failed: {str(e)}"
        )


@router.post(
    "/validate-data",
    response_model=ValidationResponse,
    summary="Validate time-series data",
    description="Validate data format and quality for anomaly detection"
)
async def validate_data(
    request: ValidationRequest,
    logger: LoggerDep
) -> ValidationResponse:
    """
    Validate time-series data for anomaly detection.
    
    Args:
        request: Validation request
        logger: Logger instance
        
    Returns:
        ValidationResponse: Validation results
    """
    try:
        issues = []
        suggestions = []
        
        # Load or use provided data
        if request.file_path:
            try:
                file_reader = FileReaderTool()
                data = file_reader.read_file(request.file_path)
            except Exception as e:
                return ValidationResponse(
                    valid=False,
                    issues=[f"File reading error: {str(e)}"],
                    suggestions=["Check file format and content"]
                )
        elif request.data:
            try:
                data = request.data.to_core_model()
            except Exception as e:
                return ValidationResponse(
                    valid=False,
                    issues=[f"Data validation error: {str(e)}"],
                    suggestions=["Check data format and values"]
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either file_path or data must be provided"
            )
        
        # Perform validation checks
        import pandas as pd
        series = pd.Series(data.values)
        
        # Check data length
        if len(data.values) < 10:
            issues.append("Insufficient data points (minimum 10 recommended)")
            suggestions.append("Collect more data points for better anomaly detection")
        
        # Check for missing values
        if series.isna().any():
            issues.append("Data contains missing values")
            suggestions.append("Remove or interpolate missing values")
        
        # Check for constant values
        if series.std() == 0:
            issues.append("Data has zero variance (all values are the same)")
            suggestions.append("Anomaly detection requires variable data")
        
        # Check for extreme outliers that might affect detection
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        extreme_outliers = series[(series < q1 - 3 * iqr) | (series > q3 + 3 * iqr)]
        
        if len(extreme_outliers) > len(series) * 0.1:
            issues.append("Data contains many extreme outliers")
            suggestions.append("Consider data cleaning or using robust detection methods")
        
        # Data characteristics
        data_info = {
            "total_points": len(data.values),
            "time_span_hours": (data.timestamp[-1] - data.timestamp[0]).total_seconds() / 3600,
            "value_range": [float(series.min()), float(series.max())],
            "mean": float(series.mean()),
            "std": float(series.std()),
            "skewness": float(series.skew()),
            "missing_values": int(series.isna().sum())
        }
        
        valid = len(issues) == 0
        
        return ValidationResponse(
            valid=valid,
            issues=issues,
            suggestions=suggestions,
            data_info=data_info
        )
        
    except Exception as e:
        logger.error(f"Data validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data validation failed: {str(e)}"
        )


@router.post(
    "/upload-file",
    response_model=FileUploadResponse,
    summary="Upload data file",
    description="Upload CSV or Excel file for analysis"
)
async def upload_file(
    file: UploadFile = File(...),
    settings: SettingsDep = None
) -> FileUploadResponse:
    """
    Upload a data file for analysis.
    
    Args:
        file: Uploaded file
        settings: Application settings
        
    Returns:
        FileUploadResponse: Upload results
    """
    try:
        # Validate file type
        allowed_extensions = {'.csv', '.xlsx', '.xls'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type: {file_extension}. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Save file
        file_path = await _handle_file_upload(file, settings)
        
        # Get file info
        file_reader = FileReaderTool()
        file_info = file_reader.get_file_info(file_path)
        
        return FileUploadResponse(
            success=True,
            file_info=file_info,
            message=f"File uploaded successfully to {file_path}",
            file_id=file_path.stem
        )
        
    except Exception as e:
        logger.error(f"File upload error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File upload failed: {str(e)}"
        )


@router.get(
    "/status",
    response_model=StatusResponse,
    summary="Get API status",
    description="Get current API status and configuration"
)
async def get_status(
    settings: SettingsDep = None
) -> StatusResponse:
    """
    Get API status and configuration information.
    
    Args:
        settings: Application settings
        
    Returns:
        StatusResponse: Status information
    """
    return StatusResponse(
        status="healthy",
        version="1.0.0",
        capabilities=[
            "anomaly_detection",
            "visualization",
            "insights_generation",
            "multi_agent_workflow",
            "file_upload",
            "method_recommendation"
        ],
        configuration={
            "supported_methods": ["z-score", "iqr", "rolling-iqr", "dbscan"],
            "supported_formats": ["csv", "xlsx", "xls"],
            "visualization_types": ["matplotlib", "plotly"],
            "llm_model": settings.llm_model
        }
    )


@router.get(
    "/download-plot/{plot_filename}",
    response_class=FileResponse,
    summary="Download generated plot",
    description="Download a generated anomaly detection plot"
)
async def download_plot(
    plot_filename: str,
    settings: SettingsDep = None
) -> FileResponse:
    """
    Download a generated plot file.
    
    Args:
        plot_filename: Name of the plot file
        settings: Application settings
        
    Returns:
        FileResponse: Plot file
    """
    try:
        plot_path = settings.plots_dir / plot_filename
        
        if not plot_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Plot file not found"
            )
        
        return FileResponse(
            path=str(plot_path),
            filename=plot_filename,
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download plot: {str(e)}"
        )


async def _handle_file_upload(file: UploadFile, settings) -> Path:
    """
    Handle file upload and save to data directory.
    
    Args:
        file: Uploaded file
        settings: Application settings
        
    Returns:
        Path: Path to saved file
    """
    import uuid
    
    # Generate unique filename
    file_id = str(uuid.uuid4())[:8]
    file_extension = Path(file.filename).suffix
    filename = f"{file_id}_{file.filename}"
    file_path = settings.data_dir / filename
    
    # Save file
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    return file_path