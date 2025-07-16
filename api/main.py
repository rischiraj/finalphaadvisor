"""
FastAPI application for time-series anomaly detection system.
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi

from core.config import get_settings
from core.exceptions import (
    AnomalyDetectionError,
    FileProcessingError,
    DataValidationError,
    LLMError,
    AgentError,
    ConfigurationError
)
from api.endpoints import router
from api.models import ErrorResponse


# Configure logging to match CLI logging setup
def setup_api_logging():
    """Setup API logging to match the CLI configuration."""
    settings = get_settings()
    
    # Configure logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler(sys.stdout)]
    
    # Add file handler if configured
    if settings.log_file:
        settings.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(settings.log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers,
        force=True  # Prevent duplicate handlers
    )

# Initialize API logging
setup_api_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    """
    # Startup
    logger.info("Starting Time-Series Anomaly Detection API")
    
    # Validate configuration
    try:
        settings = get_settings()
        
        # Create required directories
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        settings.plots_dir.mkdir(parents=True, exist_ok=True)
        
        if settings.log_file:
            settings.log_file.parent.mkdir(parents=True, exist_ok=True)
            # File handler already configured in setup_api_logging()
        
        logger.info("Configuration validated and directories created")
        
    except ConfigurationError as e:
        logger.error(f"Configuration error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Time-Series Anomaly Detection API")


# Create FastAPI application
app = FastAPI(
    title="Time-Series Anomaly Detection API",
    description="""
    ## Multi-Agent Time-Series Anomaly Detection System

    This API provides comprehensive anomaly detection capabilities for time-series data using:
    
    - **Multiple Detection Methods**: Z-score, IQR, and DBSCAN algorithms
    - **Multi-Agent Architecture**: LangChain/LangGraph-based agent coordination
    - **AI-Powered Insights**: Google Gemini LLM for generating explanations and recommendations
    - **Interactive Visualizations**: Matplotlib and Plotly chart generation
    - **File Support**: CSV and Excel file processing

    ### Supported Methods
    
    - **Z-Score**: Statistical method using standard deviations (best for normal distributions)
    - **IQR**: Interquartile range method (best for skewed data)
    - **DBSCAN**: Density-based clustering (best for complex patterns)
    
    ### Authentication
    
    Currently, the API is open for development. In production, implement proper authentication.
    """,
    version="1.0.0",
    contact={
        "name": "Time-Series Anomaly Detection API",
        "email": "support@example.com"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)


# Exception handlers
@app.exception_handler(FileProcessingError)
async def file_processing_exception_handler(request: Request, exc: FileProcessingError):
    """Handle file processing errors."""
    logger.error(f"File processing error: {exc.message}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error_code": exc.error_code,
            "error_message": exc.message,
            "details": {"filename": exc.filename} if exc.filename else None
        }
    )


@app.exception_handler(DataValidationError)
async def data_validation_exception_handler(request: Request, exc: DataValidationError):
    """Handle data validation errors."""
    logger.error(f"Data validation error: {exc.message}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error_code": exc.error_code,
            "error_message": exc.message,
            "details": {"field": exc.field} if exc.field else None
        }
    )


@app.exception_handler(LLMError)
async def llm_exception_handler(request: Request, exc: LLMError):
    """Handle LLM errors."""
    logger.error(f"LLM error: {exc.message}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error_code": exc.error_code,
            "error_message": f"AI processing error: {exc.message}",
            "details": {"model": exc.model} if exc.model else None
        }
    )


@app.exception_handler(AgentError)
async def agent_exception_handler(request: Request, exc: AgentError):
    """Handle agent workflow errors."""
    logger.error(f"Agent error: {exc.message}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error_code": exc.error_code,
            "error_message": f"Agent processing error: {exc.message}",
            "details": {"agent": exc.agent_name} if exc.agent_name else None
        }
    )


@app.exception_handler(ConfigurationError)
async def configuration_exception_handler(request: Request, exc: ConfigurationError):
    """Handle configuration errors."""
    logger.error(f"Configuration error: {exc.message}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error_code": exc.error_code,
            "error_message": f"Configuration error: {exc.message}",
            "details": {"config_key": exc.config_key} if exc.config_key else None
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error_code": "INTERNAL_ERROR",
            "error_message": "An unexpected error occurred",
            "details": {"type": type(exc).__name__}
        }
    )


# Include routers
app.include_router(
    router,
    prefix="/api/v1",
    tags=["Anomaly Detection"]
)

# Add conversation router (new multi-turn functionality)
try:
    from api.conversation_endpoints import conversation_router
    app.include_router(
        conversation_router,
        prefix="/api/v1",
        tags=["Multi-Turn Conversation"]
    )
    logger.info("Multi-turn conversation endpoints enabled")
except ImportError as e:
    logger.warning(f"Conversation endpoints not available: {e}")
except Exception as e:
    logger.error(f"Failed to load conversation endpoints: {e}")


# Custom OpenAPI schema
def custom_openapi():
    """Generate custom OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add custom schema extensions
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    
    # Add examples to schemas (if they exist)
    if "APIAnomalyDetectionRequest" in openapi_schema.get("components", {}).get("schemas", {}):
        openapi_schema["components"]["schemas"]["APIAnomalyDetectionRequest"]["example"] = {
            "method": "z-score",
            "threshold": 3.0,
            "query": "Find anomalies in my sales data",
            "plot_type": "matplotlib",
            "include_visualization": True,
            "include_insights": True
        }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Health check endpoint
@app.get(
    "/health",
    summary="Health Check",
    description="Check if the API is healthy and operational",
    tags=["Health"]
)
async def health_check():
    """Health check endpoint."""
    try:
        settings = get_settings()
        
        return {
            "status": "healthy",
            "timestamp": "2025-01-07T10:00:00Z",
            "version": app.version,
            "environment": "development" if settings.debug else "production"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )


# Root endpoint
@app.get(
    "/",
    summary="API Root",
    description="Get basic API information",
    tags=["Root"]
)
async def root():
    """Root endpoint with basic API information."""
    return {
        "name": app.title,
        "version": app.version,
        "description": "Multi-Agent Time-Series Anomaly Detection System",
        "docs_url": "/docs",
        "health_url": "/health",
        "api_base": "/api/v1"
    }


# Additional utility endpoints
@app.get(
    "/api/v1/methods",
    summary="Get Available Methods",
    description="Get information about available anomaly detection methods",
    tags=["Anomaly Detection"]
)
async def get_methods():
    """Get information about available detection methods."""
    from agents.tools.anomaly_detector import AnomalyDetectionTool
    
    detector = AnomalyDetectionTool()
    methods = {}
    
    for method in ['z-score', 'iqr', 'rolling-iqr', 'dbscan']:
        methods[method] = detector.get_method_info(method)
    
    return {
        "available_methods": methods,
        "default_method": "z-score",
        "recommendation_available": True
    }


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.reload,
        workers=settings.api_workers if not settings.reload else 1,
        log_level=settings.log_level.lower(),
        access_log=True
    )