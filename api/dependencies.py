"""
FastAPI dependencies for dependency injection.
"""

import logging
from typing import Annotated
from functools import lru_cache

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from core.config import Settings, get_settings
from core.exceptions import ConfigurationError
from agents.supervisor import AnomalyDetectionSupervisor
from agents.anomaly_agent import AnomalyAgent
from agents.enhanced_suggestion_agent import EnhancedSuggestionAgent


logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)


@lru_cache()
def get_cached_settings() -> Settings:
    """
    Get cached application settings.
    
    Returns:
        Settings: Application configuration
    """
    return get_settings()


def get_supervisor(
    settings: Annotated[Settings, Depends(get_cached_settings)] = None
) -> AnomalyDetectionSupervisor:
    """
    Get or create supervisor instance.
    
    Args:
        settings: Application settings
    
    Returns:
        AnomalyDetectionSupervisor: Supervisor instance
        
    Raises:
        HTTPException: If supervisor creation fails
    """
    try:
        # Create new supervisor instance for each request
        # In production, you might want to use a singleton pattern
        supervisor = AnomalyDetectionSupervisor(enable_llm=settings.enable_llm)
        return supervisor
    except Exception as e:
        logger.error(f"Failed to create supervisor: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize analysis supervisor: {str(e)}"
        )


def get_anomaly_agent() -> AnomalyAgent:
    """
    Get or create anomaly agent instance.
    
    Returns:
        AnomalyAgent: Anomaly agent instance
        
    Raises:
        HTTPException: If agent creation fails
    """
    try:
        agent = AnomalyAgent()
        return agent
    except Exception as e:
        logger.error(f"Failed to create anomaly agent: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize anomaly agent: {str(e)}"
        )


def get_suggestion_agent() -> EnhancedSuggestionAgent:
    """
    Get or create enhanced suggestion agent instance.
    
    Returns:
        EnhancedSuggestionAgent: Enhanced suggestion agent instance
        
    Raises:
        HTTPException: If agent creation fails
    """
    try:
        # Enhanced suggestion agent requires LLM for initialization
        from core.config import get_settings
        settings = get_settings()
        
        if settings.enable_llm:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                google_api_key=settings.google_ai_api_key
            )
            agent = EnhancedSuggestionAgent(llm)
        else:
            # Fallback to basic mode without LLM
            agent = EnhancedSuggestionAgent(None)
        
        return agent
    except Exception as e:
        logger.error(f"Failed to create enhanced suggestion agent: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize enhanced suggestion agent: {str(e)}"
        )


# Unused authentication and rate limiting functions - commented out
# def verify_api_key(
#     credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)] = None,
#     settings: Annotated[Settings, Depends(get_cached_settings)] = None
# ) -> bool:
#     """
#     Verify API key authentication (optional, for future use).
#     
#     Args:
#         credentials: HTTP authorization credentials
#         settings: Application settings
#         
#     Returns:
#         bool: Whether authentication is valid
#         
#     Raises:
#         HTTPException: If authentication fails
#     """
#     # For now, authentication is disabled
#     # In production, you would implement proper API key validation
#     return True


# def validate_request_limits(
#     settings: Annotated[Settings, Depends(get_cached_settings)] = None
# ) -> bool:
#     """
#     Validate request rate limits and quotas.
#     
#     Args:
#         settings: Application settings
#         
#     Returns:
#         bool: Whether request is within limits
#         
#     Raises:
#         HTTPException: If limits are exceeded
#     """
#     # Placeholder for rate limiting logic
#     # In production, you would implement rate limiting here
#     return True


def get_logger(name: str = "api") -> logging.Logger:
    """
    Get configured logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        logging.Logger: Configured logger
    """
    return logging.getLogger(name)


def validate_configuration(
    settings: Annotated[Settings, Depends(get_cached_settings)] = None
) -> Settings:
    """
    Validate application configuration.
    
    Args:
        settings: Application settings
        
    Returns:
        Settings: Validated settings
        
    Raises:
        HTTPException: If configuration is invalid
    """
    try:
        # Validate critical configuration
        if not settings.google_ai_api_key or settings.google_ai_api_key == "your_google_ai_api_key_here":
            raise ConfigurationError("Google AI API key not configured")
        
        return settings
        
    except ConfigurationError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration validation failed: {str(e)}"
        )


# Dependency aliases for common combinations
SupervisorDep = Annotated[AnomalyDetectionSupervisor, Depends(get_supervisor)]
AnomalyAgentDep = Annotated[AnomalyAgent, Depends(get_anomaly_agent)]
SuggestionAgentDep = Annotated[EnhancedSuggestionAgent, Depends(get_suggestion_agent)]
SettingsDep = Annotated[Settings, Depends(get_cached_settings)]
LoggerDep = Annotated[logging.Logger, Depends(get_logger)]
ValidatedSettingsDep = Annotated[Settings, Depends(validate_configuration)]