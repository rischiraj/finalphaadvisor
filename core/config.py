"""
Configuration management using pydantic-settings.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    """
    
    # Google AI API Configuration
    google_ai_api_key: str = Field(..., description="Google AI API key for Gemini")
    
    # LLM Configuration  
    llm_model: str = Field("gemini-2.0-flash", description="LLM model to use")
    llm_temperature: float = Field(0.1, ge=0.0, le=2.0, description="LLM temperature")
    enable_llm: bool = Field(True, description="Enable LLM-powered insights and analysis")
    
    # API Configuration
    api_host: str = Field("0.0.0.0", description="FastAPI host")
    api_port: int = Field(8000, ge=1, le=65535, description="FastAPI port")
    api_workers: int = Field(1, ge=1, description="Number of API workers")
    
    # Anomaly Detection Thresholds
    default_z_score_threshold: float = Field(3.0, gt=0, description="Default Z-score threshold")
    default_iqr_multiplier: float = Field(1.5, gt=0, description="Default IQR multiplier")
    default_rolling_iqr_multiplier: float = Field(1.5, gt=0, description="Default Rolling IQR multiplier")
    default_rolling_iqr_window: int = Field(20, ge=5, description="Default Rolling IQR window size")
    default_dbscan_eps: float = Field(0.5, gt=0, description="Default DBSCAN eps parameter")
    default_dbscan_min_samples: int = Field(5, ge=1, description="Default DBSCAN min_samples")
    
    # File Storage Configuration
    data_dir: Path = Field(Path("./data"), description="Data directory")
    output_dir: Path = Field(Path("./outputs"), description="Output directory")
    plots_dir: Path = Field(Path("./outputs/plots"), description="Plots directory")
    
    # Logging Configuration
    log_level: str = Field("INFO", description="Logging level")
    log_file: Optional[Path] = Field(Path("./logs/app.log"), description="Log file path")
    
    # Development Configuration
    debug: bool = Field(False, description="Debug mode")
    reload: bool = Field(False, description="Auto-reload for development")
    
    # LangChain Debugging
    langchain_verbose: bool = Field(False, description="Enable LangChain verbose logging")
    langchain_debug: bool = Field(False, description="Enable LangChain debug logging")
    
    # NEW: Multi-Turn Conversation Configuration
    # Admin & Security
    admin_password: str = Field("secure_admin_password_2024", description="Admin panel password")
    ui_secret_key: str = Field("streamlit_secret_key_12345", description="Streamlit UI secret key")
    
    # Conversation Settings
    conversation_timeout_minutes: int = Field(120, ge=15, le=480, description="Conversation session timeout in minutes")
    max_concurrent_sessions: int = Field(100, ge=1, le=1000, description="Maximum concurrent conversation sessions")
    enable_conversation_logging: bool = Field(True, description="Enable conversation logging")
    
    # File Upload Settings
    upload_dir: Path = Field(Path("./data/uploads"), description="File upload directory")
    max_upload_size_mb: int = Field(10, ge=1, le=100, description="Maximum upload file size in MB")
    
    # Streamlit UI Configuration
    streamlit_server_port: int = Field(8501, ge=1, le=65535, description="Streamlit server port")
    streamlit_server_host: str = Field("0.0.0.0", description="Streamlit server host")
    
    @validator('google_ai_api_key')
    def validate_api_key(cls, v):
        """
        Validate that the API key is provided.
        """
        if not v or v == "your_google_ai_api_key_here":
            raise ValueError(
                "Google AI API key must be provided. "
                "Get one at https://ai.google.dev/gemini-api/docs/api-key"
            )
        return v
    
    @validator('data_dir', 'output_dir', 'plots_dir', 'upload_dir')
    def create_directories(cls, v):
        """
        Create directories if they don't exist.
        """
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @validator('log_file')
    def create_log_directory(cls, v):
        """
        Create log directory if log file is specified.
        """
        if v:
            v.parent.mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance - lazy initialization
_settings = None


def get_settings() -> Settings:
    """
    Get the global settings instance.
    
    Returns:
        Settings: The application settings.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings