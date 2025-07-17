"""
Custom exceptions for the anomaly detection system.
"""


class AnomalyDetectionError(Exception):
    """
    Base exception for anomaly detection system.
    """
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code or "GENERAL_ERROR"
        super().__init__(self.message)


class FileProcessingError(AnomalyDetectionError):
    """
    Exception raised for errors in file processing.
    """
    def __init__(self, message: str, filename: str = None):
        self.filename = filename
        super().__init__(message, "FILE_PROCESSING_ERROR")


class DataValidationError(AnomalyDetectionError):
    """
    Exception raised for data validation errors.
    """
    def __init__(self, message: str, field: str = None):
        self.field = field
        super().__init__(message, "DATA_VALIDATION_ERROR")


class AnomalyDetectionMethodError(AnomalyDetectionError):
    """
    Exception raised for errors in anomaly detection methods.
    """
    def __init__(self, message: str, method: str = None):
        self.method = method
        super().__init__(message, "ANOMALY_DETECTION_METHOD_ERROR")


class VisualizationError(AnomalyDetectionError):
    """
    Exception raised for errors in visualization generation.
    """
    def __init__(self, message: str, plot_type: str = None):
        self.plot_type = plot_type
        super().__init__(message, "VISUALIZATION_ERROR")


class LLMError(AnomalyDetectionError):
    """
    Exception raised for errors in LLM communication.
    """
    def __init__(self, message: str, model: str = None):
        self.model = model
        super().__init__(message, "LLM_ERROR")


class ConfigurationError(AnomalyDetectionError):
    """
    Exception raised for configuration errors.
    """
    def __init__(self, message: str, config_key: str = None):
        self.config_key = config_key
        super().__init__(message, "CONFIGURATION_ERROR")


class APIError(AnomalyDetectionError):
    """
    Exception raised for API-related errors.
    """
    def __init__(self, message: str, status_code: int = 500):
        self.status_code = status_code
        super().__init__(message, "API_ERROR")


class AgentError(AnomalyDetectionError):
    """
    Exception raised for agent workflow errors.
    """
    def __init__(self, message: str, agent_name: str = None):
        self.agent_name = agent_name
        super().__init__(message, "AGENT_ERROR")