"""
File reading tool for CSV and Excel files with data validation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from datetime import datetime

from core.exceptions import FileProcessingError, DataValidationError
from core.models import TimeSeriesData, FileInfo

logger = logging.getLogger(__name__)


class FileReaderTool:
    """
    Tool for reading and validating time-series data files.
    """
    
    SUPPORTED_EXTENSIONS = {'.csv', '.xlsx', '.xls'}
    
    def __init__(self):
        """Initialize the file reader tool."""
        self.logger = logging.getLogger(__name__)
    
    def read_file(
        self, 
        file_path: Union[str, Path], 
        timestamp_column: Optional[str] = None,
        value_column: Optional[str] = None
    ) -> TimeSeriesData:
        """
        Read time-series data from file.
        
        Args:
            file_path: Path to the data file
            timestamp_column: Name of timestamp column (auto-detected if None)
            value_column: Name of value column (auto-detected if None)
            
        Returns:
            TimeSeriesData: Validated time-series data
            
        Raises:
            FileProcessingError: If file cannot be read or processed
            DataValidationError: If data validation fails
        """
        file_path = Path(file_path)
        
        # Validate file exists and is supported
        self._validate_file(file_path)
        
        try:
            # Read the file based on extension
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() in {'.xlsx', '.xls'}:
                df = pd.read_excel(file_path)
            else:
                raise FileProcessingError(
                    f"Unsupported file type: {file_path.suffix}",
                    str(file_path)
                )
            
            self.logger.info(f"Successfully read file {file_path} with {len(df)} rows")
            
            # Detect columns if not provided
            if not timestamp_column:
                timestamp_column = self._detect_timestamp_column(df)
            if not value_column:
                value_column = self._detect_value_column(df, timestamp_column)
            
            # Validate columns exist
            self._validate_columns(df, timestamp_column, value_column)
            
            # Extract and validate data
            return self._extract_time_series_data(df, timestamp_column, value_column)
            
        except pd.errors.EmptyDataError:
            raise FileProcessingError(f"File {file_path} is empty", str(file_path))
        except pd.errors.ParserError as e:
            raise FileProcessingError(
                f"Failed to parse file {file_path}: {str(e)}", 
                str(file_path)
            )
        except Exception as e:
            if isinstance(e, (FileProcessingError, DataValidationError)):
                raise
            raise FileProcessingError(
                f"Unexpected error reading file {file_path}: {str(e)}", 
                str(file_path)
            )
    
    def get_file_info(self, file_path: Union[str, Path]) -> FileInfo:
        """
        Get information about a file without fully loading it.
        
        Args:
            file_path: Path to the file
            
        Returns:
            FileInfo: Information about the file
        """
        file_path = Path(file_path)
        self._validate_file(file_path)
        
        try:
            # Get basic file info
            file_size = file_path.stat().st_size
            file_type = file_path.suffix.lower().lstrip('.')
            
            # Read just the header to get column info
            if file_path.suffix.lower() == '.csv':
                df_sample = pd.read_csv(file_path, nrows=0)  # Just header
                row_count = sum(1 for _ in open(file_path)) - 1  # Count lines minus header
            else:
                df_sample = pd.read_excel(file_path, nrows=0)
                # For Excel, we need to read the full file to count rows (less efficient)
                df_full = pd.read_excel(file_path)
                row_count = len(df_full)
            
            return FileInfo(
                filename=file_path.name,
                file_size=file_size,
                file_type=file_type,
                columns=df_sample.columns.tolist(),
                row_count=row_count
            )
            
        except Exception as e:
            raise FileProcessingError(
                f"Failed to get file info for {file_path}: {str(e)}", 
                str(file_path)
            )
    
    def _validate_file(self, file_path: Path) -> None:
        """
        Validate that file exists and has supported extension.
        
        Args:
            file_path: Path to validate
            
        Raises:
            FileProcessingError: If validation fails
        """
        if not file_path.exists():
            raise FileProcessingError(f"File not found: {file_path}", str(file_path))
        
        if not file_path.is_file():
            raise FileProcessingError(f"Path is not a file: {file_path}", str(file_path))
        
        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise FileProcessingError(
                f"Unsupported file extension: {file_path.suffix}. "
                f"Supported: {', '.join(self.SUPPORTED_EXTENSIONS)}",
                str(file_path)
            )
    
    def _detect_timestamp_column(self, df: pd.DataFrame) -> str:
        """
        Auto-detect timestamp column from DataFrame.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            str: Name of detected timestamp column
            
        Raises:
            DataValidationError: If no timestamp column found
        """
        # Common timestamp column names
        timestamp_candidates = [
            'timestamp', 'time', 'date', 'datetime', 'Date', 'Time', 
            'Timestamp', 'DateTime', 'TIMESTAMP', 'DATE', 'TIME'
        ]
        
        # First, try exact matches
        for col in timestamp_candidates:
            if col in df.columns:
                self.logger.info(f"Detected timestamp column: {col}")
                return col
        
        # Try partial matches
        for col in df.columns:
            col_lower = col.lower()
            if any(candidate.lower() in col_lower for candidate in timestamp_candidates):
                self.logger.info(f"Detected timestamp column: {col}")
                return col
        
        # Try to detect by data type
        for col in df.columns:
            try:
                pd.to_datetime(df[col].dropna().head(10))
                self.logger.info(f"Detected timestamp column by type: {col}")
                return col
            except (ValueError, TypeError):
                continue
        
        # If still not found, use first column if it looks like a timestamp
        if len(df.columns) > 0:
            first_col = df.columns[0]
            try:
                pd.to_datetime(df[first_col].dropna().head(10))
                self.logger.warning(f"Using first column as timestamp: {first_col}")
                return first_col
            except (ValueError, TypeError):
                pass
        
        raise DataValidationError(
            "Could not detect timestamp column. Please specify explicitly."
        )
    
    def _detect_value_column(self, df: pd.DataFrame, timestamp_column: str) -> str:
        """
        Auto-detect value column from DataFrame.
        
        Args:
            df: DataFrame to analyze
            timestamp_column: Already identified timestamp column
            
        Returns:
            str: Name of detected value column
            
        Raises:
            DataValidationError: If no suitable value column found
        """
        # Exclude timestamp column
        available_columns = [col for col in df.columns if col != timestamp_column]
        
        if not available_columns:
            raise DataValidationError("No columns available for values after excluding timestamp")
        
        # Common value column names
        value_candidates = [
            'value', 'amount', 'price', 'close', 'open', 'high', 'low',
            'Value', 'Amount', 'Price', 'Close', 'Open', 'High', 'Low',
            'VALUE', 'AMOUNT', 'PRICE'
        ]
        
        # First, try exact matches
        for col in value_candidates:
            if col in available_columns:
                self.logger.info(f"Detected value column: {col}")
                return col
        
        # Try to find numeric columns
        numeric_columns = []
        for col in available_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_columns.append(col)
        
        if numeric_columns:
            # Prefer first numeric column
            self.logger.info(f"Using first numeric column as value: {numeric_columns[0]}")
            return numeric_columns[0]
        
        # Try to convert columns to numeric
        for col in available_columns:
            try:
                pd.to_numeric(df[col].dropna().head(10))
                self.logger.info(f"Detected convertible numeric column: {col}")
                return col
            except (ValueError, TypeError):
                continue
        
        # Last resort: use first available column
        if available_columns:
            self.logger.warning(f"Using first available column as value: {available_columns[0]}")
            return available_columns[0]
        
        raise DataValidationError("Could not detect value column. Please specify explicitly.")
    
    def _validate_columns(self, df: pd.DataFrame, timestamp_column: str, value_column: str) -> None:
        """
        Validate that specified columns exist in DataFrame.
        
        Args:
            df: DataFrame to validate
            timestamp_column: Timestamp column name
            value_column: Value column name
            
        Raises:
            DataValidationError: If columns don't exist
        """
        missing_columns = []
        
        if timestamp_column not in df.columns:
            missing_columns.append(timestamp_column)
        
        if value_column not in df.columns:
            missing_columns.append(value_column)
        
        if missing_columns:
            raise DataValidationError(
                f"Columns not found in file: {missing_columns}. "
                f"Available columns: {list(df.columns)}"
            )
    
    def _extract_time_series_data(
        self, 
        df: pd.DataFrame, 
        timestamp_column: str, 
        value_column: str
    ) -> TimeSeriesData:
        """
        Extract and validate time-series data from DataFrame.
        
        Args:
            df: Source DataFrame
            timestamp_column: Timestamp column name
            value_column: Value column name
            
        Returns:
            TimeSeriesData: Validated time-series data
            
        Raises:
            DataValidationError: If data validation fails
        """
        try:
            # Extract relevant columns
            working_df = df[[timestamp_column, value_column]].copy()
            
            # Remove rows with missing values
            initial_count = len(working_df)
            working_df = working_df.dropna()
            dropped_count = initial_count - len(working_df)
            
            if dropped_count > 0:
                self.logger.warning(f"Dropped {dropped_count} rows with missing values")
            
            if len(working_df) == 0:
                raise DataValidationError("No valid data rows after removing missing values")
            
            # Convert timestamp column
            try:
                # Try common date formats to avoid ambiguity warnings
                timestamps = pd.to_datetime(working_df[timestamp_column], 
                                          infer_datetime_format=True, 
                                          dayfirst=False)
            except Exception as e:
                raise DataValidationError(f"Failed to parse timestamps: {str(e)}")
            
            # Convert value column
            try:
                values = pd.to_numeric(working_df[value_column], errors='coerce')
                # Remove any NaN values that resulted from conversion
                valid_mask = ~values.isna()
                timestamps = timestamps[valid_mask]
                values = values[valid_mask]
                
                if len(values) == 0:
                    raise DataValidationError("No valid numeric values found")
                    
            except Exception as e:
                raise DataValidationError(f"Failed to parse values as numeric: {str(e)}")
            
            # Sort by timestamp
            sort_order = timestamps.argsort()
            timestamps = timestamps.iloc[sort_order]
            values = values.iloc[sort_order]
            
            # Convert to lists
            timestamp_list = [ts.to_pydatetime() for ts in timestamps]
            value_list = values.tolist()
            
            self.logger.info(
                f"Extracted {len(timestamp_list)} valid time-series points "
                f"from column '{value_column}'"
            )
            
            return TimeSeriesData(
                timestamp=timestamp_list,
                values=value_list,
                column_name=value_column
            )
            
        except Exception as e:
            if isinstance(e, DataValidationError):
                raise
            raise DataValidationError(f"Failed to extract time-series data: {str(e)}")