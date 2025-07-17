"""
Anomaly detection tool implementing z-score, IQR, and DBSCAN methods.
"""

import logging
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from core.exceptions import AnomalyDetectionMethodError, DataValidationError
from core.models import AnomalyResult, TimeSeriesData


logger = logging.getLogger(__name__)


class AnomalyDetectionTool:
    """
    Tool for detecting anomalies in time-series data using various statistical methods.
    """
    
    def __init__(self):
        """Initialize the anomaly detection tool."""
        self.logger = logging.getLogger(__name__)
    
    def detect_anomalies(
        self, 
        data: TimeSeriesData, 
        method: str, 
        threshold: float = None,
        **kwargs
    ) -> AnomalyResult:
        """
        Detect anomalies using the specified method.
        
        Args:
            data: Time-series data to analyze
            method: Detection method ('z-score', 'iqr', 'dbscan', 'rolling-iqr')
            threshold: Threshold parameter (method-specific)
            **kwargs: Additional parameters for specific methods
            
        Returns:
            AnomalyResult: Detection results
            
        Raises:
            AnomalyDetectionMethodError: If method fails
            DataValidationError: If data is invalid
        """
        if len(data.values) < 3:
            raise DataValidationError("Need at least 3 data points for anomaly detection")
        
        # Convert to pandas Series for easier processing
        series = pd.Series(data.values)
        
        try:
            if method == 'z-score':
                return self._detect_z_score(series, data, threshold or 3.0)
            elif method == 'iqr':
                return self._detect_iqr(series, data, threshold or 1.5)
            elif method == 'rolling-iqr':
                window_size = kwargs.get('window_size', 20)
                return self._detect_rolling_iqr(series, data, threshold or 1.5, window_size)
            elif method == 'dbscan':
                eps = kwargs.get('eps', threshold or 0.5)
                min_samples = kwargs.get('min_samples', 5)
                return self._detect_dbscan(series, data, eps, min_samples)
            else:
                raise AnomalyDetectionMethodError(f"Unknown method: {method}", method)
                
        except Exception as e:
            if isinstance(e, (AnomalyDetectionMethodError, DataValidationError)):
                raise
            raise AnomalyDetectionMethodError(
                f"Error in {method} detection: {str(e)}", 
                method
            )
    
    def _detect_z_score(
        self, 
        series: pd.Series, 
        data: TimeSeriesData, 
        threshold: float
    ) -> AnomalyResult:
        """
        Detect anomalies using Z-score method.
        
        Z-score measures how many standard deviations away from the mean a data point is.
        Points with |z-score| > threshold are considered anomalies.
        
        Args:
            series: Pandas series of values
            data: Original time-series data
            threshold: Z-score threshold (typically 3.0)
            
        Returns:
            AnomalyResult: Detection results
        """
        self.logger.info(f"Running Z-score anomaly detection with threshold={threshold}")
        
        try:
            # Calculate mean and standard deviation
            mean = series.mean()
            std_dev = series.std()
            
            if std_dev == 0:
                self.logger.warning("Standard deviation is 0, no anomalies can be detected")
                return self._create_empty_result('z-score', threshold, len(series), data)
            
            # Calculate Z-scores
            z_scores = (series - mean) / std_dev
            
            # Identify outliers where |z-score| > threshold
            anomaly_mask = np.abs(z_scores) > threshold
            anomaly_indices = series[anomaly_mask].index.tolist()
            
            self.logger.info(
                f"Z-score detection found {len(anomaly_indices)} anomalies "
                f"out of {len(series)} points ({len(anomaly_indices)/len(series)*100:.1f}%)"
            )
            
            return self._create_anomaly_result(
                anomaly_indices=anomaly_indices,
                method_used='z-score',
                threshold_used=threshold,
                total_points=len(series),
                data=data
            )
            
        except Exception as e:
            raise AnomalyDetectionMethodError(f"Z-score calculation failed: {str(e)}", 'z-score')
    
    def _detect_iqr(
        self, 
        series: pd.Series, 
        data: TimeSeriesData, 
        multiplier: float
    ) -> AnomalyResult:
        """
        Detect anomalies using Interquartile Range (IQR) method.
        
        IQR method identifies outliers as points outside the range:
        [Q1 - multiplier*IQR, Q3 + multiplier*IQR]
        where IQR = Q3 - Q1
        
        Args:
            series: Pandas series of values
            data: Original time-series data
            multiplier: IQR multiplier (typically 1.5)
            
        Returns:
            AnomalyResult: Detection results
        """
        self.logger.info(f"Running IQR anomaly detection with multiplier={multiplier}")
        
        try:
            # Calculate quartiles and IQR
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:
                self.logger.warning("IQR is 0, no anomalies can be detected")
                return self._create_empty_result('iqr', multiplier, len(series), data)
            
            # Calculate bounds
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            # Identify outliers outside the bounds
            anomaly_mask = (series < lower_bound) | (series > upper_bound)
            anomaly_indices = series[anomaly_mask].index.tolist()
            
            self.logger.info(
                f"IQR detection found {len(anomaly_indices)} anomalies "
                f"out of {len(series)} points ({len(anomaly_indices)/len(series)*100:.1f}%)"
            )
            self.logger.debug(f"IQR bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
            
            return self._create_anomaly_result(
                anomaly_indices=anomaly_indices,
                method_used='iqr',
                threshold_used=multiplier,
                total_points=len(series),
                data=data
            )
            
        except Exception as e:
            raise AnomalyDetectionMethodError(f"IQR calculation failed: {str(e)}", 'iqr')
    
    def _detect_rolling_iqr(
        self, 
        series: pd.Series, 
        data: TimeSeriesData, 
        multiplier: float,
        window_size: int
    ) -> AnomalyResult:
        """
        Detect anomalies using Rolling IQR method with sliding window.
        
        This method compares each data point to a rolling window of recent values,
        making it excellent for detecting local spikes and crashes in time series.
        It's particularly effective for data with changing baseline levels.
        
        Args:
            series: Pandas series of values
            data: Original time-series data
            multiplier: IQR multiplier (typically 1.5)
            window_size: Size of rolling window (default 20 days)
            
        Returns:
            AnomalyResult: Detection results
        """
        self.logger.info(f"Running Rolling IQR anomaly detection with multiplier={multiplier}, window_size={window_size}")
        
        try:
            if len(series) < window_size:
                self.logger.warning(f"Data length ({len(series)}) is less than window size ({window_size})")
                # Fall back to standard IQR for small datasets
                return self._detect_iqr(series, data, multiplier)
            
            # Initialize results
            anomaly_indices = []
            
            # Convert series to DataFrame for easier rolling calculations
            df = pd.DataFrame({'value': series})
            
            # Calculate rolling quantiles
            rolling_q1 = df['value'].rolling(window=window_size, center=False).quantile(0.25)
            rolling_q3 = df['value'].rolling(window=window_size, center=False).quantile(0.75)
            rolling_iqr = rolling_q3 - rolling_q1
            
            # Calculate rolling bounds
            rolling_lower = rolling_q1 - multiplier * rolling_iqr
            rolling_upper = rolling_q3 + multiplier * rolling_iqr
            
            # Start detection after we have enough data for the window
            for i in range(window_size - 1, len(series)):
                current_value = series.iloc[i]
                lower_bound = rolling_lower.iloc[i]
                upper_bound = rolling_upper.iloc[i]
                
                # Check if current value is outside the rolling bounds
                if pd.isna(lower_bound) or pd.isna(upper_bound):
                    continue
                    
                if current_value < lower_bound or current_value > upper_bound:
                    anomaly_indices.append(i)
                    self.logger.debug(
                        f"Rolling IQR anomaly at index {i}: value={current_value:.2f}, "
                        f"bounds=[{lower_bound:.2f}, {upper_bound:.2f}]"
                    )
            
            self.logger.info(
                f"Rolling IQR detection found {len(anomaly_indices)} anomalies "
                f"out of {len(series)} points ({len(anomaly_indices)/len(series)*100:.1f}%)"
            )
            
            # Log additional insights about rolling detection
            if anomaly_indices:
                anomaly_values = [series.iloc[i] for i in anomaly_indices]
                self.logger.info(f"Anomaly value range: [{min(anomaly_values):.2f}, {max(anomaly_values):.2f}]")
                
                # Detect if anomalies are spikes (above normal) or crashes (below normal)
                median_value = series.median()
                spikes = sum(1 for val in anomaly_values if val > median_value)
                crashes = len(anomaly_values) - spikes
                self.logger.info(f"Detected {spikes} spikes and {crashes} crashes relative to median")
            
            return self._create_anomaly_result(
                anomaly_indices=anomaly_indices,
                method_used='rolling-iqr',
                threshold_used=multiplier,
                total_points=len(series),
                data=data
            )
            
        except Exception as e:
            raise AnomalyDetectionMethodError(f"Rolling IQR calculation failed: {str(e)}", 'rolling-iqr')
    
    def _detect_dbscan(
        self, 
        series: pd.Series, 
        data: TimeSeriesData, 
        eps: float, 
        min_samples: int
    ) -> AnomalyResult:
        """
        Detect anomalies using DBSCAN clustering method.
        
        DBSCAN groups similar points into clusters. Points that don't belong to any cluster
        (labeled as -1) are considered anomalies/outliers.
        
        Args:
            series: Pandas series of values
            data: Original time-series data
            eps: Maximum distance between samples for them to be in the same cluster
            min_samples: Minimum number of samples in a cluster
            
        Returns:
            AnomalyResult: Detection results
        """
        self.logger.info(
            f"Running DBSCAN anomaly detection with eps={eps}, min_samples={min_samples}"
        )
        
        try:
            # Reshape data for DBSCAN (needs 2D array)
            X = series.values.reshape(-1, 1)
            
            # Apply DBSCAN clustering
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(X)
            
            # Identify outliers (DBSCAN labels outliers as -1)
            anomaly_mask = cluster_labels == -1
            anomaly_indices = series[anomaly_mask].index.tolist()
            
            # Count clusters
            unique_labels = set(cluster_labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            
            self.logger.info(
                f"DBSCAN found {n_clusters} clusters and {len(anomaly_indices)} anomalies "
                f"out of {len(series)} points ({len(anomaly_indices)/len(series)*100:.1f}%)"
            )
            
            return self._create_anomaly_result(
                anomaly_indices=anomaly_indices,
                method_used='dbscan',
                threshold_used=eps,
                total_points=len(series),
                data=data
            )
            
        except Exception as e:
            raise AnomalyDetectionMethodError(f"DBSCAN clustering failed: {str(e)}", 'dbscan')
    
    def _create_anomaly_result(
        self,
        anomaly_indices: List[int],
        method_used: str,
        threshold_used: float,
        total_points: int,
        data: TimeSeriesData
    ) -> AnomalyResult:
        """
        Create AnomalyResult object from detection results with enhanced formatting.
        
        Args:
            anomaly_indices: Indices of detected anomalies
            method_used: Detection method used
            threshold_used: Threshold parameter used
            total_points: Total number of data points
            data: Original time-series data
            
        Returns:
            AnomalyResult: Formatted result object with structured anomaly points
        """
        anomaly_count = len(anomaly_indices)
        anomaly_percentage = (anomaly_count / total_points) * 100 if total_points > 0 else 0
        
        # Extract anomaly values and timestamps
        anomaly_values = [data.values[i] for i in anomaly_indices]
        anomaly_timestamps = [data.timestamp[i] for i in anomaly_indices]
        
        # Create enhanced anomaly result with structured data
        result = AnomalyResult(
            anomaly_indices=anomaly_indices,
            method_used=method_used,
            threshold_used=threshold_used,
            total_points=total_points,
            anomaly_count=anomaly_count,
            anomaly_percentage=anomaly_percentage,
            anomaly_values=anomaly_values,
            anomaly_timestamps=anomaly_timestamps
        )
        
        # Note: structured_anomaly_points functionality was removed during cleanup
        # result.structured_anomaly_points = self._create_structured_anomaly_points(
        #     anomaly_indices, data, method_used
        # )
        
        return result
    
    # Commented out - StructuredAnomalyPoint class was removed during cleanup
    # def _create_structured_anomaly_points(
    #     self, 
    #     anomaly_indices: List[int], 
    #     data: TimeSeriesData,
    #     method_used: str
    # ) -> List[StructuredAnomalyPoint]:
    #     """
    #     Create structured anomaly points in the format expected by enhanced suggestion agent.
    #     
    #     Args:
    #         anomaly_indices: Indices of detected anomalies
    #         data: Original time-series data
    #         method_used: Detection method used
    #         
    #     Returns:
    #         List[StructuredAnomalyPoint]: Structured anomaly points with severity, deviation scores, and trends
    #     """
    #     if not anomaly_indices:
    #         return []
    #     
    #     series = pd.Series(data.values)
    #     structured_points = []
    #     
    #     # Calculate statistics for severity assessment
    #     mean_val = series.mean()
    #     std_val = series.std()
    #     median_val = series.median()
    #     
    #     for idx in anomaly_indices:
    #         value = data.values[idx]
    #         timestamp = data.timestamp[idx]
    #         
    #         # Calculate deviation score based on method
    #         if method_used == 'z-score':
    #             deviation_score = abs(value - mean_val) / std_val if std_val > 0 else 0
    #         elif method_used in ['iqr', 'rolling-iqr']:
    #             # Use IQR-based deviation
    #             Q1 = series.quantile(0.25)
    #             Q3 = series.quantile(0.75)
    #             IQR = Q3 - Q1
    #             if value < Q1:
    #                 deviation_score = abs(Q1 - value) / IQR if IQR > 0 else 0
    #             elif value > Q3:
    #                 deviation_score = abs(value - Q3) / IQR if IQR > 0 else 0
    #             else:
    #                 deviation_score = 0
    #         else:  # dbscan or other methods
    #             deviation_score = abs(value - median_val) / std_val if std_val > 0 else 0
    #         
    #         # Determine severity based on deviation score
    #         if deviation_score > 3:
    #             severity = 'high'
    #         elif deviation_score > 2:
    #             severity = 'medium'
    #         else:
    #             severity = 'low'
    #         
    #         # Determine trend by looking at surrounding values
    #         trend = self._determine_trend(idx, series)
    #         
    #         structured_point = StructuredAnomalyPoint(
    #             timestamp=timestamp.isoformat(),
    #             value=float(value),
    #             severity=severity,
    #             deviation_score=round(deviation_score, 2),
    #             trend=trend
    #         )
    #         
    #         structured_points.append(structured_point)
    #     
    #     # Sort by severity and deviation score (most significant first)
    #     structured_points.sort(
    #         key=lambda x: (x.severity == 'high', x.severity == 'medium', x.deviation_score), 
    #         reverse=True
    #     )
    #     
    #     return structured_points
    
    def _determine_trend(self, idx: int, series: pd.Series, window: int = 3) -> str:
        """
        Determine trend around an anomaly point.
        
        Args:
            idx: Index of the anomaly point
            series: Full data series
            window: Window size to look at surrounding points
            
        Returns:
            str: Trend direction ('increasing', 'decreasing', 'stable', 'unknown')
        """
        try:
            # Look at surrounding values
            start_idx = max(0, idx - window)
            end_idx = min(len(series), idx + window + 1)
            
            if end_idx - start_idx < 3:
                return 'unknown'
            
            # Get surrounding values
            surrounding = series.iloc[start_idx:end_idx]
            current_value = series.iloc[idx]
            
            # Calculate trend
            before_values = surrounding.iloc[:idx-start_idx] if idx > start_idx else pd.Series()
            after_values = surrounding.iloc[idx-start_idx+1:] if idx < end_idx-1 else pd.Series()
            
            if len(before_values) > 0 and len(after_values) > 0:
                before_avg = before_values.mean()
                after_avg = after_values.mean()
                
                if current_value > before_avg and current_value > after_avg:
                    return 'spike'
                elif current_value < before_avg and current_value < after_avg:
                    return 'dip'
                elif after_avg > before_avg:
                    return 'increasing'
                elif after_avg < before_avg:
                    return 'decreasing'
                else:
                    return 'stable'
            elif len(before_values) > 0:
                # Only have before values
                if current_value > before_values.mean():
                    return 'increasing'
                else:
                    return 'decreasing'
            else:
                return 'unknown'
                
        except Exception:
            return 'unknown'
    
    def _create_empty_result(
        self, 
        method: str, 
        threshold: float, 
        total_points: int, 
        data: TimeSeriesData
    ) -> AnomalyResult:
        """
        Create empty AnomalyResult when no anomalies are detected.
        
        Args:
            method: Detection method used
            threshold: Threshold parameter used
            total_points: Total number of data points
            data: Original time-series data
            
        Returns:
            AnomalyResult: Empty result object
        """
        return AnomalyResult(
            anomaly_indices=[],
            method_used=method,
            threshold_used=threshold,
            total_points=total_points,
            anomaly_count=0,
            anomaly_percentage=0.0,
            anomaly_values=[],
            anomaly_timestamps=[]
        )
    
    def get_method_info(self, method: str) -> dict:
        """
        Get information about a specific detection method.
        
        Args:
            method: Detection method name
            
        Returns:
            dict: Method information and default parameters
        """
        method_info = {
            'z-score': {
                'name': 'Z-Score',
                'description': 'Statistical method using standard deviations from mean',
                'default_threshold': 3.0,
                'parameters': ['threshold'],
                'best_for': 'Normally distributed data with clear outliers'
            },
            'iqr': {
                'name': 'Interquartile Range',
                'description': 'Statistical method using quartile ranges',
                'default_threshold': 1.5,
                'parameters': ['multiplier'],
                'best_for': 'Skewed data distributions'
            },
            'rolling-iqr': {
                'name': 'Rolling IQR',
                'description': '20-day rolling window IQR for local anomaly detection',
                'default_threshold': 1.5,
                'parameters': ['multiplier', 'window_size'],
                'best_for': 'Time series with changing baselines, detecting local spikes and crashes'
            },
            'dbscan': {
                'name': 'DBSCAN Clustering',
                'description': 'Density-based clustering method',
                'default_threshold': 0.5,
                'parameters': ['eps', 'min_samples'],
                'best_for': 'Complex patterns and irregular distributions'
            }
        }
        
        return method_info.get(method, {})
    
    def recommend_method(self, data: TimeSeriesData) -> str:
        """
        Recommend the best detection method based on data characteristics.
        
        Args:
            data: Time-series data to analyze
            
        Returns:
            str: Recommended method name
        """
        series = pd.Series(data.values)
        
        # Calculate basic statistics
        skewness = abs(series.skew())
        kurtosis = series.kurtosis()
        cv = series.std() / series.mean() if series.mean() != 0 else float('inf')
        n_points = len(series)
        
        # Time-based considerations
        if len(data.timestamp) > 1:
            # Calculate time differences to assess if it's time-series data
            time_diffs = [(data.timestamp[i+1] - data.timestamp[i]).total_seconds() for i in range(len(data.timestamp)-1)]
            avg_time_diff = sum(time_diffs) / len(time_diffs) if time_diffs else 0
            is_regular_time_series = len(set(time_diffs)) <= 3  # Regular intervals
        else:
            is_regular_time_series = False
            avg_time_diff = 0
        
        # Enhanced decision logic
        if n_points >= 20 and is_regular_time_series:  
            # For time-series data with sufficient points, prefer rolling methods
            return 'rolling-iqr'
        elif skewness > 2:  # Highly skewed data
            return 'iqr'
        elif kurtosis > 3 and cv < 0.5:  # High kurtosis, low variability
            return 'z-score'
        else:  # Complex patterns
            return 'dbscan'