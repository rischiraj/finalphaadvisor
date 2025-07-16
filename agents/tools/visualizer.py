"""
Visualization tool for creating time-series anomaly detection plots.
"""

import base64
import io
import logging
from pathlib import Path
from typing import Optional, Union

import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd

from core.config import get_settings
from core.exceptions import VisualizationError
from core.models import AnomalyResult, TimeSeriesData, VisualizationResult


logger = logging.getLogger(__name__)


class VisualizationTool:
    """
    Tool for creating time-series visualizations with highlighted anomalies.
    """
    
    def __init__(self):
        """Initialize the visualization tool."""
        self.logger = logging.getLogger(__name__)
        self.settings = get_settings()
        
        # Ensure plots directory exists
        self.settings.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def create_anomaly_plot(
        self,
        data: TimeSeriesData,
        anomaly_result: AnomalyResult,
        plot_type: str = 'matplotlib',
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None
    ) -> VisualizationResult:
        """
        Create a time-series plot with highlighted anomalies.
        
        Args:
            data: Time-series data
            anomaly_result: Anomaly detection results
            plot_type: Type of plot ('matplotlib' or 'plotly')
            title: Custom title for the plot
            save_path: Custom save path for the plot
            
        Returns:
            VisualizationResult: Plot information and base64 encoded image
            
        Raises:
            VisualizationError: If plot creation fails
        """
        try:
            if plot_type == 'matplotlib':
                return self._create_matplotlib_plot(data, anomaly_result, title, save_path)
            elif plot_type == 'plotly':
                return self._create_plotly_plot(data, anomaly_result, title, save_path)
            else:
                raise VisualizationError(f"Unsupported plot type: {plot_type}", plot_type)
                
        except Exception as e:
            if isinstance(e, VisualizationError):
                raise
            raise VisualizationError(f"Failed to create plot: {str(e)}", plot_type)
    
    def _create_matplotlib_plot(
        self,
        data: TimeSeriesData,
        anomaly_result: AnomalyResult,
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None
    ) -> VisualizationResult:
        """
        Create matplotlib time-series plot with anomalies.
        
        Args:
            data: Time-series data
            anomaly_result: Anomaly detection results
            title: Custom title for the plot
            save_path: Custom save path for the plot
            
        Returns:
            VisualizationResult: Plot information
        """
        try:
            # Create figure with proper size
            plt.figure(figsize=(12, 8))
            
            # Plot normal data points
            plt.plot(data.timestamp, data.values, 'b-', linewidth=1.5, alpha=0.7, label='Normal Data')
            
            # Highlight anomalies if any
            if anomaly_result.anomaly_indices:
                anomaly_timestamps = [data.timestamp[i] for i in anomaly_result.anomaly_indices]
                anomaly_values = [data.values[i] for i in anomaly_result.anomaly_indices]
                
                # Plot anomaly points with larger markers
                plt.scatter(
                    anomaly_timestamps, 
                    anomaly_values, 
                    color='red', 
                    s=80, 
                    alpha=0.9, 
                    label=f'Anomalies ({anomaly_result.anomaly_count})',
                    zorder=5,
                    edgecolors='darkred',
                    linewidth=1.5
                )
                
                # Add date annotations for anomalies (limit to first 10 to avoid clutter)
                max_annotations = min(10, len(anomaly_timestamps))
                for i in range(max_annotations):
                    timestamp = anomaly_timestamps[i]
                    value = anomaly_values[i]
                    date_str = timestamp.strftime('%Y-%m-%d')
                    
                    plt.annotate(
                        f'{date_str}\n{value:.2f}',
                        xy=(timestamp, value),
                        xytext=(5, 15),
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        fontsize=8,
                        ha='left',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                    )
                
                # If more than 10 anomalies, add a note
                if len(anomaly_timestamps) > 10:
                    plt.text(
                        0.02, 0.98, 
                        f'Showing dates for first {max_annotations} of {len(anomaly_timestamps)} anomalies',
                        transform=plt.gca().transAxes,
                        fontsize=9,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
                    )
            
            # Customize plot
            if not title:
                title = (
                    f'Time Series Anomaly Detection - {anomaly_result.method_used.upper()}\n'
                    f'{anomaly_result.anomaly_count} anomalies found '
                    f'({anomaly_result.anomaly_percentage:.1f}%)'
                )
            
            plt.title(title, fontsize=14, fontweight='bold', pad=20)
            plt.xlabel('Time', fontsize=12)
            plt.ylabel(f'{data.column_name}', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            
            # Smart x-axis date formatting based on data range and number of points
            time_span = (data.timestamp[-1] - data.timestamp[0]).days
            n_points = len(data.timestamp)
            
            # Determine optimal number of x-axis labels (between 6-12 labels)
            target_labels = min(12, max(6, n_points // 10))
            
            ax = plt.gca()
            
            if time_span > 365:  # More than a year
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                # Show quarters or months depending on length
                interval = max(1, time_span // (target_labels * 30))
                plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
                plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())
                
            elif time_span > 90:  # More than 3 months
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                interval = max(1, time_span // target_labels)
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval))
                plt.gca().xaxis.set_minor_locator(mdates.WeekdayLocator())
                
            elif time_span > 7:  # More than a week
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                interval = max(1, time_span // target_labels)
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval))
                plt.gca().xaxis.set_minor_locator(mdates.DayLocator())
                
            elif time_span >= 1:  # At least one day
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                # For daily data, show every few hours
                interval = max(1, (time_span * 24) // target_labels)
                plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=interval))
                plt.gca().xaxis.set_minor_locator(mdates.HourLocator())
                
            else:  # Less than a day
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                # For intraday data, show every few hours or minutes
                total_minutes = time_span * 24 * 60
                interval = max(1, int(total_minutes // target_labels))
                plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=interval))
                plt.gca().xaxis.set_minor_locator(mdates.MinuteLocator())
            
            # Rotate labels for better readability
            plt.xticks(rotation=45, ha='right')
            
            # Add minor grid for better readability
            ax.grid(True, which='major', alpha=0.3)
            ax.grid(True, which='minor', alpha=0.1)
            
            # Adjust layout
            plt.tight_layout()
            
            # Generate filename
            if not save_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"anomaly_plot_{anomaly_result.method_used}_{timestamp}.png"
                save_path = self.settings.plots_dir / filename
            else:
                save_path = Path(save_path)
            
            # Save plot
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plot_base64 = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()  # Close buffer to free memory
            
            # Close the plot to free memory
            plt.close('all')  # Close all figures
            
            # Generate description
            description = self._generate_plot_description(data, anomaly_result)
            
            self.logger.info(f"Created matplotlib plot saved to {save_path}")
            
            return VisualizationResult(
                plot_path=str(save_path),
                plot_base64=plot_base64,
                plot_description=description,
                plot_type='matplotlib'
            )
            
        except Exception as e:
            plt.close()  # Ensure plot is closed even on error
            raise VisualizationError(f"Matplotlib plot creation failed: {str(e)}", 'matplotlib')
    
    def _create_plotly_plot(
        self,
        data: TimeSeriesData,
        anomaly_result: AnomalyResult,
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None
    ) -> VisualizationResult:
        """
        Create Plotly interactive time-series plot with anomalies.
        
        Args:
            data: Time-series data
            anomaly_result: Anomaly detection results
            title: Custom title for the plot
            save_path: Custom save path for the plot
            
        Returns:
            VisualizationResult: Plot information
        """
        try:
            # Create figure
            fig = go.Figure()
            
            # Add normal data trace
            fig.add_trace(go.Scatter(
                x=data.timestamp,
                y=data.values,
                mode='lines+markers',
                name='Normal Data',
                line=dict(color='blue', width=2),
                marker=dict(size=4, opacity=0.7),
                hovertemplate='<b>Time:</b> %{x}<br><b>Value:</b> %{y:.2f}<extra></extra>'
            ))
            
            # Add anomaly points if any
            if anomaly_result.anomaly_indices:
                anomaly_timestamps = [data.timestamp[i] for i in anomaly_result.anomaly_indices]
                anomaly_values = [data.values[i] for i in anomaly_result.anomaly_indices]
                
                fig.add_trace(go.Scatter(
                    x=anomaly_timestamps,
                    y=anomaly_values,
                    mode='markers',
                    name=f'Anomalies ({anomaly_result.anomaly_count})',
                    marker=dict(
                        color='red',
                        size=10,
                        symbol='x',
                        line=dict(width=2, color='darkred')
                    ),
                    hovertemplate='<b>ANOMALY</b><br><b>Time:</b> %{x}<br><b>Value:</b> %{y:.2f}<extra></extra>'
                ))
            
            # Customize layout
            if not title:
                title = (
                    f'Time Series Anomaly Detection - {anomaly_result.method_used.upper()}<br>'
                    f'<sub>{anomaly_result.anomaly_count} anomalies found '
                    f'({anomaly_result.anomaly_percentage:.1f}%)</sub>'
                )
            
            fig.update_layout(
                title=dict(text=title, x=0.5, font=dict(size=16)),
                xaxis_title='Time',
                yaxis_title=data.column_name,
                hovermode='x unified',
                template='plotly_white',
                width=1000,
                height=600,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            # Generate filename
            if not save_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"anomaly_plot_{anomaly_result.method_used}_{timestamp}.html"
                save_path = self.settings.plots_dir / filename
            else:
                save_path = Path(save_path)
                if save_path.suffix.lower() not in ['.html', '.png']:
                    save_path = save_path.with_suffix('.html')
            
            # Save plot
            if save_path.suffix.lower() == '.html':
                fig.write_html(str(save_path))
            else:
                fig.write_image(str(save_path), width=1000, height=600)
            
            # Convert to base64 (PNG format)
            img_bytes = fig.to_image(format="png", width=1000, height=600)
            plot_base64 = base64.b64encode(img_bytes).decode()
            
            # Generate description
            description = self._generate_plot_description(data, anomaly_result)
            
            self.logger.info(f"Created Plotly plot saved to {save_path}")
            
            return VisualizationResult(
                plot_path=str(save_path),
                plot_base64=plot_base64,
                plot_description=description,
                plot_type='plotly'
            )
            
        except Exception as e:
            raise VisualizationError(f"Plotly plot creation failed: {str(e)}", 'plotly')
    
    def _generate_plot_description(self, data: TimeSeriesData, anomaly_result: AnomalyResult) -> str:
        """
        Generate a descriptive text for the plot.
        
        Args:
            data: Time-series data
            anomaly_result: Anomaly detection results
            
        Returns:
            str: Plot description
        """
        time_range = f"{data.timestamp[0].strftime('%Y-%m-%d %H:%M')} to {data.timestamp[-1].strftime('%Y-%m-%d %H:%M')}"
        
        description = (
            f"Time-series anomaly detection plot showing {len(data.values)} data points "
            f"from {time_range}. "
            f"Using {anomaly_result.method_used.upper()} method with threshold {anomaly_result.threshold_used}, "
            f"detected {anomaly_result.anomaly_count} anomalies "
            f"({anomaly_result.anomaly_percentage:.1f}% of total data). "
        )
        
        if anomaly_result.anomaly_count > 0:
            value_stats = pd.Series(data.values)
            anomaly_values_series = pd.Series(anomaly_result.anomaly_values)
            
            description += (
                f"Normal values range from {value_stats.min():.2f} to {value_stats.max():.2f} "
                f"(mean: {value_stats.mean():.2f}). "
                f"Anomaly values range from {anomaly_values_series.min():.2f} to {anomaly_values_series.max():.2f}."
            )
        else:
            description += "No anomalies were detected in the data."
        
        return description
    
    def create_comparison_plot(
        self,
        data: TimeSeriesData,
        results: list[AnomalyResult],
        save_path: Optional[Union[str, Path]] = None
    ) -> VisualizationResult:
        """
        Create a comparison plot showing results from multiple detection methods.
        
        Args:
            data: Time-series data
            results: List of anomaly detection results from different methods
            save_path: Custom save path for the plot
            
        Returns:
            VisualizationResult: Comparison plot information
        """
        try:
            # Create subplots
            fig, axes = plt.subplots(len(results), 1, figsize=(14, 4 * len(results)))
            if len(results) == 1:
                axes = [axes]
            
            colors = ['red', 'orange', 'purple']
            
            for i, result in enumerate(results):
                ax = axes[i]
                
                # Plot normal data
                ax.plot(data.timestamp, data.values, 'b-', linewidth=1, alpha=0.7, label='Normal Data')
                
                # Plot anomalies
                if result.anomaly_indices:
                    anomaly_timestamps = [data.timestamp[j] for j in result.anomaly_indices]
                    anomaly_values = [data.values[j] for j in result.anomaly_indices]
                    
                    ax.scatter(
                        anomaly_timestamps, 
                        anomaly_values, 
                        color=colors[i % len(colors)], 
                        s=40, 
                        alpha=0.8, 
                        label=f'Anomalies ({result.anomaly_count})',
                        zorder=5
                    )
                
                # Customize subplot
                ax.set_title(
                    f'{result.method_used.upper()} - {result.anomaly_count} anomalies '
                    f'({result.anomaly_percentage:.1f}%)',
                    fontsize=12,
                    fontweight='bold'
                )
                ax.set_ylabel(data.column_name, fontsize=10)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
                
                # Smart date formatting for subplots
                time_span = (data.timestamp[-1] - data.timestamp[0]).days
                if time_span > 90:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    interval = max(1, time_span // 8)
                    ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
                elif time_span > 7:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                    interval = max(1, time_span // 6)
                    ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
                else:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                    ax.xaxis.set_major_locator(mdates.DayLocator())
                
                ax.tick_params(axis='x', rotation=45)
            
            # Overall title
            plt.suptitle(
                f'Anomaly Detection Method Comparison - {data.column_name}',
                fontsize=14,
                fontweight='bold'
            )
            
            plt.tight_layout()
            
            # Generate filename
            if not save_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"anomaly_comparison_{timestamp}.png"
                save_path = self.settings.plots_dir / filename
            else:
                save_path = Path(save_path)
            
            # Save plot
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plot_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            plt.close()
            
            # Generate description
            method_names = [result.method_used for result in results]
            total_anomalies = [result.anomaly_count for result in results]
            
            description = (
                f"Comparison plot showing anomaly detection results from {len(results)} methods: "
                f"{', '.join(method_names)}. "
                f"Anomaly counts: {', '.join(f'{method}: {count}' for method, count in zip(method_names, total_anomalies))}"
            )
            
            self.logger.info(f"Created comparison plot saved to {save_path}")
            
            return VisualizationResult(
                plot_path=str(save_path),
                plot_base64=plot_base64,
                plot_description=description,
                plot_type='matplotlib'
            )
            
        except Exception as e:
            plt.close()
            raise VisualizationError(f"Comparison plot creation failed: {str(e)}", 'matplotlib')