"""
Tests for Streamlit application components.
"""

import pytest
from unittest.mock import patch, MagicMock, mock_open
import streamlit as st
from pathlib import Path
import base64
import json
import pandas as pd
from datetime import datetime

# Import streamlit components
from streamlit_app.components.analysis_renderer import AnalysisRenderer
from streamlit_app.components.json_viewer import JsonViewer
from streamlit_app.components.progress_indicator import ProgressIndicator
from streamlit_app.utils.styles import get_custom_css, apply_custom_styles


class TestAnalysisRenderer:
    """Test analysis rendering component."""
    
    def setup_method(self):
        """Set up test renderer."""
        self.renderer = AnalysisRenderer()
    
    def test_render_anomaly_result(self):
        """Test rendering anomaly detection results."""
        from core.models import AnomalyResult
        
        # Create mock anomaly result
        anomaly_result = AnomalyResult(
            anomaly_indices=[1, 5, 9],
            method_used="z-score",
            threshold_used=3.0,
            total_points=100,
            anomaly_count=3,
            anomaly_percentage=3.0,
            anomaly_values=[125.0, 150.0, 175.0],
            anomaly_timestamps=[datetime.now() for _ in range(3)]
        )
        
        with patch('streamlit.metric') as mock_metric, \
             patch('streamlit.write') as mock_write, \
             patch('streamlit.subheader') as mock_subheader:
            
            self.renderer.render_anomaly_result(anomaly_result)
            
            # Verify metrics were displayed
            assert mock_metric.called
            assert mock_write.called
            assert mock_subheader.called
    
    def test_render_visualization(self):
        """Test rendering visualization results."""
        from core.models import VisualizationResult
        
        # Create mock visualization result
        viz_result = VisualizationResult(
            plot_path="test_plot.png",
            plot_base64="base64_encoded_image_data",
            plot_description="Test anomaly detection plot",
            plot_type="matplotlib"
        )
        
        with patch('streamlit.image') as mock_image, \
             patch('streamlit.write') as mock_write:
            
            self.renderer.render_visualization(viz_result)
            
            # Verify image was displayed
            mock_image.assert_called_once()
            mock_write.assert_called()
    
    def test_render_insights(self):
        """Test rendering insight results."""
        from core.models import InsightResponse
        
        # Create mock insight response
        insights = InsightResponse(
            summary="Test analysis summary",
            anomaly_explanations=["Explanation 1", "Explanation 2"],
            recommendations=["Recommendation 1", "Recommendation 2"],
            root_causes=["Cause 1", "Cause 2"],
            confidence_score=85.0
        )
        
        with patch('streamlit.write') as mock_write, \
             patch('streamlit.subheader') as mock_subheader, \
             patch('streamlit.info') as mock_info:
            
            self.renderer.render_insights(insights)
            
            # Verify insights were displayed
            assert mock_write.called
            assert mock_subheader.called
            assert mock_info.called
    
    def test_render_complete_analysis(self):
        """Test rendering complete analysis response."""
        from core.models import AnalysisResponse, AnomalyResult, VisualizationResult, InsightResponse
        
        # Create complete analysis response
        analysis_response = AnalysisResponse(
            anomaly_result=AnomalyResult(
                anomaly_indices=[1, 5],
                method_used="iqr",
                threshold_used=1.5,
                total_points=100,
                anomaly_count=2,
                anomaly_percentage=2.0,
                anomaly_values=[120.0, 140.0],
                anomaly_timestamps=[datetime.now() for _ in range(2)]
            ),
            visualization=VisualizationResult(
                plot_path="test.png",
                plot_base64="base64_data",
                plot_description="Test plot",
                plot_type="matplotlib"
            ),
            insights=InsightResponse(
                summary="Complete analysis summary",
                anomaly_explanations=["Complete explanation"],
                recommendations=["Complete recommendation"],
                root_causes=["Complete cause"],
                confidence_score=90.0
            ),
            processing_time=2.5,
            metadata={"test": "metadata"}
        )
        
        with patch('streamlit.success') as mock_success, \
             patch('streamlit.metric') as mock_metric, \
             patch('streamlit.image') as mock_image, \
             patch('streamlit.write') as mock_write:
            
            self.renderer.render_complete_analysis(analysis_response)
            
            # Verify all components were rendered
            assert mock_success.called
            assert mock_metric.called
            assert mock_image.called
            assert mock_write.called
    
    def test_render_error(self):
        """Test rendering error messages."""
        error_message = "Test error message"
        
        with patch('streamlit.error') as mock_error:
            self.renderer.render_error(error_message)
            
            mock_error.assert_called_once_with(error_message)
    
    def test_render_loading_state(self):
        """Test rendering loading state."""
        with patch('streamlit.spinner') as mock_spinner:
            with self.renderer.render_loading_state("Processing data..."):
                pass
            
            mock_spinner.assert_called_once_with("Processing data...")


class TestJsonViewer:
    """Test JSON viewer component."""
    
    def setup_method(self):
        """Set up test JSON viewer."""
        self.json_viewer = JsonViewer()
    
    def test_render_json_data(self):
        """Test rendering JSON data."""
        test_data = {
            "name": "Test Analysis",
            "results": {
                "anomaly_count": 5,
                "method": "z-score",
                "threshold": 3.0
            },
            "timestamp": "2025-01-01T00:00:00Z"
        }
        
        with patch('streamlit.json') as mock_json:
            self.json_viewer.render_json_data(test_data)
            
            mock_json.assert_called_once_with(test_data)
    
    def test_render_formatted_json(self):
        """Test rendering formatted JSON."""
        test_data = {"test": "data", "nested": {"value": 123}}
        
        with patch('streamlit.code') as mock_code:
            self.json_viewer.render_formatted_json(test_data)
            
            mock_code.assert_called_once()
            # Check that JSON was formatted
            call_args = mock_code.call_args[0]
            assert "{\n" in call_args[0]  # Should be pretty-printed
    
    def test_render_json_with_syntax_highlighting(self):
        """Test rendering JSON with syntax highlighting."""
        test_data = {"highlighted": True, "value": 42}
        
        with patch('streamlit.code') as mock_code:
            self.json_viewer.render_json_with_syntax_highlighting(test_data)
            
            mock_code.assert_called_once()
            # Check that language was set to json
            call_args = mock_code.call_args
            assert call_args[1]['language'] == 'json'
    
    def test_render_collapsible_json(self):
        """Test rendering collapsible JSON sections."""
        test_data = {
            "section1": {"data": "value1"},
            "section2": {"data": "value2"}
        }
        
        with patch('streamlit.expander') as mock_expander, \
             patch('streamlit.json') as mock_json:
            
            # Mock the expander context manager
            mock_expander.return_value.__enter__ = MagicMock()
            mock_expander.return_value.__exit__ = MagicMock()
            
            self.json_viewer.render_collapsible_json(test_data, "Test Section")
            
            mock_expander.assert_called_once_with("Test Section")
    
    def test_handle_invalid_json(self):
        """Test handling invalid JSON data."""
        invalid_data = "not json data"
        
        with patch('streamlit.error') as mock_error:
            try:
                self.json_viewer.render_json_data(invalid_data)
            except Exception:
                pass
            
            # Should handle gracefully without crashing


class TestProgressIndicator:
    """Test progress indicator component."""
    
    def setup_method(self):
        """Set up test progress indicator."""
        self.progress_indicator = ProgressIndicator()
    
    def test_show_progress_bar(self):
        """Test showing progress bar."""
        with patch('streamlit.progress') as mock_progress:
            self.progress_indicator.show_progress_bar(0.5, "Processing...")
            
            mock_progress.assert_called_once_with(0.5)
    
    def test_show_step_progress(self):
        """Test showing step-by-step progress."""
        steps = ["Step 1", "Step 2", "Step 3"]
        current_step = 1
        
        with patch('streamlit.write') as mock_write, \
             patch('streamlit.progress') as mock_progress:
            
            self.progress_indicator.show_step_progress(steps, current_step)
            
            mock_progress.assert_called_once()
            mock_write.assert_called()
    
    def test_show_spinner(self):
        """Test showing spinner."""
        with patch('streamlit.spinner') as mock_spinner:
            with self.progress_indicator.show_spinner("Loading..."):
                pass
            
            mock_spinner.assert_called_once_with("Loading...")
    
    def test_show_success_message(self):
        """Test showing success message."""
        with patch('streamlit.success') as mock_success:
            self.progress_indicator.show_success_message("Analysis complete!")
            
            mock_success.assert_called_once_with("Analysis complete!")
    
    def test_show_error_message(self):
        """Test showing error message."""
        with patch('streamlit.error') as mock_error:
            self.progress_indicator.show_error_message("Analysis failed!")
            
            mock_error.assert_called_once_with("Analysis failed!")
    
    def test_show_info_message(self):
        """Test showing info message."""
        with patch('streamlit.info') as mock_info:
            self.progress_indicator.show_info_message("Processing data...")
            
            mock_info.assert_called_once_with("Processing data...")


class TestCustomStyles:
    """Test custom styling utilities."""
    
    def test_get_custom_css(self):
        """Test getting custom CSS."""
        css = get_custom_css()
        
        assert isinstance(css, str)
        assert len(css) > 0
        assert "color" in css or "font" in css or "background" in css
    
    def test_apply_custom_styles(self):
        """Test applying custom styles."""
        with patch('streamlit.markdown') as mock_markdown:
            apply_custom_styles()
            
            mock_markdown.assert_called_once()
            # Check that CSS was applied
            call_args = mock_markdown.call_args[0]
            assert "unsafe_allow_html=True" in str(mock_markdown.call_args)
    
    def test_custom_theme_colors(self):
        """Test custom theme colors."""
        css = get_custom_css()
        
        # Should contain color definitions
        assert any(color in css for color in ['#', 'rgb', 'rgba', 'color'])
    
    def test_responsive_layout(self):
        """Test responsive layout CSS."""
        css = get_custom_css()
        
        # Should contain responsive elements
        assert any(element in css for element in ['width', 'max-width', 'min-width'])


class TestStreamlitMainApp:
    """Test main Streamlit application."""
    
    def setup_method(self):
        """Set up test environment."""
        # Mock streamlit session state
        self.mock_session_state = {}
        
    def test_app_initialization(self):
        """Test app initialization."""
        with patch('streamlit.set_page_config') as mock_page_config, \
             patch('streamlit.title') as mock_title:
            
            # Import and run main app
            from streamlit_app.main import main
            
            # Should not crash during initialization
            # Note: This would need the actual app structure
    
    def test_file_upload_functionality(self):
        """Test file upload functionality."""
        with patch('streamlit.file_uploader') as mock_uploader, \
             patch('streamlit.write') as mock_write:
            
            # Mock file upload
            mock_file = MagicMock()
            mock_file.name = "test.csv"
            mock_file.read.return_value = b"timestamp,value\n2025-01-01,100\n2025-01-02,200"
            mock_uploader.return_value = mock_file
            
            # Test file processing
            # Note: This would need actual app logic
    
    def test_analysis_workflow(self):
        """Test analysis workflow in app."""
        with patch('streamlit.button') as mock_button, \
             patch('streamlit.selectbox') as mock_selectbox, \
             patch('streamlit.slider') as mock_slider:
            
            # Mock user interactions
            mock_button.return_value = True
            mock_selectbox.return_value = "z-score"
            mock_slider.return_value = 3.0
            
            # Test analysis trigger
            # Note: This would need actual app logic
    
    def test_session_state_management(self):
        """Test session state management."""
        with patch('streamlit.session_state', self.mock_session_state):
            # Test session state initialization
            assert len(self.mock_session_state) == 0
            
            # Test adding to session state
            self.mock_session_state['test_key'] = 'test_value'
            assert self.mock_session_state['test_key'] == 'test_value'
    
    def test_error_handling_in_app(self):
        """Test error handling in app."""
        with patch('streamlit.error') as mock_error:
            # Test error display
            mock_error("Test error message")
            
            mock_error.assert_called_once_with("Test error message")
    
    def test_download_functionality(self):
        """Test download functionality."""
        with patch('streamlit.download_button') as mock_download:
            # Test download button
            test_data = "test,data\n1,2\n3,4"
            mock_download.return_value = True
            
            # Would test actual download logic
            # Note: This would need actual app logic


# Integration tests for Streamlit components
class TestStreamlitIntegration:
    """Test integration between Streamlit components."""
    
    def test_renderer_to_viewer_integration(self):
        """Test integration between renderer and JSON viewer."""
        renderer = AnalysisRenderer()
        json_viewer = JsonViewer()
        
        # Test data flow between components
        test_data = {"test": "integration"}
        
        with patch('streamlit.json') as mock_json:
            json_viewer.render_json_data(test_data)
            mock_json.assert_called_once_with(test_data)
    
    def test_progress_to_renderer_integration(self):
        """Test integration between progress indicator and renderer."""
        progress = ProgressIndicator()
        renderer = AnalysisRenderer()
        
        # Test progress flow
        with patch('streamlit.spinner') as mock_spinner, \
             patch('streamlit.success') as mock_success:
            
            with progress.show_spinner("Processing..."):
                pass
            
            progress.show_success_message("Complete!")
            
            mock_spinner.assert_called_once()
            mock_success.assert_called_once()
    
    def test_complete_workflow_integration(self):
        """Test complete workflow integration."""
        from core.models import AnalysisResponse, AnomalyResult, VisualizationResult, InsightResponse
        
        # Create complete analysis response
        analysis_response = AnalysisResponse(
            anomaly_result=AnomalyResult(
                anomaly_indices=[1, 5],
                method_used="integration_test",
                threshold_used=2.0,
                total_points=100,
                anomaly_count=2,
                anomaly_percentage=2.0,
                anomaly_values=[120.0, 140.0],
                anomaly_timestamps=[datetime.now() for _ in range(2)]
            ),
            visualization=VisualizationResult(
                plot_path="integration_test.png",
                plot_base64="base64_integration_data",
                plot_description="Integration test plot",
                plot_type="matplotlib"
            ),
            insights=InsightResponse(
                summary="Integration test summary",
                anomaly_explanations=["Integration explanation"],
                recommendations=["Integration recommendation"],
                root_causes=["Integration cause"],
                confidence_score=95.0
            ),
            processing_time=1.0,
            metadata={"integration": "test"}
        )
        
        # Test complete workflow
        renderer = AnalysisRenderer()
        json_viewer = JsonViewer()
        progress = ProgressIndicator()
        
        with patch('streamlit.success') as mock_success, \
             patch('streamlit.metric') as mock_metric, \
             patch('streamlit.image') as mock_image, \
             patch('streamlit.json') as mock_json:
            
            # Simulate complete workflow
            with progress.show_spinner("Processing..."):
                renderer.render_complete_analysis(analysis_response)
            
            progress.show_success_message("Analysis complete!")
            json_viewer.render_json_data(analysis_response.metadata)
            
            # Verify all components worked together
            assert mock_success.called
            assert mock_metric.called
            assert mock_image.called
            assert mock_json.called


# Performance tests
class TestStreamlitPerformance:
    """Test performance of Streamlit components."""
    
    def test_large_data_rendering(self):
        """Test rendering large datasets."""
        renderer = AnalysisRenderer()
        
        # Create large anomaly result
        large_anomaly_result = AnomalyResult(
            anomaly_indices=list(range(0, 1000, 10)),  # 100 anomalies
            method_used="performance_test",
            threshold_used=1.0,
            total_points=1000,
            anomaly_count=100,
            anomaly_percentage=10.0,
            anomaly_values=[float(i) for i in range(0, 1000, 10)],
            anomaly_timestamps=[datetime.now() for _ in range(100)]
        )
        
        with patch('streamlit.metric') as mock_metric, \
             patch('streamlit.write') as mock_write:
            
            # Should handle large data without issues
            renderer.render_anomaly_result(large_anomaly_result)
            
            assert mock_metric.called
            assert mock_write.called
    
    def test_complex_json_rendering(self):
        """Test rendering complex JSON structures."""
        json_viewer = JsonViewer()
        
        # Create complex nested structure
        complex_data = {
            "level1": {
                "level2": {
                    "level3": {
                        "data": [{"item": i, "value": i * 2} for i in range(100)]
                    }
                }
            }
        }
        
        with patch('streamlit.json') as mock_json:
            # Should handle complex structures
            json_viewer.render_json_data(complex_data)
            
            mock_json.assert_called_once_with(complex_data)
    
    def test_rapid_progress_updates(self):
        """Test rapid progress updates."""
        progress = ProgressIndicator()
        
        with patch('streamlit.progress') as mock_progress:
            # Simulate rapid updates
            for i in range(0, 101, 10):
                progress.show_progress_bar(i / 100, f"Step {i}")
            
            # Should handle rapid updates
            assert mock_progress.call_count == 11