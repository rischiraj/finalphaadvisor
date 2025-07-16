"""
Tests for anomaly detection agents and supervisor.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from pathlib import Path

from agents.supervisor import AnomalyDetectionSupervisor
from agents.anomaly_agent import AnomalyAgent
from agents.conversation_manager import ConversationManager
from agents.conversation_workflow import ConversationWorkflow
from agents.enhanced_suggestion_agent import EnhancedSuggestionAgent
from agents.llm_logger import LLMLogger
from core.models import (
    AnomalyDetectionRequest, TimeSeriesData, AnomalyResult, 
    VisualizationResult, InsightResponse, AnalysisResponse
)
from core.exceptions import AnomalyDetectionError, LLMError, AgentError
from core.config import Settings


class TestAnomalyDetectionSupervisor:
    """Test the main supervisor for anomaly detection."""
    
    def setup_method(self):
        """Set up test supervisor."""
        self.test_settings = Settings(
            google_ai_api_key="test_key_12345",
            llm_model="gemini-2.0-flash",
            llm_temperature=0.1
        )
        self.supervisor = AnomalyDetectionSupervisor(self.test_settings)
    
    @pytest.mark.asyncio
    async def test_analyze_with_file_path(self, sample_csv_file):
        """Test analysis with file path."""
        request = AnomalyDetectionRequest(
            file_path=sample_csv_file,
            method="z-score",
            threshold=3.0,
            query="Test file analysis"
        )
        
        with patch('agents.tools.file_reader.FileReaderTool.read_file') as mock_read_file, \
             patch('agents.tools.anomaly_detector.AnomalyDetectionTool.detect_anomalies') as mock_detect, \
             patch('agents.tools.visualizer.VisualizationTool.create_anomaly_plot') as mock_viz, \
             patch('agents.tools.intelligent_insight_generator.IntelligentInsightGenerator.generate_insights') as mock_insights:
            
            # Mock file reading
            mock_time_series = self._create_mock_time_series()
            mock_read_file.return_value = mock_time_series
            
            # Mock anomaly detection
            mock_anomaly_result = self._create_mock_anomaly_result()
            mock_detect.return_value = mock_anomaly_result
            
            # Mock visualization
            mock_viz_result = self._create_mock_visualization_result()
            mock_viz.return_value = mock_viz_result
            
            # Mock insights
            mock_insight_result = self._create_mock_insight_response()
            mock_insights.return_value = mock_insight_result
            
            # Run analysis
            result = await self.supervisor.analyze(request)
            
            # Verify result
            assert isinstance(result, AnalysisResponse)
            assert result.anomaly_result == mock_anomaly_result
            assert result.visualization == mock_viz_result
            assert result.insights == mock_insight_result
            assert result.processing_time > 0
    
    @pytest.mark.asyncio
    async def test_analyze_with_direct_data(self, sample_time_series_data):
        """Test analysis with direct data."""
        request = AnomalyDetectionRequest(
            data=sample_time_series_data,
            method="iqr",
            threshold=1.5,
            query="Test direct data analysis"
        )
        
        with patch('agents.tools.anomaly_detector.AnomalyDetectionTool.detect_anomalies') as mock_detect, \
             patch('agents.tools.visualizer.VisualizationTool.create_anomaly_plot') as mock_viz, \
             patch('agents.tools.intelligent_insight_generator.IntelligentInsightGenerator.generate_insights') as mock_insights:
            
            # Mock responses
            mock_detect.return_value = self._create_mock_anomaly_result()
            mock_viz.return_value = self._create_mock_visualization_result()
            mock_insights.return_value = self._create_mock_insight_response()
            
            # Run analysis
            result = await self.supervisor.analyze(request)
            
            # Verify result
            assert isinstance(result, AnalysisResponse)
            assert result.processing_time > 0
    
    @pytest.mark.asyncio
    async def test_analyze_error_handling(self):
        """Test error handling in analysis."""
        request = AnomalyDetectionRequest(
            file_path="non_existent_file.csv",
            method="z-score",
            query="Test error handling"
        )
        
        with patch('agents.tools.file_reader.FileReaderTool.read_file') as mock_read_file:
            mock_read_file.side_effect = FileNotFoundError("File not found")
            
            with pytest.raises(AnomalyDetectionError):
                await self.supervisor.analyze(request)
    
    def _create_mock_time_series(self):
        """Create mock time series data."""
        timestamps = [datetime.now() + timedelta(hours=i) for i in range(10)]
        values = [10.0 + i + (50 if i == 5 else 0) for i in range(10)]
        return TimeSeriesData(
            timestamp=timestamps,
            values=values,
            column_name="mock_values"
        )
    
    def _create_mock_anomaly_result(self):
        """Create mock anomaly result."""
        return AnomalyResult(
            anomaly_indices=[5],
            method_used="z-score",
            threshold_used=3.0,
            total_points=10,
            anomaly_count=1,
            anomaly_percentage=10.0,
            anomaly_values=[60.0],
            anomaly_timestamps=[datetime.now()]
        )
    
    def _create_mock_visualization_result(self):
        """Create mock visualization result."""
        return VisualizationResult(
            plot_path="test_plot.png",
            plot_base64="base64_encoded_image",
            plot_description="Test plot description",
            plot_type="matplotlib"
        )
    
    def _create_mock_insight_response(self):
        """Create mock insight response."""
        return InsightResponse(
            summary="Test analysis summary",
            anomaly_explanations=["Test explanation"],
            recommendations=["Test recommendation"],
            root_causes=["Test cause"],
            confidence_score=85.0
        )


class TestAnomalyAgent:
    """Test the anomaly detection agent."""
    
    def setup_method(self):
        """Set up test agent."""
        self.test_settings = Settings(
            google_ai_api_key="test_key_12345",
            llm_model="gemini-2.0-flash"
        )
        self.agent = AnomalyAgent(self.test_settings)
    
    @pytest.mark.asyncio
    async def test_analyze_anomalies(self, sample_time_series_data):
        """Test anomaly analysis."""
        with patch('agents.tools.anomaly_detector.AnomalyDetectionTool.detect_anomalies') as mock_detect:
            mock_result = AnomalyResult(
                anomaly_indices=[2, 5],
                method_used="z-score",
                threshold_used=3.0,
                total_points=100,
                anomaly_count=2,
                anomaly_percentage=2.0,
                anomaly_values=[125.0, 150.0],
                anomaly_timestamps=[datetime.now(), datetime.now()]
            )
            mock_detect.return_value = mock_result
            
            result = await self.agent.analyze_anomalies(
                sample_time_series_data,
                method="z-score",
                threshold=3.0
            )
            
            assert isinstance(result, AnomalyResult)
            assert result.method_used == "z-score"
            assert result.anomaly_count == 2
    
    @pytest.mark.asyncio
    async def test_recommend_method(self, sample_time_series_data):
        """Test method recommendation."""
        with patch('agents.tools.anomaly_detector.AnomalyDetectionTool.recommend_method') as mock_recommend:
            mock_recommend.return_value = "iqr"
            
            recommended = await self.agent.recommend_method(sample_time_series_data)
            
            assert recommended == "iqr"
            mock_recommend.assert_called_once_with(sample_time_series_data)
    
    @pytest.mark.asyncio
    async def test_analyze_error_handling(self, sample_time_series_data):
        """Test error handling in agent."""
        with patch('agents.tools.anomaly_detector.AnomalyDetectionTool.detect_anomalies') as mock_detect:
            mock_detect.side_effect = Exception("Detection failed")
            
            with pytest.raises(AgentError):
                await self.agent.analyze_anomalies(
                    sample_time_series_data,
                    method="z-score"
                )


class TestConversationManager:
    """Test conversation management functionality."""
    
    def setup_method(self):
        """Set up test conversation manager."""
        self.test_settings = Settings(
            google_ai_api_key="test_key_12345",
            llm_model="gemini-2.0-flash"
        )
        self.conversation_manager = ConversationManager(self.test_settings)
    
    @pytest.mark.asyncio
    async def test_start_conversation(self):
        """Test starting a new conversation."""
        session_id = await self.conversation_manager.start_conversation()
        
        assert isinstance(session_id, str)
        assert len(session_id) > 0
        assert session_id in self.conversation_manager.conversations
    
    @pytest.mark.asyncio
    async def test_process_message(self):
        """Test processing a message."""
        session_id = await self.conversation_manager.start_conversation()
        
        with patch('agents.enhanced_suggestion_agent.EnhancedSuggestionAgent.process_message') as mock_process:
            mock_process.return_value = "Test response"
            
            response = await self.conversation_manager.process_message(
                session_id,
                "Test message",
                context={"test": "context"}
            )
            
            assert response == "Test response"
            mock_process.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_conversation_history(self):
        """Test getting conversation history."""
        session_id = await self.conversation_manager.start_conversation()
        
        # Add some messages
        await self.conversation_manager.process_message(session_id, "Message 1")
        await self.conversation_manager.process_message(session_id, "Message 2")
        
        history = await self.conversation_manager.get_conversation_history(session_id)
        
        assert isinstance(history, list)
        assert len(history) >= 2
    
    @pytest.mark.asyncio
    async def test_invalid_session_id(self):
        """Test handling of invalid session ID."""
        with pytest.raises(ValueError, match="Session not found"):
            await self.conversation_manager.process_message(
                "invalid_session_id",
                "Test message"
            )


class TestConversationWorkflow:
    """Test conversation workflow functionality."""
    
    def setup_method(self):
        """Set up test workflow."""
        self.test_settings = Settings(
            google_ai_api_key="test_key_12345",
            llm_model="gemini-2.0-flash"
        )
        self.workflow = ConversationWorkflow(self.test_settings)
    
    @pytest.mark.asyncio
    async def test_process_user_input(self):
        """Test processing user input."""
        with patch('agents.enhanced_suggestion_agent.EnhancedSuggestionAgent.process_message') as mock_process:
            mock_process.return_value = "Test workflow response"
            
            response = await self.workflow.process_user_input(
                "Test input",
                context={"analysis_result": "test_result"}
            )
            
            assert response == "Test workflow response"
    
    @pytest.mark.asyncio
    async def test_workflow_state_management(self):
        """Test workflow state management."""
        initial_state = self.workflow.get_current_state()
        
        assert isinstance(initial_state, dict)
        assert "messages" in initial_state
        assert "context" in initial_state
        
        # Update state
        self.workflow.update_state({"test_key": "test_value"})
        
        updated_state = self.workflow.get_current_state()
        assert updated_state["test_key"] == "test_value"


class TestEnhancedSuggestionAgent:
    """Test enhanced suggestion agent functionality."""
    
    def setup_method(self):
        """Set up test agent."""
        self.test_settings = Settings(
            google_ai_api_key="test_key_12345",
            llm_model="gemini-2.0-flash"
        )
        self.agent = EnhancedSuggestionAgent(self.test_settings)
    
    @pytest.mark.asyncio
    async def test_process_message(self, disable_llm_calls):
        """Test processing a message."""
        response = await self.agent.process_message(
            "What insights can you provide about this anomaly?",
            context={"anomaly_count": 5, "method": "z-score"}
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    @pytest.mark.asyncio
    async def test_generate_suggestions(self, disable_llm_calls):
        """Test generating suggestions."""
        analysis_context = {
            "anomaly_count": 10,
            "method_used": "iqr",
            "data_points": 1000
        }
        
        suggestions = await self.agent.generate_suggestions(
            "Test query",
            analysis_context
        )
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert all(isinstance(s, str) for s in suggestions)
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in suggestion agent."""
        with patch('langchain_google_genai.ChatGoogleGenerativeAI.ainvoke') as mock_llm:
            mock_llm.side_effect = Exception("LLM Error")
            
            with pytest.raises(LLMError):
                await self.agent.process_message(
                    "Test message",
                    context={"test": "context"}
                )


class TestLLMLogger:
    """Test LLM logging functionality."""
    
    def setup_method(self):
        """Set up test logger."""
        self.test_settings = Settings(
            google_ai_api_key="test_key_12345",
            llm_model="gemini-2.0-flash"
        )
        self.logger = LLMLogger(self.test_settings)
    
    def test_log_request(self):
        """Test logging LLM requests."""
        request_data = {
            "prompt": "Test prompt",
            "model": "gemini-2.0-flash",
            "temperature": 0.1
        }
        
        self.logger.log_request("test_session", request_data)
        
        # Check that request was logged
        assert len(self.logger.get_session_logs("test_session")) > 0
    
    def test_log_response(self):
        """Test logging LLM responses."""
        response_data = {
            "content": "Test response",
            "usage": {"tokens": 100},
            "response_time": 1.5
        }
        
        self.logger.log_response("test_session", response_data)
        
        # Check that response was logged
        logs = self.logger.get_session_logs("test_session")
        assert len(logs) > 0
        assert logs[0]["type"] == "response"
    
    def test_get_session_stats(self):
        """Test getting session statistics."""
        # Log some requests and responses
        self.logger.log_request("test_session", {"prompt": "Test 1"})
        self.logger.log_response("test_session", {"content": "Response 1", "usage": {"tokens": 50}})
        self.logger.log_request("test_session", {"prompt": "Test 2"})
        self.logger.log_response("test_session", {"content": "Response 2", "usage": {"tokens": 75}})
        
        stats = self.logger.get_session_stats("test_session")
        
        assert stats["total_requests"] == 2
        assert stats["total_responses"] == 2
        assert stats["total_tokens"] == 125
        assert stats["avg_response_time"] >= 0
    
    def test_clear_session_logs(self):
        """Test clearing session logs."""
        self.logger.log_request("test_session", {"prompt": "Test"})
        assert len(self.logger.get_session_logs("test_session")) > 0
        
        self.logger.clear_session_logs("test_session")
        assert len(self.logger.get_session_logs("test_session")) == 0
    
    def test_export_logs(self):
        """Test exporting logs."""
        self.logger.log_request("test_session", {"prompt": "Test"})
        self.logger.log_response("test_session", {"content": "Response"})
        
        exported = self.logger.export_logs("test_session")
        
        assert isinstance(exported, str)
        assert "test_session" in exported
        assert "Request" in exported
        assert "Response" in exported


# Integration tests
class TestAgentIntegration:
    """Test integration between different agents."""
    
    def setup_method(self):
        """Set up integration tests."""
        self.test_settings = Settings(
            google_ai_api_key="test_key_12345",
            llm_model="gemini-2.0-flash"
        )
    
    @pytest.mark.asyncio
    async def test_supervisor_to_agent_integration(self, sample_time_series_data):
        """Test integration between supervisor and agents."""
        supervisor = AnomalyDetectionSupervisor(self.test_settings)
        
        request = AnomalyDetectionRequest(
            data=sample_time_series_data,
            method="z-score",
            query="Integration test"
        )
        
        with patch('agents.tools.anomaly_detector.AnomalyDetectionTool.detect_anomalies') as mock_detect, \
             patch('agents.tools.visualizer.VisualizationTool.create_anomaly_plot') as mock_viz, \
             patch('agents.tools.intelligent_insight_generator.IntelligentInsightGenerator.generate_insights') as mock_insights:
            
            # Mock all the tools
            mock_detect.return_value = AnomalyResult(
                anomaly_indices=[5],
                method_used="z-score",
                threshold_used=3.0,
                total_points=100,
                anomaly_count=1,
                anomaly_percentage=1.0,
                anomaly_values=[60.0],
                anomaly_timestamps=[datetime.now()]
            )
            
            mock_viz.return_value = VisualizationResult(
                plot_path="test.png",
                plot_base64="base64_data",
                plot_description="Test plot",
                plot_type="matplotlib"
            )
            
            mock_insights.return_value = InsightResponse(
                summary="Integration test summary",
                anomaly_explanations=["Integration explanation"],
                recommendations=["Integration recommendation"],
                root_causes=["Integration cause"],
                confidence_score=90.0
            )
            
            # Run integration test
            result = await supervisor.analyze(request)
            
            assert isinstance(result, AnalysisResponse)
            assert result.anomaly_result.method_used == "z-score"
            assert result.insights.confidence_score == 90.0
    
    @pytest.mark.asyncio
    async def test_conversation_to_analysis_integration(self):
        """Test integration between conversation and analysis."""
        conversation_manager = ConversationManager(self.test_settings)
        session_id = await conversation_manager.start_conversation()
        
        with patch('agents.enhanced_suggestion_agent.EnhancedSuggestionAgent.process_message') as mock_process:
            mock_process.return_value = "Analysis complete. Found 3 anomalies."
            
            response = await conversation_manager.process_message(
                session_id,
                "Analyze this data for anomalies",
                context={"analysis_result": "test_result"}
            )
            
            assert "anomalies" in response
            assert isinstance(response, str)