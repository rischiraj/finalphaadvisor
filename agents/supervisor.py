"""
Supervisor agent for coordinating multi-agent anomaly detection workflow using LangGraph.
"""

import logging
from typing import Dict, Any, Optional
import time
import asyncio

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI

from core.config import get_settings
from core.exceptions import AgentError
from core.models import (
    AnomalyDetectionRequest,
    AnomalyResult,
    TimeSeriesData,
    VisualizationResult,
    InsightResponse,
    AnalysisResponse,
    AgentState
)
from agents.anomaly_agent import AnomalyAgent
from agents.enhanced_suggestion_agent import EnhancedSuggestionAgent


logger = logging.getLogger(__name__)


class AnomalyDetectionSupervisor:
    """
    Supervisor agent that orchestrates the multi-agent anomaly detection workflow.
    """
    
    def __init__(self, enable_llm: bool = False):
        """Initialize the supervisor with LangGraph workflow."""
        self.logger = logging.getLogger(__name__)
        self.settings = get_settings()
        self.enable_llm = enable_llm
        
        
        # Initialize LLM only if enabled
        if enable_llm:
            self.llm = ChatGoogleGenerativeAI(
                model=self.settings.llm_model,
                temperature=self.settings.llm_temperature,
                google_api_key=self.settings.google_ai_api_key
            )
        else:
            self.llm = None
        
        # Initialize agents
        self.anomaly_agent = AnomalyAgent(self.llm)
        self.suggestion_agent = EnhancedSuggestionAgent(self.llm) if enable_llm else None
        
        # Create workflow
        self.workflow = self._create_workflow()
        self.app = self.workflow.compile(checkpointer=MemorySaver())
        
        self.logger.info(f"Anomaly detection supervisor initialized successfully (LLM: {'enabled' if enable_llm else 'disabled'})")
    
    def _create_workflow(self) -> StateGraph:
        """
        Create the LangGraph workflow for multi-agent coordination.
        
        Returns:
            StateGraph: Configured workflow
        """
        # Define the workflow state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("anomaly_detection", self._run_anomaly_detection)
        workflow.add_node("insight_generation", self._run_insight_generation)
        workflow.add_node("error_handling", self._handle_errors)
        workflow.add_node("finalize_results", self._finalize_results)
        
        # Set entry point
        workflow.set_entry_point("anomaly_detection")
        
        # Add edges - Include insight generation if LLM is enabled
        if self.enable_llm:
            workflow.add_conditional_edges(
                "anomaly_detection",
                self._should_continue_after_anomaly_detection,
                {
                    "continue": "insight_generation",  # Include insight generation
                    "error": "error_handling",
                    "end": END
                }
            )
        else:
            workflow.add_conditional_edges(
                "anomaly_detection",
                self._should_continue_after_anomaly_detection,
                {
                    "continue": "finalize_results",  # Skip insight generation
                    "error": "error_handling",
                    "end": END
                }
            )
        
        # Add insight generation edges if LLM is enabled
        if self.enable_llm:
            workflow.add_conditional_edges(
                "insight_generation",
                self._should_continue_after_insight_generation,
                {
                    "continue": "finalize_results",
                    "error": "error_handling",
                    "end": END
                }
            )
        
        workflow.add_edge("error_handling", END)
        workflow.add_edge("finalize_results", END)
        
        return workflow
    
    async def _run_anomaly_detection(self, state: AgentState) -> AgentState:
        """
        Run the anomaly detection agent.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        try:
            self.logger.info("Starting anomaly detection phase")
            
            # Process the request using anomaly agent
            result = await self.anomaly_agent.process_request(state.request)
            
            # Extract results
            if result.get("successful") and "anomaly_result" in result:
                state.anomaly_result = result["anomaly_result"]
                state.visualization = result.get("visualization")
                
                # Store data if processed from file
                if "data_info" in result and state.data is None:
                    # Reconstruct data object from available info
                    # In a real implementation, this would be passed more directly
                    if hasattr(self.anomaly_agent, '_current_data'):
                        state.data = self.anomaly_agent._current_data
                
                state.status = "processing"
                self.logger.info(f"Anomaly detection completed: {state.anomaly_result.anomaly_count} anomalies found")
            else:
                error_msg = result.get("error") or "Unknown error in anomaly detection"
                if error_msg and error_msg != "None":  # Don't add None or "None" strings
                    state.errors.append(str(error_msg))
                state.status = "failed"
                self.logger.error(f"Anomaly detection failed: {error_msg}")
            
        except Exception as e:
            error_msg = f"Error in anomaly detection: {str(e)}"
            state.errors.append(error_msg)
            state.status = "failed"
            self.logger.error(error_msg)
        
        return state
    
    async def _run_insight_generation(self, state: AgentState) -> AgentState:
        """
        Run the suggestion agent to generate insights.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        try:
            self.logger.info("Starting insight generation phase")
            
            if not state.data or not state.anomaly_result:
                error_msg = "Missing data or anomaly results for insight generation"
                state.errors.append(error_msg)
                state.status = "failed"
                return state
            
            # Generate insights using enhanced suggestion agent
            result = await self.suggestion_agent.process_results(
                state.data,
                state.anomaly_result,
                state.request.query,
                f"Analysis using {state.anomaly_result.method_used} method"
            )
            
            # Extract structured insights directly (no parsing needed!)
            if result.get("successful") and "insights" in result:
                state.insights = result["insights"]
                self.logger.info("Enhanced insight generation completed successfully")
            else:
                error_msg = f"Failed to generate insights: {result.get('error', 'Unknown error')}"
                state.errors.append(error_msg)
                self.logger.error(error_msg)
            
        except Exception as e:
            error_msg = f"Error in insight generation: {str(e)}"
            state.errors.append(error_msg)
            self.logger.error(error_msg)
        
        return state
    
    async def _handle_errors(self, state: AgentState) -> AgentState:
        """
        Handle errors in the workflow.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        self.logger.warning(f"Handling workflow errors: {state.errors}")
        
        # Create minimal results for failed cases
        if not state.anomaly_result and state.data:
            # Create empty anomaly result
            state.anomaly_result = AnomalyResult(
                anomaly_indices=[],
                method_used=state.request.method,
                threshold_used=0.0,
                total_points=len(state.data.values),
                anomaly_count=0,
                anomaly_percentage=0.0,
                anomaly_values=[],
                anomaly_timestamps=[]
            )
        
        if not state.insights:
            # Create basic insights
            state.insights = InsightResponse(
                summary=f"Analysis encountered errors: {'; '.join(state.errors)}",
                anomaly_explanations=["Analysis could not be completed due to errors"],
                recommendations=["Please check the data and try again"],
                root_causes=["Technical issues prevented complete analysis"],
                confidence_score=0
            )
        
        state.status = "failed"
        return state
    
    async def _finalize_results(self, state: AgentState) -> AgentState:
        """
        Finalize the analysis results.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        try:
            self.logger.info("Finalizing analysis results")
            
            # Create default insights without LLM calls
            if not state.insights and state.anomaly_result:
                state.insights = InsightResponse(
                    summary=f"Anomaly detection completed using {state.anomaly_result.method_used} method. "
                           f"Found {state.anomaly_result.anomaly_count} anomalies out of {state.anomaly_result.total_points} data points "
                           f"({state.anomaly_result.anomaly_percentage:.2f}%).",
                    anomaly_explanations=[
                        f"Anomalies detected using {state.anomaly_result.method_used} method with threshold {state.anomaly_result.threshold_used}"
                    ],
                    recommendations=[
                        "Review the identified anomalies to understand patterns",
                        "Consider adjusting threshold parameters if needed",
                        "Investigate data quality around anomaly periods"
                    ],
                    root_causes=[
                        "Statistical outliers in the data",
                        "Data collection irregularities",
                        "Natural variations in the process"
                    ],
                    confidence_score=75.0  # Default confidence without LLM analysis
                )
            
            # Ensure we have all required components
            if state.anomaly_result and state.insights:
                state.status = "completed"
                self.logger.info("Analysis completed successfully")
            else:
                state.status = "failed"
                state.errors.append("Missing required results for completion")
                self.logger.error("Analysis completion failed: missing required results")
            
        except Exception as e:
            error_msg = f"Error finalizing results: {str(e)}"
            state.errors.append(error_msg)
            state.status = "failed"
            self.logger.error(error_msg)
        
        return state
    
    def _should_continue_after_anomaly_detection(self, state: AgentState) -> str:
        """
        Determine next step after anomaly detection.
        
        Args:
            state: Current workflow state
            
        Returns:
            Next step decision
        """
        if state.errors:
            return "error"
        elif state.anomaly_result:
            return "continue"
        else:
            return "error"
    
    def _should_continue_after_insight_generation(self, state: AgentState) -> str:
        """
        Determine next step after insight generation.
        
        Args:
            state: Current workflow state
            
        Returns:
            Next step decision
        """
        if state.errors and not state.insights:
            return "error"
        else:
            return "continue"
    
    async def analyze(self, request: AnomalyDetectionRequest) -> AnalysisResponse:
        """
        Run the complete multi-agent anomaly detection analysis.
        
        Args:
            request: Anomaly detection request
            
        Returns:
            Complete analysis response
            
        Raises:
            AgentError: If analysis fails
        """
        try:
            start_time = time.time()
            self.logger.info(f"Starting multi-agent analysis for {request.method} method")
            
            # Create initial state
            initial_state = AgentState(
                request=request,
                data=request.data,
                status="pending"
            )
            
            # Run the workflow
            config = {"configurable": {"thread_id": f"analysis_{int(start_time)}"}}
            result = await self.app.ainvoke(initial_state, config=config)
            
            # LangGraph returns a dict, so we need to convert back to AgentState
            if isinstance(result, dict):
                final_state = AgentState(**result)
            else:
                final_state = result
            
            processing_time = time.time() - start_time
            
            # Validate final state
            if final_state.status == "failed":
                raise AgentError(
                    f"Multi-agent analysis failed: {'; '.join(final_state.errors)}", 
                    "supervisor"
                )
            
            # Ensure required components exist
            if not final_state.anomaly_result:
                raise AgentError("Missing anomaly detection results", "supervisor")
            
            if not final_state.insights:
                raise AgentError("Missing insight generation results", "supervisor")
            
            # Create visualization if missing
            if not final_state.visualization and final_state.data:
                try:
                    from agents.tools.visualizer import VisualizationTool
                    visualizer = VisualizationTool()
                    final_state.visualization = visualizer.create_anomaly_plot(
                        final_state.data,
                        final_state.anomaly_result
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to create visualization: {str(e)}")
                    # Create minimal visualization result
                    final_state.visualization = VisualizationResult(
                        plot_path="",
                        plot_base64="",
                        plot_description="Visualization could not be generated",
                        plot_type="matplotlib"
                    )
            
            # Build response
            response = AnalysisResponse(
                anomaly_result=final_state.anomaly_result,
                visualization=final_state.visualization,
                insights=final_state.insights,
                processing_time=processing_time,
                metadata={
                    "workflow_status": final_state.status,
                    "errors": final_state.errors,
                    "agents_used": ["anomaly_agent", "suggestion_agent"],
                    "method": request.method
                }
            )
            
            self.logger.info(f"Multi-agent analysis completed in {processing_time:.2f} seconds")
            return response
            
        except Exception as e:
            self.logger.error(f"Multi-agent analysis failed: {str(e)}")
            if isinstance(e, AgentError):
                raise
            raise AgentError(f"Supervisor workflow failed: {str(e)}", "supervisor")
    
    async def analyze_with_streaming(
        self, 
        request: AnomalyDetectionRequest,
        callback: Optional[callable] = None
    ) -> AnalysisResponse:
        """
        Run analysis with streaming updates.
        
        Args:
            request: Anomaly detection request
            callback: Optional callback for streaming updates
            
        Returns:
            Complete analysis response
        """
        try:
            start_time = time.time()
            
            # Create initial state
            initial_state = AgentState(
                request=request,
                data=request.data,
                status="pending"
            )
            
            config = {"configurable": {"thread_id": f"stream_{int(start_time)}"}}
            
            # Stream the workflow execution
            async for step in self.app.astream(initial_state, config=config):
                if callback:
                    await callback(step)
                
                # Log progress
                for node_name, node_state in step.items():
                    if hasattr(node_state, 'status'):
                        self.logger.info(f"Node {node_name}: {node_state.status}")
            
            # Get final result
            final_state = await self.app.ainvoke(initial_state, config=config)
            processing_time = time.time() - start_time
            
            # Convert to AnalysisResponse (similar to analyze method)
            return AnalysisResponse(
                anomaly_result=final_state.anomaly_result,
                visualization=final_state.visualization or VisualizationResult(
                    plot_path="", plot_base64="", plot_description="No visualization", plot_type="matplotlib"
                ),
                insights=final_state.insights,
                processing_time=processing_time,
                metadata={
                    "workflow_status": final_state.status,
                    "errors": final_state.errors,
                    "streaming": True
                }
            )
            
        except Exception as e:
            self.logger.error(f"Streaming analysis failed: {str(e)}")
            raise AgentError(f"Streaming workflow failed: {str(e)}", "supervisor")
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """
        Get information about the workflow structure.
        
        Returns:
            Workflow information
        """
        return {
            "name": "Multi-Agent Anomaly Detection Workflow",
            "nodes": ["anomaly_detection", "insight_generation", "error_handling", "finalize_results"],
            "agents": {
                "anomaly_agent": self.anomaly_agent.get_capabilities(),
                "suggestion_agent": self.suggestion_agent.get_capabilities()
            },
            "flow": [
                "anomaly_detection -> insight_generation -> finalize_results",
                "error paths available at each step"
            ]
        }