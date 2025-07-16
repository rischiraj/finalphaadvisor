"""
Anomaly detection agent using LangChain tools for time-series analysis.
"""

import logging
from typing import Dict, Any, Optional
import asyncio

from langchain.agents import create_tool_calling_agent
from langchain.agents.agent import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from core.config import get_settings
from core.prompt_manager import get_prompt_manager
from core.exceptions import AgentError, FileProcessingError, AnomalyDetectionMethodError
from core.models import (
    AnomalyDetectionRequest, 
    AnomalyResult, 
    TimeSeriesData,
    VisualizationResult,
    AgentState
)
from agents.tools.file_reader import FileReaderTool
from agents.tools.anomaly_detector import AnomalyDetectionTool
from agents.tools.visualizer import VisualizationTool


logger = logging.getLogger(__name__)


class AnomalyAgent:
    """
    Primary agent responsible for anomaly detection workflow.
    """
    
    def __init__(self, llm: Optional[ChatGoogleGenerativeAI] = None):
        """
        Initialize the anomaly detection agent.
        
        Args:
            llm: Optional LLM instance (unused - kept for compatibility)
        """
        self.logger = logging.getLogger(__name__)
        self.settings = get_settings()
        self.prompt_manager = get_prompt_manager()
        self.llm = llm
        self.enable_llm = False  # Always disabled - use direct tool execution
        
        # Initialize core tools - no LLM needed for deterministic tasks
        self.file_reader = FileReaderTool()
        self.anomaly_detector = AnomalyDetectionTool()
        self.visualizer = VisualizationTool()
        
        self.logger.info("Anomaly agent initialized successfully (direct tool execution mode)")
    
    def _create_tools(self):
        """
        Create LangChain tools from the detection tools.
        
        Returns:
            List of LangChain tools
        """
        @tool
        def read_file_data(file_path: str, timestamp_column: str = None, value_column: str = None) -> str:
            """
            Read time-series data from a file.
            
            Args:
                file_path: Path to the data file (CSV or Excel)
                timestamp_column: Name of timestamp column (optional, auto-detected)
                value_column: Name of value column (optional, auto-detected)
                
            Returns:
                String description of the loaded data
            """
            try:
                data = self.file_reader.read_file(file_path, timestamp_column, value_column)
                self._current_data = data  # Store data for other tools to access
                return (
                    f"Successfully loaded {len(data.values)} data points from {file_path}. "
                    f"Column: {data.column_name}, "
                    f"Time range: {data.timestamp[0]} to {data.timestamp[-1]}, "
                    f"Value range: {min(data.values):.2f} to {max(data.values):.2f}"
                )
            except Exception as e:
                return f"Error reading file: {str(e)}"
        
        @tool
        def analyze_complete(file_path: str, method: str, threshold: float = None, plot_type: str = 'matplotlib') -> str:
            """
            Complete end-to-end analysis: load data, detect anomalies, and create visualization.
            
            Args:
                file_path: Path to the data file
                method: Detection method ('z-score', 'iqr', 'dbscan')
                threshold: Threshold parameter (optional)
                plot_type: Type of plot ('matplotlib' or 'plotly')
                
            Returns:
                String description of complete analysis results
            """
            try:
                # Step 1: Load data
                data = self.file_reader.read_file(file_path)
                self._current_data = data
                
                # Step 2: Detect anomalies
                result = self.anomaly_detector.detect_anomalies(data, method, threshold)
                self._current_anomaly_result = result
                
                # Step 3: Create visualization
                viz_result = self.visualizer.create_anomaly_plot(data, result, plot_type)
                self._current_visualization = viz_result
                
                return (
                    f"Complete analysis finished successfully!\n"
                    f"Data: {len(data.values)} points from {data.timestamp[0]} to {data.timestamp[-1]}\n"
                    f"Method: {method.upper()} with threshold {result.threshold_used}\n"
                    f"Results: {result.anomaly_count} anomalies found ({result.anomaly_percentage:.1f}%)\n"
                    f"Visualization: {viz_result.plot_path}"
                )
            except Exception as e:
                return f"Error in complete analysis: {str(e)}"
        
        @tool
        def detect_anomalies(method: str, threshold: float = None, **kwargs) -> str:
            """
            Detect anomalies in the loaded time-series data.
            
            Args:
                method: Detection method ('z-score', 'iqr', 'dbscan')
                threshold: Threshold parameter (method-specific)
                **kwargs: Additional parameters for specific methods
                
            Returns:
                String description of detection results
            """
            try:
                if not hasattr(self, '_current_data'):
                    return "No data loaded. Please load data first using read_file_data."
                
                result = self.anomaly_detector.detect_anomalies(
                    self._current_data, method, threshold, **kwargs
                )
                
                self._current_anomaly_result = result
                
                return (
                    f"Anomaly detection completed using {method.upper()} method. "
                    f"Found {result.anomaly_count} anomalies out of {result.total_points} points "
                    f"({result.anomaly_percentage:.1f}%). "
                    f"Threshold used: {result.threshold_used}"
                )
            except Exception as e:
                return f"Error in anomaly detection: {str(e)}"
        
        @tool
        def create_visualization(plot_type: str = 'matplotlib', title: str = None) -> str:
            """
            Create a visualization of the time-series data with highlighted anomalies.
            
            Args:
                plot_type: Type of plot ('matplotlib' or 'plotly')
                title: Custom title for the plot
                
            Returns:
                String description of the created visualization
            """
            try:
                if not hasattr(self, '_current_data'):
                    return "No data loaded. Please load data first."
                
                if not hasattr(self, '_current_anomaly_result'):
                    return "No anomaly detection results. Please run anomaly detection first."
                
                viz_result = self.visualizer.create_anomaly_plot(
                    self._current_data,
                    self._current_anomaly_result,
                    plot_type,
                    title
                )
                
                self._current_visualization = viz_result
                
                return (
                    f"Created {plot_type} visualization saved to {viz_result.plot_path}. "
                    f"{viz_result.plot_description}"
                )
            except Exception as e:
                return f"Error creating visualization: {str(e)}"
        
        @tool
        def get_method_recommendation(data_description: str = None) -> str:
            """
            Get a recommendation for the best anomaly detection method.
            
            Args:
                data_description: Optional description of the data characteristics
                
            Returns:
                Recommended method and explanation
            """
            try:
                if hasattr(self, '_current_data'):
                    recommended = self.anomaly_detector.recommend_method(self._current_data)
                    method_info = self.anomaly_detector.get_method_info(recommended)
                    
                    return (
                        f"Recommended method: {method_info.get('name', recommended)} "
                        f"({recommended}). "
                        f"Reason: {method_info.get('best_for', 'General purpose')}. "
                        f"Default threshold: {method_info.get('default_threshold', 'N/A')}"
                    )
                else:
                    return (
                        "No data loaded for analysis. Available methods: "
                        "z-score (normal distributions), iqr (skewed data), "
                        "dbscan (complex patterns)"
                    )
            except Exception as e:
                return f"Error getting recommendation: {str(e)}"
        
        return [analyze_complete, read_file_data, detect_anomalies, create_visualization, get_method_recommendation]
    
    def _create_agent(self) -> Optional[AgentExecutor]:
        """
        Create the LangChain agent executor.
        
        Returns:
            AgentExecutor: Configured agent executor or None if LLM disabled
        """
        if not self.enable_llm:
            return None
            
        # Load system prompt from YAML configuration
        try:
            system_prompt = self.prompt_manager.get_agent_system_prompt('anomaly_agent')
        except Exception as e:
            self.logger.warning(f"Failed to load system prompt from config: {e}, using fallback")
            system_prompt = "You are an expert anomaly detection agent specializing in time-series analysis."
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        # Create agent
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        
        # Create executor with enhanced logging
        import os
        verbose_mode = os.getenv("LANGCHAIN_VERBOSE", "false").lower() == "true"
        debug_mode = os.getenv("LANGCHAIN_DEBUG", "false").lower() == "true"
        
        # Setup callbacks for detailed logging
        callbacks = []
        if debug_mode:
            from agents.llm_logger import LLMDebugCallback
            callbacks.append(LLMDebugCallback("agents.llm_calls"))
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=verbose_mode or True,  # Always verbose for LLM mode
            handle_parsing_errors=True,
            max_iterations=10,
            return_intermediate_steps=debug_mode,  # Return detailed steps in debug mode
            callbacks=callbacks  # Add our custom callback
        )
    
    async def process_request(self, request: AnomalyDetectionRequest) -> Dict[str, Any]:
        """
        Process an anomaly detection request using direct tool calls (no LLM needed).
        
        Args:
            request: Anomaly detection request
            
        Returns:
            Dict containing processing results
            
        Raises:
            AgentError: If processing fails
        """
        try:
            self.logger.info(f"Processing anomaly detection request: {request.method}")
            
            # Always use direct tool execution - LLM not needed for deterministic tasks
            return await self._process_request_direct(request)
            
        except Exception as e:
            self.logger.error(f"Error processing request: {str(e)}")
            raise AgentError(f"Failed to process anomaly detection request: {str(e)}", "anomaly_agent")
    
    def _prepare_agent_input(self, request: AnomalyDetectionRequest) -> str:
        """
        Prepare input text for the agent.
        
        Args:
            request: Anomaly detection request
            
        Returns:
            Formatted input text
        """
        # Store data if provided directly
        if request.data:
            self._current_data = request.data
            data_info = (
                f"Data provided directly: {len(request.data.values)} points "
                f"from {request.data.timestamp[0]} to {request.data.timestamp[-1]}"
            )
        else:
            data_info = f"Data file: {request.file_path}"
        
        # Prepare instructions
        instructions = [
            f"User query: {request.query}",
            f"Data source: {data_info}",
            f"Requested method: {request.method}",
        ]
        
        if request.threshold:
            instructions.append(f"Requested threshold: {request.threshold}")
        
        instructions.extend([
            "",
            "Please perform the following:",
            "1. Load the data (if from file) or acknowledge provided data",
            "2. Analyze the data characteristics",
            "3. Perform anomaly detection using the requested method",
            "4. Create a visualization of the results",
            "5. Provide a summary of findings"
        ])
        
        return "\n".join(instructions)
    
    def _extract_results(self, agent_result: Dict[str, Any], request: AnomalyDetectionRequest) -> Dict[str, Any]:
        """
        Extract results from agent execution.
        
        Args:
            agent_result: Agent execution result
            request: Original request
            
        Returns:
            Processed results dictionary
        """
        results = {
            "agent_output": agent_result.get("output", ""),
            "successful": True,
            "error": None
        }
        
        # Add anomaly detection results if available
        if hasattr(self, '_current_anomaly_result'):
            results["anomaly_result"] = self._current_anomaly_result
        
        # Add visualization if available
        if hasattr(self, '_current_visualization'):
            results["visualization"] = self._current_visualization
        
        # Add data info if available
        if hasattr(self, '_current_data'):
            results["data_info"] = {
                "total_points": len(self._current_data.values),
                "column_name": self._current_data.column_name,
                "time_range": {
                    "start": self._current_data.timestamp[0],
                    "end": self._current_data.timestamp[-1]
                }
            }
        
        return results
    
    async def _process_request_direct(self, request: AnomalyDetectionRequest) -> Dict[str, Any]:
        """
        Process request directly using tools without LLM.
        
        Args:
            request: Anomaly detection request
            
        Returns:
            Processing results
        """
        try:
            # Step 1: Load data
            if request.data:
                data = request.data
                self._current_data = data
                self.logger.info(f"Using provided data: {len(data.values)} points")
            else:
                data = self.file_reader.read_file(request.file_path)
                self._current_data = data
                self.logger.info(f"Loaded data from {request.file_path}: {len(data.values)} points")
            
            # Step 2: Detect anomalies
            anomaly_result = self.anomaly_detector.detect_anomalies(
                data, 
                request.method, 
                threshold=request.threshold
            )
            self._current_anomaly_result = anomaly_result
            self.logger.info(f"Detected {anomaly_result.anomaly_count} anomalies using {request.method}")
            
            # Step 3: Create visualization
            try:
                visualization = self.visualizer.create_anomaly_plot(data, anomaly_result)
                self._current_visualization = visualization
                self.logger.info(f"Created visualization: {visualization.plot_path}")
            except Exception as e:
                self.logger.warning(f"Failed to create visualization: {str(e)}")
                visualization = VisualizationResult(
                    plot_path="",
                    plot_base64="",
                    plot_description="Visualization could not be generated",
                    plot_type="matplotlib"
                )
                self._current_visualization = visualization
            
            return {
                "successful": True,
                "anomaly_result": anomaly_result,
                "visualization": visualization,
                "data_info": {
                    "total_points": len(data.values),
                    "column_name": data.column_name,
                    "time_range": {
                        "start": data.timestamp[0],
                        "end": data.timestamp[-1]
                    }
                },
                "agent_output": f"Anomaly detection completed: {anomaly_result.anomaly_count} anomalies found using {request.method} method"
            }
            
        except Exception as e:
            self.logger.error(f"Direct processing failed: {str(e)}")
            return {
                "successful": False,
                "error": str(e),
                "agent_output": f"Processing failed: {str(e)}"
            }
    
    async def run_standalone(
        self, 
        file_path: str = None, 
        data: TimeSeriesData = None,
        method: str = "z-score",
        threshold: float = None,
        query: str = "Detect anomalies in this time-series data"
    ) -> Dict[str, Any]:
        """
        Run anomaly detection as a standalone operation.
        
        Args:
            file_path: Path to data file
            data: Direct time-series data
            method: Detection method
            threshold: Detection threshold
            query: Analysis query
            
        Returns:
            Complete analysis results
        """
        request = AnomalyDetectionRequest(
            file_path=file_path,
            data=data,
            method=method,
            threshold=threshold,
            query=query
        )
        
        return await self.process_request(request)
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get information about agent capabilities.
        
        Returns:
            Capabilities information
        """
        return {
            "name": "Anomaly Detection Agent",
            "description": "Detects anomalies in time-series data using statistical methods",
            "methods": ["z-score", "iqr", "dbscan"],
            "file_formats": ["csv", "xlsx", "xls"],
            "visualization_types": ["matplotlib", "plotly"],
            "tools": [tool.name for tool in self.tools]
        }