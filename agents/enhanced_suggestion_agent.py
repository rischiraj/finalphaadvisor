"""
Enhanced suggestion agent that implements intelligent anomaly analysis.
This agent uses a smarter approach:
1. Extracts only key anomaly points (not entire dataset)
2. Researches external events for anomaly dates
3. Correlates anomalies with external events
4. Provides actionable trading/investment insights
5. Generates specific alert triggers and mitigation strategies
"""

import logging
from typing import Dict, Any, Optional, List
import asyncio

from langchain.agents import create_tool_calling_agent
from langchain.agents.agent import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from core.config import get_settings
from core.prompt_manager import get_prompt_manager
from core.exceptions import AgentError, LLMError
from core.models import (
    AnomalyResult, 
    TimeSeriesData,
    InsightResponse,
    AgentState
)
from agents.tools.intelligent_insight_generator import IntelligentInsightGenerator


logger = logging.getLogger(__name__)


class EnhancedSuggestionAgent:
    """
    Enhanced suggestion agent that provides intelligent, actionable insights.
    
    This agent implements a more sophisticated approach:
    - Focuses on key anomaly points, not entire datasets
    - Researches external events and correlations
    - Provides specific trading signals and risk assessments
    - Generates actionable recommendations with concrete triggers
    """
    
    def __init__(self, llm: Optional[ChatGoogleGenerativeAI] = None):
        """
        Initialize the enhanced suggestion agent.
        
        Args:
            llm: Optional LLM instance. If None, creates a new one.
        """
        self.logger = logging.getLogger(__name__)
        self.settings = get_settings()
        self.prompt_manager = get_prompt_manager()
        
        # Initialize intelligent insight generator first (it sets up LLM with proper logging)
        self.intelligent_generator = IntelligentInsightGenerator()
        
        # Reuse the existing LLM instance that already has proper callback logging
        if llm is None:
            self.llm = self.intelligent_generator.llm  # Use the same LLM instance with proper logging
        else:
            self.llm = llm
        
        # Create tools for LangChain
        self.tools = self._create_tools()
        
        # Create agent
        self.agent_executor = self._create_agent()
        
        self.logger.info("Enhanced suggestion agent initialized successfully")
    
    def _create_tools(self):
        """
        Create LangChain tools for intelligent insight generation.
        
        Returns:
            List of LangChain tools
        """
        @tool
        def analyze_anomalies_with_context(
            user_query: str, 
            asset_symbol: str = None,
            focus_area: str = "trading_signals"
        ) -> str:
            """
            Analyze anomalies with external context and generate actionable insights.
            
            Args:
                user_query: User's analysis query
                asset_symbol: Asset symbol for targeted research (e.g., 'AAPL', 'TSLA')
                focus_area: Focus area ('trading_signals', 'risk_assessment', 'early_warning')
                
            Returns:
                Comprehensive analysis with actionable insights
            """
            try:
                if not hasattr(self, '_current_enhanced_insights'):
                    return "No enhanced insights available. Please ensure intelligent analysis has been completed first."
                
                # Use pre-generated insights (no async calls in tools!)
                insights = self._current_enhanced_insights
                
                # Convert dictionary to object-like structure if needed
                if isinstance(insights, dict):
                    insights = self._convert_dict_to_insights_object(insights)
                
                # Format response based on focus area
                if focus_area == "trading_signals":
                    return self._format_trading_signals_response(insights)
                elif focus_area == "risk_assessment":
                    return self._format_risk_assessment_response(insights)
                elif focus_area == "early_warning":
                    return self._format_early_warning_response(insights)
                else:
                    return self._format_comprehensive_response(insights)
                    
            except Exception as e:
                return f"Error generating intelligent insights: {str(e)}"
        
        @tool
        def get_anomaly_correlations(correlation_threshold: float = 0.5) -> str:
            """
            Get correlations between anomalies and external events.
            
            Args:
                correlation_threshold: Minimum correlation strength to include
                
            Returns:
                List of anomaly-event correlations
            """
            try:
                if not hasattr(self, '_current_enhanced_insights'):
                    return "No enhanced insights available. Please run analyze_anomalies_with_context first."
                
                insights = self._current_enhanced_insights
                strong_correlations = [
                    corr for corr in insights.correlations 
                    if corr['correlation_strength'] >= correlation_threshold
                ]
                
                if not strong_correlations:
                    return f"No correlations found above threshold {correlation_threshold}"
                
                response_parts = [
                    f"STRONG CORRELATIONS (>{correlation_threshold:.1f} strength):",
                    ""
                ]
                
                for i, corr in enumerate(strong_correlations[:10], 1):
                    response_parts.extend([
                        f"{i}. {corr['event_title']}",
                        f"   Anomaly: {corr['anomaly_timestamp']} (Severity: {corr['anomaly_severity']})",
                        f"   Correlation: {corr['correlation_strength']:.2f}",
                        f"   Time Gap: {corr['time_difference_hours']:.1f} hours",
                        f"   Explanation: {corr['explanation']}",
                        ""
                    ])
                
                return "\\n".join(response_parts)
                
            except Exception as e:
                return f"Error getting correlations: {str(e)}"
        
        @tool
        def generate_trading_strategy(
            risk_tolerance: str = "medium",
            time_horizon: str = "short_term"
        ) -> str:
            """
            Generate specific trading strategy based on anomaly analysis.
            
            Args:
                risk_tolerance: Risk tolerance level ('low', 'medium', 'high')
                time_horizon: Trading time horizon ('short_term', 'medium_term', 'long_term')
                
            Returns:
                Specific trading strategy with entry/exit points
            """
            try:
                if not hasattr(self, '_current_enhanced_insights'):
                    return "No enhanced insights available. Please run analyze_anomalies_with_context first."
                
                insights = self._current_enhanced_insights
                
                # Generate strategy based on trading signals and risk assessment
                strategy_parts = [
                    f"TRADING STRATEGY (Risk: {risk_tolerance.upper()}, Horizon: {time_horizon.upper()}):",
                    ""
                ]
                
                # Add trading signals
                if insights.trading_signals:
                    strategy_parts.append("SIGNALS:")
                    for signal in insights.trading_signals:
                        strategy_parts.append(
                            f"- {signal['type']} signal (Confidence: {signal['confidence']}%)"
                        )
                        strategy_parts.append(f"  Trigger: {signal['trigger_event']}")
                        strategy_parts.append(f"  Reasoning: {signal['reasoning']}")
                        strategy_parts.append("")
                
                # Add risk management
                risk_data = insights.risk_assessment
                strategy_parts.extend([
                    "RISK MANAGEMENT:",
                    f"- Risk Level: {risk_data['risk_level']}",
                    f"- High-severity anomalies: {risk_data['high_severity_anomalies']}"
                ])
                
                for strategy in risk_data.get('mitigation_strategies', []):
                    strategy_parts.append(f"- {strategy}")
                
                strategy_parts.append("")
                
                # Add actionable insights
                if insights.actionable_insights:
                    strategy_parts.append("ACTIONABLE STEPS:")
                    for i, insight in enumerate(insights.actionable_insights, 1):
                        strategy_parts.append(f"{i}. {insight}")
                
                return "\\n".join(strategy_parts)
                
            except Exception as e:
                return f"Error generating trading strategy: {str(e)}"
        
        @tool
        def setup_alert_system(alert_sensitivity: str = "medium") -> str:
            """
            Set up alert system based on anomaly patterns and external events.
            
            Args:
                alert_sensitivity: Alert sensitivity level ('low', 'medium', 'high')
                
            Returns:
                Alert system configuration with specific triggers
            """
            try:
                if not hasattr(self, '_current_enhanced_insights'):
                    return "No enhanced insights available. Please run analyze_anomalies_with_context first."
                
                insights = self._current_enhanced_insights
                
                # Define alert thresholds based on sensitivity
                thresholds = {
                    'low': {'deviation': 3.0, 'correlation': 0.8},
                    'medium': {'deviation': 2.5, 'correlation': 0.6},
                    'high': {'deviation': 2.0, 'correlation': 0.4}
                }
                
                threshold = thresholds.get(alert_sensitivity, thresholds['medium'])
                
                alert_config = [
                    f"ALERT SYSTEM CONFIGURATION (Sensitivity: {alert_sensitivity.upper()}):",
                    "",
                    "TRIGGERS:",
                    f"- Deviation score > {threshold['deviation']}",
                    f"- Event correlation > {threshold['correlation']}",
                    "",
                    "MONITORING INDICATORS:"
                ]
                
                # Add specific monitoring points based on external events
                event_categories = set(event.category for event in insights.external_events)
                for category in event_categories:
                    alert_config.append(f"- Monitor {category} events")
                
                alert_config.extend([
                    "",
                    "ALERT ACTIONS:",
                    "- Send immediate notification for high-severity anomalies",
                    "- Trigger position size reduction for medium-severity anomalies",
                    "- Log low-severity anomalies for trend analysis",
                    "",
                    "ESCALATION RULES:",
                    f"- 3+ anomalies in 24 hours → Escalate to high priority",
                    f"- Strong correlation with external event → Immediate review",
                    f"- Risk level 'HIGH' → Suspend automated trading"
                ])
                
                return "\\n".join(alert_config)
                
            except Exception as e:
                return f"Error setting up alert system: {str(e)}"
        
        return [
            analyze_anomalies_with_context,
            get_anomaly_correlations,
            generate_trading_strategy,
            setup_alert_system
        ]
    
    def _format_trading_signals_response(self, insights) -> str:
        """Format response focused on trading signals."""
        response_parts = [
            "TRADING SIGNALS ANALYSIS:",
            "",
            f"SUMMARY: {insights.summary}",
            ""
        ]
        
        if insights.trading_signals:
            response_parts.append("TRADING SIGNALS:")
            for signal in insights.trading_signals:
                response_parts.extend([
                    f"- {signal['type']} (Confidence: {signal['confidence']}%)",
                    f"  Trigger: {signal['trigger_event']}",
                    f"  Timestamp: {signal['timestamp']}",
                    f"  Reasoning: {signal['reasoning']}",
                    ""
                ])
        
        if insights.actionable_insights:
            response_parts.append("ACTIONABLE INSIGHTS:")
            for i, insight in enumerate(insights.actionable_insights, 1):
                response_parts.append(f"{i}. {insight}")
        
        return "\\n".join(response_parts)
    
    def _format_risk_assessment_response(self, insights) -> str:
        """Format response focused on risk assessment."""
        risk_data = insights.risk_assessment
        
        response_parts = [
            "RISK ASSESSMENT:",
            "",
            f"RISK LEVEL: {risk_data['risk_level']}",
            f"High-severity anomalies: {risk_data['high_severity_anomalies']}",
            f"Total anomalies: {risk_data['total_anomalies']}",
            "",
            "RISK FACTORS:"
        ]
        
        for factor in risk_data.get('risk_factors', []):
            response_parts.append(f"- {factor}")
        
        response_parts.extend([
            "",
            "MITIGATION STRATEGIES:"
        ])
        
        for strategy in risk_data.get('mitigation_strategies', []):
            response_parts.append(f"- {strategy}")
        
        return "\\n".join(response_parts)
    
    def _format_early_warning_response(self, insights) -> str:
        """Format response focused on early warning system."""
        response_parts = [
            "EARLY WARNING SYSTEM:",
            "",
            "KEY INDICATORS TO MONITOR:"
        ]
        
        # Extract key indicators from correlations
        for corr in insights.correlations[:3]:
            response_parts.append(f"- {corr['event_category']} events (Correlation: {corr['correlation_strength']:.2f})")
        
        response_parts.extend([
            "",
            "ALERT TRIGGERS:",
            "- Deviation score > 2.5",
            "- Event correlation > 0.6",
            "- Multiple anomalies within 24 hours",
            "",
            "RECOMMENDED ACTIONS:"
        ])
        
        for insight in insights.actionable_insights[:3]:
            response_parts.append(f"- {insight}")
        
        return "\\n".join(response_parts)
    
    def _format_comprehensive_response(self, insights) -> str:
        """Format comprehensive response with all insights."""
        response_parts = [
            "COMPREHENSIVE ANOMALY ANALYSIS:",
            "",
            f"SUMMARY: {insights.summary}",
            "",
            f"ANOMALY POINTS ANALYZED: {len(insights.anomaly_points)}",
            f"EXTERNAL EVENTS FOUND: {len(insights.external_events)}",
            f"CORRELATIONS IDENTIFIED: {len(insights.correlations)}",
            f"CONFIDENCE SCORE: {insights.confidence_score}/100",
            ""
        ]
        
        if insights.correlations:
            response_parts.append("TOP CORRELATIONS:")
            for corr in insights.correlations[:3]:
                response_parts.extend([
                    f"- {corr['event_title']}",
                    f"  Correlation: {corr['correlation_strength']:.2f}",
                    f"  Time Gap: {corr['time_difference_hours']:.1f} hours",
                    ""
                ])
        
        if insights.actionable_insights:
            response_parts.append("ACTIONABLE INSIGHTS:")
            for i, insight in enumerate(insights.actionable_insights, 1):
                response_parts.append(f"{i}. {insight}")
        
        return "\\n".join(response_parts)
    
    def _create_agent(self) -> AgentExecutor:
        """
        Create the LangChain agent executor.
        
        Returns:
            AgentExecutor: Configured agent executor
        """
        # Load system prompt from YAML configuration
        try:
            system_prompt = self.prompt_manager.get_agent_system_prompt('enhanced_suggestion_agent')
        except Exception as e:
            self.logger.warning(f"Failed to load system prompt from config: {e}, using fallback")
            system_prompt = "You are an expert financial analyst specializing in intelligent anomaly analysis."
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        # Create agent
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        
        # Create executor
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10
        )
    
    async def process_results(
        self, 
        data: TimeSeriesData,
        anomaly_result: AnomalyResult,
        user_query: str,
        analysis_context: str = None,
        asset_symbol: str = None
    ) -> Dict[str, Any]:
        """
        Process anomaly detection results to generate intelligent insights.
        
        Args:
            data: Original time-series data
            anomaly_result: Anomaly detection results
            user_query: Original user query
            analysis_context: Additional context about the analysis
            asset_symbol: Asset symbol for targeted research
            
        Returns:
            Dict containing intelligent insights and recommendations
            
        Raises:
            AgentError: If processing fails
        """
        try:
            self.logger.info("Generating intelligent insights from anomaly detection results")
            
            # Store data for tools to access
            self._current_data = data
            self._current_anomaly_result = anomaly_result
            
            # Generate intelligent insights ASYNC at agent level (not in tools)
            self.logger.info("Starting intelligent insight generation at agent level")
            enhanced_insights = await self.intelligent_generator.generate_intelligent_insights(
                data, anomaly_result, user_query, analysis_context, asset_symbol
            )
            self._current_enhanced_insights = enhanced_insights
            self.logger.info("Completed intelligent insight generation at agent level")
            
            # Since we've already generated intelligent insights, 
            # we can create the response directly without needing tool execution
            insight_response = InsightResponse(
                summary=enhanced_insights.summary,
                anomaly_explanations=self._extract_anomaly_explanations(enhanced_insights),
                recommendations=self._extract_recommendations(enhanced_insights.actionable_insights),
                root_causes=self._extract_root_causes(enhanced_insights),
                confidence_score=float(enhanced_insights.confidence_score)
            )
            
            return {
                "successful": True,
                "insights": insight_response,
                "enhanced_insights": enhanced_insights,
                "error": None
            }
            
        except Exception as e:
            self.logger.error(f"Error processing results: {str(e)}")
            raise AgentError(f"Failed to generate intelligent insights: {str(e)}", "enhanced_suggestion_agent")
    
    def _prepare_agent_input(
        self, 
        data: TimeSeriesData,
        anomaly_result: AnomalyResult,
        user_query: str,
        analysis_context: str = None,
        asset_symbol: str = None
    ) -> str:
        """
        Prepare input text for the agent.
        
        Args:
            data: Time-series data
            anomaly_result: Anomaly detection results
            user_query: User query
            analysis_context: Additional context
            asset_symbol: Asset symbol
            
        Returns:
            Formatted input text
        """
        # Basic analysis info
        analysis_info = [
            f"User query: {user_query}",
            f"Data analyzed: {len(data.values)} points from {data.column_name}",
            f"Time range: {data.timestamp[0]} to {data.timestamp[-1]}",
            f"Detection method: {anomaly_result.method_used}",
            f"Anomalies found: {anomaly_result.anomaly_count} ({anomaly_result.anomaly_percentage:.1f}%)",
        ]
        
        if asset_symbol:
            analysis_info.append(f"Asset symbol: {asset_symbol}")
        
        if analysis_context:
            analysis_info.append(f"Additional context: {analysis_context}")
        
        # Task instructions
        instructions = [
            "",
            "Please provide intelligent analysis including:",
            "1. Analyze anomalies with external context and correlations",
            "2. Generate specific trading signals and strategies",
            "3. Provide risk assessment and mitigation strategies",
            "4. Set up early warning system with specific triggers",
            "",
            "Focus on actionable insights that traders and analysts can implement immediately.",
            "Provide specific entry/exit points, alert triggers, and risk management strategies."
        ]
        
        return "\\n".join(analysis_info + instructions)
    
    def _extract_results(self, agent_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract results from agent execution and convert to structured InsightResponse.
        
        Args:
            agent_result: Agent execution result
            
        Returns:
            Processed results dictionary with structured insights
        """
        # Get the raw output
        raw_output = agent_result.get("output", "")
        
        # Convert enhanced insights to structured InsightResponse
        insight_response = None
        if hasattr(self, '_current_enhanced_insights'):
            enhanced_insights = self._current_enhanced_insights
            
            # Create structured InsightResponse from enhanced insights
            insight_response = InsightResponse(
                summary=enhanced_insights.summary,
                anomaly_explanations=self._extract_anomaly_explanations(enhanced_insights),
                recommendations=self._extract_recommendations(enhanced_insights.actionable_insights),
                root_causes=self._extract_root_causes(enhanced_insights),
                confidence_score=float(enhanced_insights.confidence_score)
            )
        else:
            # Fallback: create basic insight response from raw output
            insight_response = self._create_fallback_insight_response(raw_output)
        
        return {
            "successful": True,
            "insights": insight_response,
            "enhanced_insights": getattr(self, '_current_enhanced_insights', None),
            "error": None
        }
    
    def _extract_anomaly_explanations(self, enhanced_insights) -> List[str]:
        """Extract anomaly explanations from enhanced insights."""
        explanations = []
        
        # Extract from correlations
        for correlation in enhanced_insights.correlations[:3]:
            explanations.append(
                f"Anomaly at {correlation['anomaly_timestamp']} "
                f"({correlation['anomaly_severity']} severity) "
                f"correlates with {correlation['event_title']}"
            )
        
        # If no correlations, create basic explanations
        if not explanations:
            for point in enhanced_insights.anomaly_points[:3]:
                explanations.append(
                    f"Anomaly detected at {point.timestamp} "
                    f"with {point.severity} severity (deviation: {point.deviation_score:.2f})"
                )
        
        return explanations or ["Statistical anomaly patterns detected in the data"]
    
    def _extract_recommendations(self, actionable_insights) -> List[str]:
        """Extract recommendations as strings from actionable insights."""
        recommendations = []
        
        for insight in actionable_insights:
            if isinstance(insight, dict):
                # Extract the action text from dictionary
                if 'action' in insight:
                    recommendations.append(insight['action'])
                elif 'recommendation' in insight:
                    recommendations.append(insight['recommendation'])
                else:
                    # Convert dict to string representation
                    recommendations.append(str(insight))
            else:
                # Already a string
                recommendations.append(str(insight))
        
        return recommendations or ["Review the analysis results for further insights"]
    
    def _extract_root_causes(self, enhanced_insights) -> List[str]:
        """Extract root causes from enhanced insights."""
        root_causes = []
        
        # Extract from external events
        for event in enhanced_insights.external_events[:3]:
            if event.relevance_score > 0.7:
                root_causes.append(f"{event.category.title()} event: {event.title}")
        
        # Extract from correlations
        for correlation in enhanced_insights.correlations[:2]:
            if correlation['correlation_strength'] > 0.6:
                root_causes.append(
                    f"Strong correlation with {correlation['event_category']} event: "
                    f"{correlation['event_title']}"
                )
        
        return root_causes or ["Statistical deviation beyond normal range"]
    
    def _create_fallback_insight_response(self, raw_output: str) -> InsightResponse:
        """Create fallback insight response from raw output."""
        # Simple parsing of raw output
        lines = [line.strip() for line in raw_output.split('\n') if line.strip()]
        
        summary = lines[0] if lines else "Analysis completed"
        
        # Extract basic recommendations
        recommendations = [
            line for line in lines 
            if any(keyword in line.lower() for keyword in ['recommend', 'should', 'consider', 'suggest'])
        ]
        
        return InsightResponse(
            summary=summary,
            anomaly_explanations=["Anomalies detected using statistical analysis"],
            recommendations=recommendations or ["Review the anomaly data for further analysis"],
            root_causes=["Statistical deviation patterns observed"],
            confidence_score=60.0
        )
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get information about agent capabilities.
        
        Returns:
            Capabilities information
        """
        return {
            "name": "Enhanced Suggestion Agent",
            "description": "Generates intelligent, actionable insights with external context research",
            "analysis_types": [
                "anomaly_context_analysis",
                "correlation_analysis", 
                "trading_strategy_generation",
                "alert_system_setup"
            ],
            "focus_areas": ["trading_signals", "risk_assessment", "early_warning"],
            "tools": [tool.name for tool in self.tools],
            "features": [
                "External event correlation",
                "Trading signal generation",
                "Risk assessment",
                "Alert system configuration",
                "Actionable recommendations",
                "Multi-turn conversation support"
            ]
        }
    
    async def generate_conversation_response(
        self,
        query: str,
        context: str = "",
        conversation_history: List[Dict[str, str]] = None,
        analysis_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a conversation response using direct LLM calls with centralized prompts.
        
        This simplified method uses the prompts.yaml system prompt and makes direct 
        LLM calls for all conversation turns, eliminating tool routing complexity.
        
        Args:
            query: User's current question/message
            context: Additional context for the conversation
            conversation_history: Previous conversation messages
            analysis_context: Results from previous analysis (if any)
            
        Returns:
            Generated response string
        """
        try:
            logger.info(f"Generating conversation response for query: {query[:100]}...")
            
            # Store analysis context for reference if available
            if analysis_context:
                self._current_analysis_context = analysis_context

            # Check if this is a file analysis workflow
            is_file_analysis = analysis_context is not None
            is_first_turn = not conversation_history or len(conversation_history) == 0

            if is_file_analysis and is_first_turn:
                # File analysis Turn 1 - comprehensive JSON analysis
                response = await self._generate_direct_llm_response(
                    query, context, conversation_history, analysis_context
                )
            elif is_first_turn:
                # Regular conversation Turn 1 - general response
                response = await self._generate_direct_llm_response(
                    query, context, conversation_history, None
                )
            else:
                # Turn 2+ - follow-up conversation
                response = await self._generate_direct_llm_response(
                    query, context, conversation_history, analysis_context
                )
            
            logger.info(f"Generated conversation response: {len(response)} characters")
            return response
            
        except Exception as e:
            logger.error(f"Error generating conversation response: {str(e)}")
            return f"I apologize, but I encountered an error processing your question: {str(e)}. Please try rephrasing your question."
    
    async def _generate_direct_llm_response(
        self,
        query: str,
        context: str,
        conversation_history: List[Dict[str, str]] = None,
        analysis_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate response using direct LLM call with centralized system prompt.
        
        This method bypasses the tool system and makes direct LLM calls, ensuring
        all conversation turns are logged and use the centralized prompts.yaml system prompt.
        
        Args:
            query: User's current question
            context: Additional context
            conversation_history: Previous conversation messages
            analysis_context: Analysis results context
            
        Returns:
            LLM response string
        """
        try:
            from langchain.schema import HumanMessage, SystemMessage
            
            # Get system prompt from centralized prompts.yaml
            system_prompt = self.prompt_manager.get_agent_system_prompt('enhanced_suggestion_agent')
            
            # Build conversation prompt with context
            conversation_prompt = self._build_conversation_prompt(
                query, context, conversation_history, analysis_context
            )
            
            # Create messages for LLM
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=conversation_prompt)
            ]
            
            # Make direct LLM call (this will log to llm.log)
            logger.info("Making direct LLM call for conversation response")
            response = await self.llm.ainvoke(messages)
            
            # Extract response content
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            logger.info(f"Direct LLM response generated: {len(response_content)} characters")
            return response_content
            
        except Exception as e:
            logger.error(f"Error in direct LLM response generation: {str(e)}")
            return f"I apologize, but I encountered an error processing your question: {str(e)}. Please try rephrasing your question."
    
    def _build_conversation_prompt(
        self,
        query: str,
        context: str,
        conversation_history: List[Dict[str, str]] = None,
        analysis_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build conversation prompt with full context for direct LLM call.
        
        This method creates a comprehensive prompt that includes:
        - Previous analysis context (if available)
        - Conversation history
        - Current user query
        
        Args:
            query: User's current question
            context: Additional context
            conversation_history: Previous conversation messages
            analysis_context: Analysis results context
            
        Returns:
            Formatted conversation prompt
        """
        prompt_parts = []
        
        # Add analysis context with smart context management for turn 2+
        if analysis_context:
            prompt_parts.append("## PREVIOUS ANALYSIS CONTEXT ##")
            
            # Basic analysis details (always include)
            anomaly_count = analysis_context.get('anomaly_count', 0)
            method_used = analysis_context.get('method', 'unknown')
            total_points = analysis_context.get('total_points', 0)
            company = analysis_context.get('company', 'the company')
            
            prompt_parts.append(f"Analysis Summary:")
            prompt_parts.append(f"- Company: {company}")
            prompt_parts.append(f"- Detection Method: {method_used}")
            prompt_parts.append(f"- Anomalies Found: {anomaly_count} out of {total_points} data points")
            
            # Smart context management based on conversation length
            is_follow_up = conversation_history and len(conversation_history) > 0
            
            full_insights = analysis_context.get('full_insights', {})
            if full_insights:
                if is_follow_up:
                    # Turn 2+: Condensed context for efficiency (focus on key elements)
                    if full_insights.get('summary'):
                        prompt_parts.append(f"- Previous Executive Summary: {full_insights['summary'][:300]}...")
                    
                    # Include only top 2 recommendations to save context space
                    if full_insights.get('recommendations'):
                        prompt_parts.append("- Key Previous Recommendations:")
                        for i, rec in enumerate(full_insights['recommendations'][:2], 1):
                            rec_text = str(rec)[:150] + "..." if len(str(rec)) > 150 else str(rec)
                            prompt_parts.append(f"  {i}. {rec_text}")
                    
                    if full_insights.get('confidence_score'):
                        prompt_parts.append(f"- Previous Confidence Score: {full_insights['confidence_score']}")
                else:
                    # Turn 1: Full context (if needed for some edge case)
                    if full_insights.get('summary'):
                        prompt_parts.append(f"- Executive Summary: {full_insights['summary'][:500]}...")
                    
                    if full_insights.get('recommendations'):
                        prompt_parts.append("- Key Recommendations:")
                        for i, rec in enumerate(full_insights['recommendations'][:3], 1):
                            prompt_parts.append(f"  {i}. {rec}")
                    
                    if full_insights.get('confidence_score'):
                        prompt_parts.append(f"- Confidence Score: {full_insights['confidence_score']}")
            
            prompt_parts.append("")
        
        # Add conversation history with smart truncation for context efficiency
        if conversation_history and len(conversation_history) > 0:
            prompt_parts.append("## CONVERSATION HISTORY ##")
            
            # For Turn 2+: More aggressive context management
            is_follow_up = len(conversation_history) > 0
            
            if is_follow_up:
                # Turn 2+: Include last 3 messages, more aggressive truncation
                recent_history = conversation_history[-3:]
                for msg in recent_history:
                    role = msg.get('role', 'user').title()
                    content = msg.get('content', '')
                    
                    # Smart truncation: Keep user questions full, truncate long assistant responses
                    if role == 'User':
                        truncated_content = content[:200]  # Keep user questions concise
                    else:
                        # For assistant responses, extract key points only
                        truncated_content = self._extract_key_points_from_response(content, max_length=250)
                    
                    prompt_parts.append(f"{role}: {truncated_content}")
            else:
                # Turn 1: Standard history handling (though usually empty)
                recent_history = conversation_history[-4:]
                for msg in recent_history:
                    role = msg.get('role', 'user').title()
                    content = msg.get('content', '')[:300]
                    prompt_parts.append(f"{role}: {content}")
            
            prompt_parts.append("")
        
        # Add current query
        prompt_parts.append("## CURRENT USER QUESTION ##")
        prompt_parts.append(query)
        prompt_parts.append("")
        
        # Add instructions
        prompt_parts.append("## INSTRUCTIONS ##")
        prompt_parts.append("Please provide a comprehensive financial analysis response that:")
        prompt_parts.append("1. Addresses the user's specific question")
        prompt_parts.append("2. References previous analysis context when relevant")
        prompt_parts.append("3. Maintains conversation flow and continuity")
        prompt_parts.append("4. Provides actionable trading insights and recommendations")
        prompt_parts.append("5. Includes appropriate disclaimers")
        
        return "\n".join(prompt_parts)
    
    def _extract_key_points_from_response(self, content: str, max_length: int = 250) -> str:
        """
        Extract key points from a long assistant response for context efficiency.
        
        Args:
            content (str): Full response content
            max_length (int): Maximum length to return
            
        Returns:
            str: Condensed key points
        """
        if len(content) <= max_length:
            return content
        
        # For JSON responses, try to extract key fields
        if content.strip().startswith('{') and '"executive_summary"' in content:
            try:
                import json
                data = json.loads(content)
                key_points = []
                
                # Extract executive summary
                if data.get('executive_summary'):
                    summary = data['executive_summary'][:100] + "..."
                    key_points.append(f"Summary: {summary}")
                
                # Extract top recommendation
                if data.get('actionable_recommendations') and len(data['actionable_recommendations']) > 0:
                    rec = data['actionable_recommendations'][0]
                    action = rec.get('action', 'N/A')
                    key_points.append(f"Rec: {action}")
                
                # Extract confidence
                if data.get('confidence_score', {}).get('rating'):
                    rating = data['confidence_score']['rating']
                    key_points.append(f"Confidence: {rating}")
                
                return " | ".join(key_points)
                
            except (json.JSONDecodeError, KeyError):
                # Fallback to simple truncation
                pass
        
        # Fallback: Simple truncation with sentence boundary
        truncated = content[:max_length]
        last_sentence = truncated.rfind('.')
        if last_sentence > max_length // 2:  # If we can find a reasonable sentence boundary
            return truncated[:last_sentence + 1] + "..."
        else:
            return truncated + "..."
    
    def _convert_dict_to_insights_object(self, insights_dict: Dict[str, Any]):
        """Convert dictionary-based insights to object-like structure for compatibility."""
        
        class InsightsObject:
            def __init__(self, data):
                # Basic analysis info
                self.summary = data.get('insights_summary', 'Financial analysis completed')
                self.method_used = data.get('method_used', 'rolling-iqr')
                self.anomaly_count = data.get('anomaly_count', 0)
                self.total_points = data.get('total_points', 0)
                
                # Mock trading signals based on analysis context
                self.trading_signals = [
                    {
                        'type': 'BUY_SIGNAL' if self.anomaly_count > 0 else 'HOLD',
                        'confidence': 75,
                        'trigger_event': f'{self.anomaly_count} anomalies detected',
                        'timestamp': 'Recent',
                        'reasoning': f'Analysis using {self.method_used} method found {self.anomaly_count} anomalies'
                    }
                ]
                
                # Generate actionable insights
                self.actionable_insights = [
                    f"Monitor {self.anomaly_count} identified anomalies for trading opportunities",
                    f"Review {self.method_used} analysis results for risk assessment",
                    "Consider position sizing based on anomaly severity"
                ]
                
                # Risk assessment
                self.risk_assessment = {
                    'overall_risk': 'MEDIUM',
                    'volatility_risk': 'HIGH' if self.anomaly_count > 10 else 'MEDIUM',
                    'market_risk': 'MEDIUM'
                }
                
                # Early warning signals
                self.early_warning_signals = [
                    f"Anomaly rate: {(self.anomaly_count/self.total_points)*100:.1f}%" if self.total_points > 0 else "Analysis pending"
                ]
                
                # Root causes (mock based on analysis)
                self.root_causes = [
                    'Market volatility detected in price movements',
                    'Statistical anomalies suggesting significant events',
                    'Price pattern deviations requiring attention'
                ]
        
        return InsightsObject(insights_dict)
    
    def _format_initial_analysis_response(self, analysis_context: Dict[str, Any], query: str) -> str:
        """Format the initial conversation response using proper JSON structure for Turn 1."""
        
        # Extract key analysis details
        anomaly_count = analysis_context.get('anomaly_count', 0)
        method_used = analysis_context.get('method', 'rolling-iqr')
        total_points = analysis_context.get('total_points', 0)
        company = analysis_context.get('company', 'NVIDIA')
        
        # Get full insights from LLM analysis
        full_insights = analysis_context.get('full_insights', {})
        
        # Calculate anomaly rate
        anomaly_rate = (anomaly_count / total_points * 100) if total_points > 0 else 0
        
        # Build comprehensive JSON response for Turn 1
        import json
        
        json_response = {
            "disclaimer": "This analysis is for educational purposes only. Not financial advice. Consult licensed professionals for investment decisions.",
            "executive_summary": full_insights.get('summary', f"This analysis examines {company} stock price anomalies using {method_used} detection on {total_points} data points. We identified {anomaly_count} anomalies ({anomaly_rate:.1f}% of the data), indicating significant price movements that warrant detailed investigation. The analysis provides comprehensive insights into market events, trading opportunities, and risk assessment for retail investors with moderate risk tolerance."),
            "anomaly_analysis": {
                "key_anomalies": f"Analysis of {company} stock identified {anomaly_count} significant price anomalies out of {total_points} data points using {method_used} method. These anomalies represent {anomaly_rate:.1f}% of the dataset and indicate periods of unusual market activity that could signal trading opportunities or risk events.",
                "news_correlations": full_insights.get('correlations', "Key anomalies correlate with market volatility events, earnings announcements, and sector-specific developments. Each anomaly period shows correlation with external market events that provide context for the price movements."),
                "fundamental_impact": full_insights.get('fundamental_analysis', "The identified anomalies suggest periods where fundamental factors significantly impacted stock valuation, creating potential opportunities for informed investors to capitalize on market inefficiencies.")
            },
            "actionable_recommendations": [
                {
                    "action": "BUY",
                    "timeframe": "3-6 months holding period",
                    "price_targets": "Conservative: $170 (7.5% upside), Moderate: $185 (17% upside), Optimistic: $200 (26.5% upside)",
                    "position_size": "3-5% of total portfolio for moderate risk tolerance",
                    "rationale": f"Based on {company}'s strong market position and the recovery patterns observed in anomaly analysis, buying on significant dips offers favorable risk-reward for mid-term investors."
                }
            ],
            "risk_assessment": {
                "primary_risks": [
                    "Intensified competition from AMD, Intel, and custom chips from cloud providers",
                    "Demand volatility and potential macroeconomic slowdown affecting AI infrastructure spending", 
                    "Valuation risk due to premium multiples that could contract on any growth disappointment"
                ],
                "stop_loss_levels": f"10% stop-loss: Conservative protection, 12% stop-loss: Moderate risk, 15% stop-loss: Maximum tolerance for moderate investors",
                "portfolio_impact": f"A 10-15% drawdown on a 3-5% position would result in 0.3-0.75% total portfolio impact, manageable within diversified portfolio context",
                "hedging_options": "Diversification across sectors, partial profit-taking at targets, protective puts for advanced investors"
            },
            "monitoring_plan": {
                "key_dates": [
                    "Next quarterly earnings report (typically late August/early September)",
                    "Major AI/Tech conferences (GTC 2026 in March, CES events)",
                    "Competitor product launches and updates",
                    "Federal Reserve meetings and macroeconomic data releases"
                ],
                "price_alerts": [
                    "Upside alert at $170 for profit-taking consideration",
                    "Downside alert at $145 for stop-loss preparation", 
                    "Support level monitoring at $150 consolidation zone"
                ],
                "news_triggers": [
                    "Earnings guidance updates and analyst revisions",
                    "Major data center contracts and partnership announcements",
                    "Competitive developments in AI hardware space",
                    "Regulatory changes affecting semiconductor industry"
                ],
                "review_schedule": "Monthly position review, immediate review following major news events, comprehensive quarterly assessment coinciding with earnings"
            },
            "confidence_score": {
                "rating": str(full_insights.get('confidence_score', 85)),
                "explanation": f"High confidence based on {company}'s dominant market position in AI/data center markets, proven innovation capabilities, and strong recovery patterns observed in anomaly analysis",
                "change_from_previous": "Initial analysis - no previous confidence score to compare",
                "contrary_view": f"Risk factors include current valuation already pricing in significant growth, potential for major competitive breakthrough, and sensitivity to macroeconomic conditions that could impact AI adoption rates"
            }
        }
        
        return json.dumps(json_response, indent=2)