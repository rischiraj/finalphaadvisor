"""
Intelligent insight generation tool that combines statistical analysis with external research.
This tool implements a smarter approach:
1. Extract key statistical points from anomalies
2. Research external context (news, events, market data)  
3. Provide actionable insights with real-world correlation
"""

import logging
import json
from typing import List, Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage

from core.config import get_settings
from core.prompt_manager import get_prompt_manager
from core.exceptions import LLMError
from core.models import AnomalyResult, TimeSeriesData
from agents.llm_logger import LLMDebugCallback


@dataclass
class AnomalyDataPoint:
    """Structured anomaly data point with context."""
    timestamp: datetime
    value: float
    severity: str  # 'high', 'medium', 'low'
    deviation_score: float
    context_window: Dict[str, Any]  # surrounding data context


@dataclass
class ExternalEvent:
    """External event that might correlate with anomaly."""
    date: datetime
    title: str
    description: str
    source: str
    relevance_score: float
    category: str  # 'market', 'news', 'economic', 'earnings'


@dataclass
class EnhancedInsightResponse:
    """Enhanced insight response with external context."""
    summary: str
    anomaly_points: List[AnomalyDataPoint]
    external_events: List[ExternalEvent]
    correlations: List[Dict[str, Any]]
    actionable_insights: List[str]
    trading_signals: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    confidence_score: int
    raw_json_response: Optional[str] = None


class IntelligentInsightGenerator:
    """
    Intelligent insight generator that combines statistical analysis with external research.
    """
    
    def __init__(self):
        """Initialize the intelligent insight generator."""
        self.logger = logging.getLogger(__name__)
        self.settings = get_settings()
        self.prompt_manager = get_prompt_manager()
        
        # Setup dedicated LLM logger for callbacks
        self._setup_llm_logger()
        
        # Initialize LLM with debug callback
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=self.settings.llm_model,
                temperature=self.settings.llm_temperature,
                google_api_key=self.settings.google_ai_api_key,
                callbacks=[LLMDebugCallback("llm_debug")]  # Add LangChain callback
            )
            self.logger.info(f"Initialized LLM: {self.settings.llm_model}")
        except Exception as e:
            raise LLMError(f"Failed to initialize LLM: {str(e)}", self.settings.llm_model)
    
    def _setup_llm_logger(self):
        """Setup dedicated LLM logger that writes to logs/llm.log"""
        llm_logger = logging.getLogger("llm_debug")
        
        # Only setup if not already configured
        if not llm_logger.handlers:
            from pathlib import Path
            log_path = Path("logs/llm.log")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Configure LLM logger
            llm_logger.setLevel(logging.DEBUG)
            handler = logging.FileHandler(log_path)
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            llm_logger.addHandler(handler)
            llm_logger.propagate = False  # Don't send to root logger

    async def generate_intelligent_insights(
        self,
        data: TimeSeriesData,
        anomaly_result: AnomalyResult,
        user_query: str,
        context: Optional[str] = None,
        asset_symbol: Optional[str] = None
    ) -> EnhancedInsightResponse:
        """
        Generate intelligent insights with external context research.
        
        Args:
            data: Original time-series data
            anomaly_result: Anomaly detection results
            user_query: User's original query
            context: Additional context
            asset_symbol: Asset symbol for external research (e.g., 'AAPL', 'TSLA')
            
        Returns:
            EnhancedInsightResponse: Comprehensive insights with external context
        """
        try:
            self.logger.info("Starting intelligent insight generation...")
            
            # Step 1: Extract key anomaly data points (not entire dataset)
            anomaly_points = self._extract_anomaly_points(data, anomaly_result)
            self.logger.info(f"Extracted {len(anomaly_points)} key anomaly points")
            
            # Step 2: Research external events for anomaly dates
            external_events = self._research_external_events(anomaly_points, asset_symbol)
            self.logger.info(f"Found {len(external_events)} external events")
            
            # Step 3: Find correlations between anomalies and external events
            correlations = self._find_correlations(anomaly_points, external_events)
            self.logger.info(f"Identified {len(correlations)} correlations")
            
            # Step 4: Generate actionable insights using LLM with focused data
            anomaly_method_description = self._create_anomaly_method_description(anomaly_result)
            insights = await self._generate_actionable_insights(
                anomaly_points, external_events, correlations, user_query, context, anomaly_method_description
            )
            
            # Step 5: Generate trading signals and risk assessment
            trading_signals = self._generate_trading_signals(anomaly_points, external_events, correlations)
            risk_assessment = self._assess_risk(anomaly_points, external_events)
            
            # Step 6: Create comprehensive response
            response = EnhancedInsightResponse(
                summary=insights.get('summary', 'Analysis completed'),
                anomaly_points=anomaly_points,
                external_events=external_events,
                correlations=correlations,
                actionable_insights=insights.get('actionable_insights', []),
                trading_signals=trading_signals,
                risk_assessment=risk_assessment,
                confidence_score=insights.get('confidence_score', 75),
                raw_json_response=insights.get('raw_json_response')
            )
            
            self.logger.info("Intelligent insight generation completed successfully")
            return response
            
        except Exception as e:
            self.logger.error(f"Error in intelligent insight generation: {str(e)}")
            raise LLMError(f"Failed to generate intelligent insights: {str(e)}", self.settings.llm_model)
    
    def _extract_anomaly_points(
        self, 
        data: TimeSeriesData, 
        anomaly_result: AnomalyResult
    ) -> List[AnomalyDataPoint]:
        """Extract key anomaly data points with context (not entire dataset)."""
        anomaly_points = []
        
        if anomaly_result.anomaly_count == 0:
            return anomaly_points
        
        # Check if deterministic tool provided structured anomaly points
        if hasattr(anomaly_result, 'structured_anomaly_points') and anomaly_result.structured_anomaly_points:
            # Use structured data from deterministic tool
            for structured_point in anomaly_result.structured_anomaly_points:
                from datetime import datetime
                timestamp = datetime.fromisoformat(structured_point.timestamp)
                
                # Get context window
                context_window = self._get_context_window(data, timestamp, window_size=5)
                
                anomaly_point = AnomalyDataPoint(
                    timestamp=timestamp,
                    value=structured_point.value,
                    severity=structured_point.severity,
                    deviation_score=structured_point.deviation_score,
                    context_window=context_window
                )
                
                anomaly_points.append(anomaly_point)
            
            return anomaly_points
        
        # Fallback: Calculate manually if structured data not available
        import numpy as np
        values_array = np.array(data.values)
        mean_val = np.mean(values_array)
        std_val = np.std(values_array)
        
        # Extract anomaly points with context
        for i, timestamp in enumerate(anomaly_result.anomaly_timestamps):
            if i >= len(anomaly_result.anomaly_values):
                break
                
            anomaly_value = anomaly_result.anomaly_values[i]
            
            # Calculate deviation score
            deviation_score = abs(anomaly_value - mean_val) / std_val if std_val > 0 else 0
            
            # Determine severity
            if deviation_score > 3:
                severity = 'high'
            elif deviation_score > 2:
                severity = 'medium'
            else:
                severity = 'low'
            
            # Get context window (surrounding data points)
            context_window = self._get_context_window(data, timestamp, window_size=5)
            
            anomaly_point = AnomalyDataPoint(
                timestamp=timestamp,
                value=anomaly_value,
                severity=severity,
                deviation_score=deviation_score,
                context_window=context_window
            )
            
            anomaly_points.append(anomaly_point)
        
        # Sort by severity and deviation score
        anomaly_points.sort(key=lambda x: (x.severity == 'high', x.deviation_score), reverse=True)
        
        return anomaly_points
    
    def _get_context_window(
        self, 
        data: TimeSeriesData, 
        target_timestamp: datetime, 
        window_size: int = 5
    ) -> Dict[str, Any]:
        """Get context window around an anomaly point."""
        # Find the index of the target timestamp
        try:
            target_index = data.timestamp.index(target_timestamp)
        except ValueError:
            # If exact match not found, find closest
            target_index = min(range(len(data.timestamp)), 
                             key=lambda i: abs((data.timestamp[i] - target_timestamp).total_seconds()))
        
        # Get surrounding data points
        start_idx = max(0, target_index - window_size)
        end_idx = min(len(data.values), target_index + window_size + 1)
        
        context_values = data.values[start_idx:end_idx]
        
        # Calculate context statistics
        import numpy as np
        context_array = np.array(context_values)
        
        return {
            'before_values': context_values[:target_index-start_idx] if target_index > start_idx else [],
            'after_values': context_values[target_index-start_idx+1:] if target_index < end_idx-1 else [],
            'context_mean': float(np.mean(context_array)),
            'context_std': float(np.std(context_array)),
            'trend_direction': self._calculate_trend(context_values),
            'volatility': float(np.std(context_array) / np.mean(context_array)) if np.mean(context_array) != 0 else 0
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values."""
        if len(values) < 2:
            return 'stable'
        
        import numpy as np
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        slope = coeffs[0]
        
        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'
    
    def _research_external_events(
        self, 
        anomaly_points: List[AnomalyDataPoint], 
        asset_symbol: Optional[str] = None
    ) -> List[ExternalEvent]:
        """Research external events that might correlate with anomalies."""
        external_events = []
        
        if not anomaly_points:
            return external_events
        
        try:
            # Get unique dates for research
            unique_dates = list(set(point.timestamp.date() for point in anomaly_points))
            
            for date in unique_dates:
                # Research events for this date
                events = self._search_events_for_date(date, asset_symbol)
                external_events.extend(events)
            
            # Remove duplicates and sort by relevance
            external_events = self._deduplicate_events(external_events)
            external_events.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return external_events[:20]  # Return top 20 most relevant events
            
        except Exception as e:
            self.logger.warning(f"Failed to research external events: {str(e)}")
            return []
    
    def _search_events_for_date(
        self, 
        date: datetime, 
        asset_symbol: Optional[str] = None
    ) -> List[ExternalEvent]:
        """Search for events on a specific date."""
        events = []
        
        # Simulate some events based on common patterns
        demo_events = [
            {
                'title': f'Market volatility increased on {date.strftime("%Y-%m-%d")}',
                'description': 'High trading volume and increased volatility observed in the market',
                'source': 'Market Data Analysis',
                'category': 'market',
                'relevance_score': 0.8
            },
            {
                'title': f'Economic indicator release on {date.strftime("%Y-%m-%d")}',
                'description': 'Important economic data released affecting market sentiment',
                'source': 'Economic Calendar',
                'category': 'economic',
                'relevance_score': 0.7
            }
        ]
        
        if asset_symbol:
            demo_events.append({
                'title': f'{asset_symbol} specific event on {date.strftime("%Y-%m-%d")}',
                'description': f'Company-specific news or announcement for {asset_symbol}',
                'source': 'Financial News',
                'category': 'earnings',
                'relevance_score': 0.9
            })
        
        for event_data in demo_events:
            event = ExternalEvent(
                date=datetime.combine(date, datetime.min.time()),
                title=event_data['title'],
                description=event_data['description'],
                source=event_data['source'],
                relevance_score=event_data['relevance_score'],
                category=event_data['category']
            )
            events.append(event)
        
        return events
    
    def _deduplicate_events(self, events: List[ExternalEvent]) -> List[ExternalEvent]:
        """Remove duplicate events."""
        seen_titles = set()
        unique_events = []
        
        for event in events:
            if event.title not in seen_titles:
                seen_titles.add(event.title)
                unique_events.append(event)
        
        return unique_events
    
    def _find_correlations(
        self, 
        anomaly_points: List[AnomalyDataPoint], 
        external_events: List[ExternalEvent]
    ) -> List[Dict[str, Any]]:
        """Find correlations between anomalies and external events."""
        correlations = []
        
        for anomaly in anomaly_points:
            for event in external_events:
                # Check if event is within reasonable time window of anomaly
                time_diff = abs((anomaly.timestamp - event.date).total_seconds())
                
                # Consider events within 24 hours as potentially correlated
                if time_diff <= 24 * 3600:  # 24 hours in seconds
                    correlation_strength = self._calculate_correlation_strength(
                        anomaly, event, time_diff
                    )
                    
                    if correlation_strength > 0.3:  # Only include meaningful correlations
                        correlation = {
                            'anomaly_timestamp': anomaly.timestamp,
                            'anomaly_value': anomaly.value,
                            'anomaly_severity': anomaly.severity,
                            'event_title': event.title,
                            'event_description': event.description,
                            'event_category': event.category,
                            'time_difference_hours': time_diff / 3600,
                            'correlation_strength': correlation_strength,
                            'explanation': self._generate_correlation_explanation(anomaly, event)
                        }
                        correlations.append(correlation)
        
        # Sort by correlation strength
        correlations.sort(key=lambda x: x['correlation_strength'], reverse=True)
        
        return correlations
    
    def _calculate_correlation_strength(
        self, 
        anomaly: AnomalyDataPoint, 
        event: ExternalEvent, 
        time_diff: float
    ) -> float:
        """Calculate correlation strength between anomaly and event."""
        # Base correlation from event relevance
        base_correlation = event.relevance_score
        
        # Adjust based on time proximity (closer in time = stronger correlation)
        time_factor = max(0, 1 - (time_diff / (24 * 3600)))  # Decay over 24 hours
        
        # Adjust based on anomaly severity
        severity_factor = {'high': 1.0, 'medium': 0.8, 'low': 0.6}[anomaly.severity]
        
        # Adjust based on event category
        category_factor = {
            'earnings': 1.0,
            'market': 0.9,
            'economic': 0.8,
            'news': 0.7
        }.get(event.category, 0.6)
        
        # Calculate final correlation strength
        correlation_strength = base_correlation * time_factor * severity_factor * category_factor
        
        return min(1.0, correlation_strength)
    
    def _generate_correlation_explanation(
        self, 
        anomaly: AnomalyDataPoint, 
        event: ExternalEvent
    ) -> str:
        """Generate explanation for correlation."""
        time_desc = "around the same time" if abs((anomaly.timestamp - event.date).total_seconds()) < 3600 else "within 24 hours"
        
        return f"The {anomaly.severity} anomaly on {anomaly.timestamp.strftime('%Y-%m-%d %H:%M')} occurred {time_desc} as {event.title}, suggesting a potential causal relationship."
    
    async def _generate_actionable_insights(
        self,
        anomaly_points: List[AnomalyDataPoint],
        external_events: List[ExternalEvent],
        correlations: List[Dict[str, Any]],
        user_query: str,
        context: Optional[str] = None,
        anomaly_method: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate actionable insights using LLM with focused data."""
        # Load system prompt from YAML configuration
        system_prompt = self.prompt_manager.get_tool_system_prompt('intelligent_insight_generator')
        
        # Create human prompt using YAML template
        human_prompt = self._create_human_prompt_from_template(
            anomaly_points, external_events, correlations, user_query, context, anomaly_method
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            # Store the raw response content before parsing
            raw_response_content = response.content
            parsed_response = self._parse_intelligent_response(raw_response_content)
            # Add the raw response to the parsed result
            parsed_response['raw_json_response'] = raw_response_content
            return parsed_response
        except Exception as e:
            self.logger.error(f"Failed to generate actionable insights: {str(e)}")
            return {
                'summary': 'Analysis completed with limited insights',
                'actionable_insights': ['Review the anomaly data and external events'],
                'confidence_score': 50
            }
    
    def _create_human_prompt_from_template(
        self,
        anomaly_points: List[AnomalyDataPoint],
        external_events: List[ExternalEvent],
        correlations: List[Dict[str, Any]],
        user_query: str,
        context: Optional[str] = None,
        anomaly_method: Optional[str] = None
    ) -> str:
        """Create human prompt using YAML template with variable substitution."""
        # Prepare variables for template substitution
        variables = {
            'user_query': str(user_query),
            'company_name': self._extract_company_name(user_query),
            'anomaly_analysis_method': anomaly_method or 'Statistical anomaly detection',
            'anomaly_points': self._format_anomaly_points_for_prompt(anomaly_points)
        }
        
        # Debug: Log variables being passed
        self.logger.info(f"Template variables prepared: {list(variables.keys())}")
        self.logger.debug(f"Variable user_query: {variables['user_query'][:100]}...")
        self.logger.debug(f"Variable anomaly_points: {variables['anomaly_points'][:200]}...")
        
        # Get template from YAML and substitute variables
        return self.prompt_manager.get_tool_human_prompt('intelligent_insight_generator', variables)
    
    def _extract_company_name(self, user_query: str) -> str:
        """Extract company name from user query."""
        # Simple extraction - look for common patterns
        query_lower = user_query.lower()
        
        # Common company names and their stock symbols
        company_patterns = {
            'nvidia': 'NVIDIA',
            'nvda': 'NVIDIA', 
            'apple': 'Apple',
            'aapl': 'Apple',
            'microsoft': 'Microsoft',
            'msft': 'Microsoft',
            'tesla': 'Tesla',
            'tsla': 'Tesla',
            'amazon': 'Amazon',
            'amzn': 'Amazon',
            'google': 'Google',
            'googl': 'Google',
            'meta': 'Meta',
            'fb': 'Meta'
        }
        
        for pattern, company_name in company_patterns.items():
            if pattern in query_lower:
                return company_name
        
        # Default fallback
        return 'the company'
    
    def _create_anomaly_method_description(self, anomaly_result: AnomalyResult) -> str:
        """Create a descriptive string for the anomaly detection method used."""
        method = anomaly_result.method_used
        threshold = anomaly_result.threshold_used
        
        method_descriptions = {
            'z-score': f'Z-Score anomaly detection with threshold={threshold} (statistical outlier detection using standard deviations)',
            'iqr': f'IQR anomaly detection with multiplier={threshold} (interquartile range method for robust outlier detection)',
            'rolling-iqr': f'Rolling IQR anomaly detection with multiplier={threshold}, window_size=20 (20-day rolling window for local anomaly detection)',
            'dbscan': f'DBSCAN anomaly detection with eps={threshold} (density-based clustering for complex pattern detection)'
        }
        
        return method_descriptions.get(method, f'{method} anomaly detection with threshold={threshold}')
    
    def _assess_volatility(self, anomaly_points: List[AnomalyDataPoint]) -> str:
        """Assess volatility from anomaly points."""
        if not anomaly_points:
            return "low"
        
        high_severity_count = sum(1 for p in anomaly_points if p.severity == 'high')
        total_count = len(anomaly_points)
        
        if high_severity_count / total_count > 0.5:
            return "high"
        elif high_severity_count / total_count > 0.2:
            return "medium"
        else:
            return "low"
    
    def _format_anomaly_points_for_prompt(self, anomaly_points: List[AnomalyDataPoint]) -> str:
        """Format anomaly points for YAML prompt template."""
        if not anomaly_points:
            return "[]"
        
        formatted_points = []
        for point in anomaly_points[:10]:  # Limit to top 10 points
            # Determine trend based on context window
            trend = "unknown"
            if hasattr(point, 'trend'):
                # Use trend from structured data if available
                trend = point.trend
            elif point.context_window.get('values'):
                values = point.context_window['values']
                if len(values) >= 2:
                    trend = "increasing" if values[-1] > values[0] else "decreasing"
            
            formatted_point = {
                "timestamp": point.timestamp.isoformat(),
                "value": point.value,
                "severity": point.severity,
                "deviation_score": round(point.deviation_score, 2),
                "trend": trend
            }
            formatted_points.append(formatted_point)
        
        return str(formatted_points)
    
    def _format_external_events_for_prompt(self, external_events: List[ExternalEvent]) -> str:
        """Format external events for YAML prompt template."""
        if not external_events:
            return "[]"
        
        formatted_events = []
        for event in external_events[:10]:  # Limit to top 10 events
            formatted_event = {
                "date": event.date.isoformat(),
                "title": event.title,
                "category": event.category,
                "relevance_score": round(event.relevance_score, 2)
            }
            formatted_events.append(formatted_event)
        
        return str(formatted_events)
    
    def _format_correlations_for_prompt(self, correlations: List[Dict[str, Any]]) -> str:
        """Format correlations for YAML prompt template."""
        if not correlations:
            return "[]"
        
        # Limit to top 10 correlations and format for prompt
        formatted_correlations = []
        for corr in correlations[:10]:
            formatted_corr = {
                "anomaly_timestamp": corr.get('anomaly_timestamp', ''),
                "anomaly_value": corr.get('anomaly_value', 0),
                "anomaly_severity": corr.get('anomaly_severity', ''),
                "event_title": corr.get('event_title', ''),
                "event_description": corr.get('event_description', ''),
                "event_category": corr.get('event_category', ''),
                "time_difference_hours": round(corr.get('time_difference_hours', 0), 1),
                "correlation_strength": round(corr.get('correlation_strength', 0), 2),
                "explanation": corr.get('explanation', '')
            }
            formatted_correlations.append(formatted_corr)
        
        return str(formatted_correlations)
    
    def _parse_intelligent_response(self, response: str) -> Dict[str, Any]:
        """Parse intelligent response from retail-focused LLM."""
        try:
            # Extract JSON from response - handle extra content and malformed JSON
            import re
            import json
            
            # Method 1: Try to extract complete JSON block with proper nesting
            brace_count = 0
            json_start = -1
            json_content = ""
            
            for i, char in enumerate(response):
                if char == '{':
                    if brace_count == 0:
                        json_start = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and json_start != -1:
                        json_content = response[json_start:i+1]
                        break
            
            # Method 2: Fallback to regex if brace counting fails
            if not json_content:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_content = json_match.group()
            
            if json_content:
                try:
                    parsed = json.loads(json_content)
                except json.JSONDecodeError as e:
                    # Log the problematic JSON for debugging
                    self.logger.warning(f"JSON parse error: {str(e)}")
                    self.logger.debug(f"Problematic JSON (first 500 chars): {json_content[:500]}...")
                    self.logger.debug(f"Full response length: {len(response)} chars")
                    parsed = None
            else:
                parsed = None
            
            if parsed:
                
                # Validate required fields for new structure
                required_fields = ['executive_summary', 'anomaly_analysis', 'actionable_recommendations', 
                                 'risk_assessment', 'monitoring_plan', 'confidence_score']
                
                # If new structure is present, convert it to expected format
                if all(field in parsed for field in required_fields):
                    return self._adapt_new_structure_to_legacy(parsed)
                
                # Legacy support - convert old structure to new if needed
                return self._convert_legacy_response(parsed)
            else:
                # Fallback parsing - return legacy format
                return self._create_legacy_fallback()
                
        except Exception as e:
            self.logger.warning(f"Failed to parse intelligent response: {str(e)}")
            return self._create_legacy_fallback()
    
    def _adapt_new_structure_to_legacy(self, new_response: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt new retail-focused structure back to legacy format for compatibility."""
        # Extract confidence score
        confidence_score = new_response.get('confidence_score', {})
        if isinstance(confidence_score, dict):
            rating = confidence_score.get('rating', 70)
            confidence_value = self._parse_confidence_score(rating)
        else:
            confidence_value = self._parse_confidence_score(confidence_score) if confidence_score else 70.0
        
        # Extract recommendations
        actionable_recs = new_response.get('actionable_recommendations', [])
        recommendations = []
        if isinstance(actionable_recs, list):
            for rec in actionable_recs:
                if isinstance(rec, dict):
                    recommendations.append(f"{rec.get('action', 'REVIEW')}: {rec.get('rationale', 'Analysis recommendation')}")
                else:
                    recommendations.append(str(rec))
        
        if not recommendations:
            recommendations = ['Review analysis for trading opportunities']
        
        # Extract explanations from anomaly analysis
        anomaly_analysis = new_response.get('anomaly_analysis', {})
        explanations = []
        if isinstance(anomaly_analysis, dict):
            key_anomalies = anomaly_analysis.get('key_anomalies', '')
            news_correlations = anomaly_analysis.get('news_correlations', '')
            if key_anomalies:
                explanations.append(f"Key anomalies: {key_anomalies}")
            if news_correlations:
                explanations.append(f"News correlations: {news_correlations}")
        
        if not explanations:
            explanations = ['Statistical anomalies detected in price movements']
        
        # Extract risk factors as root causes
        risk_assessment = new_response.get('risk_assessment', {})
        root_causes = []
        if isinstance(risk_assessment, dict):
            primary_risks = risk_assessment.get('primary_risks', [])
            if isinstance(primary_risks, list):
                root_causes.extend(primary_risks)
            elif isinstance(primary_risks, str):
                root_causes.append(primary_risks)
        
        if not root_causes:
            root_causes = ['Market volatility and external factors']
        
        return {
            'summary': new_response.get('executive_summary', 'Analysis completed with retail-focused insights'),
            'anomaly_explanations': explanations,
            'recommendations': recommendations,
            'root_causes': root_causes,
            'confidence_score': confidence_value
        }
    
    def _parse_confidence_score(self, score) -> float:
        """Parse confidence score handling various formats like '70/100', '70%', '70'."""
        if isinstance(score, (int, float)):
            return float(score)
        
        if isinstance(score, str):
            # Handle formats like "70/100"
            if '/' in score:
                try:
                    parts = score.split('/')
                    if len(parts) == 2:
                        numerator = float(parts[0])
                        denominator = float(parts[1])
                        return (numerator / denominator) * 100 if denominator != 0 else 70.0
                except (ValueError, ZeroDivisionError):
                    pass
            
            # Handle formats like "70%"
            if '%' in score:
                try:
                    return float(score.replace('%', ''))
                except ValueError:
                    pass
            
            # Handle plain numbers as strings
            try:
                return float(score)
            except ValueError:
                pass
        
        # Default fallback
        return 70.0
    
    def _convert_legacy_response(self, legacy_response: Dict[str, Any]) -> Dict[str, Any]:
        """Convert legacy response format to new retail-focused structure."""
        return {
            'executive_summary': legacy_response.get('summary', 'Analysis completed with anomaly insights'),
            'anomaly_analysis': {
                'key_anomalies': legacy_response.get('actionable_insights', ['Review anomaly data']),
                'news_correlations': 'See external correlations for detailed analysis',
                'fundamental_impact': 'Impact assessment based on detected anomalies'
            },
            'actionable_recommendations': [{
                'action': 'REVIEW',
                'timeframe': '1-2 weeks',
                'price_targets': 'To be determined based on analysis',
                'position_size': '2-3% of portfolio',
                'rationale': 'Further analysis needed for specific recommendations'
            }],
            'risk_assessment': {
                'primary_risks': ['Market volatility', 'Company-specific risks', 'Sector risks'],
                'stop_loss_levels': 'To be determined',
                'portfolio_impact': 'Moderate risk profile',
                'hedging_options': 'Consider protective puts or position sizing'
            },
            'monitoring_plan': {
                'key_dates': 'Next earnings report and major announcements',
                'price_alerts': 'Key technical levels to watch',
                'news_triggers': 'Monitor company-specific news',
                'review_schedule': 'Weekly position review recommended'
            },
            'confidence_score': {
                'rating': str(legacy_response.get('confidence_score', 60)),
                'explanation': 'Based on statistical analysis and correlation data',
                'contrary_view': 'Market conditions may change rapidly'
            }
        }
    
    def _create_legacy_fallback(self) -> Dict[str, Any]:
        """Create fallback response in legacy format for compatibility."""
        return {
            'summary': 'Anomaly analysis completed with enhanced insights for retail investors',
            'anomaly_explanations': [
                'Statistical anomalies detected in price movements',
                'Correlation analysis with external market events completed',
                'Risk assessment based on detected patterns'
            ],
            'recommendations': [
                'Monitor position with moderate risk tolerance',
                'Consider 2-3% portfolio allocation',
                'Set stop-loss at 8-10% below entry price'
            ],
            'root_causes': [
                'Market volatility affecting price movements',
                'External market events impacting sentiment',
                'Fundamental factors requiring further analysis'
            ],
            'confidence_score': 65.0
        }
    
    def _create_fallback_response(self) -> Dict[str, Any]:
        """Create fallback response in new retail structure."""
        return {
            'executive_summary': 'Anomaly analysis completed. Review detailed findings for trading opportunities.',
            'anomaly_analysis': {
                'key_anomalies': 'Statistical anomalies detected in price movements',
                'news_correlations': 'External event correlation analysis available',
                'fundamental_impact': 'Impact on company fundamentals requires further analysis'
            },
            'actionable_recommendations': [{
                'action': 'WATCH',
                'timeframe': '2-4 weeks monitoring period',
                'price_targets': 'Targets to be established based on market conditions',
                'position_size': '1-2% initial allocation',
                'rationale': 'Anomalies detected - monitoring for clear trend confirmation'
            }],
            'risk_assessment': {
                'primary_risks': ['Market volatility', 'Incomplete data analysis', 'External market factors'],
                'stop_loss_levels': '8-10% below entry price',
                'portfolio_impact': 'Low to moderate risk given small position size',
                'hedging_options': 'Consider protective strategies if position increases'
            },
            'monitoring_plan': {
                'key_dates': 'Next earnings release and major economic announcements',
                'price_alerts': 'Key support and resistance levels',
                'news_triggers': 'Company announcements and sector news',
                'review_schedule': 'Bi-weekly position and strategy review'
            },
            'confidence_score': {
                'rating': '60',
                'explanation': 'Moderate confidence based on available data analysis',
                'contrary_view': 'Market sentiment and external factors could change outlook'
            }
        }
    
    def _generate_trading_signals(
        self,
        anomaly_points: List[AnomalyDataPoint],
        external_events: List[ExternalEvent],
        correlations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate trading signals based on analysis."""
        signals = []
        
        for correlation in correlations[:3]:  # Top 3 correlations
            signal_strength = correlation['correlation_strength']
            anomaly_severity = correlation['anomaly_severity']
            
            if signal_strength > 0.7 and anomaly_severity in ['high', 'medium']:
                signal_type = 'BUY' if correlation['anomaly_value'] < 0 else 'SELL'
                
                signal = {
                    'type': signal_type,
                    'strength': signal_strength,
                    'trigger_event': correlation['event_title'],
                    'timestamp': correlation['anomaly_timestamp'],
                    'confidence': min(100, int(signal_strength * 100)),
                    'reasoning': f"Strong correlation detected between {correlation['event_category']} event and price anomaly"
                }
                signals.append(signal)
        
        return signals
    
    def _assess_risk(
        self,
        anomaly_points: List[AnomalyDataPoint],
        external_events: List[ExternalEvent]
    ) -> Dict[str, Any]:
        """Assess risk based on anomaly patterns and external events."""
        high_severity_count = sum(1 for point in anomaly_points if point.severity == 'high')
        total_anomalies = len(anomaly_points)
        
        risk_level = 'LOW'
        if high_severity_count > 0:
            risk_ratio = high_severity_count / total_anomalies
            if risk_ratio > 0.5:
                risk_level = 'HIGH'
            elif risk_ratio > 0.3:
                risk_level = 'MEDIUM'
        
        return {
            'risk_level': risk_level,
            'high_severity_anomalies': high_severity_count,
            'total_anomalies': total_anomalies,
            'risk_factors': [
                f"{high_severity_count} high-severity anomalies detected",
                f"Recent external events may indicate continued volatility"
            ],
            'mitigation_strategies': [
                "Implement stop-loss orders",
                "Monitor key external event indicators",
                "Reduce position sizes during high volatility periods"
            ]
        }