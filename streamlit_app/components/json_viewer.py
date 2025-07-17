"""
Professional JSON analysis viewer for Financial Analyst AI.

This component renders structured JSON analysis responses in a readable,
professional format instead of raw JSON or truncated text.
"""

import streamlit as st
import json
from typing import Dict, Any, Optional
import re


class AnalysisJSONViewer:
    """Professional viewer for JSON analysis responses."""
    
    def __init__(self):
        """Initialize the JSON viewer."""
        pass
    
    def render(self, json_content: str, container=None) -> None:
        """
        Render JSON analysis in structured format.
        
        Args:
            json_content (str): JSON string from LLM response
            container: Streamlit container to render in
        """
        if container is None:
            container = st
            
        try:
            # Try to parse as JSON
            if json_content.strip().startswith('{'):
                data = json.loads(json_content)
                self._render_structured_analysis(data, container)
            else:
                # Fallback to text format
                self._render_text_analysis(json_content, container)
                
        except json.JSONDecodeError:
            # Try to extract JSON from mixed content
            json_match = self._extract_json_from_text(json_content)
            if json_match:
                try:
                    data = json.loads(json_match)
                    self._render_structured_analysis(data, container)
                    return
                except json.JSONDecodeError:
                    pass
            
            # Fallback to text format
            self._render_text_analysis(json_content, container)
    
    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """
        Extract JSON from mixed text content.
        
        Args:
            text (str): Mixed content that may contain JSON
            
        Returns:
            str: Extracted JSON string or None
        """
        # First try to find JSON in markdown code blocks
        if '```json' in text:
            start = text.find('```json') + 7
            end = text.find('```', start)
            if end != -1:
                potential_json = text[start:end].strip()
                try:
                    json.loads(potential_json)
                    return potential_json
                except json.JSONDecodeError:
                    pass
        
        # Look for JSON block starting with { and ending with }
        # More robust pattern that handles nested objects
        start_pos = text.find('{')
        if start_pos != -1:
            brace_count = 0
            for i, char in enumerate(text[start_pos:], start_pos):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        potential_json = text[start_pos:i+1]
                        try:
                            json.loads(potential_json)
                            return potential_json
                        except json.JSONDecodeError:
                            break
        
        return None
    
    def _render_structured_analysis(self, data: Dict[str, Any], container) -> None:
        """
        Render structured JSON analysis with professional formatting.
        
        Args:
            data (Dict): Parsed JSON data
            container: Streamlit container
        """
        # Skip the title since it's provided by the chat message container
        
        # Disclaimer
        if 'disclaimer' in data:
            container.warning(f"⚠️ **Disclaimer:** {data['disclaimer']}")
        
        # Executive Summary
        if 'executive_summary' in data:
            with container.expander("📋 Executive Summary", expanded=True):
                container.markdown(f"""
                <div style="
                    background-color: rgba(6, 78, 59, 0.1);
                    color: #1f2937;
                    padding: 1rem;
                    border-radius: 6px;
                    border-left: 4px solid #10b981;
                    line-height: 1.5;
                ">
                    {data['executive_summary']}
                </div>
                """, unsafe_allow_html=True)
        
        # Anomaly Analysis
        if 'anomaly_analysis' in data:
            with container.expander("🔍 Anomaly Analysis", expanded=True):
                anomaly_data = data['anomaly_analysis']
                
                if 'key_anomalies' in anomaly_data:
                    container.markdown("### 📈 Key Anomalies")
                    self._render_text_section_simple(anomaly_data['key_anomalies'], container)
                
                if 'news_correlations' in anomaly_data:
                    container.markdown("### 📰 News Correlations")
                    self._render_text_section_simple(anomaly_data['news_correlations'], container)
                
                if 'fundamental_impact' in anomaly_data:
                    container.markdown("### 💼 Fundamental Impact")
                    self._render_text_section_simple(anomaly_data['fundamental_impact'], container)
        
        # Actionable Recommendations
        if 'actionable_recommendations' in data:
            with container.expander("🎯 Trading Recommendations", expanded=True):
                recommendations = data['actionable_recommendations']
                
                for i, rec in enumerate(recommendations):
                    action_color = self._get_action_color(rec.get('action', ''))
                    
                    st.markdown(f"""
                    <div style="
                        background: {action_color['bg']};
                        border-left: 5px solid {action_color['border']};
                        padding: 1.5rem;
                        margin: 1rem 0;
                        border-radius: 8px;
                    ">
                        <h4 style="color: {action_color['text']}; margin-top: 0;">
                            {action_color['icon']} {rec.get('action', 'ACTION')}
                        </h4>
                        <p><strong>Timeframe:</strong> {rec.get('timeframe', 'N/A')}</p>
                        <p><strong>Price Targets:</strong> {rec.get('price_targets', 'N/A')}</p>
                        <p><strong>Position Size:</strong> {rec.get('position_size', 'N/A')}</p>
                        <p><strong>Rationale:</strong> {rec.get('rationale', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Risk Assessment
        if 'risk_assessment' in data:
            with container.expander("⚠️ Risk Assessment", expanded=True):
                risk_data = data['risk_assessment']
                
                col1, col2 = container.columns(2)
                
                with col1:
                    if 'primary_risks' in risk_data:
                        col1.markdown("### 🚨 Primary Risks")
                        if isinstance(risk_data['primary_risks'], list):
                            for risk in risk_data['primary_risks']:
                                col1.markdown(f"• {risk}")
                        else:
                            self._render_text_section_simple(risk_data['primary_risks'], col1)
                    
                    if 'stop_loss_levels' in risk_data:
                        col1.markdown("### 🛑 Stop Loss Levels")
                        self._render_text_section_simple(risk_data['stop_loss_levels'], col1)
                
                with col2:
                    if 'portfolio_impact' in risk_data:
                        col2.markdown("### 📊 Portfolio Impact")
                        self._render_text_section_simple(risk_data['portfolio_impact'], col2)
                    
                    if 'hedging_options' in risk_data:
                        col2.markdown("### 🛡️ Hedging Options")
                        self._render_text_section_simple(risk_data['hedging_options'], col2)
        
        # Monitoring Plan
        if 'monitoring_plan' in data:
            with container.expander("📅 Monitoring Plan", expanded=False):
                monitoring_data = data['monitoring_plan']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'key_dates' in monitoring_data:
                        st.markdown("### 📅 Key Dates")
                        if isinstance(monitoring_data['key_dates'], list):
                            for date in monitoring_data['key_dates']:
                                st.markdown(f"• {date}")
                        else:
                            self._render_text_section(monitoring_data['key_dates'])
                    
                    if 'price_alerts' in monitoring_data:
                        st.markdown("### 🔔 Price Alerts")
                        if isinstance(monitoring_data['price_alerts'], list):
                            for alert in monitoring_data['price_alerts']:
                                st.markdown(f"• {alert}")
                        else:
                            self._render_text_section(monitoring_data['price_alerts'])
                
                with col2:
                    if 'news_triggers' in monitoring_data:
                        st.markdown("### 📰 News Triggers")
                        if isinstance(monitoring_data['news_triggers'], list):
                            for trigger in monitoring_data['news_triggers']:
                                st.markdown(f"• {trigger}")
                        else:
                            self._render_text_section(monitoring_data['news_triggers'])
                    
                    if 'review_schedule' in monitoring_data:
                        st.markdown("### 🔄 Review Schedule")
                        self._render_text_section(monitoring_data['review_schedule'])
        
        # Confidence Score
        if 'confidence_score' in data:
            with container.expander("🎯 Confidence Assessment", expanded=True):
                confidence_data = data['confidence_score']
                
                # Display confidence score with visual indicator
                rating = confidence_data.get('rating', 0)
                # Convert string to int if needed
                if isinstance(rating, str):
                    try:
                        rating = int(rating)
                    except ValueError:
                        rating = 0
                rating_color = self._get_confidence_color(rating)
                
                st.markdown(f"""
                <div style="
                    background: linear-gradient(90deg, {rating_color['bg']}, {rating_color['bg2']});
                    color: white;
                    padding: 1.5rem;
                    border-radius: 8px;
                    text-align: center;
                    margin-bottom: 1rem;
                ">
                    <h3 style="margin: 0; color: white;">Confidence Score: {rating}/100</h3>
                    <div style="
                        background-color: rgba(255,255,255,0.3);
                        height: 20px;
                        border-radius: 10px;
                        margin: 1rem 0;
                        overflow: hidden;
                    ">
                        <div style="
                            background-color: white;
                            height: 100%;
                            width: {rating}%;
                            border-radius: 10px;
                        "></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if 'explanation' in confidence_data:
                    container.markdown("### 📝 Explanation")
                    self._render_text_section_simple(confidence_data['explanation'], container)
                
                if 'contrary_view' in confidence_data:
                    container.markdown("### 🤔 Contrary View")
                    container.warning(confidence_data['contrary_view'])
    
    def _render_text_analysis(self, text: str, container) -> None:
        """
        Render text-based analysis as fallback.
        
        Args:
            text (str): Text content
            container: Streamlit container
        """
        container.markdown(f"""
        <div style="
            background-color: #064e3b;
            color: #ffffff;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 5px solid #34d399;
            line-height: 1.6;
            white-space: pre-wrap;
        ">
            {text}
        </div>
        """, unsafe_allow_html=True)
    
    def _render_text_section(self, text: str) -> None:
        """
        Render a text section with proper formatting.
        
        Args:
            text (str): Text to render
        """
        st.markdown(f"""
        <div style="
            background-color: #f8f9fa;
            color: #2c3e50;
            padding: 1rem;
            border-radius: 6px;
            border-left: 3px solid #3498db;
            line-height: 1.6;
            white-space: pre-wrap;
        ">
            {text}
        </div>
        """, unsafe_allow_html=True)
    
    def _render_text_section_simple(self, text: str, container) -> None:
        """
        Render a simple text section for chat containers.
        
        Args:
            text (str): Text to render
            container: Streamlit container
        """
        container.markdown(f"""
        <div style="
            background-color: rgba(248, 249, 250, 0.8);
            color: #374151;
            padding: 0.75rem;
            border-radius: 4px;
            border-left: 3px solid #10b981;
            line-height: 1.4;
            margin: 0.5rem 0;
            white-space: pre-wrap;
        ">
            {text}
        </div>
        """, unsafe_allow_html=True)
    
    def _get_action_color(self, action: str) -> Dict[str, str]:
        """
        Get color scheme for trading action.
        
        Args:
            action (str): Trading action (BUY, SELL, HOLD, WATCH)
            
        Returns:
            Dict: Color configuration
        """
        action_upper = action.upper()
        
        color_schemes = {
            'BUY': {
                'bg': 'rgba(16, 185, 129, 0.1)',
                'border': '#10b981',
                'text': '#065f46',
                'icon': '📈'
            },
            'SELL': {
                'bg': 'rgba(239, 68, 68, 0.1)',
                'border': '#ef4444',
                'text': '#7f1d1d',
                'icon': '📉'
            },
            'HOLD': {
                'bg': 'rgba(59, 130, 246, 0.1)',
                'border': '#3b82f6',
                'text': '#1e3a8a',
                'icon': '🤝'
            },
            'WATCH': {
                'bg': 'rgba(245, 158, 11, 0.1)',
                'border': '#f59e0b',
                'text': '#78350f',
                'icon': '👀'
            }
        }
        
        return color_schemes.get(action_upper, color_schemes['WATCH'])
    
    def _get_confidence_color(self, rating: int) -> Dict[str, str]:
        """
        Get color scheme for confidence rating.
        
        Args:
            rating (int): Confidence rating 0-100
            
        Returns:
            Dict: Color configuration
        """
        if rating >= 80:
            return {'bg': '#10b981', 'bg2': '#34d399'}
        elif rating >= 60:
            return {'bg': '#3b82f6', 'bg2': '#60a5fa'}
        elif rating >= 40:
            return {'bg': '#f59e0b', 'bg2': '#fbbf24'}
        else:
            return {'bg': '#ef4444', 'bg2': '#f87171'}


def render_analysis_json(json_content: str, container=None) -> None:
    """
    Convenience function to render analysis JSON.
    
    Args:
        json_content (str): JSON string from LLM
        container: Streamlit container
    """
    viewer = AnalysisJSONViewer()
    viewer.render(json_content, container)


class JsonViewer:
    """Simple JSON viewer for testing compatibility."""
    
    def __init__(self):
        """Initialize the JSON viewer."""
        pass
    
    def render_json_data(self, data):
        """Render JSON data."""
        st.json(data)
    
    def render_formatted_json(self, data):
        """Render formatted JSON."""
        import json
        formatted = json.dumps(data, indent=2)
        st.code(formatted, language='json')
    
    def render_json_with_syntax_highlighting(self, data):
        """Render JSON with syntax highlighting."""
        import json
        formatted = json.dumps(data, indent=2)
        st.code(formatted, language='json')
    
    def render_collapsible_json(self, data, title):
        """Render collapsible JSON sections."""
        with st.expander(title):
            st.json(data)