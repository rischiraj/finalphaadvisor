"""
Analysis renderer for displaying JSON analysis in a single contained format.
This creates a professional, single-container analysis display.
"""

import streamlit as st
import json
import logging
from typing import Dict, Any
from contextlib import contextmanager

# Create a dedicated logger for JSON rendering debugging
json_logger = logging.getLogger('json_renderer')
json_logger.setLevel(logging.DEBUG)

# Create file handler if it doesn't exist
if not json_logger.handlers:
    from pathlib import Path
    # Use project root to build path
    project_root = Path(__file__).parent.parent.parent
    log_path = project_root / "logs" / "json_renderer.log"
    log_path.parent.mkdir(exist_ok=True)  # Create logs directory if it doesn't exist
    
    handler = logging.FileHandler(str(log_path))
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    json_logger.addHandler(handler)


def render_analysis_single_container(json_content: str) -> None:
    """
    Render JSON analysis in a single contained format like the old working version.
    
    Args:
        json_content (str): JSON string from LLM response
    """
    json_logger.info("=== JSON RENDERER START ===")
    json_logger.info(f"Input JSON length: {len(json_content)}")
    json_logger.info(f"Input JSON first 500 chars: {json_content[:500]}")
    json_logger.info(f"Input JSON last 500 chars: {json_content[-500:]}")
    
    try:
        json_logger.info("Attempting to parse JSON...")
        data = json.loads(json_content)
        json_logger.info("JSON parsing successful!")
        json_logger.info(f"Parsed data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        
        # Build content as single markdown string like the old working version
        content_parts = []
        
        # Disclaimer
        if 'disclaimer' in data:
            content_parts.append(f"⚠️ **Disclaimer:** {data['disclaimer']}")
            content_parts.append("")
        
        # Executive Summary  
        if 'executive_summary' in data:
            content_parts.append("**📋 Executive Summary**")
            content_parts.append(data['executive_summary'])
            content_parts.append("")
        
        # Anomaly Analysis
        if 'anomaly_analysis' in data:
            content_parts.append("**🔍 Anomaly Analysis**")
            anomaly = data['anomaly_analysis']
            
            if 'key_anomalies' in anomaly:
                content_parts.append("**📈 Key Anomalies:**")
                content_parts.append(anomaly['key_anomalies'])
                content_parts.append("")
            
            if 'news_correlations' in anomaly:
                content_parts.append("**📰 News Correlations:**")
                content_parts.append(anomaly['news_correlations'])
                content_parts.append("")
            
            if 'fundamental_impact' in anomaly:
                content_parts.append("**💼 Fundamental Impact:**")
                content_parts.append(anomaly['fundamental_impact'])
                content_parts.append("")
        
        # Trading Recommendations
        if 'actionable_recommendations' in data:
            content_parts.append("**🎯 Trading Recommendations:**")
            for i, rec in enumerate(data['actionable_recommendations']):
                content_parts.append(f"**{rec.get('action', 'ACTION')} 📈**")
                content_parts.append(f"- **Timeframe:** {rec.get('timeframe', 'N/A')}")
                content_parts.append(f"- **Price Targets:** {rec.get('price_targets', 'N/A')}")
                content_parts.append(f"- **Position Size:** {rec.get('position_size', 'N/A')}")
                content_parts.append(f"- **Rationale:** {rec.get('rationale', 'N/A')}")
                content_parts.append("")
        
        # Risk Assessment
        if 'risk_assessment' in data:
            content_parts.append("**⚠️ Risk Assessment:**")
            risk = data['risk_assessment']
            
            if 'primary_risks' in risk:
                content_parts.append("**🚨 Primary Risks:**")
                if isinstance(risk['primary_risks'], list):
                    for r in risk['primary_risks']:
                        content_parts.append(f"• {r}")
                else:
                    content_parts.append(risk['primary_risks'])
                content_parts.append("")
            
            if 'stop_loss_levels' in risk:
                content_parts.append(f"**🛑 Stop Loss:** {risk['stop_loss_levels']}")
            
            if 'portfolio_impact' in risk:
                content_parts.append(f"**📊 Portfolio Impact:** {risk['portfolio_impact']}")
            
            if 'hedging_options' in risk:
                content_parts.append(f"**🛡️ Hedging:** {risk['hedging_options']}")
            
            content_parts.append("")
        
        # Monitoring Plan
        if 'monitoring_plan' in data:
            content_parts.append("**📅 Monitoring Plan:**")
            monitoring = data['monitoring_plan']
            
            if 'key_dates' in monitoring:
                content_parts.append("**📅 Key Dates:**")
                if isinstance(monitoring['key_dates'], list):
                    for date in monitoring['key_dates']:
                        content_parts.append(f"• {date}")
                else:
                    content_parts.append(monitoring['key_dates'])
                content_parts.append("")
            
            if 'price_alerts' in monitoring:
                content_parts.append(f"**🔔 Price Alerts:** {monitoring['price_alerts']}")
            
            if 'news_triggers' in monitoring:
                content_parts.append(f"**📰 News Triggers:** {monitoring['news_triggers']}")
            
            if 'review_schedule' in monitoring:
                content_parts.append(f"**🔄 Review Schedule:** {monitoring['review_schedule']}")
            
            content_parts.append("")
        
        # Confidence Score
        if 'confidence_score' in data:
            confidence = data['confidence_score']
            
            rating = confidence.get('rating', 0)
            if isinstance(rating, str):
                try:
                    rating = int(rating)
                except ValueError:
                    rating = 0
            
            content_parts.append(f"**🎯 Confidence Score: {rating}/100**")
            
            if 'explanation' in confidence:
                content_parts.append(f"**📝 Explanation:** {confidence['explanation']}")
            
            if 'contrary_view' in confidence:
                content_parts.append(f"**🤔 Contrary View:** {confidence['contrary_view']}")
        
        # Join all parts and render as single markdown - like the old working version
        formatted_content = "\n".join(content_parts)
        st.markdown(formatted_content)
                    
    except json.JSONDecodeError as e:
        json_logger.error(f"JSON parsing failed: {e}")
        json_logger.error(f"JSON content that failed: {json_content}")
        
        st.error("⚠️ **Incomplete Analysis Response**")
        st.warning("The LLM response was truncated and contains incomplete JSON. This usually means the response exceeded token limits.")
        
        # Show available parts as text
        if json_content:
            st.markdown("**Available Analysis (Partial):**")
            st.text(json_content[:1000] + "..." if len(json_content) > 1000 else json_content)
        
        st.info("💡 **Tip:** Try asking a more specific question to get a complete response within token limits.")
    except Exception as e:
        json_logger.error(f"Unexpected error in JSON renderer: {e}")
        json_logger.error(f"JSON content that caused error: {json_content}")
        
        st.error(f"JSON Parsing Error: {e}")
        st.text("Raw content:")
        st.text(json_content[:500] + "..." if len(json_content) > 500 else json_content)


class AnalysisRenderer:
    """Analysis renderer for displaying anomaly detection results."""
    
    def __init__(self):
        """Initialize the analysis renderer."""
        pass
    
    def render_anomaly_result(self, anomaly_result):
        """Render anomaly detection results."""
        st.subheader("🔍 Anomaly Detection Results")
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Points", anomaly_result.total_points)
        with col2:
            st.metric("Anomalies Found", anomaly_result.anomaly_count)
        with col3:
            st.metric("Percentage", f"{anomaly_result.anomaly_percentage:.1f}%")
        
        # Display method information
        st.write(f"**Method Used:** {anomaly_result.method_used}")
        st.write(f"**Threshold:** {anomaly_result.threshold_used}")
        
        # Display anomaly details if any
        if anomaly_result.anomaly_count > 0:
            st.write("**Anomaly Details:**")
            for i, (idx, value, timestamp) in enumerate(zip(
                anomaly_result.anomaly_indices,
                anomaly_result.anomaly_values,
                anomaly_result.anomaly_timestamps
            )):
                st.write(f"• Point {idx}: {value} at {timestamp}")
    
    def render_visualization(self, visualization_result):
        """Render visualization results."""
        st.subheader("📊 Visualization")
        
        # Display plot
        if visualization_result.plot_base64:
            import base64
            image_bytes = base64.b64decode(visualization_result.plot_base64)
            st.image(image_bytes, caption=visualization_result.plot_description)
        else:
            st.info("No visualization available")
        
        st.write(f"**Plot Type:** {visualization_result.plot_type}")
        st.write(f"**Description:** {visualization_result.plot_description}")
    
    def render_insights(self, insights):
        """Render insight results."""
        st.subheader("💡 Insights")
        
        # Summary
        st.write("**Summary:**")
        st.write(insights.summary)
        
        # Explanations
        st.write("**Anomaly Explanations:**")
        for explanation in insights.anomaly_explanations:
            st.write(f"• {explanation}")
        
        # Recommendations
        st.write("**Recommendations:**")
        for recommendation in insights.recommendations:
            st.write(f"• {recommendation}")
        
        # Root causes
        st.write("**Root Causes:**")
        for cause in insights.root_causes:
            st.write(f"• {cause}")
        
        # Confidence score
        st.info(f"**Confidence Score:** {insights.confidence_score}%")
    
    def render_complete_analysis(self, analysis_response):
        """Render complete analysis response."""
        st.success("✅ Analysis Complete!")
        
        # Display processing time
        st.metric("Processing Time", f"{analysis_response.processing_time:.2f}s")
        
        # Render each component
        self.render_anomaly_result(analysis_response.anomaly_result)
        self.render_visualization(analysis_response.visualization)
        self.render_insights(analysis_response.insights)
        
        # Display metadata if available
        if analysis_response.metadata:
            with st.expander("📋 Metadata"):
                st.json(analysis_response.metadata)
    
    def render_error(self, error_message):
        """Render error message."""
        st.error(error_message)
    
    @contextmanager
    def render_loading_state(self, message="Processing..."):
        """Context manager for rendering loading state."""
        with st.spinner(message):
            yield