"""
Professional progress indicator components for Streamlit app.

This module provides reusable progress indicators and status displays
for a professional user experience.
"""

import streamlit as st
import time
from typing import Dict, List, Optional


class ProgressIndicator:
    """Professional progress indicator with status messages."""
    
    def __init__(self, title: str = "Processing"):
        """
        Initialize progress indicator.
        
        Args:
            title (str): Title to display during processing.
        """
        self.title = title
        self.steps: List[Dict] = []
        self.current_step = 0
        
    def add_step(self, name: str, description: str = ""):
        """
        Add a processing step.
        
        Args:
            name (str): Step name.
            description (str): Step description.
        """
        self.steps.append({
            "name": name,
            "description": description,
            "status": "pending",
            "start_time": None,
            "end_time": None
        })
    
    def start_step(self, step_index: int):
        """
        Start a processing step.
        
        Args:
            step_index (int): Index of step to start.
        """
        if 0 <= step_index < len(self.steps):
            self.steps[step_index]["status"] = "running"
            self.steps[step_index]["start_time"] = time.time()
            self.current_step = step_index
    
    def complete_step(self, step_index: int):
        """
        Complete a processing step.
        
        Args:
            step_index (int): Index of step to complete.
        """
        if 0 <= step_index < len(self.steps):
            self.steps[step_index]["status"] = "completed"
            self.steps[step_index]["end_time"] = time.time()
    
    def fail_step(self, step_index: int, error_msg: str = ""):
        """
        Mark a step as failed.
        
        Args:
            step_index (int): Index of step that failed.
            error_msg (str): Error message.
        """
        if 0 <= step_index < len(self.steps):
            self.steps[step_index]["status"] = "failed"
            self.steps[step_index]["error"] = error_msg
            self.steps[step_index]["end_time"] = time.time()
    
    def render(self, container=None):
        """
        Render the progress indicator.
        
        Args:
            container: Streamlit container to render in.
        """
        if container is None:
            container = st
        
        # Create a centered container
        col1, col2, col3 = container.columns([1, 2, 1])
        
        with col2:
            # Progress header - compact and centered
            st.markdown(f"""
            <div style="
                background: linear-gradient(90deg, #1f77b4, #2196f3);
                color: white;
                padding: 0.75rem 1rem;
                border-radius: 8px;
                margin: 0.5rem 0;
                text-align: center;
                font-weight: bold;
                font-size: 1rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">
                🔄 {self.title}
            </div>
            """, unsafe_allow_html=True)
            
            # Overall progress
            completed_steps = sum(1 for step in self.steps if step["status"] == "completed")
            total_steps = len(self.steps)
            progress_percent = (completed_steps / total_steps) if total_steps > 0 else 0
            
            st.progress(progress_percent)
            st.text(f"Progress: {completed_steps}/{total_steps} steps completed")
            
            # Individual steps - compact display
            for i, step in enumerate(self.steps):
                self._render_step_compact(st, i, step)
    
    def _render_step(self, container, index: int, step: Dict):
        """
        Render an individual step.
        
        Args:
            container: Streamlit container.
            index (int): Step index.
            step (Dict): Step information.
        """
        status = step["status"]
        
        # Status icons and colors
        status_config = {
            "pending": {"icon": "⏳", "color": "#9e9e9e", "bg": "#f5f5f5"},
            "running": {"icon": "🔄", "color": "#ff9800", "bg": "#fff3e0"},
            "completed": {"icon": "✅", "color": "#4caf50", "bg": "#e8f5e9"},
            "failed": {"icon": "❌", "color": "#f44336", "bg": "#ffebee"}
        }
        
        config = status_config.get(status, status_config["pending"])
        
        # Calculate duration if available
        duration_text = ""
        if step.get("start_time") and step.get("end_time"):
            duration = step["end_time"] - step["start_time"]
            duration_text = f"({duration:.1f}s)"
        elif step.get("start_time") and status == "running":
            duration = time.time() - step["start_time"]
            duration_text = f"({duration:.1f}s)"
        
        # Render step
        container.markdown(f"""
        <div style="
            display: flex;
            align-items: center;
            padding: 0.75rem;
            margin: 0.5rem 0;
            border-radius: 8px;
            background-color: {config['bg']};
            border-left: 4px solid {config['color']};
        ">
            <span style="font-size: 1.2rem; margin-right: 0.75rem;">{config['icon']}</span>
            <div style="flex: 1;">
                <div style="font-weight: bold; color: {config['color']};">
                    {step['name']} {duration_text}
                </div>
                {f'<div style="font-size: 0.9rem; color: #666; margin-top: 0.25rem;">{step["description"]}</div>' if step.get("description") else ''}
                {f'<div style="font-size: 0.9rem; color: #d32f2f; margin-top: 0.25rem;">Error: {step.get("error", "")}</div>' if step.get("error") else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_step_compact(self, container, index: int, step: Dict):
        """
        Render an individual step in compact format.
        
        Args:
            container: Streamlit container.
            index (int): Step index.
            step (Dict): Step information.
        """
        status = step["status"]
        
        # Status icons and colors
        status_config = {
            "pending": {"icon": "⏳", "color": "#6b7280"},
            "running": {"icon": "🔄", "color": "#f59e0b"},
            "completed": {"icon": "✅", "color": "#10b981"},
            "failed": {"icon": "❌", "color": "#ef4444"}
        }
        
        config = status_config.get(status, status_config["pending"])
        
        # Calculate duration if available
        duration_text = ""
        if step.get("start_time") and step.get("end_time"):
            duration = step["end_time"] - step["start_time"]
            duration_text = f" ({duration:.1f}s)"
        elif step.get("start_time") and status == "running":
            duration = time.time() - step["start_time"]
            duration_text = f" ({duration:.1f}s)"
        
        # Compact step display
        container.markdown(f"""
        <div style="
            display: flex;
            align-items: center;
            padding: 0.4rem 0.8rem;
            margin: 0.2rem 0;
            border-radius: 6px;
            background-color: rgba(255,255,255,0.05);
            border-left: 3px solid {config['color']};
            font-size: 0.9rem;
        ">
            <span style="margin-right: 0.5rem;">{config['icon']}</span>
            <span style="color: {config['color']}; font-weight: 500;">
                {step['name']}{duration_text}
            </span>
        </div>
        """, unsafe_allow_html=True)


def show_loading_spinner(message: str = "Processing...", container=None):
    """
    Show a simple loading spinner - compact and centered.
    
    Args:
        message (str): Loading message.
        container: Streamlit container.
    """
    if container is None:
        container = st
    
    # Create centered columns
    col1, col2, col3 = container.columns([1, 1, 1])
    
    with col2:
        st.markdown(f"""
        <div style="
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 1rem;
            background: rgba(31, 119, 180, 0.1);
            border-radius: 8px;
            margin: 0.5rem 0;
        ">
            <div style="
                width: 20px;
                height: 20px;
                border: 2px solid rgba(31, 119, 180, 0.3);
                border-top: 2px solid #1f77b4;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-right: 0.75rem;
            "></div>
            <span style="color: #1f77b4; font-weight: 500; font-size: 0.9rem;">{message}</span>
        </div>
        
        <style>
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        </style>
        """, unsafe_allow_html=True)


def show_status_card(title: str, status: str, details: Optional[Dict] = None, container=None):
    """
    Show a professional status card.
    
    Args:
        title (str): Card title.
        status (str): Status (success, warning, error, info).
        details (Dict): Additional details to display.
        container: Streamlit container.
    """
    if container is None:
        container = st
    
    status_config = {
        "success": {"icon": "✅", "color": "#4caf50", "bg": "#e8f5e9"},
        "warning": {"icon": "⚠️", "color": "#ff9800", "bg": "#fff3e0"},
        "error": {"icon": "❌", "color": "#f44336", "bg": "#ffebee"},
        "info": {"icon": "ℹ️", "color": "#2196f3", "bg": "#e3f2fd"}
    }
    
    config = status_config.get(status, status_config["info"])
    
    details_html = ""
    if details:
        details_html = "<div style='margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #ddd;'>"
        for key, value in details.items():
            details_html += f"<div style='margin: 0.25rem 0;'><strong>{key}:</strong> {value}</div>"
        details_html += "</div>"
    
    container.markdown(f"""
    <div style="
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 8px;
        background-color: {config['bg']};
        border-left: 6px solid {config['color']};
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    ">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <span style="font-size: 1.5rem; margin-right: 0.75rem;">{config['icon']}</span>
            <h3 style="margin: 0; color: {config['color']};">{title}</h3>
        </div>
        {details_html}
    </div>
    """, unsafe_allow_html=True)


def create_file_analysis_progress():
    """
    Create a progress indicator specifically for file analysis.
    
    Returns:
        ProgressIndicator: Configured progress indicator.
    """
    progress = ProgressIndicator("File Analysis & Processing")
    progress.add_step("File Upload", "Validating and reading file data")
    progress.add_step("Data Processing", "Parsing and cleaning data")
    progress.add_step("Anomaly Detection", "Running analysis algorithms") 
    progress.add_step("Visualization", "Generating charts and plots")
    progress.add_step("AI Analysis", "Getting intelligent insights")
    progress.add_step("Conversation Setup", "Preparing chat interface")
    
    return progress


def create_conversation_progress():
    """
    Create a progress indicator for conversation processing.
    
    Returns:
        ProgressIndicator: Configured progress indicator.
    """
    progress = ProgressIndicator("AI Response Generation")
    progress.add_step("Message Processing", "Analyzing your question")
    progress.add_step("Context Retrieval", "Accessing conversation history")
    progress.add_step("AI Generation", "Generating intelligent response")
    progress.add_step("Response Formatting", "Preparing final output")
    
    return progress


# Simple methods for testing compatibility - Note: This creates a duplicate class name
class SimpleProgressIndicator:
    """Simple progress indicator for testing."""
    
    def __init__(self):
        """Initialize progress indicator."""
        pass
    
    def show_progress_bar(self, progress, message=""):
        """Show progress bar."""
        st.progress(progress)
        if message:
            st.write(message)
    
    def show_step_progress(self, steps, current_step):
        """Show step progress."""
        total_steps = len(steps)
        progress = (current_step + 1) / total_steps if total_steps > 0 else 0
        st.progress(progress)
        
        st.write(f"Step {current_step + 1} of {total_steps}: {steps[current_step] if current_step < len(steps) else 'Complete'}")
    
    def show_spinner(self, message="Loading..."):
        """Show spinner context manager."""
        return st.spinner(message)
    
    def show_success_message(self, message):
        """Show success message."""
        st.success(message)
    
    def show_error_message(self, message):
        """Show error message."""
        st.error(message)
    
    def show_info_message(self, message):
        """Show info message."""
        st.info(message)


# Note: ProgressIndicator and SimpleProgressIndicator are separate classes
# Use ProgressIndicator for full functionality, SimpleProgressIndicator for basic use