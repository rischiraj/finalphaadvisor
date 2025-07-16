"""
Professional CSS styling for Financial Analyst AI Streamlit app.

This module provides modular CSS styles with proper text contrast
and professional appearance.
"""

def get_base_styles() -> str:
    """
    Get base application styles with dark theme and proper contrast.
    
    Returns:
        str: CSS styles for base application styling.
    """
    return """
    <style>
    /* Dark theme base with high contrast */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Fix general text visibility throughout app */
    .main .block-container {
        color: #f9fafb;
    }
    .main .block-container p {
        color: #e5e7eb;
    }
    .main .block-container h1, .main .block-container h2, .main .block-container h3 {
        color: #ffffff;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1f2937;
    }
    .css-1d391kg .stMarkdown {
        color: #f9fafb;
    }
    </style>
    """


def get_header_styles() -> str:
    """
    Get header and title styles.
    
    Returns:
        str: CSS styles for headers and titles.
    """
    return """
    <style>
    /* Main header styling */
    .main-header { 
        font-size: 2.8rem; 
        color: #ffffff; 
        margin-bottom: 2rem; 
        text-align: center;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.3);
        background: linear-gradient(90deg, #1f77b4, #2196f3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Section headers */
    .section-header {
        color: #60a5fa;
        font-size: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    </style>
    """


def get_chat_styles() -> str:
    """
    Get chat message styles with high contrast.
    
    Returns:
        str: CSS styles for chat messages.
    """
    return """
    <style>
    /* Enhanced chat messages with better contrast */
    .chat-message { 
        padding: 1.2rem; 
        margin: 0.8rem 0; 
        border-radius: 12px; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        font-size: 1rem;
        line-height: 1.6;
    }
    .user-message { 
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); 
        margin-left: 8%;
        border-left: 5px solid #60a5fa;
        color: #ffffff;
        font-weight: 500;
    }
    .assistant-message { 
        background: linear-gradient(135deg, #064e3b 0%, #10b981 100%); 
        margin-right: 8%;
        border-left: 5px solid #34d399;
        color: #ffffff;
        font-weight: 500;
    }
    
    /* Professional token counter */
    .token-counter { 
        font-size: 0.75rem; 
        color: rgba(255,255,255,0.7); 
        text-align: right;
        margin-top: 0.8rem;
        padding-top: 0.5rem;
        border-top: 1px solid rgba(255,255,255,0.2);
    }
    </style>
    """


def get_component_styles() -> str:
    """
    Get styles for UI components (buttons, cards, etc.).
    
    Returns:
        str: CSS styles for UI components.
    """
    return """
    <style>
    /* Enhanced feature cards */
    .feature-card {
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.2rem 0;
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        color: #f9fafb;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    }
    .feature-card h4 {
        color: #60a5fa;
        margin-bottom: 0.8rem;
        font-size: 1.1rem;
    }
    
    /* Professional button styling */
    .stButton > button {
        background: linear-gradient(90deg, #1f77b4, #2196f3);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s;
        box-shadow: 0 2px 8px rgba(31, 119, 180, 0.3);
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #1565c0, #1976d2);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.4);
    }
    
    /* Metrics */
    .metric-container {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #4b5563;
        color: #f9fafb;
    }
    </style>
    """


def get_form_styles() -> str:
    """
    Get styles for forms and input elements.
    
    Returns:
        str: CSS styles for form elements.
    """
    return """
    <style>
    /* Input areas with high contrast */
    .stTextArea textarea {
        background-color: #374151;
        color: #f9fafb;
        border: 1px solid #4b5563;
        border-radius: 8px;
    }
    .stTextInput input {
        background-color: #374151;
        color: #f9fafb;
        border: 1px solid #4b5563;
        border-radius: 8px;
    }
    .stSelectbox select {
        background-color: #374151;
        color: #f9fafb;
        border: 1px solid #4b5563;
        border-radius: 8px;
    }
    .stNumberInput input {
        background-color: #374151;
        color: #f9fafb;
        border: 1px solid #4b5563;
        border-radius: 8px;
    }
    
    /* File uploader */
    .stFileUploader > div {
        background-color: #374151;
        border: 2px dashed #6b7280;
        border-radius: 8px;
        color: #f9fafb;
    }
    .stFileUploader label {
        color: #e5e7eb;
    }
    </style>
    """


def get_status_styles() -> str:
    """
    Get styles for status indicators and alerts.
    
    Returns:
        str: CSS styles for status components.
    """
    return """
    <style>
    /* Status indicators with proper contrast */
    .stAlert {
        border-radius: 8px;
    }
    .stSuccess {
        background-color: rgba(16, 185, 129, 0.1);
        border: 1px solid #10b981;
        color: #6ee7b7;
    }
    .stError {
        background-color: rgba(239, 68, 68, 0.1);
        border: 1px solid #ef4444;
        color: #fca5a5;
    }
    .stWarning {
        background-color: rgba(245, 158, 11, 0.1);
        border: 1px solid #f59e0b;
        color: #fcd34d;
    }
    .stInfo {
        background-color: rgba(59, 130, 246, 0.1);
        border: 1px solid #3b82f6;
        color: #93c5fd;
    }
    
    /* Progress bars */
    .stProgress .st-bo {
        background-color: #374151;
    }
    .stProgress .st-bp {
        background: linear-gradient(90deg, #1f77b4, #2196f3);
    }
    </style>
    """


def get_data_display_styles() -> str:
    """
    Get styles for data display components.
    
    Returns:
        str: CSS styles for dataframes, tables, etc.
    """
    return """
    <style>
    /* Dataframe styling with proper contrast */
    .stDataFrame {
        background-color: #374151;
        border-radius: 8px;
    }
    .stDataFrame table {
        color: #f9fafb;
    }
    .stDataFrame th {
        background-color: #1f2937;
        color: #ffffff;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #374151;
        color: #f9fafb;
        border-radius: 8px;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background-color: #1f2937;
        border: 1px solid #374151;
        border-radius: 8px;
        color: #f9fafb;
    }
    </style>
    """


def get_animation_styles() -> str:
    """
    Get animation and loading styles.
    
    Returns:
        str: CSS styles for animations.
    """
    return """
    <style>
    /* Professional loading animations */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .loading-pulse {
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .loading-spin {
        animation: spin 1s linear infinite;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .fade-in {
        animation: fadeIn 0.3s ease-out;
    }
    
    /* Analysis header styles */
    .analysis-header {
        background: linear-gradient(90deg, #064e3b, #10b981);
        color: white;
        padding: 0.5rem 0.75rem;
        border-radius: 6px;
        font-weight: bold;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        text-align: center;
        max-width: 300px;
        margin: 0 auto 0.5rem auto;
    }
    
    /* User message styles */
    .user-message-compact {
        background: linear-gradient(90deg, #1f2937, #374151);
        color: white;
        padding: 0.5rem 0.75rem;
        border-radius: 6px;
        font-weight: normal;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        text-align: left;
        max-width: 400px;
        margin: 0 0 0.5rem 0;
    }
    </style>
    """


def get_all_styles() -> str:
    """
    Get all CSS styles combined.
    
    Returns:
        str: All CSS styles combined.
    """
    return (
        get_base_styles() +
        get_header_styles() +
        get_chat_styles() +
        get_component_styles() +
        get_form_styles() +
        get_status_styles() +
        get_data_display_styles() +
        get_animation_styles()
    )


# Simple functions for testing compatibility
def get_custom_css():
    """Get custom CSS for testing compatibility."""
    return get_all_styles()


def apply_custom_styles():
    """Apply custom styles for testing compatibility."""
    import streamlit as st
    st.markdown(get_custom_css(), unsafe_allow_html=True)