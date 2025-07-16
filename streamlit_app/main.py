"""
Main Streamlit application for Financial Analyst AI multi-turn conversations.
This is completely separate from CLI functionality and won't interfere with it.
"""

import streamlit as st
import os
import sys
import time
import logging
from pathlib import Path

# Create a dedicated logger for JSON extraction debugging
json_extraction_logger = logging.getLogger('json_extraction')
json_extraction_logger.setLevel(logging.DEBUG)

# Create main logger
logger = logging.getLogger(__name__)

# Create file handler if it doesn't exist
if not json_extraction_logger.handlers:
    # Use project root to build path
    project_root = Path(__file__).parent.parent
    log_path = project_root / "logs" / "json_extraction.log"
    log_path.parent.mkdir(exist_ok=True)  # Create logs directory if it doesn't exist
    
    handler = logging.FileHandler(str(log_path))
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    json_extraction_logger.addHandler(handler)

# Add the parent directory to Python path to import project modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import modular styling and components
from utils.styles import get_all_styles
from components import (
    ProgressIndicator, 
    show_loading_spinner, 
    show_status_card,
    create_file_analysis_progress,
    create_conversation_progress,
    render_analysis_json
)
from components.analysis_renderer import render_analysis_single_container

# Configure Streamlit page
st.set_page_config(
    page_title="FinAlphaAdvisor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# FinAlphaAdvisor\n\nFinAlpha Advisor is an enterprise AI agent that flags outliers across your financial dashboards and instantly explains them through relevant macroeconomic headlines or micro‑level corporate events—so you can turn raw anomalies into alpha‑driving decisions."
    }
)

# Apply professional modular CSS styling
st.markdown(get_all_styles(), unsafe_allow_html=True)

def main():
    # Main header
    st.markdown('<h1 class="main-header">📊 FinAlphaAdvisor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.1rem; color: #666; font-style: italic;">From anomalies to alpha—contextualized by real‑time macro insights.</p>', unsafe_allow_html=True)
    
    # Check if API is available
    api_status = check_api_status()
    
    # Sidebar
    with st.sidebar:
        st.title("🎯 Quick Actions")
        st.markdown("---")
        
        # API Status
        if api_status:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Unavailable")
            st.warning("Please start the FastAPI server:\n```bash\nuvicorn api.main:app --reload\n```")
        
        st.markdown("### 🚀 Get Started")
        
        if st.button("💬 Start Conversation", use_container_width=True):
            st.session_state['current_page'] = 'conversation'
            st.rerun()
        
        if st.button("📊 Analyze File", use_container_width=True):
            st.session_state['current_page'] = 'analysis'  
            st.rerun()
        
        if st.button("📋 Sample Prompts", use_container_width=True):
            st.session_state['show_samples'] = True
            st.rerun()
        
        st.markdown("---")
        st.markdown("### ⚙️ System")
        
        if st.button("🔧 Admin Panel", use_container_width=True):
            st.session_state['current_page'] = 'admin'
            st.rerun()
        
        # Session info
        if 'conversation_id' in st.session_state:
            st.markdown("### 📊 Current Session")
            st.info(f"**Session ID:** {st.session_state['conversation_id'][:8]}...")
            
            if 'message_count' in st.session_state:
                st.metric("Messages", st.session_state['message_count'])
            
            if 'total_tokens' in st.session_state:
                st.metric("Tokens Used", st.session_state['total_tokens'])
            
            if st.button("🔄 New Session", use_container_width=True):
                clear_session()
                st.rerun()
    
    # Main content area
    if not api_status:
        show_api_setup_instructions()
    elif st.session_state.get('current_page') == 'conversation':
        show_conversation_page()
    elif st.session_state.get('current_page') == 'analysis':
        show_analysis_page()
    elif st.session_state.get('current_page') == 'admin':
        show_admin_page()
    elif st.session_state.get('show_samples'):
        show_sample_prompts()
    else:
        show_home_page()

def check_api_status():
    """Check if the API server is running."""
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def show_api_setup_instructions():
    """Show instructions to start the API server."""
    st.error("🚨 API Server Not Running")
    
    st.markdown("""
    ### 📋 Setup Instructions
    
    To use the Financial Analyst AI, please start the API server first:
    
    1. **Open a terminal** in your project directory
    2. **Run the API server:**
    ```bash
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
    ```
    3. **Refresh this page** once the server is running
    
    ### 🔧 Alternative Setup
    If you're having issues, try:
    ```bash
    cd /mnt/e/Agents2025/Assignment25
    python -m uvicorn api.main:app --reload
    ```
    """)
    
    if st.button("🔄 Check API Status"):
        st.rerun()

def show_home_page():
    """Show the home page with feature overview."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 What Can FinAlphaAdvisor Do?
        
        FinAlphaAdvisor is an **enterprise AI agent** that transforms financial anomalies into alpha-driving insights:
        """)
        
        # Feature cards
        st.markdown("""
        <div class="feature-card">
            <h4>🚨 Anomaly Detection</h4>
            <p>Automatically flags outliers across your financial dashboards and time-series data. 
            Spot unusual patterns, spikes, and deviations that matter for your trading decisions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>📰 Macro/Micro Context</h4>
            <p>Instantly explains detected anomalies through relevant macroeconomic headlines 
            and micro-level corporate events. Understand the "why" behind market movements.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>💡 Alpha-Driving Insights</h4>
            <p>Transforms raw anomalies into actionable trading opportunities. 
            Get specific recommendations, risk assessments, and timing guidance.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🚀 Quick Start")
        
        st.markdown("""
        **Option 1: Start Chatting**
        - Click "Start Conversation"
        - Ask any financial analysis question
        - Build on responses with follow-ups
        
        **Option 2: Analyze Data**
        - Click "Analyze File" 
        - Upload your CSV/Excel file
        - Automatic conversation starts with results
        
        **Option 3: Use Templates**
        - Click "Sample Prompts"
        - Choose from pre-built analysis templates
        - Customize for your specific needs
        """)
        
        # Quick demo button
        if st.button("🎬 Try Demo Conversation", use_container_width=True):
            # Set up a demo conversation
            st.session_state['demo_mode'] = True
            st.session_state['current_page'] = 'conversation'
            st.rerun()

def show_conversation_page():
    """Show the main conversation interface."""
    st.markdown("### 💬 FinAlphaAdvisor Chat")
    
    # Initialize conversation history
    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = []
    
    # Chat history display
    chat_container = st.container()
    
    with chat_container:
        if st.session_state['conversation_history']:
            for i, message in enumerate(st.session_state['conversation_history']):
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div class="user-message-compact">
                        <strong>You:</strong> {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Check if this is a JSON analysis response (both Turn 1 and Turn 2+)
                    content = message['content'].strip()
                    json_extraction_logger.info(f"=== JSON EXTRACTION START ===")
                    json_extraction_logger.info(f"Message content length: {len(content)}")
                    json_extraction_logger.info(f"Message content first 500 chars: {content[:500]}")
                    json_extraction_logger.info(f"Message content last 500 chars: {content[-500:]}")
                    
                    is_json_response = (
                        content.startswith('{') or 
                        '```json' in content or
                        ('"disclaimer"' in content and '"executive_summary"' in content) or
                        ('"anomaly_analysis"' in content and '"actionable_recommendations"' in content)
                    )
                    json_extraction_logger.info(f"is_json_response: {is_json_response}")
                    
                    if is_json_response:
                        # Show smaller header for JSON responses using CSS class
                        st.markdown("""
                        <div class="analysis-header">
                            📊 FinAlphaAdvisor - Complete Analysis
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Extract JSON from content (now consistent for both flows)
                        json_content = message['content']
                        json_extraction_logger.info(f"Starting JSON extraction...")
                        json_extraction_logger.info(f"Original json_content length: {len(json_content)}")
                        
                        # Remove markdown code blocks if present
                        if '```json' in json_content:
                            json_extraction_logger.info("Found ```json markers, extracting...")
                            start = json_content.find('```json') + 7
                            end = json_content.rfind('```')  # Use rfind to get the closing ```
                            if end != -1 and end > start:
                                json_content = json_content[start:end].strip()
                                json_extraction_logger.info(f"After markdown extraction: {len(json_content)} chars")
                        elif json_content.strip().startswith('```') and json_content.strip().endswith('```'):
                            # Handle generic code blocks
                            json_extraction_logger.info("Found generic code blocks, extracting...")
                            lines = json_content.strip().split('\n')
                            if len(lines) > 2:
                                json_content = '\n'.join(lines[1:-1])
                                json_extraction_logger.info(f"After generic block extraction: {len(json_content)} chars")
                        
                        # Additional cleanup for any remaining markdown artifacts
                        json_content = json_content.strip()
                        if json_content.startswith('```'):
                            json_content = json_content[3:].strip()
                            json_extraction_logger.info("Removed leading ```")
                        if json_content.endswith('```'):
                            json_content = json_content[:-3].strip()
                            json_extraction_logger.info("Removed trailing ```")
                        
                        # Clean up any remaining language identifier (like "json")
                        if json_content.startswith('json'):
                            json_content = json_content[4:].strip()
                            json_extraction_logger.info("Removed json language identifier")
                        
                        # Find the actual JSON object start
                        json_start = json_content.find('{')
                        if json_start != -1:
                            json_content = json_content[json_start:]
                            json_extraction_logger.info(f"Found JSON start at position {json_start}")
                        
                        json_extraction_logger.info(f"Final extracted JSON length: {len(json_content)}")
                        json_extraction_logger.info(f"Final JSON first 500 chars: {json_content[:500]}")
                        json_extraction_logger.info(f"Final JSON last 500 chars: {json_content[-500:]}")
                        
                        # Use the single container analysis renderer
                        render_analysis_single_container(json_content)
                        
                        # Add token counter
                        st.markdown(f"""
                        <div class="token-counter">~{len(message['content'])//4} tokens (Full Analysis)</div>
                        """, unsafe_allow_html=True)
                    else:
                        # Regular chat message format for follow-ups
                        st.markdown(f"""
                        <div class="chat-message assistant-message">
                            <strong>FinAlphaAdvisor:</strong> {message['content']}
                            <div class="token-counter">~{len(message['content'])//4} tokens</div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("👋 Welcome! Start a conversation by typing your financial analysis question below.")
    
    # Input area
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Check if we have a preset prompt from sample questions
        initial_value = ""
        if 'preset_prompt' in st.session_state:
            initial_value = st.session_state['preset_prompt']
            del st.session_state['preset_prompt']  # Clear it after using
            
        user_input = st.text_area(
            "Ask your financial analysis question:",
            value=initial_value,
            height=100,
            placeholder="e.g., 'Analyze NVIDIA stock anomalies' or 'What are the risks of this investment?'",
            key="user_input_main"
        )
    
    with col2:
        # Create collapsible Quick Actions pane
        with st.expander("🎯 Quick Actions", expanded=False):
            st.markdown("📋 Sample Questions")
            if st.button("📋 View Sample Questions", use_container_width=True):
                st.session_state['show_sample_questions'] = True
            
            st.markdown("📊 Chart Download")
            if st.session_state.get('last_plot_filename'):
                download_latest_chart(st.session_state['last_plot_filename'])
            else:
                st.info("No chart available yet. Run file analysis first.")
            
            st.markdown("🧪 Testing")
            if st.button("🧪 Test JSON Viewer", use_container_width=True):
                # Add a mock JSON response to test the viewer
                mock_json = '''{
  "disclaimer": "This analysis is for educational purposes only. Not financial advice.",
  "executive_summary": "Test analysis of NVIDIA stock showing strong performance in AI sector.",
  "anomaly_analysis": {
    "key_anomalies": "Major price spikes during Q2 2023 earnings and AI announcements.",
    "news_correlations": "Strong correlation with AI sector developments and earnings beats.",
    "fundamental_impact": "Reinforces NVIDIA's market leadership in AI hardware."
  },
  "actionable_recommendations": [
    {
      "action": "BUY",
      "timeframe": "Long-term (12+ months)",
      "price_targets": "$1200-$1500 within 12-18 months",
      "position_size": "Moderate to High",
      "rationale": "Strong AI market position and growth prospects"
    },
    {
      "action": "HOLD",
      "timeframe": "Medium-term (6-12 months)",
      "price_targets": "Current levels with upside to $1300",
      "position_size": "Maintain current allocation",
      "rationale": "Strong fundamentals support current valuation"
    }
  ],
  "risk_assessment": {
    "primary_risks": ["High valuation", "Market volatility", "Competition risk"],
    "stop_loss_levels": "10-15% below entry point",
    "portfolio_impact": "High beta stock, significant portfolio impact",
    "hedging_options": "Put options or sector diversification"
  },
  "monitoring_plan": {
    "key_dates": ["Q3 2024 earnings", "AI conference dates", "Fed meeting schedules"],
    "price_alerts": "Watch for breaks above $1100 or below $900",
    "news_triggers": "AI sector developments, competitor announcements",
    "review_schedule": "Weekly technical review, monthly fundamental assessment"
  },
  "confidence_score": {
    "rating": "85",
    "explanation": "High confidence based on strong fundamentals and market position",
    "contrary_view": "High valuation could lead to significant corrections"
  }
}'''
                if 'conversation_history' not in st.session_state:
                    st.session_state['conversation_history'] = []
                st.session_state['conversation_history'].append({
                    'role': 'assistant',
                    'content': mock_json
                })
                st.rerun()
            
            st.markdown("📁 File Analysis")
            if st.button("📁 Upload & Analyze File", use_container_width=True):
                st.session_state['current_page'] = 'analysis'
                st.rerun()
            
            st.markdown("⚙️ Settings")
            if st.button("🔧 Advanced Options", use_container_width=True):
                st.session_state['current_page'] = 'admin'
                st.rerun()
    
    # Send button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("💬 Send Message", use_container_width=True):
            if user_input.strip():
                # Show processing progress
                message_progress = create_conversation_progress()
                progress_container = st.empty()
                
                try:
                    # Step 1: Message Processing
                    message_progress.start_step(0)
                    message_progress.render(progress_container)
                    time.sleep(0.3)
                    message_progress.complete_step(0)
                    
                    # Step 2: Context Retrieval
                    message_progress.start_step(1)
                    message_progress.render(progress_container)
                    time.sleep(0.5)
                    message_progress.complete_step(1)
                    
                    # Step 3: AI Generation
                    message_progress.start_step(2)
                    message_progress.render(progress_container)
                    
                    # Process the actual message
                    process_user_message(user_input.strip())
                    
                    message_progress.complete_step(2)
                    
                    # Step 4: Response Formatting
                    message_progress.start_step(3)
                    message_progress.render(progress_container)
                    time.sleep(0.3)
                    message_progress.complete_step(3)
                    
                    # Clear progress and show result
                    progress_container.empty()
                    st.rerun()
                    
                except Exception as e:
                    message_progress.fail_step(message_progress.current_step, f"Error: {str(e)}")
                    message_progress.render(progress_container)
    
    # Show sample questions if requested
    if st.session_state.get('show_sample_questions'):
        show_sample_questions()

def show_analysis_page():
    """Show the file analysis page."""
    st.markdown("### 📊 File Analysis & Conversation Starter")
    
    st.info("Upload your financial data file to automatically start an analysis and begin a conversation about the results.")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload time-series financial data for anomaly detection analysis"
    )
    
    if uploaded_file is not None:
        # Show file preview
        try:
            import pandas as pd
            
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File loaded: {uploaded_file.name}")
            st.write(f"📏 **Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
            
            with st.expander("👀 Preview Data"):
                st.dataframe(df.head(10))
            
            # Analysis configuration
            st.markdown("### ⚙️ Analysis Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                method = st.selectbox(
                    "Detection Method",
                    ["rolling-iqr", "z-score", "iqr", "dbscan"],
                    index=0,
                    help="Choose the anomaly detection method"
                )
            
            with col2:
                threshold = st.number_input(
                    "Threshold",
                    min_value=0.1,
                    max_value=5.0,
                    value=1.5,
                    step=0.1,
                    help="Detection sensitivity threshold"
                )
            
            initial_query = st.text_area(
                "Initial Analysis Question",
                value="Analyze these anomalies for trading opportunities and provide actionable insights.",
                height=100,
                help="This will be your first question in the conversation"
            )
            
            if st.button("🔍 Analyze & Start Conversation", use_container_width=True):
                # Create professional progress indicator
                progress = create_file_analysis_progress()
                progress_container = st.empty()
                
                # Show progress indicator
                progress.render(progress_container)
                
                try:
                    # Step 1: File Upload
                    progress.start_step(0)
                    progress.render(progress_container)
                    time.sleep(0.5)  # Simulate processing time
                    progress.complete_step(0)
                    
                    # Step 2: Data Processing
                    progress.start_step(1)
                    progress.render(progress_container)
                    time.sleep(1.0)
                    progress.complete_step(1)
                    
                    # Step 3: Anomaly Detection
                    progress.start_step(2)
                    progress.render(progress_container)
                    
                    # Call the actual analysis
                    result = analyze_file_and_start_chat(uploaded_file, method, threshold, initial_query)
                    
                    if result:
                        progress.complete_step(2)
                        
                        # Step 4: Visualization
                        progress.start_step(3)
                        progress.render(progress_container)
                        time.sleep(0.5)
                        progress.complete_step(3)
                        
                        # Step 5: AI Analysis
                        progress.start_step(4)
                        progress.render(progress_container)
                        time.sleep(1.0)
                        progress.complete_step(4)
                        
                        # Step 6: Conversation Setup
                        progress.start_step(5)
                        progress.render(progress_container)
                        time.sleep(0.5)
                        progress.complete_step(5)
                        
                        # Final render
                        progress.render(progress_container)
                        
                        # Show success status
                        show_status_card(
                            "Analysis Complete!",
                            "success", 
                            {
                                "File": uploaded_file.name,
                                "Method": method,
                                "Status": "Ready for conversation"
                            }
                        )
                        
                        st.session_state['current_page'] = 'conversation'
                        time.sleep(2)  # Brief pause to show completion
                        st.rerun()
                    else:
                        progress.fail_step(2, "Analysis failed")
                        progress.render(progress_container)
                        
                except Exception as e:
                    current_step = progress.current_step
                    progress.fail_step(current_step, f"Error: {str(e)}")
                    progress.render(progress_container)
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

def show_admin_page():
    """Show the admin configuration page."""
    st.markdown("### ⚙️ Admin Panel")
    
    # Simple password check
    if 'admin_authenticated' not in st.session_state:
        st.session_state['admin_authenticated'] = False
    
    if not st.session_state['admin_authenticated']:
        password = st.text_input("Admin Password", type="password")
        if st.button("🔐 Login"):
            expected_password = os.getenv('ADMIN_PASSWORD', 'secure_admin_password_2024')
            if password == expected_password:
                st.session_state['admin_authenticated'] = True
                st.rerun()
            else:
                st.error("❌ Invalid password")
        return
    
    # Admin content
    st.success("✅ Admin access granted")
    
    tab1, tab2, tab3 = st.tabs(["🔧 Configuration", "📊 Sessions", "📋 Logs"])
    
    with tab1:
        st.markdown("#### System Configuration")
        
        # API status
        api_status = check_api_status()
        if api_status:
            st.success("✅ API Server: Running")
        else:
            st.error("❌ API Server: Not responding")
        
        # Basic config display
        st.markdown("#### Environment Variables")
        config_data = {
            "API_HOST": os.getenv('API_HOST', 'localhost'),
            "API_PORT": os.getenv('API_PORT', '8000'),
            "LLM_MODEL": os.getenv('LLM_MODEL', 'gemini-2.5-flash'),
            "ENABLE_LLM": os.getenv('ENABLE_LLM', 'true'),
            "MAX_CONCURRENT_SESSIONS": os.getenv('MAX_CONCURRENT_SESSIONS', '100'),
            "CONVERSATION_TIMEOUT_MINUTES": os.getenv('CONVERSATION_TIMEOUT_MINUTES', '120')
        }
        
        for key, value in config_data.items():
            col1, col2 = st.columns([1, 2])
            with col1:
                st.text(key)
            with col2:
                st.code(value)
    
    with tab2:
        st.markdown("#### Active Sessions")
        if st.button("🔄 Refresh Session Data"):
            st.rerun()
        
        # This would show session data if API is available
        st.info("Session monitoring will be available when conversation system is active.")
    
    with tab3:
        st.markdown("#### System Logs")
        st.info("Log viewing functionality will be implemented in future updates.")

def show_sample_prompts():
    """Show sample prompts and templates."""
    st.markdown("### 📋 Sample Analysis Prompts")
    
    # Load sample prompts
    sample_prompts = load_sample_prompts()
    
    for category, prompts in sample_prompts.items():
        st.markdown(f"#### 📁 {category}")
        
        for prompt in prompts:
            with st.expander(f"💡 {prompt['title']}"):
                st.markdown(f"**Description:** {prompt['description']}")
                st.markdown("**Template:**")
                st.code(prompt['template'], language="text")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Use This Prompt", key=f"use_{prompt['title']}"):
                        # Set up conversation with this prompt
                        st.session_state['preset_prompt'] = prompt['template']
                        st.session_state['current_page'] = 'conversation'
                        st.session_state['show_samples'] = False
                        st.rerun()
                
                with col2:
                    if 'recommended_method' in prompt:
                        st.info(f"Recommended: {prompt['recommended_method']}")

def show_sample_questions():
    """Show quick sample questions."""
    st.markdown("#### 💡 Sample Questions to Get Started")
    
    sample_questions = [
        "Analyze NVIDIA stock price anomalies for trading opportunities",
        "What are the key risk factors I should consider for this investment?",
        "How do these price anomalies correlate with earnings announcements?",
        "Provide a comprehensive risk assessment for this position",
        "What would be an appropriate position size for moderate risk tolerance?",
        "How does this stock compare to its sector peers?"
    ]
    
    for i, question in enumerate(sample_questions):
        if st.button(f"💬 {question}", key=f"sample_q_{i}"):
            st.session_state['preset_prompt'] = question
            st.session_state['show_sample_questions'] = False
            st.rerun()  # Trigger rerun to apply the preset prompt

def load_sample_prompts():
    """Load sample prompts from JSON files."""
    sample_prompts = {
        "Stock Analysis": [],
        "Risk Management": [],
        "Market Analysis": []
    }
    
    # This is a simplified version - in full implementation, would load from JSON files
    sample_prompts["Stock Analysis"].append({
        "title": "NVIDIA Analysis",
        "description": "Comprehensive NVIDIA stock analysis with AI market focus",
        "template": "Analyze NVIDIA stock price anomalies for mid-term trading opportunities. Focus on AI market trends, semiconductor sector performance, and provide specific entry points and risk management strategies.",
        "recommended_method": "rolling-iqr"
    })
    
    return sample_prompts

def process_user_message(message):
    """Process user message and get AI response."""
    # Add user message to history
    st.session_state['conversation_history'].append({
        'role': 'user',
        'content': message
    })
    
    # Simulate AI response (in full implementation, would call API)
    ai_response = get_ai_response(message)
    
    # Add AI response to history
    st.session_state['conversation_history'].append({
        'role': 'assistant', 
        'content': ai_response
    })
    
    # Update session stats
    st.session_state['message_count'] = len(st.session_state['conversation_history'])
    st.session_state['total_tokens'] = sum(len(msg['content'])//4 for msg in st.session_state['conversation_history'])

def get_ai_response(message):
    """Get AI response by calling the conversation API."""
    try:
        import requests
        
        # Check if we have an existing conversation session
        if 'conversation_id' in st.session_state:
            # Continue existing conversation
            response = requests.post(
                f"http://localhost:8000/api/v1/conversation/{st.session_state['conversation_id']}/message",
                json={"message": message},
                timeout=120
            )
        else:
            # Start new conversation
            response = requests.post(
                "http://localhost:8000/api/v1/conversation/start",
                json={
                    "initial_query": message,
                    "user_id": "streamlit_user"
                },
                timeout=120
            )
        
        if response.status_code == 200:
            data = response.json()
            
            # Store session info
            st.session_state['conversation_id'] = data['session_id']
            st.session_state['message_count'] = data['message_count']
            st.session_state['total_tokens'] = data['total_tokens']
            
            return data['response']
        else:
            return f"❌ API Error ({response.status_code}): {response.text}"
            
    except requests.exceptions.ConnectionError:
        return """❌ **Connection Error**: Cannot connect to the API server.

**To fix this:**
1. Start the API server in a separate terminal:
   ```bash
   cd "/mnt/e/Agents2025/Assignment25"
   ./client_venv/Scripts/activate
   ./client_venv/Scripts/python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
2. Refresh this page and try again

**Expected API endpoint:** http://localhost:8000/api/v1/conversation/start"""
        
    except Exception as e:
        return f"❌ **Unexpected Error**: {str(e)}\n\nPlease check the API server status and try again."

def download_latest_chart(plot_filename):
    """Download the latest chart using the API endpoint."""
    try:
        import requests
        
        # Call the download-plot API endpoint
        response = requests.get(
            f"http://localhost:8000/api/v1/download-plot/{plot_filename}",
            timeout=30
        )
        
        if response.status_code == 200:
            # Create download link using streamlit
            st.download_button(
                label="📥 Download Chart",
                data=response.content,
                file_name=plot_filename,
                mime="image/png",
                use_container_width=True
            )
            st.success(f"✅ Chart ready for download: {plot_filename}")
        else:
            st.error(f"❌ Failed to download chart: {response.status_code}")
            
    except Exception as e:
        st.error(f"❌ Download error: {str(e)}")

def analyze_file_and_start_chat(file, method, threshold, query):
    """Analyze file and start conversation using the real API."""
    try:
        import requests
        
        # Prepare file for upload
        files = {'uploaded_file': (file.name, file.getvalue(), file.type)}
        data = {
            'method': method,
            'threshold': threshold,
            'initial_query': query
        }
        
        # Call the analyze-and-chat API endpoint
        response = requests.post(
            "http://localhost:8000/api/v1/conversation/analyze-and-chat",
            files=files,
            data=data,
            timeout=120  # Longer timeout for file processing
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Store conversation info
            st.session_state['conversation_id'] = result['conversation']['session_id']
            st.session_state['message_count'] = result['conversation']['message_count']
            
            # Extract and store plot filename for download functionality
            if 'analysis_result' in result and 'visualization' in result['analysis_result']:
                plot_path = result['analysis_result']['visualization']['plot_path']
                # Extract just the filename from the full path
                import os
                plot_filename = os.path.basename(plot_path)
                st.session_state['last_plot_filename'] = plot_filename
                
                # Log for debugging
                logger.info(f"Stored plot filename for download: {plot_filename}")
            
            # Set up conversation history with analysis result
            # Store the clean JSON response directly for proper rendering
            st.session_state['conversation_history'] = [
                {
                    'role': 'assistant',
                    'content': result['conversation']['initial_response'],  # Direct JSON response
                    'file_analysis_metadata': {
                        'file_name': file.name,
                        'method_used': result['analysis_result']['method_used'],
                        'total_points': result['analysis_result']['total_points'],
                        'anomaly_count': result['analysis_result']['anomaly_count'],
                        'anomaly_percentage': result['analysis_result']['anomaly_percentage'],
                        'original_name': result['file_info']['original_name'],
                        'file_size': result['file_info']['file_size'],
                        'processing_time_ms': result['processing_time_ms']
                    }
                }
            ]
            return True
        else:
            st.error(f"❌ API Error ({response.status_code}): {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        st.error("""❌ **Connection Error**: Cannot connect to the API server.
        
**To fix this:**
1. Start the API server in a separate terminal:
   ```bash
   cd "/mnt/e/Agents2025/Assignment25"
   ./client_venv/Scripts/activate
   ./client_venv/Scripts/python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
2. Try uploading your file again""")
        return False
        
    except Exception as e:
        st.error(f"❌ **Unexpected Error**: {str(e)}")
        return False

def clear_session():
    """Clear conversation session."""
    keys_to_clear = ['conversation_history', 'conversation_id', 'message_count', 'total_tokens']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

if __name__ == "__main__":
    main()