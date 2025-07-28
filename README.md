# FinAlphaAdvisor - AI-Powered Financial Anomaly Detection System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-🦜🔗-green.svg)](https://langchain.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated **multi-agent AI system** for detecting anomalies in financial time-series data with **multi-turn conversation capabilities**. Built with **LangChain/LangGraph** and **Google Gemini 2.0 Flash**, featuring an interactive **Streamlit UI** and comprehensive **REST API**.

## 🚀 Key Features

### 🤖 **Multi-Agent Architecture**
- **LangChain/LangGraph-based** agent coordination
- **Multi-turn conversation** capabilities for interactive analysis
- **Intelligent suggestion agent** for contextual recommendations
- **Enhanced conversation workflow** with memory and context tracking

### 🔍 **Advanced Detection Methods**
- **Z-Score Analysis**: Statistical outlier detection
- **IQR (Interquartile Range)**: Robust outlier detection for skewed data
- **Rolling IQR**: Time-window based dynamic threshold detection
- **DBSCAN Clustering**: Density-based anomaly detection

### 💬 **Interactive Interfaces**
- **Streamlit Web UI**: Professional, interactive dashboard
- **Multi-turn Chat**: Contextual conversations about your data
- **REST API**: Programmatic access with comprehensive endpoints
- **CLI Interface**: Command-line tools for automation

### 📊 **Rich Visualizations**
- **Matplotlib & Plotly** interactive charts
- **Anomaly highlighting** with detailed annotations
- **Real-time plot generation** and download capabilities
- **Professional financial data visualization**

### 🧠 **AI-Powered Insights**
- **Google Gemini 2.0 Flash** integration for intelligent analysis
- **Financial expertise**: Specialized in market data interpretation
- **Root cause analysis** and actionable recommendations
- **Confidence scoring** for reliability assessment

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      FinAlphaAdvisor System                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐  │
│  │  Streamlit UI   │    │   FastAPI       │    │     CLI     │  │
│  │                 │    │                 │    │             │  │
│  │ • Interactive   │    │ • REST API      │    │ • Batch     │  │
│  │ • Multi-turn    │    │ • File Upload   │    │ • Scripts   │  │
│  │ • Visualizations│    │ • Swagger Docs  │    │ • Automation│  │
│  └─────────────────┘    └─────────────────┘    └─────────────┘  │
│           │                        │                     │      │
│           └────────────────────────┼─────────────────────┘      │
│                                    │                            │
│                     ┌─────────────────────────────┐             │
│                     │     Conversation Manager    │             │
│                     │                             │             │
│                     │ • Multi-turn conversations  │             │
│                     │ • Context management        │             │
│                     │ • Session handling          │             │
│                     └─────────────────────────────┘             │
│                                    │                            │
│         ┌──────────────────────────┼──────────────────────────┐ │
│         │                          │                          │ │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ 
│  │ Anomaly Agent   │    │ Suggestion Agent│    │   Supervisor    │ 
│  │                 │    │                 │    │                 │ 
│  │ • File Reading  │    │ • Financial     │    │ • LangGraph     │ 
│  │ • Detection     │    │   Insights      │    │ • Workflow      │
│  │ • Visualization │    │ • Multi-turn    │    │ • Orchestration │ 
│  │ • Analysis      │    │   Suggestions   │    │ • Error Handling│ 
│  └─────────────────┘    └─────────────────┘    └─────────────────┘ 
│         │                          │                          │ │
│         └──────────────────────────┼──────────────────────────┘ │
│                                    │                            │
│                     ┌─────────────────────────────┐             │
│                     │         Tool Suite          │             │
│                     │                             │             │
│                     │ • FileReader                │             │
│                     │ • AnomalyDetector           │             │
│                     │ • Visualizer                │             │
│                     │ • IntelligentInsightGen     │             │
│                     └─────────────────────────────┘             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📋 Requirements

- **Python 3.9+** (Recommended: 3.11+)
- **Google AI API Key** (for Gemini 2.0 Flash)
- **Virtual Environment** (recommended)
- **Dependencies** listed in `requirements.txt`

## ⚡ Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd FinAlphaAdvisor

# Create and activate virtual environment
python -m venv venv_linux
source venv_linux/bin/activate  # On Windows: venv_linux\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the project root:

```env
# Required: Google AI API Key
GOOGLE_AI_API_KEY=your_google_ai_api_key_here

# Optional: Model Configuration
LLM_MODEL=gemini-2.0-flash
LLM_TEMPERATURE=0.1

# Optional: API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Optional: Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HOST=0.0.0.0

# Optional: Directory Configuration
DATA_DIR=./data
OUTPUT_DIR=./outputs
PLOTS_DIR=./outputs/plots
LOG_LEVEL=INFO
```

**Get Google AI API Key:**
1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Create/sign in to your account
3. Generate an API key
4. Add the key to your `.env` file

### 3. Usage Options

#### Option A: Streamlit Web Interface (Recommended)

```bash
# Start the Streamlit app
streamlit run streamlit_app/main.py

# Or use the Python module
python -m streamlit run streamlit_app/main.py
```

**Access the web interface at: http://localhost:8501**

**Features:**
- 📁 **File Upload**: Drag & drop CSV/Excel files
- 🔍 **Interactive Analysis**: Select detection methods and parameters
- 💬 **Multi-turn Chat**: Ask questions about your data
- 📊 **Live Visualizations**: Interactive charts and plots
- 💾 **Download Results**: Save plots and analysis reports

#### Option B: REST API

```bash
# Start the FastAPI server
python -m api.main

# Or with uvicorn
s
```

**Access the API documentation at: http://localhost:8000/docs**

#### Option C: Command Line Interface

```bash
# Interactive mode
python -m cli.main interactive

# Direct analysis
python -m cli.main analyze data.csv --method z-score --threshold 3.0

# Validate data
python -m cli.main validate data.csv
```

## 🔍 Detection Methods

### 1. Z-Score Method
- **Best for**: Normally distributed financial data
- **How it works**: Identifies points > N standard deviations from mean
- **Parameters**: `threshold` (default: 3.0)
- **Use case**: Stock prices, trading volumes, returns

### 2. IQR (Interquartile Range) Method
- **Best for**: Skewed financial distributions
- **How it works**: Uses quartile ranges to identify outliers
- **Parameters**: `multiplier` (default: 1.5)
- **Use case**: Revenue data, profit margins, volatile assets

### 3. Rolling IQR Method
- **Best for**: Time-sensitive financial data
- **How it works**: Dynamic thresholds using rolling windows
- **Parameters**: `window_size` (default: 20), `multiplier` (default: 1.5)
- **Use case**: Intraday trading, real-time monitoring

### 4. DBSCAN Method
- **Best for**: Complex pattern detection
- **How it works**: Density-based clustering for anomaly detection
- **Parameters**: `eps` (default: 0.5), `min_samples` (default: 5)
- **Use case**: Fraud detection, irregular market behavior

## 🌐 API Endpoints

### Core Analysis Endpoints
- `POST /api/v1/analyze` - Complete anomaly detection with insights
- `POST /api/v1/upload-file` - Upload and process data files
- `GET /api/v1/methods` - Available detection methods
- `GET /api/v1/download-plot/{filename}` - Download generated plots
- `GET /health` - System health check

### Conversation Endpoints
- `POST /api/v1/conversation/start` - Start new conversation session
- `POST /api/v1/conversation/{session_id}/message` - Send message in conversation
- `GET /api/v1/conversation/{session_id}/history` - Get conversation history
- `DELETE /api/v1/conversation/{session_id}` - End conversation session

### Example API Usage

```bash
# Start analysis
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "data/Sample.csv",
    "method": "rolling-iqr",
    "threshold": 1.5,
    "query": "Analyze 1 year data NVIDIA stock for trading anomalies"
  }'

# Start conversation
curl -X POST "http://localhost:8000/api/v1/conversation/start" \
  -H "Content-Type: application/json" \
  -d '{
    "initial_message": "What caused the anomalies in my data?",
    "context": {"analysis_id": "analysis_123"}
  }'
```

## 💬 Multi-turn Conversation Features

### Conversation Capabilities
- **Context Awareness**: Remembers previous questions and analysis
- **Financial Expertise**: Specialized knowledge about market data
- **Follow-up Questions**: Natural conversation flow
- **Actionable Insights**: Specific recommendations based on your data

### Example Conversation Flow
```
User: "What anomalies did you find in my stock data?"
AI: "I found 5 significant anomalies in your NVIDIA stock data representing 2.3% of trading days. These include notable spikes on..."

User: "What might have caused the spike on March 15th?"
AI: "The March 15th spike coincided with NVIDIA's quarterly earnings announcement. The 23% price increase suggests positive market reaction to better-than-expected results..."

User: "Should I be concerned about these anomalies?"
AI: "These anomalies appear to be event-driven rather than concerning. I recommend monitoring for unusual patterns without fundamental catalysts..."
```

## 📊 Streamlit Interface Features

### Dashboard Components
- **📁 File Upload Area**: Drag & drop interface with format validation
- **⚙️ Analysis Controls**: Method selection, parameter tuning
- **📈 Live Visualization**: Real-time chart updates
- **💬 Chat Interface**: Multi-turn conversations
- **📋 Results Summary**: Comprehensive analysis reports
- **💾 Export Options**: Download plots and data

### Professional UI Elements
- **Dark Theme**: Professional financial interface
- **Responsive Design**: Works on desktop and mobile
- **Progress Indicators**: Real-time analysis progress
- **Error Handling**: User-friendly error messages
- **Session Management**: Persistent conversation history

## 🛠️ Development

### Project Structure

```
FinAlphaAdvisor/
├── streamlit_app/              # Streamlit web interface
│   ├── main.py                # Main Streamlit application
│   ├── components/            # UI components
│   │   ├── analysis_renderer.py  # Analysis display
│   │   ├── json_viewer.py     # JSON visualization
│   │   └── progress_indicator.py # Progress displays
│   └── utils/                 # Utility functions
│       └── styles.py          # CSS styling
├── agents/                    # Multi-agent system
│   ├── supervisor.py         # LangGraph workflow coordinator
│   ├── anomaly_agent.py      # Detection agent
│   ├── conversation_manager.py # Multi-turn conversation
│   ├── conversation_workflow.py # Conversation orchestration
│   ├── enhanced_suggestion_agent.py # AI suggestions
│   ├── llm_logger.py         # LLM interaction logging
│   └── tools/                # Agent tools
│       ├── file_reader.py    # Data ingestion
│       ├── anomaly_detector.py # Detection algorithms
│       ├── visualizer.py     # Chart generation
│       └── intelligent_insight_generator.py # AI insights
├── api/                      # FastAPI application
│   ├── main.py              # FastAPI app
│   ├── endpoints.py         # Main API routes
│   ├── conversation_endpoints.py # Conversation API
│   ├── models.py            # API models
│   └── dependencies.py      # Dependency injection
├── cli/                     # Command-line interface
│   └── main.py              # CLI implementation
├── core/                    # Core utilities
│   ├── config.py           # Configuration management
│   ├── models.py           # Data models
│   ├── conversation_models.py # Conversation models
│   ├── exceptions.py       # Custom exceptions
│   └── prompt_manager.py   # Prompt management
├── tests/                   # Comprehensive test suite
│   ├── test_core.py        # Core functionality tests
│   ├── test_api.py         # API endpoint tests
│   ├── test_agents.py      # Agent system tests
│   ├── test_tools.py       # Tool tests
│   ├── test_streamlit.py   # UI component tests
│   └── conftest.py         # Test configuration
├── config/                  # Configuration files
├── data/                    # Sample data files
├── outputs/plots/           # Generated visualizations
└── logs/                    # Application logs
```

### Running Tests

```bash
# Install test dependencies (already in requirements.txt)
pip install pytest pytest-asyncio pytest-cov

# Run all tests
pytest

# Run with coverage report
pytest --cov=. --cov-report=html --cov-report=term-missing

# Run specific test modules
pytest tests/test_core.py -v
pytest tests/test_agents.py -v
pytest tests/test_api.py -v

# Open HTML coverage report
# Navigate to htmlcov/index.html in your browser
```

**Test Coverage Summary:**
- **Core Models**: 100% coverage ✅
- **API Endpoints**: 90% coverage ✅
- **Agent Tools**: 76% coverage ✅
- **Overall**: 47% coverage (2,818/5,333 lines)

### Code Quality

```bash
# Format code
black .

# Lint code
ruff check . --fix

# Type checking
mypy .
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `GOOGLE_AI_API_KEY` | Google AI API key for Gemini | - | ✅ Yes |
| `LLM_MODEL` | LLM model to use | gemini-2.0-flash | No |
| `LLM_TEMPERATURE` | LLM temperature (0-2) | 0.1 | No |
| `API_HOST` | FastAPI host | 0.0.0.0 | No |
| `API_PORT` | FastAPI port | 8000 | No |
| `STREAMLIT_SERVER_PORT` | Streamlit port | 8501 | No |
| `STREAMLIT_SERVER_HOST` | Streamlit host | 0.0.0.0 | No |
| `DATA_DIR` | Data directory | ./data | No |
| `OUTPUT_DIR` | Output directory | ./outputs | No |
| `PLOTS_DIR` | Plots directory | ./outputs/plots | No |
| `LOG_LEVEL` | Logging level | INFO | No |

### Detection Parameters

| Method | Parameter | Description | Default |
|--------|-----------|-------------|---------|
| Z-Score | `threshold` | Standard deviations from mean | 3.0 |
| IQR | `multiplier` | IQR multiplier for bounds | 1.5 |
| Rolling IQR | `window_size` | Rolling window size | 20 |
| Rolling IQR | `multiplier` | IQR multiplier | 1.5 |
| DBSCAN | `eps` | Maximum distance between points | 0.5 |
| DBSCAN | `min_samples` | Minimum samples in cluster | 5 |

## 📈 Use Cases - CLI usage
 
### 1. Financial Market Analysis
```python
# Analyze stock price anomalies
python -m cli.main analyze stock_data.csv --method rolling-iqr --threshold 1.5
```

### 2. Trading Volume Monitoring
```python
# Detect unusual trading patterns
python -m cli.main analyze trading_volume.csv --method z-score --threshold 3.0
```

### 3. Revenue Anomaly Detection
```python
# Identify revenue outliers
python -m cli.main analyze revenue_data.csv --method iqr --threshold 2.0
```

### 4. Real-time Market Monitoring
- Use the **Streamlit interface** for live monitoring
- **Multi-turn conversations** for contextual analysis
- **Rolling detection methods** for dynamic thresholds

## 🚀 Deployment

### Local Development

```bash
# Start all services
# Terminal 1: API Server
python -m api.main

# Terminal 2: Streamlit UI
streamlit run streamlit_app/main.py

# Terminal 3: CLI usage
python -m cli.main interactive
```

### Production Considerations

1. **Security**: API authentication, rate limiting, CORS configuration
2. **Scaling**: Load balancing, horizontal scaling, caching
3. **Monitoring**: Application metrics, error tracking, performance monitoring
4. **Data Management**: Persistent storage, backup strategies, data retention
5. **Performance**: Redis caching, database optimization, CDN for static assets

## 🔍 Troubleshooting

### Common Issues

**1. API Key Configuration**
```
Error: Google AI API key must be provided
```
- Set `GOOGLE_AI_API_KEY` in `.env` file
- Verify key is valid at [Google AI Studio](https://aistudio.google.com/)
- Check for extra spaces or quotes

**2. File Upload Issues**
```
Error: Invalid file format
```
- Ensure file is CSV or Excel format
- Check for proper timestamp column
- Two column must for Date and Value
- Verify data has minimum required rows (10+)

**3. Streamlit Connection Error**
```
Error: Connection refused
```
- Check if port 8501 is available
- Verify `STREAMLIT_SERVER_PORT` configuration
- Try different port: `streamlit run streamlit_app/main.py --server.port 8502`

**4. Multi-turn Conversation Issues**
```
Error: Session not found
```
- Conversation sessions expire after inactivity
- Start new session if needed
- Check session ID format

# Check logs
tail -f logs/app.log
```


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[LangChain](https://langchain.com/)** - Multi-agent framework and orchestration
- **[Google Gemini](https://ai.google.dev/)** - Advanced AI language model
- **[FastAPI](https://fastapi.tiangolo.com/)** - High-performance web framework
- **[Streamlit](https://streamlit.io/)** - Interactive web application framework
- **[Matplotlib](https://matplotlib.org/)** - Visualization libraries
- **[Pydantic](https://pydantic.dev/)** - Data validation and settings management

*Powered by LangChain , LangGraph, Google Gemini AI, and Streamlit*

**Version**: 2.0.0 | **Test Coverage**: 47% | **Python**: 3.9+