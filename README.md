# Time-Series Anomaly Detection Multi-Agent System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-🦜🔗-green.svg)](https://langchain.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated **multi-agent AI system** for detecting anomalies in time-series data, built with **LangChain/LangGraph** and **Google Gemini 2.0 Flash**. The system combines statistical analysis with intelligent LLM-powered insights to provide actionable recommendations for financial, operational, and business data.

## 🚀 Features

- **Multi-Agent Architecture**: LangChain/LangGraph-based agent coordination
- **Multiple Detection Methods**: Z-score, IQR, and DBSCAN algorithms
- **AI-Powered Insights**: Google Gemini LLM for generating explanations and recommendations
- **Interactive Visualizations**: Matplotlib and Plotly chart generation
- **Dual Interfaces**: Both CLI and REST API access
- **File Support**: CSV and Excel file processing
- **Production Ready**: Comprehensive error handling, logging, and testing

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Anomaly Agent │    │ Suggestion Agent│    │   Supervisor    │
│                 │    │                 │    │                 │
│ • File Reading  │    │ • Insight Gen   │    │ • Workflow      │
│ • Detection     │    │ • Recommendations│    │ • Coordination  │
│ • Visualization │    │ • Root Causes   │    │ • Error Handling│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                     ┌─────────────────┐
                     │   LangGraph     │
                     │   Workflow      │
                     └─────────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         │                        │                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │      CLI        │    │   Tools         │
│                 │    │                 │    │                 │
│ • REST Endpoints│    │ • Interactive   │    │ • FileReader    │
│ • Swagger Docs  │    │ • Commands      │    │ • AnomalyDetector│
│ • File Upload   │    │ • Rich Output   │    │ • Visualizer    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 Requirements

- Python 3.9+
- Google AI API Key (for Gemini LLM)
- Dependencies listed in `requirements.txt`

## ⚡ Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>


cd Context-Engineering-Intro

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your configuration
# Required: GOOGLE_AI_API_KEY=your_api_key_here
```

**Get Google AI API Key:**
1. Visit [Google AI Studio](https://ai.google.dev/)
2. Create an account and generate an API key
3. Add the key to your `.env` file

### 3. Usage Options

#### Option A: CLI Interface

```bash
# Interactive mode
python -m cli.main interactive

# Direct analysis
python -m cli.main analyze data.csv --method z-score --threshold 3.0

# Validate data
python -m cli.main validate data.csv

# Get method recommendation
python -m cli.main methods
```

#### Option B: REST API

```bash
# Start the API server
python -m api.main

# Or with uvicorn
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Access the interactive API documentation at: http://localhost:8000/docs

## 🔍 Detection Methods

### Z-Score Method
- **Best for**: Normally distributed data
- **How it works**: Identifies points more than N standard deviations from mean
- **Default threshold**: 3.0
- **Use case**: Financial data, sensor readings with known distributions

### IQR (Interquartile Range) Method  
- **Best for**: Skewed data distributions
- **How it works**: Uses quartile ranges to identify outliers
- **Default multiplier**: 1.5
- **Use case**: Sales data, web traffic, asymmetric distributions

### DBSCAN Method
- **Best for**: Complex patterns and irregular distributions
- **How it works**: Density-based clustering to find isolated points
- **Default parameters**: eps=0.5, min_samples=5
- **Use case**: IoT sensor networks, irregular time series

## 📊 API Endpoints

### Core Endpoints

- `POST /api/v1/analyze` - Complete anomaly detection analysis
- `POST /api/v1/quick-analyze` - Fast detection without insights
- `POST /api/v1/recommend-method` - Get method recommendation
- `POST /api/v1/validate-data` - Validate data compatibility
- `POST /api/v1/upload-file` - Upload data files
- `GET /api/v1/status` - API health check
- `GET /api/v1/methods` - Available detection methods

### Example API Request

```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "data/sample.csv",
    "method": "z-score",
    "threshold": 3.0,
    "query": "Find anomalies in sales data",
    "include_visualization": true,
    "include_insights": true
  }'
```

### Example API Response

```json
{
  "success": true,
  "anomaly_result": {
    "anomaly_count": 5,
    "anomaly_percentage": 2.5,
    "method_used": "z-score",
    "threshold_used": 3.0,
    "total_points": 200,
    "anomaly_indices": [45, 67, 123, 156, 189],
    "anomaly_values": [150.5, 145.8, 162.3, 158.9, 171.2]
  },
  "visualization": {
    "plot_path": "/outputs/plots/anomaly_plot_20250107_123456.png",
    "plot_description": "Time-series plot with 5 highlighted anomalies"
  },
  "insights": {
    "summary": "Analysis detected 5 significant anomalies representing 2.5% of the data...",
    "recommendations": [
      "Investigate data collection process during anomaly periods",
      "Consider implementing real-time monitoring for similar patterns"
    ],
    "confidence_score": 85
  },
  "processing_time": 3.45
}
```

## 🛠️ Development

### Project Structure

```
Context-Engineering-Intro/
├── agents/                    # Multi-agent system
│   ├── anomaly_agent.py      # Primary detection agent
│   ├── suggestion_agent.py   # Insights generation agent
│   ├── supervisor.py         # LangGraph coordinator
│   └── tools/                # Agent tools
│       ├── file_reader.py    # Data ingestion
│       ├── anomaly_detector.py # Detection algorithms
│       ├── visualizer.py     # Chart generation
│       └── insight_generator.py # LLM insights
├── api/                      # FastAPI application
│   ├── main.py              # FastAPI app
│   ├── endpoints.py         # API routes
│   ├── models.py            # API models
│   └── dependencies.py      # Dependency injection
├── cli/                     # Command-line interface
│   └── main.py              # CLI implementation
├── core/                    # Core utilities
│   ├── config.py           # Configuration management
│   ├── models.py           # Data models
│   └── exceptions.py       # Custom exceptions
├── tests/                   # Test suite
├── DataSource/             # Sample data
└── examples/               # Usage examples
```

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=agents --cov=api --cov=core

# Run specific test module
pytest tests/test_tools.py -v
```

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
| `GOOGLE_AI_API_KEY` | Google AI API key for Gemini | - | Yes |
| `LLM_MODEL` | LLM model to use | gemini-2.0-flash | No |
| `LLM_TEMPERATURE` | LLM temperature (0-2) | 0.1 | No |
| `API_HOST` | FastAPI host | 0.0.0.0 | No |
| `API_PORT` | FastAPI port | 8000 | No |
| `DATA_DIR` | Data directory | ./data | No |
| `OUTPUT_DIR` | Output directory | ./outputs | No |
| `PLOTS_DIR` | Plots directory | ./outputs/plots | No |
| `LOG_LEVEL` | Logging level | INFO | No |

### Detection Parameters

| Method | Parameter | Description | Default |
|--------|-----------|-------------|---------|
| Z-Score | `threshold` | Standard deviations from mean | 3.0 |
| IQR | `multiplier` | IQR multiplier for bounds | 1.5 |
| DBSCAN | `eps` | Maximum distance between points | 0.5 |
| DBSCAN | `min_samples` | Minimum samples in cluster | 5 |

## 📈 Use Cases

### 1. Financial Monitoring
```bash
# Detect anomalies in stock prices
python -m cli.main analyze stock_data.csv --method z-score --threshold 3.0
```

### 2. IoT Sensor Monitoring
```bash
# Complex pattern detection in sensor data
python -m cli.main analyze sensor_data.csv --method dbscan
```

### 3. Web Analytics
```bash
# Traffic anomaly detection
python -m cli.main analyze traffic_data.csv --method iqr
```

### 4. Sales Analysis
```bash
# Revenue anomaly detection with insights
python -m cli.main analyze sales_data.csv --method iqr --threshold 1.5
```

## 🚀 Deployment

### Docker Deployment

```bash
# Build image
docker build -t anomaly-detector .

# Run container
docker run -p 8000:8000 -e GOOGLE_AI_API_KEY=your_key anomaly-detector
```

### Production Considerations

1. **Security**: Implement proper authentication and rate limiting
2. **Scaling**: Use multiple workers and load balancing
3. **Monitoring**: Add metrics collection and alerting
4. **Data**: Implement data persistence and backup strategies
5. **Caching**: Add Redis/Memcached for performance optimization

## 🔍 Troubleshooting

### Common Issues

**1. API Key Not Working**
```
Error: Configuration error: Google AI API key not configured
```
- Verify API key is set in `.env` file
- Check key validity at Google AI Studio
- Ensure no extra spaces or quotes around the key

**2. File Reading Errors**
```
Error: File processing error: Could not detect timestamp column
```
- Ensure CSV has timestamp column (Date, Time, Timestamp, etc.)
- Check data format and encoding
- Verify file has at least 10 data points

**3. Memory Issues with Large Files**
```
Error: Memory error during processing
```
- Process data in chunks for large files
- Reduce visualization complexity
- Consider using data sampling

**4. LLM Timeout Errors**
```
Error: LLM processing timeout
```
- Check internet connectivity
- Verify API key has sufficient quota
- Reduce data complexity or use quick-analyze endpoint

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
python -m cli.main analyze data.csv --verbose
```

### Log Files

- Application logs: `./logs/app.log`
- Error details: Check console output
- API access logs: Uvicorn logs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run tests and linting (`pytest && ruff check .`)
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain** - Agent framework and orchestration
- **Google Gemini** - Advanced language model capabilities  
- **FastAPI** - High-performance web framework
- **Plotly/Matplotlib** - Visualization libraries
- **Pydantic** - Data validation and settings management

## 📞 Support

- **Documentation**: Check this README and API docs at `/docs`
- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Use GitHub Discussions for questions
- **Examples**: See `examples/` directory for sample usage

---

Built with ❤️ using LangChain, LangGraph, and Google Gemini AI