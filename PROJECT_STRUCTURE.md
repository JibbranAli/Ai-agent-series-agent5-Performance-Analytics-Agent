# AI Performance Analytics Agent - Project Structure

## 📁 Project Organization

```
ai_agent/
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── env.example            # Environment variables template
├── README.md              # Project documentation
├── PROJECT_STRUCTURE.md   # This file
│
├── config/                # Configuration files
│   ├── __init__.py
│   └── settings.py        # Application settings
│
├── src/                   # Core source code
│   ├── __init__.py
│   ├── agent.py          # AI Agent implementation
│   ├── data_processor.py # Data processing logic
│   └── visualizer.py     # Data visualization
│
├── dashboards/           # Streamlit dashboard implementations
│   ├── __init__.py
│   ├── professional_dashboard.py  # Main production dashboard
│   ├── beautiful_dashboard.py     # Alternative UI design
│   ├── compact_dark_dashboard.py  # Compact dark theme
│   ├── dark_tabs_dashboard.py     # Tabbed dark interface
│   ├── redesigned_dashboard.py    # Redesigned version
│   ├── unified_dashboard.py       # Unified interface
│   ├── final_dashboard.py         # Final version
│   ├── production_dashboard.py    # Production ready
│   ├── chat_dashboard.py          # Chat-focused interface
│   ├── app.py                     # Legacy app
│   └── simple_app.py              # Simple version
│
├── data/                 # Data storage
│   ├── sample_data.csv   # Sample datasets
│   └── uploads/          # User uploaded files
│
├── reports/              # Generated reports
│   ├── pdf/              # PDF reports
│   ├── html/             # HTML reports
│   └── docx/             # Word documents
│
├── logs/                 # Application logs
│   ├── app.log           # Main application log
│   └── error.log         # Error logs
│
├── tests/                # Test files
│   ├── __init__.py
│   ├── test_agent.py     # Agent tests
│   ├── test_data_processor.py  # Data processor tests
│   └── test_visualizer.py      # Visualizer tests
│
├── docs/                 # Documentation
│   ├── api.md            # API documentation
│   ├── user_guide.md     # User guide
│   └── deployment.md     # Deployment guide
│
├── assets/               # Static assets
│   ├── images/           # Images and icons
│   ├── styles/           # CSS files
│   └── templates/        # Report templates
│
└── cache/                # Temporary cache files
    ├── data_cache/       # Cached data
    └── model_cache/      # Cached AI models
```

## 🚀 Quick Start

### 1. Installation
```bash
cd ai_agent
pip install -r requirements.txt
```

### 2. Configuration
```bash
cp env.example .env
# Edit .env with your API keys
```

### 3. Run the Application
```bash
streamlit run main.py
```

## 📋 Key Components

### Core Modules
- **agent.py**: Main AI agent with Gemini integration
- **data_processor.py**: Data ingestion, cleaning, and validation
- **visualizer.py**: Chart generation and data visualization

### Dashboards
- **professional_dashboard.py**: Main production dashboard (recommended)
- **beautiful_dashboard.py**: Alternative UI with enhanced styling
- **compact_dark_dashboard.py**: Compact dark theme interface

### Configuration
- **settings.py**: Centralized configuration management
- **env.example**: Environment variables template

## 🔧 Development

### Adding New Features
1. Add core logic to `src/` modules
2. Create new dashboard in `dashboards/`
3. Update configuration in `config/settings.py`
4. Add tests in `tests/`

### Testing
```bash
python -m pytest tests/
```

### Logging
Logs are stored in `logs/` directory with rotation and different levels.

## 📊 Data Flow

1. **Upload**: User uploads data via dashboard
2. **Processing**: Data is cleaned and validated
3. **Analysis**: AI agent analyzes the data
4. **Visualization**: Charts and insights are generated
5. **Export**: Reports can be exported in multiple formats

## 🎯 Best Practices

- Keep dashboards in `dashboards/` directory
- Use configuration files for settings
- Implement proper error handling
- Add logging for debugging
- Write tests for new features
- Follow PEP 8 coding standards
