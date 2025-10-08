# 🤖 Performance Analytics AI Agent

An **Agentic AI** project designed for educational purposes, demonstrating how AI agents can autonomously analyze data, generate insights, and provide intelligent assistance. Built with Python and Streamlit for the **Agentic AI Course**.

## 🎯 Project Focus: Agentic AI Education

This project is specifically designed to teach students about **Agentic AI** concepts:

- **🧠 Memory System**: Persistent memory for learning and context
- **🎯 Planning Engine**: Step-by-step task decomposition
- **🔍 Reasoning Module**: Logical analysis and inference
- **🔧 Tool Interface**: Modular capability system
- **💬 Conversational AI**: Natural language interaction
- **📊 Data Analysis**: Intelligent data processing and insights

## 🏗️ Project Structure

```
ai_agent/
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── config/                # Configuration files
├── src/                   # Core source code
├── dashboards/           # Streamlit dashboard implementations
├── data/                 # Data storage
├── reports/              # Generated reports
├── logs/                 # Application logs
├── tests/                # Test files
├── docs/                 # Documentation
├── assets/               # Static assets
└── cache/                # Temporary cache files
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed organization.

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone or navigate to the project
cd ai_agent

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Get your free API key from: https://makersuite.google.com/app/apikey

### 3. Run the Application

```bash
streamlit run main.py
```

Visit: http://localhost:8501

## 🎓 Key Features

### 🤖 AI Agent Capabilities
- **Memory System**: Remembers interactions and analysis history
- **Planning Engine**: Creates step-by-step plans for complex tasks
- **Reasoning Module**: Makes logical inferences about data
- **Tool Interface**: Uses specialized tools for analysis
- **Conversational AI**: Natural language data queries
- **Learning System**: Improves through user interactions

### 📊 Data Analysis Features
- **Intelligent Data Processing**: Automatic data validation and cleaning
- **AI-Powered Insights**: Google Gemini integration for advanced analysis
- **Interactive Visualizations**: Dynamic charts and graphs
- **Anomaly Detection**: Automatic identification of outliers
- **Trend Analysis**: Pattern recognition and forecasting
- **Automated Reporting**: Generate insights and recommendations

### 🎓 Educational Components
- **Interactive Demos**: Hands-on agent capability demonstrations
- **Learning Guide**: Comprehensive explanation of agentic AI concepts
- **Code Examples**: Well-documented, educational codebase
- **Real-world Applications**: Practical use cases and scenarios

## 🔧 Configuration

The application uses a comprehensive configuration system in `config/settings.py`:

### AI Configuration
```python
ai_config = AIConfig(
    gemini_api_key="your_api_key",
    gemini_model="gemini-1.5-flash",
    max_tokens=2048,
    temperature=0.7,
    enable_ai=True
)
```

### Agent Configuration
```python
agent_config = AgentConfig(
    memory_limit=100,
    enable_planning=True,
    enable_reasoning=True,
    enable_tools=True,
    auto_insights=True,
    confidence_threshold=0.7
)
```

### UI Configuration
```python
ui_config = UIConfig(
    page_title="Performance Analytics AI Agent",
    page_icon="🤖",
    layout="wide",
    theme="dark",
    max_file_size=200  # MB
)
```

## 🎯 Usage Examples

### 1. Upload and Analyze Data
1. Upload a CSV or Excel file in the sidebar
2. The agent automatically processes and validates the data
3. View data overview and quality metrics
4. Ask questions in natural language

### 2. Ask Questions
Try these example queries:
- "What are the key trends in this data?"
- "Which team performed best and why?"
- "Generate insights about performance patterns"
- "What anomalies do you detect?"
- "Create a plan to improve performance"

### 3. Explore Agent Capabilities
- **Planning**: Watch the agent create step-by-step plans
- **Reasoning**: See logical analysis in action
- **Tool Usage**: Test different analysis capabilities
- **Memory**: Explore persistent context storage

### 4. Generate Insights
- **Automated Analysis**: AI-powered data insights
- **Interactive Visualizations**: Dynamic charts and graphs
- **Anomaly Detection**: Automatic outlier identification
- **Trend Analysis**: Pattern recognition and forecasting

## 🧠 Agent Architecture

### Core Components

1. **PerformanceAnalyticsAgent** (`src/agent.py`)
   - Memory management
   - Planning and reasoning
   - AI integration
   - Tool orchestration

2. **DataProcessor** (`src/data_processor.py`)
   - File handling and validation
   - Data cleaning and preprocessing
   - Quality assessment
   - Metadata generation

3. **DataVisualizer** (`src/visualizer.py`)
   - Interactive chart creation
   - Visualization recommendations
   - Insight-based graphics
   - Custom chart generation

### Agent Capabilities

```python
# Initialize agent
agent = PerformanceAnalyticsAgent(config)

# Add to memory
agent.add_to_memory("User question", "user")

# Plan a task
plan = agent.plan_task("Analyze performance trends", data)

# Reason about data
reasoning = agent.reason_about_data(data, "Which team is best?")

# Generate insights
insights = agent.generate_insights(data)

# Get agent status
status = agent.get_agent_status()
```

## 📊 Sample Data

The application includes sample data generation:

```python
# Generate sample data
processor = DataProcessor(config)
sample_data = processor.create_sample_data(n_rows=100)

# Sample data includes:
# - Date ranges
# - Team performance metrics
# - Sales data
# - Customer satisfaction scores
# - Categories and classifications
```

## 🎓 Learning Objectives

After completing this project, students will understand:

### Core Agentic AI Concepts
- **Autonomy**: How agents make independent decisions
- **Planning**: Breaking down complex tasks into steps
- **Reasoning**: Logical analysis and inference
- **Tool Usage**: Leveraging external capabilities
- **Memory**: Persistent context and learning

### Technical Skills
- **AI Integration**: Working with language models
- **Data Analysis**: Processing and analyzing datasets
- **Interactive UI**: Building user-friendly interfaces
- **API Integration**: Connecting to external services
- **Error Handling**: Robust application design

### Practical Applications
- **Business Intelligence**: Automated data analysis
- **Decision Support**: AI-powered recommendations
- **Process Automation**: Streamlined workflows
- **User Experience**: Natural language interfaces
- **Educational Tools**: Interactive learning systems

## 🔧 Development

### Running in Development Mode
```bash
# Enable debug mode
export DEBUG=true

# Run with auto-reload
streamlit run app.py --server.runOnSave true
```

### Adding New Capabilities
1. Extend the `PerformanceAnalyticsAgent` class
2. Add new tools to the agent's toolset
3. Update the configuration system
4. Add UI components in `app.py`

### Testing
```bash
# Run with sample data
python -c "from src.data_processor import DataProcessor; from config.settings import config; dp = DataProcessor(config); print(dp.create_sample_data())"
```

## 📚 Educational Resources

### Learning Modules
1. **Introduction to Agentic AI**: Core concepts and principles
2. **Agent Architecture**: Memory, planning, and reasoning systems
3. **Tool Integration**: Using external APIs and services
4. **Conversational AI**: Natural language processing
5. **Practical Applications**: Real-world use cases

### Code Examples
- **Agent Class**: Complete agent implementation
- **Memory System**: Persistent storage and retrieval
- **Planning Engine**: Task decomposition and execution
- **Tool Interface**: Modular capability system
- **UI Integration**: Streamlit interface design

## 🤝 Contributing

This is an educational project. Contributions welcome:
1. Fork the repository
2. Create educational examples
3. Add new agent capabilities
4. Improve documentation
5. Submit pull requests

## 📄 License

Educational use - Part of the **Agentic AI Course** series.

## 🆘 Support

For questions and support:
1. Check the configuration in `config/settings.py`
2. Review the agent status in the sidebar
3. Ensure your Gemini API key is properly configured
4. Test with sample data first

## 🎯 Next Steps

After mastering this project:
1. **Extend Agent Capabilities**: Add new tools and features
2. **Integrate More AI Models**: Try different language models
3. **Build Specialized Agents**: Create domain-specific agents
4. **Deploy to Production**: Learn about deployment and scaling
5. **Advanced Topics**: Explore multi-agent systems and coordination

---

**🎓 Perfect for Agentic AI Course demonstrations and hands-on learning!**
