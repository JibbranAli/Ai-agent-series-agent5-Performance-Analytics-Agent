"""
Performance Analytics AI Agent - Main Streamlit Application
An intelligent agent that analyzes data, generates insights, and provides AI-powered assistance.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from datetime import datetime
import json

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from config.settings import config, validate_config
from src.agent import PerformanceAnalyticsAgent
from src.visualizer import DataVisualizer
from src.data_processor import DataProcessor

# Configure Streamlit page
st.set_page_config(
    page_title=config.ui.page_title,
    page_icon=config.ui.page_icon,
    layout=config.ui.layout,
    initial_sidebar_state=config.ui.sidebar_state
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-message {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-online { background-color: #4CAF50; }
    .status-offline { background-color: #f44336; }
    .status-warning { background-color: #ff9800; }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'agent' not in st.session_state:
        st.session_state.agent = PerformanceAnalyticsAgent(config)
    
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = DataProcessor(config)
    
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = DataVisualizer(config)
    
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    
    if 'current_metadata' not in st.session_state:
        st.session_state.current_metadata = None
    
    if 'insights' not in st.session_state:
        st.session_state.insights = []

def render_sidebar():
    """Render the sidebar with controls and information"""
    with st.sidebar:
        st.header("🎛️ Agent Controls")
        
        # Agent status
        st.subheader("🤖 Agent Status")
        agent_status = st.session_state.agent.get_agent_status()
        
        # Status indicators
        ai_status = "🟢 Online" if agent_status['ai_enabled'] else "🔴 Offline"
        st.write(f"**AI Model:** {ai_status}")
        st.write(f"**Memory:** {agent_status['memory_entries']} entries")
        st.write(f"**Analyses:** {agent_status['analysis_count']}")
        st.write(f"**Plans:** {agent_status['plans_created']}")
        
        # Data upload section
        st.subheader("📊 Data Upload")
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=config.data.supported_formats,
            help="Upload your performance data for analysis"
        )
        
        if uploaded_file is not None:
            # Save uploaded file
            file_path = config.data_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process the file
            with st.spinner("Processing data..."):
                result = st.session_state.data_processor.process_uploaded_file(file_path)
                
                if result["success"]:
                    st.session_state.current_data = result["data"]
                    st.session_state.current_metadata = result["metadata"]
                    st.success(f"✅ Data loaded: {result['data'].shape[0]} rows, {result['data'].shape[1]} columns")
                    
                    # Generate initial insights
                    if config.agent.auto_insights:
                        st.session_state.insights = st.session_state.agent.generate_insights(result["data"])
                else:
                    st.error(f"❌ {result['error']}")
        
        # Agent settings
        st.subheader("⚙️ Agent Settings")
        
        # AI settings
        use_ai = st.checkbox("Enable AI Analysis", value=config.ai.enable_ai, help="Use Gemini AI for advanced analysis")
        config.ai.enable_ai = use_ai
        
        auto_insights = st.checkbox("Auto-generate Insights", value=config.agent.auto_insights)
        config.agent.auto_insights = auto_insights
        
        # Memory management
        st.subheader("🧠 Memory Management")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Memory"):
                st.session_state.agent.clear_memory()
                st.success("Memory cleared!")
        
        with col2:
            if st.button("Export Memory"):
                export_path = config.data_dir / f"agent_memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                st.session_state.agent.export_memory(export_path)
                st.success(f"Memory exported to {export_path.name}")
        
        # Show recent memory
        if agent_status['memory_entries'] > 0:
            with st.expander("View Recent Memory"):
                recent_memory = st.session_state.agent.memory[-5:]
                for entry in recent_memory:
                    st.write(f"**{entry['role'].title()}:** {entry['message'][:100]}...")
        
        # Configuration info
        with st.expander("🔧 Configuration"):
            st.json({
                "AI Model": config.ai.gemini_model,
                "Memory Limit": config.agent.memory_limit,
                "Max File Size": f"{config.ui.max_file_size} MB",
                "Supported Formats": config.data.supported_formats
            })

def render_main_content():
    """Render the main content area"""
    # Header
    st.markdown('<h1 class="main-header">🤖 Performance Analytics AI Agent</h1>', unsafe_allow_html=True)
    st.markdown("**An intelligent agent that analyzes your data and provides AI-powered insights**")
    
    # Check if data is loaded
    if st.session_state.current_data is None:
        render_welcome_screen()
    else:
        render_data_analysis()

def render_welcome_screen():
    """Render welcome screen when no data is loaded"""
    st.info("👆 Upload a CSV or Excel file in the sidebar to get started with the AI Agent analysis!")
    
    # Demo section
    st.header("🎓 Agentic AI Learning Demo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 What is an AI Agent?")
        st.markdown("""
        This application demonstrates key concepts of **Agentic AI**:
        
        - **🧠 Memory**: Remembers interactions and analysis history
        - **🎯 Planning**: Creates step-by-step plans for complex tasks
        - **🔍 Reasoning**: Makes logical inferences about data
        - **🔧 Tool Usage**: Uses specialized tools for analysis
        - **💬 Conversation**: Natural language interface
        - **📊 Learning**: Improves through interactions
        """)
    
    with col2:
        st.subheader("🚀 Try These Features")
        st.markdown("""
        1. **Upload Data**: Use sample data or your own files
        2. **Ask Questions**: Natural language data queries
        3. **Generate Insights**: Automatic pattern detection
        4. **Create Visualizations**: Interactive charts
        5. **Explore Capabilities**: Test agent tools
        """)
    
    # Sample data section
    st.subheader("📥 Sample Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Generate Sample Data"):
            with st.spinner("Creating sample data..."):
                sample_data = st.session_state.data_processor.create_sample_data(100)
                st.session_state.current_data = sample_data
                st.session_state.current_metadata = st.session_state.data_processor._generate_metadata(sample_data, Path("sample_data.csv"))
                st.success("Sample data generated successfully!")
                st.rerun()
    
    with col2:
        if st.button("Download Sample CSV"):
            sample_data = st.session_state.data_processor.create_sample_data(100)
            csv = sample_data.to_csv(index=False)
            st.download_button(
                label="Download sample_data.csv",
                data=csv,
                file_name="sample_data.csv",
                mime="text/csv"
            )

def render_data_analysis():
    """Render data analysis interface"""
    data = st.session_state.current_data
    metadata = st.session_state.current_metadata
    
    # Data overview
    st.header("📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", len(data))
    with col2:
        st.metric("Columns", len(data.columns))
    with col3:
        st.metric("Numeric Columns", len(data.select_dtypes(include=[np.number]).columns))
    with col4:
        st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    # Data preview
    with st.expander("📋 Data Preview"):
        st.dataframe(data.head(10))
    
    # Data validation
    if st.button("🔍 Validate Data Quality"):
        with st.spinner("Validating data..."):
            validation = st.session_state.data_processor.validate_data(data)
            
            if validation["is_valid"]:
                st.success("✅ Data validation passed!")
            else:
                st.error("❌ Data validation failed!")
            
            if validation["warnings"]:
                for warning in validation["warnings"]:
                    st.warning(f"⚠️ {warning}")
            
            if validation["recommendations"]:
                for rec in validation["recommendations"]:
                    st.info(f"💡 {rec}")
    
    # AI Analysis Section
    st.header("🧠 AI Agent Analysis")
    
    # Chat interface
    st.subheader("💬 Ask the Agent")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        user_question = st.text_input(
            "Ask a question about your data:",
            placeholder="e.g., What are the key trends? Which team performed best?",
            help="The AI agent will analyze your data and provide insights"
        )
    
    with col2:
        ask_button = st.button("Ask Agent", type="primary")
    
    if ask_button and user_question:
        with st.spinner("🤖 Agent is thinking..."):
            # Add to memory
            st.session_state.agent.add_to_memory(user_question, "user")
            
            # Get AI analysis
            if config.ai.enable_ai and st.session_state.agent.model:
                analysis = st.session_state.agent.analyze_data_with_ai(data, user_question)
                
                if "error" not in analysis:
                    st.markdown(f'<div class="agent-message"><strong>🤖 AI Agent:</strong><br>{analysis["analysis"]}</div>', unsafe_allow_html=True)
                else:
                    st.error(f"AI Analysis Error: {analysis['error']}")
            else:
                st.info("AI analysis is disabled. Enable it in the sidebar to get AI-powered insights.")
    
    # Automated Insights
    if st.session_state.insights:
        st.subheader("🔍 Automated Insights")
        
        # Display insights by category
        insight_types = {}
        for insight in st.session_state.insights:
            insight_type = insight.get('type', 'other')
            if insight_type not in insight_types:
                insight_types[insight_type] = []
            insight_types[insight_type].append(insight)
        
        for insight_type, insights in insight_types.items():
            with st.expander(f"{insight_type.title()} Insights ({len(insights)})"):
                for insight in insights:
                    severity = insight.get('severity', 'info')
                    if severity == 'success':
                        st.success(f"✅ {insight['insight']}")
                    elif severity == 'warning':
                        st.warning(f"⚠️ {insight['insight']}")
                    elif severity == 'error':
                        st.error(f"❌ {insight['insight']}")
                    else:
                        st.info(f"ℹ️ {insight['insight']}")
    
    # Visualizations
    st.header("📈 Data Visualizations")
    
    if st.button("Create Visualizations"):
        with st.spinner("📊 Creating visualizations..."):
            figures = st.session_state.visualizer.create_visualizations(data, st.session_state.insights)
            
            for name, fig in figures.items():
                st.plotly_chart(fig, use_container_width=True)
    
    # Agent Capabilities Demo
    st.header("🎯 Agent Capabilities")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Data Summary"):
            summary = {
                "shape": data.shape,
                "columns": list(data.columns),
                "data_types": dict(data.dtypes),
                "missing_values": data.isnull().sum().to_dict(),
                "memory_usage": f"{data.memory_usage(deep=True).sum() / 1024:.1f} KB"
            }
            st.json(summary)
    
    with col2:
        if st.button("🔍 Find Anomalies"):
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            anomalies = []
            
            for col in numeric_cols:
                if col in data.columns:
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
                    if len(outliers) > 0:
                        anomalies.append(f"{len(outliers)} outliers in {col}")
            
            if anomalies:
                for anomaly in anomalies:
                    st.warning(anomaly)
            else:
                st.success("No significant anomalies detected")
    
    with col3:
        if st.button("📈 Performance Metrics"):
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                metrics = {
                    "Mean": data[col].mean(),
                    "Median": data[col].median(),
                    "Std Dev": data[col].std(),
                    "Min": data[col].min(),
                    "Max": data[col].max()
                }
                st.json(metrics)
    
    # Planning and Reasoning Demo
    st.subheader("🎯 Advanced Agent Capabilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📋 Create Analysis Plan"):
            task = "Analyze team performance and identify improvement opportunities"
            with st.spinner("Creating plan..."):
                plan = st.session_state.agent.plan_task(task, data)
                
                st.write("**Agent's Plan:**")
                for i, step in enumerate(plan, 1):
                    if step.strip():
                        st.write(f"{i}. {step}")
    
    with col2:
        if st.button("🧠 Reason About Data"):
            question = "Which team is performing best and why?"
            with st.spinner("Reasoning..."):
                reasoning = st.session_state.agent.reason_about_data(data, question)
                
                st.write("**Agent's Reasoning:**")
                st.write(reasoning)

def main():
    """Main application function"""
    # Validate configuration
    if not validate_config():
        st.error("Configuration validation failed. Please check your settings.")
        return
    
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Render main content
    render_main_content()

if __name__ == "__main__":
    main()
