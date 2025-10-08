"""
Simple Performance Analytics AI Agent Dashboard
A clean, simple interface for the AI agent with graphs and core functionality.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
from datetime import datetime
import google.generativeai as genai
import os

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

# Configure Streamlit page
st.set_page_config(
    page_title="AI Performance Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .agent-response {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'agent_memory' not in st.session_state:
    st.session_state.agent_memory = []
if 'gemini_model' not in st.session_state:
    st.session_state.gemini_model = None

# Initialize Gemini AI
def init_gemini():
    api_key = "AIzaSyDQMTPH6kMlgJa2WnzkwiYib5qzLLC-CVs"
    if api_key:
        try:
            genai.configure(api_key=api_key)
            st.session_state.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
            return True
        except Exception as e:
            st.error(f"Failed to initialize AI: {e}")
            return False
    return False

# Initialize AI
if st.session_state.gemini_model is None:
    init_gemini()

def create_sample_data():
    """Create sample performance data"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    teams = np.random.choice(['Team A', 'Team B', 'Team C'], 50)
    sales = np.random.normal(1000, 200, 50)
    performance = np.random.uniform(60, 100, 50)
    
    return pd.DataFrame({
        'date': dates,
        'team': teams,
        'sales': sales,
        'performance_score': performance
    })

def ask_ai_agent(question, data):
    """Ask the AI agent a question about the data"""
    if not st.session_state.gemini_model:
        return "AI model not available. Please check your configuration."
    
    try:
        # Prepare data summary
        data_summary = f"""
        Dataset: {data.shape[0]} rows, {data.shape[1]} columns
        Columns: {list(data.columns)}
        Sample data: {data.head(3).to_string()}
        Statistics: {data.describe().to_string()}
        """
        
        prompt = f"""
        As an AI Performance Analytics Agent, analyze this data and answer the question.
        
        {data_summary}
        
        Question: {question}
        
        Please provide a clear, concise answer with insights and recommendations.
        """
        
        response = st.session_state.gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def create_charts(data):
    """Create interactive charts"""
    charts = {}
    
    try:
        # Sales over time
        if 'date' in data.columns and 'sales' in data.columns:
            fig_sales = px.line(data, x='date', y='sales', 
                              title='Sales Over Time',
                              markers=True)
            fig_sales.update_layout(height=400)
            charts['sales_trend'] = fig_sales
        
        # Performance by team
        if 'team' in data.columns and 'performance_score' in data.columns:
            team_perf = data.groupby('team')['performance_score'].mean().reset_index()
            fig_team = px.bar(team_perf, x='team', y='performance_score',
                            title='Average Performance by Team',
                            color='performance_score',
                            color_continuous_scale='Viridis')
            fig_team.update_layout(height=400)
            charts['team_performance'] = fig_team
        
        # Sales distribution
        if 'sales' in data.columns:
            fig_dist = px.histogram(data, x='sales', 
                                  title='Sales Distribution',
                                  nbins=20)
            fig_dist.update_layout(height=400)
            charts['sales_distribution'] = fig_dist
        
        # Performance vs Sales scatter
        if 'performance_score' in data.columns and 'sales' in data.columns:
            fig_scatter = px.scatter(data, x='performance_score', y='sales',
                                   color='team' if 'team' in data.columns else None,
                                   title='Performance vs Sales',
                                   trendline='ols')
            fig_scatter.update_layout(height=400)
            charts['performance_sales'] = fig_scatter
            
    except Exception as e:
        st.error(f"Chart creation error: {e}")
    
    return charts

def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Performance Analytics Agent</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Data Upload")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=['csv', 'xlsx', 'xls']
        )
        
        # Sample data button
        if st.button("📈 Generate Sample Data"):
            st.session_state.data = create_sample_data()
            st.success("Sample data generated!")
        
        # AI Status
        st.header("🤖 AI Status")
        if st.session_state.gemini_model:
            st.success("✅ AI Model Ready")
        else:
            st.error("❌ AI Model Not Available")
        
        # Memory
        st.header("🧠 Agent Memory")
        if st.button("Clear Memory"):
            st.session_state.agent_memory = []
            st.success("Memory cleared!")
        
        st.write(f"Memory entries: {len(st.session_state.agent_memory)}")
    
    # Main content
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Data overview
        st.header("📊 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", len(data))
        with col2:
            st.metric("Columns", len(data.columns))
        with col3:
            st.metric("Teams", data['team'].nunique() if 'team' in data.columns else "N/A")
        with col4:
            st.metric("Date Range", f"{data['date'].min().strftime('%Y-%m-%d') if 'date' in data.columns else 'N/A'} to {data['date'].max().strftime('%Y-%m-%d') if 'date' in data.columns else 'N/A'}")
        
        # Data preview
        with st.expander("📋 View Data"):
            st.dataframe(data.head(10))
        
        # AI Chat
        st.header("💬 Ask the AI Agent")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            question = st.text_input(
                "Ask a question about your data:",
                placeholder="e.g., Which team performed best? What are the trends?"
            )
        
        with col2:
            ask_button = st.button("Ask", type="primary")
        
        if ask_button and question:
            with st.spinner("🤖 AI is thinking..."):
                response = ask_ai_agent(question, data)
                
                # Add to memory
                st.session_state.agent_memory.append({
                    "question": question,
                    "response": response,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                
                # Display response
                st.markdown(f'<div class="agent-response"><strong>🤖 AI Agent:</strong><br>{response}</div>', unsafe_allow_html=True)
        
        # Show recent conversations
        if st.session_state.agent_memory:
            with st.expander("💭 Recent Conversations"):
                for i, memory in enumerate(st.session_state.agent_memory[-3:]):
                    st.write(f"**Q:** {memory['question']}")
                    st.write(f"**A:** {memory['response'][:200]}...")
                    st.write(f"*Time: {memory['timestamp']}*")
                    st.divider()
        
        # Charts
        st.header("📈 Interactive Charts")
        
        if st.button("Generate Charts"):
            with st.spinner("Creating charts..."):
                charts = create_charts(data)
                
                if charts:
                    # Display charts in columns
                    chart_items = list(charts.items())
                    
                    for i in range(0, len(chart_items), 2):
                        cols = st.columns(2)
                        for j, (name, fig) in enumerate(chart_items[i:i+2]):
                            with cols[j]:
                                st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No charts could be generated with the current data.")
        
        # Quick Analysis
        st.header("🔍 Quick Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Data Summary"):
                st.json({
                    "Shape": data.shape,
                    "Columns": list(data.columns),
                    "Data Types": dict(data.dtypes),
                    "Missing Values": data.isnull().sum().to_dict()
                })
        
        with col2:
            if st.button("🔍 Find Anomalies"):
                if 'sales' in data.columns:
                    Q1 = data['sales'].quantile(0.25)
                    Q3 = data['sales'].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = data[(data['sales'] < Q1 - 1.5*IQR) | (data['sales'] > Q3 + 1.5*IQR)]
                    
                    if len(outliers) > 0:
                        st.warning(f"Found {len(outliers)} outliers in sales data")
                        st.dataframe(outliers[['date', 'team', 'sales']].head())
                    else:
                        st.success("No significant outliers found")
        
        with col3:
            if st.button("📈 Performance Metrics"):
                if 'performance_score' in data.columns:
                    metrics = {
                        "Average": data['performance_score'].mean(),
                        "Median": data['performance_score'].median(),
                        "Best": data['performance_score'].max(),
                        "Worst": data['performance_score'].min()
                    }
                    st.json(metrics)
    
    else:
        # Welcome screen
        st.info("👆 Upload a file or generate sample data to get started!")
        
        # Demo section
        st.header("🎓 AI Agent Demo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🤖 What is an AI Agent?")
            st.markdown("""
            This AI agent can:
            - **Analyze** your data automatically
            - **Answer questions** about your data
            - **Generate insights** and recommendations
            - **Create visualizations** and charts
            - **Remember** our conversations
            """)
        
        with col2:
            st.subheader("🚀 Try These Features")
            st.markdown("""
            1. **Upload Data**: CSV or Excel files
            2. **Ask Questions**: Natural language queries
            3. **View Charts**: Interactive visualizations
            4. **Get Insights**: AI-powered analysis
            5. **Explore Data**: Quick analysis tools
            """)
        
        # Sample data preview
        if st.button("📊 Preview Sample Data"):
            sample_data = create_sample_data()
            st.dataframe(sample_data.head())
            st.info("Click 'Generate Sample Data' in the sidebar to use this data!")

if __name__ == "__main__":
    main()
