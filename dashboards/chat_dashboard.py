"""
AI Performance Analytics Agent - Conversational Dashboard
Natural language interface with automatic graph generation based on user questions.
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
import re

# Configure Streamlit page
st.set_page_config(
    page_title="AI Performance Agent - Chat Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 5px 0;
        text-align: right;
    }
    .ai-message {
        background-color: #e9ecef;
        color: #333;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 5px 0;
        text-align: left;
    }
    .chart-container {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
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
    """Create comprehensive sample performance data"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    teams = np.random.choice(['Team Alpha', 'Team Beta', 'Team Gamma', 'Team Delta'], 100)
    sales = np.random.normal(1000, 200, 100)
    performance = np.random.uniform(60, 100, 100)
    customer_satisfaction = np.random.uniform(3, 5, 100)
    projects_completed = np.random.poisson(5, 100)
    
    return pd.DataFrame({
        'date': dates,
        'team': teams,
        'sales': sales,
        'performance_score': performance,
        'customer_satisfaction': customer_satisfaction,
        'projects_completed': projects_completed,
        'quarter': ['Q1' if d.month <= 3 else 'Q2' if d.month <= 6 else 'Q3' if d.month <= 9 else 'Q4' for d in dates]
    })

def generate_smart_questions(data):
    """Generate intelligent questions based on the data structure"""
    questions = []
    
    # Get column information
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    date_cols = [col for col in data.columns if any(keyword in col.lower() for keyword in ['date', 'time', 'period'])]
    
    # Team-based questions
    if 'team' in categorical_cols:
        questions.extend([
            "Which team performed best overall?",
            "Compare performance across all teams",
            "Which team has the most consistent results?",
            "Show me team performance trends over time"
        ])
    
    # Time-based questions
    if date_cols and numeric_cols:
        questions.extend([
            "Show me trends over time",
            "What's the performance trend this quarter?",
            "Which month had the best results?",
            "Are there any seasonal patterns?"
        ])
    
    # Sales questions
    if 'sales' in numeric_cols:
        questions.extend([
            "What are the sales trends?",
            "Which team has the highest sales?",
            "Show me sales distribution",
            "Are there any sales outliers?"
        ])
    
    # Performance questions
    if 'performance' in ' '.join(numeric_cols).lower():
        questions.extend([
            "What's the performance distribution?",
            "Which factors affect performance most?",
            "Show me performance vs sales relationship",
            "What's the average performance score?"
        ])
    
    # Customer satisfaction questions
    if 'customer' in ' '.join(numeric_cols).lower() or 'satisfaction' in ' '.join(numeric_cols).lower():
        questions.extend([
            "How satisfied are our customers?",
            "Which team has the best customer satisfaction?",
            "What affects customer satisfaction?",
            "Show customer satisfaction trends"
        ])
    
    # Correlation questions
    if len(numeric_cols) >= 2:
        questions.extend([
            "What's the relationship between sales and performance?",
            "How do different metrics correlate?",
            "Which factors are most important?",
            "Show me correlation analysis"
        ])
    
    # Outlier detection
    if numeric_cols:
        questions.extend([
            "Are there any outliers in the data?",
            "Which data points are unusual?",
            "Show me anomaly detection",
            "What are the extreme values?"
        ])
    
    # Distribution questions
    if numeric_cols:
        questions.extend([
            "What's the data distribution like?",
            "Show me histograms of key metrics",
            "How is the data spread?",
            "What are the data patterns?"
        ])
    
    # Summary questions
    questions.extend([
        "Give me a data summary",
        "What are the key insights?",
        "What should I focus on?",
        "What recommendations do you have?"
    ])
    
    # Remove duplicates and return top 12
    unique_questions = list(dict.fromkeys(questions))  # Preserves order while removing duplicates
    return unique_questions[:12]

def analyze_question_and_generate_chart(question, data):
    """Analyze the question and generate appropriate chart"""
    if not st.session_state.gemini_model:
        return None, "AI model not available"
    
    try:
        # Prepare data summary
        data_summary = f"""
        Dataset: {data.shape[0]} rows, {data.shape[1]} columns
        Columns: {list(data.columns)}
        Sample data: {data.head(3).to_string()}
        """
        
        # Ask AI to determine what chart to create
        chart_prompt = f"""
        Based on this question: "{question}"
        And this data: {data_summary}
        
        Determine what type of chart would best answer this question. Choose from:
        - line_chart: For trends over time
        - bar_chart: For comparing categories
        - scatter_plot: For relationships between variables
        - histogram: For distributions
        - pie_chart: For proportions
        - box_plot: For distributions and outliers
        
        Respond with just the chart type and a brief explanation.
        """
        
        chart_response = st.session_state.gemini_model.generate_content(chart_prompt)
        chart_type = chart_response.text.lower()
        
        # Generate the appropriate chart
        fig = None
        
        if 'line' in chart_type and 'date' in data.columns:
            # Find numeric column for line chart
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                y_col = numeric_cols[0]
                fig = px.line(data, x='date', y=y_col, 
                            title=f'{y_col} Over Time',
                            markers=True)
        
        elif 'bar' in chart_type:
            # Bar chart for team comparison
            if 'team' in data.columns:
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    y_col = numeric_cols[0]
                    team_data = data.groupby('team')[y_col].mean().reset_index()
                    fig = px.bar(team_data, x='team', y=y_col,
                               title=f'Average {y_col} by Team',
                               color=y_col,
                               color_continuous_scale='Viridis')
        
        elif 'scatter' in chart_type:
            # Scatter plot for relationships
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                x_col, y_col = numeric_cols[0], numeric_cols[1]
                fig = px.scatter(data, x=x_col, y=y_col,
                               color='team' if 'team' in data.columns else None,
                               title=f'{x_col} vs {y_col}',
                               trendline='ols')
        
        elif 'histogram' in chart_type or 'distribution' in chart_type:
            # Histogram for distribution
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                fig = px.histogram(data, x=col,
                                 title=f'Distribution of {col}',
                                 nbins=20)
        
        elif 'pie' in chart_type:
            # Pie chart for proportions
            if 'team' in data.columns:
                team_counts = data['team'].value_counts()
                fig = px.pie(values=team_counts.values, 
                           names=team_counts.index,
                           title='Team Distribution')
        
        elif 'box' in chart_type:
            # Box plot for distributions
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                fig = px.box(data, y=col,
                           title=f'Distribution of {col}',
                           color='team' if 'team' in data.columns else None)
        
        # If no specific chart type detected, create a default one
        if fig is None:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                fig = px.histogram(data, x=col, title=f'Distribution of {col}')
        
        return fig, chart_response.text
        
    except Exception as e:
        return None, f"Error generating chart: {str(e)}"

def get_ai_response(question, data):
    """Get AI response to the question"""
    if not st.session_state.gemini_model:
        return "AI model not available. Please check your configuration."
    
    try:
        # Prepare comprehensive data summary
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        data_summary = f"""
        Dataset Overview:
        - Shape: {data.shape[0]} rows, {data.shape[1]} columns
        - Columns: {list(data.columns)}
        - Data types: {dict(data.dtypes)}
        - Numeric columns: {list(numeric_cols)}
        - Categorical columns: {list(categorical_cols)}
        
        Sample Data:
        {data.head(5).to_string()}
        
        Statistical Summary:
        {data.describe().to_string() if len(numeric_cols) > 0 else 'No numeric columns for statistical summary'}
        """
        
        # Add categorical information if available
        if len(categorical_cols) > 0:
            data_summary += f"""
        
        Categorical Information:
        """
            for col in categorical_cols[:3]:  # Limit to first 3 categorical columns
                data_summary += f"\n{col}: {data[col].value_counts().head().to_string()}"
        
        prompt = f"""
        You are an AI Performance Analytics Agent. Analyze this data and answer the user's question in a conversational, helpful way.
        
        {data_summary}
        
        User Question: {question}
        
        Please provide:
        1. A direct answer to the question
        2. Key insights from the data
        3. Specific numbers and statistics
        4. Actionable recommendations
        5. Any interesting patterns you notice
        
        Be conversational and engaging in your response.
        """
        
        response = st.session_state.gemini_model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    """Main application"""
    
    # Header
    st.title("🤖 AI Performance Analytics Agent")
    st.markdown("**Ask questions about your data in natural language and get instant insights with visualizations!**")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Data Setup")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload your data file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload CSV or Excel file with your performance data"
        )
        
        # Process uploaded file
        if uploaded_file is not None:
            try:
                # Load data based on file type
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file)
                
                # Store in session state
                st.session_state.data = data
                st.success(f"✅ Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
        
        # Sample data button
        if st.button("📈 Generate Sample Data", type="primary"):
            st.session_state.data = create_sample_data()
            st.success("✅ Sample data generated!")
            st.rerun()
        
        # Data info
        if st.session_state.data is not None:
            st.header("📋 Data Info")
            data = st.session_state.data
            st.write(f"**Rows:** {len(data)}")
            st.write(f"**Columns:** {len(data.columns)}")
            
            # Show column information
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            categorical_cols = data.select_dtypes(include=['object']).columns
            st.write(f"**Numeric Columns:** {len(numeric_cols)}")
            st.write(f"**Categorical Columns:** {len(categorical_cols)}")
            
            # Show specific info if available
            if 'team' in data.columns:
                st.write(f"**Teams:** {data['team'].nunique()}")
            if 'date' in data.columns:
                st.write(f"**Date Range:** {data['date'].min()} to {data['date'].max()}")
            
            with st.expander("View Data"):
                st.dataframe(data.head())
        
        # AI Status
        st.header("🤖 AI Status")
        if st.session_state.gemini_model:
            st.success("✅ AI Ready")
        else:
            st.error("❌ AI Not Available")
        
        # Clear chat
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main content
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Chat interface
        st.header("💬 Ask Me Anything About Your Data")
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            for chat in st.session_state.chat_history:
                # User message
                st.markdown(f'<div class="user-message"><strong>You:</strong> {chat["question"]}</div>', unsafe_allow_html=True)
                
                # AI response
                st.markdown(f'<div class="ai-message"><strong>AI Agent:</strong> {chat["response"]}</div>', unsafe_allow_html=True)
                
                # Chart if available
                if chat.get("chart"):
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(chat["chart"], use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("---")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Question input
        col1, col2 = st.columns([4, 1])
        
        with col1:
            question = st.text_input(
                "Ask a question about your data:",
                placeholder="e.g., Which team has the best performance? Show me sales trends over time. What's the relationship between performance and customer satisfaction?",
                key="question_input"
            )
        
        with col2:
            ask_button = st.button("Ask", type="primary", use_container_width=True)
        
        # Process question
        if ask_button and question:
            with st.spinner("🤖 AI is analyzing your data and creating visualizations..."):
                # Get AI response
                response = get_ai_response(question, data)
                
                # Generate chart
                chart, chart_explanation = analyze_question_and_generate_chart(question, data)
                
                # Add to chat history
                chat_entry = {
                    "question": question,
                    "response": response,
                    "chart": chart,
                    "chart_explanation": chart_explanation,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                }
                
                st.session_state.chat_history.append(chat_entry)
                st.rerun()
        
        # Smart suggested questions based on data
        st.header("💡 Smart Questions Based on Your Data")
        
        # Generate questions based on data structure
        smart_questions = generate_smart_questions(data)
        
        # Display questions in a grid
        num_questions = len(smart_questions)
        cols_per_row = 3
        
        for i in range(0, num_questions, cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < num_questions:
                    question = smart_questions[i + j]
                    with col:
                        if st.button(question, key=f"smart_q_{i+j}"):
                            # Process the question directly
                            with st.spinner("🤖 AI is analyzing your data and creating visualizations..."):
                                response = get_ai_response(question, data)
                                chart, chart_explanation = analyze_question_and_generate_chart(question, data)
                                
                                chat_entry = {
                                    "question": question,
                                    "response": response,
                                    "chart": chart,
                                    "chart_explanation": chart_explanation,
                                    "timestamp": datetime.now().strftime("%H:%M:%S")
                                }
                                
                                st.session_state.chat_history.append(chat_entry)
                                st.rerun()
    
    else:
        # Welcome screen
        st.info("👆 Upload a data file or generate sample data to get started!")
        
        # Demo section
        st.header("🎓 How It Works")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🤖 Natural Language Processing")
            st.markdown("""
            - **Ask in plain English**: "Which team is performing best?"
            - **Get instant answers**: AI analyzes your data and responds
            - **Automatic visualizations**: Charts are generated based on your questions
            - **Conversational interface**: Chat-like experience
            """)
        
        with col2:
            st.subheader("📊 Smart Visualizations")
            st.markdown("""
            - **Automatic chart selection**: AI chooses the best chart type
            - **Interactive graphs**: Plotly-powered visualizations
            - **Context-aware**: Charts match your questions
            - **Multiple chart types**: Line, bar, scatter, histogram, pie, box plots
            """)
        
        # Example questions
        st.header("💬 Example Questions You Can Ask")
        
        examples = [
            "Which team has the highest sales?",
            "Show me the performance trends over time",
            "What's the relationship between sales and customer satisfaction?",
            "Which quarter had the best performance?",
            "Are there any outliers in the performance data?",
            "Compare the average performance across all teams",
            "What's the distribution of sales values?",
            "Show me a breakdown of projects by team"
        ]
        
        for i, example in enumerate(examples):
            if i % 2 == 0:
                col1, col2 = st.columns(2)
            
            with col1 if i % 2 == 0 else col2:
                st.write(f"💡 *{example}*")

if __name__ == "__main__":
    main()
