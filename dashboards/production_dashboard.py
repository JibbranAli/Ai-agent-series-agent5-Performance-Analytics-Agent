"""
Production-Ready AI Performance Analytics Agent Dashboard
Single-page interface with upload, chat, and progress indicators.
Optimized for performance and reliability.
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
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="AI Performance Agent - Production Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for production-ready styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .upload-section {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border: 2px dashed #007bff;
        min-height: 300px;
    }
    .chat-container {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        min-height: 500px;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 18px;
        border-radius: 18px;
        margin: 8px 0;
        text-align: right;
        max-width: 80%;
        margin-left: auto;
    }
    .ai-message {
        background-color: #f1f3f4;
        color: #333;
        padding: 12px 18px;
        border-radius: 18px;
        margin: 8px 0;
        text-align: left;
        max-width: 80%;
        border-left: 4px solid #007bff;
    }
    .chart-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    .progress-container {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid #bbdefb;
    }
    .data-info-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #28a745;
    }
    .status-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #17a2b8;
    }
    .error-card {
        background-color: #fff5f5;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #dc3545;
    }
    .success-card {
        background-color: #f0fff4;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #28a745;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with proper defaults
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'data': None,
        'chat_history': [],
        'gemini_model': None,
        'upload_progress': 0,
        'processing_progress': 0,
        'ai_initialized': False,
        'last_upload_time': None,
        'data_processed': False,
        'error_message': None,
        'success_message': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Initialize session state
init_session_state()

# Initialize Gemini AI with error handling
def init_gemini():
    """Initialize Gemini AI with proper error handling"""
    if st.session_state.ai_initialized:
        return st.session_state.gemini_model is not None
    
    try:
        api_key = "AIzaSyDQMTPH6kMlgJa2WnzkwiYib5qzLLC-CVs"
        if not api_key:
            st.session_state.error_message = "API key not configured"
            return False
        
        genai.configure(api_key=api_key)
        st.session_state.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        st.session_state.ai_initialized = True
        logger.info("Gemini AI initialized successfully")
        return True
        
    except Exception as e:
        st.session_state.error_message = f"Failed to initialize AI: {str(e)}"
        logger.error(f"Gemini initialization failed: {e}")
        return False

# Initialize AI
if not st.session_state.ai_initialized:
    init_gemini()

def create_sample_data():
    """Create comprehensive sample performance data"""
    try:
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        teams = np.random.choice(['Team Alpha', 'Team Beta', 'Team Gamma', 'Team Delta'], 100)
        sales = np.random.normal(1000, 200, 100)
        performance = np.random.uniform(60, 100, 100)
        customer_satisfaction = np.random.uniform(3, 5, 100)
        projects_completed = np.random.poisson(5, 100)
        
        data = pd.DataFrame({
            'date': dates,
            'team': teams,
            'sales': sales,
            'performance_score': performance,
            'customer_satisfaction': customer_satisfaction,
            'projects_completed': projects_completed,
            'quarter': ['Q1' if d.month <= 3 else 'Q2' if d.month <= 6 else 'Q3' if d.month <= 9 else 'Q4' for d in dates]
        })
        
        logger.info(f"Sample data created: {data.shape}")
        return data
        
    except Exception as e:
        logger.error(f"Error creating sample data: {e}")
        st.session_state.error_message = f"Error creating sample data: {str(e)}"
        return None

def process_upload_with_progress(uploaded_file):
    """Process uploaded file with progress indicators and error handling"""
    try:
        # Reset progress and messages
        st.session_state.upload_progress = 0
        st.session_state.processing_progress = 0
        st.session_state.error_message = None
        st.session_state.success_message = None
        
        # Validate file
        if uploaded_file is None:
            return False, "No file provided"
        
        # Step 1: File validation (20%)
        st.session_state.upload_progress = 20
        time.sleep(0.3)
        
        # Check file type
        if not (uploaded_file.name.endswith('.csv') or uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls')):
            return False, "Unsupported file type. Please upload CSV or Excel files."
        
        # Step 2: File loading (60%)
        st.session_state.upload_progress = 60
        time.sleep(0.3)
        
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
        except Exception as e:
            return False, f"Error reading file: {str(e)}"
        
        # Step 3: Data validation (80%)
        st.session_state.upload_progress = 80
        time.sleep(0.3)
        
        # Basic data validation
        if data.empty:
            return False, "File is empty or contains no data"
        
        if len(data.columns) < 2:
            return False, "File must contain at least 2 columns"
        
        # Step 4: Data processing and storage (100%)
        st.session_state.upload_progress = 100
        st.session_state.data = data
        st.session_state.data_processed = True
        st.session_state.last_upload_time = datetime.now()
        
        logger.info(f"File processed successfully: {data.shape}")
        return True, f"✅ Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns"
        
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        return False, f"❌ Error processing file: {str(e)}"

def generate_smart_questions(data):
    """Generate intelligent questions based on the data structure"""
    try:
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
                "Which team has the most consistent results?"
            ])
        
        # Time-based questions
        if date_cols and numeric_cols:
            questions.extend([
                "Show me trends over time",
                "What's the performance trend this quarter?",
                "Which month had the best results?"
            ])
        
        # Sales questions
        if 'sales' in numeric_cols:
            questions.extend([
                "What are the sales trends?",
                "Which team has the highest sales?",
                "Show me sales distribution"
            ])
        
        # Performance questions
        if 'performance' in ' '.join(numeric_cols).lower():
            questions.extend([
                "What's the performance distribution?",
                "Which factors affect performance most?",
                "Show me performance vs sales relationship"
            ])
        
        # Correlation questions
        if len(numeric_cols) >= 2:
            questions.extend([
                "What's the relationship between different metrics?",
                "How do different metrics correlate?",
                "Which factors are most important?"
            ])
        
        # Outlier detection
        if numeric_cols:
            questions.extend([
                "Are there any outliers in the data?",
                "Which data points are unusual?",
                "Show me anomaly detection"
            ])
        
        # Summary questions
        questions.extend([
            "Give me a data summary",
            "What are the key insights?",
            "What recommendations do you have?"
        ])
        
        # Remove duplicates and return top 9
        unique_questions = list(dict.fromkeys(questions))
        return unique_questions[:9]
        
    except Exception as e:
        logger.error(f"Error generating smart questions: {e}")
        return ["Give me a data summary", "What are the key insights?", "Show me the data overview"]

def analyze_question_and_generate_chart(question, data):
    """Analyze the question and generate appropriate chart with error handling"""
    if not st.session_state.gemini_model:
        return None, "AI model not available"
    
    try:
        # Prepare data summary
        data_summary = f"""
        Dataset: {data.shape[0]} rows, {data.shape[1]} columns
        Columns: {list(data.columns)}
        Sample data: {data.head(3).to_string()}
        """
        
        # Generate the appropriate chart based on question keywords
        fig = None
        chart_explanation = ""
        
        # Simple keyword-based chart selection for reliability
        question_lower = question.lower()
        
        if 'trend' in question_lower or 'time' in question_lower:
            # Line chart for trends
            if 'date' in data.columns:
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    y_col = numeric_cols[0]
                    fig = px.line(data, x='date', y=y_col, 
                                title=f'{y_col} Over Time',
                                markers=True)
                    chart_explanation = f"Line chart showing {y_col} trends over time"
        
        elif 'team' in question_lower and ('compare' in question_lower or 'best' in question_lower):
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
                    chart_explanation = f"Bar chart comparing {y_col} across teams"
        
        elif 'relationship' in question_lower or 'correlation' in question_lower:
            # Scatter plot for relationships
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                x_col, y_col = numeric_cols[0], numeric_cols[1]
                fig = px.scatter(data, x=x_col, y=y_col,
                               color='team' if 'team' in data.columns else None,
                               title=f'{x_col} vs {y_col}',
                               trendline='ols')
                chart_explanation = f"Scatter plot showing relationship between {x_col} and {y_col}"
        
        elif 'distribution' in question_lower or 'histogram' in question_lower:
            # Histogram for distributions
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                fig = px.histogram(data, x=col,
                                 title=f'Distribution of {col}',
                                 nbins=20)
                chart_explanation = f"Histogram showing distribution of {col}"
        
        elif 'outlier' in question_lower or 'anomaly' in question_lower:
            # Box plot for outliers
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                fig = px.box(data, y=col,
                           title=f'Distribution of {col} (showing outliers)',
                           color='team' if 'team' in data.columns else None)
                chart_explanation = f"Box plot showing distribution and outliers for {col}"
        
        # Default chart if no specific type detected
        if fig is None:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                fig = px.histogram(data, x=col, title=f'Distribution of {col}')
                chart_explanation = f"Default histogram for {col}"
        
        return fig, chart_explanation
        
    except Exception as e:
        logger.error(f"Error generating chart: {e}")
        return None, f"Error generating chart: {str(e)}"

def get_ai_response(question, data):
    """Get AI response to the question with error handling"""
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
        - Numeric columns: {list(numeric_cols)}
        - Categorical columns: {list(categorical_cols)}
        
        Sample Data:
        {data.head(3).to_string()}
        """
        
        # Add statistical summary if numeric columns exist
        if len(numeric_cols) > 0:
            data_summary += f"""
        
        Statistical Summary:
        {data.describe().to_string()}
        """
        
        # Add categorical information if available
        if len(categorical_cols) > 0:
            data_summary += f"""
        
        Categorical Information:
        """
            for col in categorical_cols[:2]:  # Limit to first 2 categorical columns
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
        
        Be conversational and engaging in your response. Keep it concise but informative.
        """
        
        response = st.session_state.gemini_model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        logger.error(f"Error getting AI response: {e}")
        return f"Error generating response: {str(e)}"

def process_question(question):
    """Process a question with progress indicators"""
    if not st.session_state.data_processed or st.session_state.data is None:
        st.session_state.error_message = "Please upload data first"
        return
    
    try:
        # Show processing progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Analyzing question (25%)
        status_text.text("🤖 AI is analyzing your question...")
        progress_bar.progress(25)
        time.sleep(0.5)
        
        # Step 2: Processing data (50%)
        status_text.text("📊 Processing your data...")
        progress_bar.progress(50)
        time.sleep(0.5)
        
        # Step 3: Generating response (75%)
        status_text.text("💭 Generating AI response...")
        progress_bar.progress(75)
        response = get_ai_response(question, st.session_state.data)
        
        # Step 4: Creating visualization (100%)
        status_text.text("📈 Creating visualization...")
        progress_bar.progress(100)
        chart, chart_explanation = analyze_question_and_generate_chart(question, st.session_state.data)
        
        # Clear progress
        progress_bar.empty()
        status_text.empty()
        
        # Add to chat history
        chat_entry = {
            "question": question,
            "response": response,
            "chart": chart,
            "chart_explanation": chart_explanation,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        
        st.session_state.chat_history.append(chat_entry)
        st.session_state.success_message = "Question processed successfully!"
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        st.session_state.error_message = f"Error processing question: {str(e)}"

def main():
    """Main application with production-ready error handling"""
    
    # Header
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🤖 AI Performance Analytics Agent")
    st.markdown("**Production-Ready Dashboard - Upload your data and chat with AI to get instant insights!**")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display messages
    if st.session_state.error_message:
        st.markdown(f'<div class="error-card">❌ {st.session_state.error_message}</div>', unsafe_allow_html=True)
        st.session_state.error_message = None
    
    if st.session_state.success_message:
        st.markdown(f'<div class="success-card">✅ {st.session_state.success_message}</div>', unsafe_allow_html=True)
        st.session_state.success_message = None
    
    # Create two main columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Upload Section
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.header("📊 Data Upload")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose your data file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload CSV or Excel file with your performance data",
            key="file_uploader"
        )
        
        # Process uploaded file
        if uploaded_file is not None:
            success, message = process_upload_with_progress(uploaded_file)
            
            if success:
                st.markdown(f'<div class="success-card">{message}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="error-card">{message}</div>', unsafe_allow_html=True)
        
        # Sample data button
        if st.button("📈 Generate Sample Data", type="primary", use_container_width=True):
            with st.spinner("Generating sample data..."):
                sample_data = create_sample_data()
                if sample_data is not None:
                    st.session_state.data = sample_data
                    st.session_state.data_processed = True
                    st.session_state.last_upload_time = datetime.now()
                    st.markdown('<div class="success-card">✅ Sample data generated successfully!</div>', unsafe_allow_html=True)
        
        # Progress indicators
        if st.session_state.upload_progress > 0:
            st.markdown('<div class="progress-container">', unsafe_allow_html=True)
            st.write("**Upload Progress:**")
            st.progress(st.session_state.upload_progress / 100)
            st.write(f"{st.session_state.upload_progress}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Data info
        if st.session_state.data is not None:
            st.markdown('<div class="data-info-card">', unsafe_allow_html=True)
            st.header("📋 Data Info")
            data = st.session_state.data
            st.write(f"**Rows:** {len(data)}")
            st.write(f"**Columns:** {len(data.columns)}")
            
            # Show column information
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            categorical_cols = data.select_dtypes(include=['object']).columns
            st.write(f"**Numeric:** {len(numeric_cols)}")
            st.write(f"**Categorical:** {len(categorical_cols)}")
            
            # Show specific info if available
            if 'team' in data.columns:
                st.write(f"**Teams:** {data['team'].nunique()}")
            if 'date' in data.columns:
                st.write(f"**Date Range:** {data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}")
            
            with st.expander("View Data"):
                st.dataframe(data.head(10))
            st.markdown('</div>', unsafe_allow_html=True)
        
        # AI Status
        st.markdown('<div class="status-card">', unsafe_allow_html=True)
        st.header("🤖 AI Status")
        if st.session_state.gemini_model:
            st.success("✅ AI Ready")
        else:
            st.error("❌ AI Not Available")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Chat Section
        if st.session_state.data_processed and st.session_state.data is not None:
            data = st.session_state.data
            
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            st.header("💬 Chat with AI Agent")
            
            # Display chat history
            if st.session_state.chat_history:
                for i, chat in enumerate(st.session_state.chat_history):
                    # User message
                    st.markdown(f'<div class="user-message"><strong>You:</strong> {chat["question"]}</div>', unsafe_allow_html=True)
                    
                    # AI response
                    st.markdown(f'<div class="ai-message"><strong>AI Agent:</strong> {chat["response"]}</div>', unsafe_allow_html=True)
                    
                    # Chart if available
                    if chat.get("chart"):
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        st.plotly_chart(chat["chart"], use_container_width=True)
                        if chat.get("chart_explanation"):
                            st.caption(f"📊 {chat['chart_explanation']}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown("---")
            
            # Question input
            col_input1, col_input2 = st.columns([4, 1])
            
            with col_input1:
                question = st.text_input(
                    "Ask a question about your data:",
                    placeholder="e.g., Which team has the best performance? Show me sales trends over time.",
                    key="question_input"
                )
            
            with col_input2:
                ask_button = st.button("Ask", type="primary", use_container_width=True)
            
            # Process question
            if ask_button and question:
                process_question(question)
                st.rerun()
            
            # Smart suggested questions
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
                            if st.button(question, key=f"smart_q_{i+j}", use_container_width=True):
                                process_question(question)
                                st.rerun()
            
            # Clear chat button
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        else:
            # Welcome message
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            st.info("👆 Upload a data file or generate sample data to start chatting with the AI agent!")
            
            # Demo section
            st.header("🎓 How It Works")
            
            col_demo1, col_demo2 = st.columns(2)
            
            with col_demo1:
                st.subheader("🤖 Natural Language Processing")
                st.markdown("""
                - **Ask in plain English**: "Which team is performing best?"
                - **Get instant answers**: AI analyzes your data and responds
                - **Automatic visualizations**: Charts are generated based on your questions
                - **Conversational interface**: Chat-like experience
                """)
            
            with col_demo2:
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
                "Compare the average performance across all teams"
            ]
            
            for i, example in enumerate(examples):
                if i % 2 == 0:
                    col_ex1, col_ex2 = st.columns(2)
                
                with col_ex1 if i % 2 == 0 else col_ex2:
                    st.write(f"💡 *{example}*")
            
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
