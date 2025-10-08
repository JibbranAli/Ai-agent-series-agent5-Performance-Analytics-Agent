"""
Compact Dark Theme AI Performance Analytics Agent Dashboard
Eliminates empty spaces and gaps with a content-rich, compact design.
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
    page_title="AI Performance Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Compact dark theme CSS
st.markdown("""
<style>
    /* Hide Streamlit default elements */
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    .stApp > header {background-color: transparent;}
    
    /* Dark theme global styles */
    .stApp {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a1a 50%, #2d2d2d 100%);
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        padding: 0;
        margin: 0;
    }
    
    /* Main container - compact */
    .main-container {
        max-width: 100%;
        margin: 0;
        padding: 0;
        background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
        min-height: 100vh;
        border: 1px solid #333;
    }
    
    /* Header - compact */
    .header-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px 30px;
        text-align: center;
        color: white;
        margin: 0;
    }
    
    /* Custom tabs - compact */
    .tab-container {
        background: #1a1a1a;
        padding: 0;
        margin: 0;
    }
    
    .tab-buttons {
        display: flex;
        background: #1a1a1a;
        margin: 0;
        padding: 0;
    }
    
    .tab-button {
        flex: 1;
        padding: 12px 15px;
        background: #2d2d2d;
        border: none;
        color: #ccc;
        cursor: pointer;
        transition: all 0.3s ease;
        border-right: 1px solid #333;
        font-weight: 600;
        font-size: 14px;
    }
    
    .tab-button:hover {
        background: #3d3d3d;
        color: #fff;
    }
    
    .tab-button.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .tab-content {
        padding: 20px;
        background: #1e1e1e;
        min-height: calc(100vh - 200px);
        margin: 0;
    }
    
    /* Compact sections */
    .section {
        background: #2d2d2d;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid #444;
    }
    
    .section-title {
        color: #fff;
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 15px;
    }
    
    /* Upload section - compact */
    .upload-section {
        background: #2d2d2d;
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 25px;
        text-align: center;
        margin: 15px 0;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #764ba2;
        background: #3d3d3d;
    }
    
    /* Chat messages - compact */
    .chat-message {
        margin: 10px 0;
        padding: 15px 20px;
        border-radius: 15px;
        max-width: 80%;
        word-wrap: break-word;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: auto;
        text-align: right;
    }
    
    .ai-message {
        background: #3d3d3d;
        color: #fff;
        border-left: 3px solid #667eea;
    }
    
    .system-message {
        background: #3d2d1a;
        color: #ffd93d;
        border: 1px solid #ffd93d;
        text-align: center;
        margin: 20px auto;
        max-width: 90%;
        border-radius: 10px;
        padding: 20px;
    }
    
    /* Input area - compact */
    .input-area {
        background: #2d2d2d;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border: 1px solid #444;
    }
    
    /* Buttons - compact */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
        font-size: 14px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Cards - compact */
    .info-card {
        background: #1a3d4d;
        border: 1px solid #17a2b8;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: #b3e5fc;
    }
    
    .success-card {
        background: #1a4d2a;
        border: 1px solid #28a745;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: #c3e6cb;
    }
    
    .error-card {
        background: #4d1a1a;
        border: 1px solid #dc3545;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: #f5c6cb;
    }
    
    /* Data info - compact */
    .data-info {
        background: #2d2d2d;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid #444;
    }
    
    /* Chart container - compact */
    .chart-container {
        background: #1e1e1e;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid #333;
    }
    
    /* Progress bar - compact */
    .progress-container {
        background: #1a2d3d;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid #444;
    }
    
    /* Metrics - compact */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 5px;
    }
    
    /* Question buttons - compact */
    .question-button {
        background: #2d2d2d;
        border: 1px solid #667eea;
        border-radius: 10px;
        padding: 12px 15px;
        margin: 5px 0;
        transition: all 0.3s ease;
        text-align: left;
        color: #fff;
        font-size: 13px;
    }
    
    .question-button:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        transform: translateX(3px);
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        background: #2d2d2d;
        color: #fff;
        border: 1px solid #444;
        border-radius: 8px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    /* File uploader styling */
    .stFileUploader > div > div {
        background: #2d2d2d;
        border: 1px solid #444;
        border-radius: 8px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #2d2d2d;
        color: #fff;
        border: 1px solid #444;
    }
    
    .streamlit-expanderContent {
        background: #1e1e1e;
        border: 1px solid #444;
    }
    
    /* Remove extra margins and padding */
    .stMarkdown {
        margin: 0;
        padding: 0;
    }
    
    .stHeader {
        margin: 0;
        padding: 0;
    }
    
    /* Compact grid */
    .compact-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 10px;
        margin: 10px 0;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .tab-content {
            padding: 15px;
        }
        
        .chat-message {
            max-width: 90%;
        }
        
        .section {
            padding: 15px;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'data': None,
        'chat_history': [],
        'gemini_model': None,
        'upload_progress': 0,
        'ai_initialized': False,
        'data_processed': False,
        'error_message': None,
        'success_message': None,
        'active_tab': 'upload'
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Initialize session state
init_session_state()

# Initialize Gemini AI
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
    """Process uploaded file with progress indicators"""
    try:
        st.session_state.upload_progress = 0
        st.session_state.error_message = None
        st.session_state.success_message = None
        
        if uploaded_file is None:
            return False, "No file provided"
        
        # Progress steps
        progress_steps = [
            (20, "Validating file..."),
            (40, "Reading file..."),
            (60, "Processing data..."),
            (80, "Validating data..."),
            (100, "Complete!")
        ]
        
        for progress, message in progress_steps:
            st.session_state.upload_progress = progress
            time.sleep(0.1)
        
        # Check file type
        if not (uploaded_file.name.endswith('.csv') or uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls')):
            return False, "Unsupported file type. Please upload CSV or Excel files."
        
        # Load file
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
        except Exception as e:
            return False, f"Error reading file: {str(e)}"
        
        # Validate data
        if data.empty:
            return False, "File is empty or contains no data"
        
        if len(data.columns) < 2:
            return False, "File must contain at least 2 columns"
        
        # Store data
        st.session_state.data = data
        st.session_state.data_processed = True
        
        logger.info(f"File processed successfully: {data.shape}")
        return True, f"✅ Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns"
        
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        return False, f"❌ Error processing file: {str(e)}"

def generate_smart_questions(data):
    """Generate intelligent questions based on the data structure"""
    try:
        questions = []
        
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
        
        # Summary questions
        questions.extend([
            "Give me a data summary",
            "What are the key insights?",
            "What recommendations do you have?"
        ])
        
        unique_questions = list(dict.fromkeys(questions))
        return unique_questions[:9]
        
    except Exception as e:
        logger.error(f"Error generating smart questions: {e}")
        return ["Give me a data summary", "What are the key insights?", "Show me the data overview"]

def analyze_question_and_generate_chart(question, data):
    """Analyze the question and generate appropriate chart"""
    if not st.session_state.gemini_model:
        return None, "AI model not available"
    
    try:
        question_lower = question.lower()
        fig = None
        chart_explanation = ""
        
        if 'trend' in question_lower or 'time' in question_lower:
            if 'date' in data.columns:
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    y_col = numeric_cols[0]
                    fig = px.line(data, x='date', y=y_col, 
                                title=f'{y_col} Over Time',
                                markers=True)
                    fig.update_layout(
                        plot_bgcolor='#1e1e1e',
                        paper_bgcolor='#1e1e1e',
                        font_color='white'
                    )
                    chart_explanation = f"Line chart showing {y_col} trends over time"
        
        elif 'team' in question_lower and ('compare' in question_lower or 'best' in question_lower):
            if 'team' in data.columns:
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    y_col = numeric_cols[0]
                    team_data = data.groupby('team')[y_col].mean().reset_index()
                    fig = px.bar(team_data, x='team', y=y_col,
                               title=f'Average {y_col} by Team',
                               color=y_col,
                               color_continuous_scale='Viridis')
                    fig.update_layout(
                        plot_bgcolor='#1e1e1e',
                        paper_bgcolor='#1e1e1e',
                        font_color='white'
                    )
                    chart_explanation = f"Bar chart comparing {y_col} across teams"
        
        elif 'relationship' in question_lower or 'correlation' in question_lower:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                x_col, y_col = numeric_cols[0], numeric_cols[1]
                fig = px.scatter(data, x=x_col, y=y_col,
                               color='team' if 'team' in data.columns else None,
                               title=f'{x_col} vs {y_col}',
                               trendline='ols')
                fig.update_layout(
                    plot_bgcolor='#1e1e1e',
                    paper_bgcolor='#1e1e1e',
                    font_color='white'
                )
                chart_explanation = f"Scatter plot showing relationship between {x_col} and {y_col}"
        
        elif 'distribution' in question_lower or 'histogram' in question_lower:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                fig = px.histogram(data, x=col,
                                 title=f'Distribution of {col}',
                                 nbins=20)
                fig.update_layout(
                    plot_bgcolor='#1e1e1e',
                    paper_bgcolor='#1e1e1e',
                    font_color='white'
                )
                chart_explanation = f"Histogram showing distribution of {col}"
        
        # Default chart
        if fig is None:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                fig = px.histogram(data, x=col, title=f'Distribution of {col}')
                fig.update_layout(
                    plot_bgcolor='#1e1e1e',
                    paper_bgcolor='#1e1e1e',
                    font_color='white'
                )
                chart_explanation = f"Default histogram for {col}"
        
        return fig, chart_explanation
        
    except Exception as e:
        logger.error(f"Error generating chart: {e}")
        return None, f"Error generating chart: {str(e)}"

def get_ai_response(question, data):
    """Get AI response to the question"""
    if not st.session_state.gemini_model:
        return "AI model not available. Please check your configuration."
    
    try:
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
        
        if len(numeric_cols) > 0:
            data_summary += f"""
        
        Statistical Summary:
        {data.describe().to_string()}
        """
        
        if len(categorical_cols) > 0:
            data_summary += f"""
        
        Categorical Information:
        """
            for col in categorical_cols[:2]:
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
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🤖 AI is analyzing your question...")
        progress_bar.progress(25)
        time.sleep(0.2)
        
        status_text.text("📊 Processing your data...")
        progress_bar.progress(50)
        time.sleep(0.2)
        
        status_text.text("💭 Generating AI response...")
        progress_bar.progress(75)
        response = get_ai_response(question, st.session_state.data)
        
        status_text.text("📈 Creating visualization...")
        progress_bar.progress(100)
        chart, chart_explanation = analyze_question_and_generate_chart(question, st.session_state.data)
        
        progress_bar.empty()
        status_text.empty()
        
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
    """Main application with compact dark theme"""
    
    # Main container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="header-section">', unsafe_allow_html=True)
    st.title("🤖 AI Performance Analytics Agent")
    st.markdown("**Dark Theme Dashboard - Upload your data and chat with AI to get instant insights!**")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display messages
    if st.session_state.error_message:
        st.markdown(f'<div class="error-card">❌ {st.session_state.error_message}</div>', unsafe_allow_html=True)
        st.session_state.error_message = None
    
    if st.session_state.success_message:
        st.markdown(f'<div class="success-card">✅ {st.session_state.success_message}</div>', unsafe_allow_html=True)
        st.session_state.success_message = None
    
    # Custom tabs
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    
    # Tab buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Upload Data", key="tab_upload", use_container_width=True):
            st.session_state.active_tab = 'upload'
            st.rerun()
    
    with col2:
        if st.button("💬 Chat with AI", key="tab_chat", use_container_width=True):
            st.session_state.active_tab = 'chat'
            st.rerun()
    
    with col3:
        if st.button("📈 Analytics", key="tab_analytics", use_container_width=True):
            st.session_state.active_tab = 'analytics'
            st.rerun()
    
    with col4:
        if st.button("⚙️ Settings", key="tab_settings", use_container_width=True):
            st.session_state.active_tab = 'settings'
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab content
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    if st.session_state.active_tab == 'upload':
        # Upload tab - compact layout
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Upload Your Data</div>', unsafe_allow_html=True)
        st.markdown("**Start by uploading your performance data to begin chatting with the AI agent**")
        
        # Upload area
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose your data file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload CSV or Excel file with your performance data",
            key="file_uploader"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process uploaded file
        if uploaded_file is not None:
            success, message = process_upload_with_progress(uploaded_file)
            
            if success:
                st.markdown(f'<div class="success-card">{message}</div>', unsafe_allow_html=True)
                st.rerun()
            else:
                st.markdown(f'<div class="error-card">{message}</div>', unsafe_allow_html=True)
        
        # Sample data button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📈 Generate Sample Data", type="primary", use_container_width=True):
                with st.spinner("Generating sample data..."):
                    sample_data = create_sample_data()
                    if sample_data is not None:
                        st.session_state.data = sample_data
                        st.session_state.data_processed = True
                        st.markdown('<div class="success-card">✅ Sample data generated successfully!</div>', unsafe_allow_html=True)
                        st.rerun()
        
        # Progress indicators
        if st.session_state.upload_progress > 0:
            st.markdown('<div class="progress-container">', unsafe_allow_html=True)
            st.write("**Upload Progress:**")
            st.progress(st.session_state.upload_progress / 100)
            st.write(f"{st.session_state.upload_progress}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Data info if available
        if st.session_state.data is not None:
            st.markdown('<div class="data-info">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📋 Data Overview</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Rows", len(st.session_state.data))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Columns", len(st.session_state.data.columns))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns
                st.metric("Numeric Columns", len(numeric_cols))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                categorical_cols = st.session_state.data.select_dtypes(include=['object']).columns
                st.metric("Categorical Columns", len(categorical_cols))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with st.expander("View Data"):
                st.dataframe(st.session_state.data.head(10))
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif st.session_state.active_tab == 'chat':
        # Chat tab - compact layout
        if not st.session_state.data_processed:
            st.markdown('<div class="system-message">', unsafe_allow_html=True)
            st.markdown('''
            <strong>📋 Please upload your data first!</strong><br>
            To start chatting with the AI agent, you need to upload a data file or generate sample data. 
            Once your data is loaded, you'll be able to ask questions and get instant insights with visualizations.
            ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="section">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">💬 Chat with AI Agent</div>', unsafe_allow_html=True)
            
            # Display chat history
            if st.session_state.chat_history:
                for i, chat in enumerate(st.session_state.chat_history):
                    # User message
                    st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {chat["question"]}</div>', unsafe_allow_html=True)
                    
                    # AI response
                    st.markdown(f'<div class="chat-message ai-message"><strong>AI Agent:</strong> {chat["response"]}</div>', unsafe_allow_html=True)
                    
                    # Chart if available
                    if chat.get("chart"):
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        st.plotly_chart(chat["chart"], use_container_width=True)
                        if chat.get("chart_explanation"):
                            st.caption(f"📊 {chat['chart_explanation']}")
                        st.markdown('</div>', unsafe_allow_html=True)
            
            # Input area
            st.markdown('<div class="input-area">', unsafe_allow_html=True)
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
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Smart suggested questions
            st.markdown('<div class="section-title">💡 Smart Questions Based on Your Data</div>', unsafe_allow_html=True)
            
            smart_questions = generate_smart_questions(st.session_state.data)
            
            # Display questions in a grid
            cols = st.columns(3)
            for i, question in enumerate(smart_questions):
                with cols[i % 3]:
                    if st.button(question, key=f"smart_q_{i}", use_container_width=True):
                        process_question(question)
                        st.rerun()
            
            # Clear chat button
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif st.session_state.active_tab == 'analytics':
        # Analytics tab - compact layout
        if not st.session_state.data_processed:
            st.markdown('<div class="system-message">', unsafe_allow_html=True)
            st.markdown('''
            <strong>📋 Please upload your data first!</strong><br>
            To view analytics, you need to upload a data file or generate sample data.
            ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="section">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📈 Data Analytics</div>', unsafe_allow_html=True)
            
            # Quick analytics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="section-title">📊 Quick Stats</div>', unsafe_allow_html=True)
                numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    for col in numeric_cols[:3]:
                        st.metric(
                            f"Average {col}",
                            f"{st.session_state.data[col].mean():.2f}",
                            f"±{st.session_state.data[col].std():.2f}"
                        )
            
            with col2:
                st.markdown('<div class="section-title">📋 Data Summary</div>', unsafe_allow_html=True)
                st.write(f"**Total Records:** {len(st.session_state.data)}")
                st.write(f"**Date Range:** {st.session_state.data['date'].min().strftime('%Y-%m-%d')} to {st.session_state.data['date'].max().strftime('%Y-%m-%d')}")
                if 'team' in st.session_state.data.columns:
                    st.write(f"**Teams:** {', '.join(st.session_state.data['team'].unique())}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif st.session_state.active_tab == 'settings':
        # Settings tab - compact layout
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">⚙️ Settings</div>', unsafe_allow_html=True)
        
        # AI Status
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        if st.session_state.gemini_model:
            st.success("✅ AI Agent Ready")
        else:
            st.error("❌ AI Agent Not Available")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # App Info
        st.markdown('<div class="section-title">📱 App Information</div>', unsafe_allow_html=True)
        st.write("**Version:** 1.0.0")
        st.write("**Theme:** Dark")
        st.write("**AI Model:** Gemini 2.0 Flash")
        
        # Clear data button
        if st.button("🗑️ Clear All Data", type="secondary"):
            st.session_state.data = None
            st.session_state.chat_history = []
            st.session_state.data_processed = False
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
