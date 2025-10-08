"""
Beautiful AI Performance Analytics Agent Dashboard
Modern, clean design with thin line separators and stunning UI/UX.
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

# Beautiful CSS with thin line separators
st.markdown("""
<style>
    /* Hide Streamlit default elements */
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    .stApp > header {background-color: transparent;}
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Main container */
    .main-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 0;
        background: white;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        overflow: hidden;
    }
    
    /* Header */
    .header-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px 30px;
        text-align: center;
        color: white;
        position: relative;
        overflow: hidden;
    }
    
    .header-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="white" opacity="0.1"/><circle cx="75" cy="75" r="1" fill="white" opacity="0.1"/><circle cx="50" cy="10" r="0.5" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        opacity: 0.3;
    }
    
    .header-content {
        position: relative;
        z-index: 1;
    }
    
    /* Thin line separator */
    .separator {
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, #e0e0e0 20%, #e0e0e0 80%, transparent 100%);
        margin: 30px 0;
    }
    
    .separator-thick {
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #667eea 20%, #764ba2 80%, transparent 100%);
        margin: 40px 0;
    }
    
    /* Upload section */
    .upload-section {
        padding: 40px 30px;
        text-align: center;
        background: white;
        position: relative;
    }
    
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 40px 20px;
        margin: 20px 0;
        background: linear-gradient(135deg, #f8f9ff 0%, #f0f2ff 100%);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .upload-area:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, #f0f2ff 0%, #e8ebff 100%);
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.15);
    }
    
    /* Chat container */
    .chat-container {
        background: white;
        padding: 30px;
        min-height: 500px;
    }
    
    /* Chat messages */
    .chat-message {
        margin: 20px 0;
        padding: 20px 25px;
        border-radius: 25px;
        max-width: 75%;
        word-wrap: break-word;
        position: relative;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: auto;
        text-align: right;
        border-bottom-right-radius: 8px;
    }
    
    .ai-message {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        color: #333;
        border-left: 4px solid #667eea;
        border-bottom-left-radius: 8px;
    }
    
    .system-message {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        color: #856404;
        border: 1px solid #ffd93d;
        text-align: center;
        margin: 30px auto;
        max-width: 90%;
        border-radius: 15px;
        padding: 25px;
    }
    
    /* Input area */
    .input-area {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 25px;
        border-radius: 20px;
        margin-top: 30px;
        border: 1px solid #e0e0e0;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 25px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Cards */
    .info-card {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: 1px solid #17a2b8;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        color: #0c5460;
        box-shadow: 0 4px 15px rgba(23, 162, 184, 0.1);
    }
    
    .success-card {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #28a745;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        color: #155724;
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.1);
    }
    
    .error-card {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 1px solid #dc3545;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        color: #721c24;
        box-shadow: 0 4px 15px rgba(220, 53, 69, 0.1);
    }
    
    /* Data info */
    .data-info {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    
    /* Chart container */
    .chart-container {
        background: white;
        border-radius: 20px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    
    /* Progress bar */
    .progress-container {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        border: 1px solid #bbdefb;
    }
    
    /* Metrics */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
    }
    
    /* Question buttons */
    .question-button {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 1px solid #667eea;
        border-radius: 15px;
        padding: 15px 20px;
        margin: 8px 0;
        transition: all 0.3s ease;
        text-align: left;
    }
    
    .question-button:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        transform: translateX(5px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-container {
            margin: 10px;
            border-radius: 15px;
        }
        
        .header-section {
            padding: 30px 20px;
        }
        
        .chat-message {
            max-width: 90%;
        }
        
        .upload-section {
            padding: 30px 20px;
        }
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
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
        'show_upload_prompt': True
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
            time.sleep(0.2)
        
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
        st.session_state.show_upload_prompt = False
        
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
                    chart_explanation = f"Bar chart comparing {y_col} across teams"
        
        elif 'relationship' in question_lower or 'correlation' in question_lower:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                x_col, y_col = numeric_cols[0], numeric_cols[1]
                fig = px.scatter(data, x=x_col, y=y_col,
                               color='team' if 'team' in data.columns else None,
                               title=f'{x_col} vs {y_col}',
                               trendline='ols')
                chart_explanation = f"Scatter plot showing relationship between {x_col} and {y_col}"
        
        elif 'distribution' in question_lower or 'histogram' in question_lower:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                fig = px.histogram(data, x=col,
                                 title=f'Distribution of {col}',
                                 nbins=20)
                chart_explanation = f"Histogram showing distribution of {col}"
        
        # Default chart
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
        time.sleep(0.3)
        
        status_text.text("📊 Processing your data...")
        progress_bar.progress(50)
        time.sleep(0.3)
        
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
    """Main application with beautiful design"""
    
    # Main container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="header-section">', unsafe_allow_html=True)
    st.markdown('<div class="header-content">', unsafe_allow_html=True)
    st.title("🤖 AI Performance Analytics Agent")
    st.markdown("**Upload your data and chat with AI to get instant insights and visualizations!**")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display messages
    if st.session_state.error_message:
        st.markdown(f'<div class="error-card">❌ {st.session_state.error_message}</div>', unsafe_allow_html=True)
        st.session_state.error_message = None
    
    if st.session_state.success_message:
        st.markdown(f'<div class="success-card">✅ {st.session_state.success_message}</div>', unsafe_allow_html=True)
        st.session_state.success_message = None
    
    # Main content area
    if not st.session_state.data_processed:
        # Upload section
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.header("📊 Upload Your Data")
        st.markdown("**Start by uploading your performance data to begin chatting with the AI agent**")
        
        # Upload area
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
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
                        st.session_state.show_upload_prompt = False
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
        
        # Thin line separator
        st.markdown('<div class="separator-thick"></div>', unsafe_allow_html=True)
        
        # Chat interface with upload prompt
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        st.header("💬 Chat Interface")
        
        # System message prompting for data upload
        st.markdown('''
        <div class="system-message">
            <strong>📋 Please upload your data first!</strong><br>
            To start chatting with the AI agent, you need to upload a data file or generate sample data. 
            Once your data is loaded, you'll be able to ask questions and get instant insights with visualizations.
        </div>
        ''', unsafe_allow_html=True)
        
        # Disabled chat input
        st.text_input(
            "Ask a question about your data:",
            placeholder="Upload data first to enable chat...",
            disabled=True,
            key="disabled_question_input"
        )
        
        st.button("Ask", disabled=True, use_container_width=True)
        
        # Example questions (disabled)
        st.header("💡 Example Questions (Available after data upload)")
        examples = [
            "Which team has the highest sales?",
            "Show me the performance trends over time",
            "What's the relationship between sales and customer satisfaction?",
            "Which quarter had the best performance?",
            "Are there any outliers in the performance data?",
            "Compare the average performance across all teams"
        ]
        
        for example in examples:
            st.button(example, disabled=True, use_container_width=True, key=f"disabled_{example}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Data is loaded - show chat interface
        data = st.session_state.data
        
        # Data info section
        st.markdown('<div class="data-info">', unsafe_allow_html=True)
        st.header("📋 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Rows", len(data))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Columns", len(data.columns))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            st.metric("Numeric Columns", len(numeric_cols))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            categorical_cols = data.select_dtypes(include=['object']).columns
            st.metric("Categorical Columns", len(categorical_cols))
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Additional info
        if 'team' in data.columns:
            st.write(f"**Teams:** {data['team'].nunique()}")
        if 'date' in data.columns:
            try:
                if not pd.api.types.is_datetime64_any_dtype(data['date']):
                    data['date'] = pd.to_datetime(data['date'])
                date_min = data['date'].min()
                date_max = data['date'].max()
                st.write(f"**Date Range:** {date_min.strftime('%Y-%m-%d')} to {date_max.strftime('%Y-%m-%d')}")
            except:
                st.write("**Date Column:** Present")
        
        with st.expander("View Data"):
            st.dataframe(data.head(10))
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Thin line separator
        st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
        
        # Chat interface
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        st.header("💬 Chat with AI Agent")
        
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
                
                # Thin line separator between messages
                if i < len(st.session_state.chat_history) - 1:
                    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
        
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
        
        # Thin line separator
        st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
        
        # Smart suggested questions
        st.header("💡 Smart Questions Based on Your Data")
        
        smart_questions = generate_smart_questions(data)
        
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
    
    # Thin line separator
    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
    
    # AI Status
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    if st.session_state.gemini_model:
        st.success("✅ AI Agent Ready")
    else:
        st.error("❌ AI Agent Not Available")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
