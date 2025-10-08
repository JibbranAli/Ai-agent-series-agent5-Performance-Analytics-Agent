"""
Professional AI Performance Analytics Agent Dashboard
Clean, streamlined design without unnecessary boxes or empty spaces.
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

# Professional CSS - minimal and clean
st.markdown("""
<style>
    /* Hide Streamlit default elements */
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    .stApp > header {background-color: transparent;}
    
    /* Professional dark theme */
    .stApp {
        background: #0f0f0f;
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Main container - clean */
    .main-container {
        max-width: 100%;
        margin: 0;
        padding: 0;
        background: #0f0f0f;
    }
    
    /* Header - minimal */
    .header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        text-align: center;
        color: white;
        margin-bottom: 0;
    }
    
    /* Tabs - clean */
    .tabs {
        display: flex;
        background: #1a1a1a;
        border-bottom: 1px solid #333;
        margin: 0;
        padding: 0;
    }
    
    .tab {
        flex: 1;
        padding: 15px;
        background: #1a1a1a;
        border: none;
        color: #ccc;
        cursor: pointer;
        transition: all 0.3s ease;
        border-right: 1px solid #333;
        font-weight: 600;
    }
    
    .tab:hover {
        background: #2a2a2a;
        color: #fff;
    }
    
    .tab.active {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Content area - clean */
    .content {
        padding: 20px;
        background: #0f0f0f;
        min-height: calc(100vh - 150px);
    }
    
    /* Buttons - professional */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Text inputs - clean */
    .stTextInput > div > div > input {
        background: #1a1a1a;
        color: #fff;
        border: 1px solid #333;
        border-radius: 8px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    /* File uploader - clean */
    .stFileUploader > div > div {
        background: #1a1a1a;
        border: 1px solid #333;
        border-radius: 8px;
    }
    
    /* Chat messages - clean */
    .user-msg {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 18px;
        border-radius: 18px;
        margin: 10px 0 10px auto;
        max-width: 70%;
        text-align: right;
    }
    
    .ai-msg {
        background: #1a1a1a;
        color: #fff;
        padding: 12px 18px;
        border-radius: 18px;
        margin: 10px auto 10px 0;
        max-width: 70%;
        border-left: 3px solid #667eea;
    }
    
    /* Metrics - clean */
    .metric {
        background: #1a1a1a;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #333;
    }
    
    /* Progress bar - clean */
    .progress {
        background: #1a1a1a;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #333;
    }
    
    /* Success/Error messages - clean */
    .success {
        background: #1a4d2a;
        color: #c3e6cb;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #28a745;
        margin: 10px 0;
    }
    
    .error {
        background: #4d1a1a;
        color: #f5c6cb;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #dc3545;
        margin: 10px 0;
    }
    
    /* Info message - clean */
    .info {
        background: #1a3d4d;
        color: #b3e5fc;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #17a2b8;
        margin: 10px 0;
    }
    
    /* Remove extra margins */
    .stMarkdown {
        margin: 0;
        padding: 0;
    }
    
    .stHeader {
        margin: 0;
        padding: 0;
    }
    
    /* Expander - clean */
    .streamlit-expanderHeader {
        background: #1a1a1a;
        color: #fff;
        border: 1px solid #333;
    }
    
    .streamlit-expanderContent {
        background: #0f0f0f;
        border: 1px solid #333;
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
        return True, f"Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns"
        
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        return False, f"Error processing file: {str(e)}"

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
                        plot_bgcolor='#0f0f0f',
                        paper_bgcolor='#0f0f0f',
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
                        plot_bgcolor='#0f0f0f',
                        paper_bgcolor='#0f0f0f',
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
                    plot_bgcolor='#0f0f0f',
                    paper_bgcolor='#0f0f0f',
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
                    plot_bgcolor='#0f0f0f',
                    paper_bgcolor='#0f0f0f',
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
                    plot_bgcolor='#0f0f0f',
                    paper_bgcolor='#0f0f0f',
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
    """Main application with professional design"""
    
    # Header
    st.markdown('<div class="header">', unsafe_allow_html=True)
    st.title("🤖 AI Performance Analytics Agent")
    st.markdown("**Professional Dashboard - Upload your data and chat with AI to get instant insights!**")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display messages
    if st.session_state.error_message:
        st.markdown(f'<div class="error">❌ {st.session_state.error_message}</div>', unsafe_allow_html=True)
        st.session_state.error_message = None
    
    if st.session_state.success_message:
        st.markdown(f'<div class="success">✅ {st.session_state.success_message}</div>', unsafe_allow_html=True)
        st.session_state.success_message = None
    
    # Tabs
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
    
    # Content
    if st.session_state.active_tab == 'upload':
        st.header("📊 Upload Your Data")
        st.markdown("**Start by uploading your performance data to begin chatting with the AI agent**")
        
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
                st.markdown(f'<div class="success">{message}</div>', unsafe_allow_html=True)
                st.rerun()
            else:
                st.markdown(f'<div class="error">{message}</div>', unsafe_allow_html=True)
        
        # Sample data button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📈 Generate Sample Data", type="primary", use_container_width=True):
                with st.spinner("Generating sample data..."):
                    sample_data = create_sample_data()
                    if sample_data is not None:
                        st.session_state.data = sample_data
                        st.session_state.data_processed = True
                        st.markdown('<div class="success">Sample data generated successfully!</div>', unsafe_allow_html=True)
                        st.rerun()
        
        # Progress indicators
        if st.session_state.upload_progress > 0:
            st.markdown('<div class="progress">', unsafe_allow_html=True)
            st.write("**Upload Progress:**")
            st.progress(st.session_state.upload_progress / 100)
            st.write(f"{st.session_state.upload_progress}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Data info if available
        if st.session_state.data is not None:
            st.header("📋 Data Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric">', unsafe_allow_html=True)
                st.metric("Rows", len(st.session_state.data))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric">', unsafe_allow_html=True)
                st.metric("Columns", len(st.session_state.data.columns))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric">', unsafe_allow_html=True)
                numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns
                st.metric("Numeric Columns", len(numeric_cols))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric">', unsafe_allow_html=True)
                categorical_cols = st.session_state.data.select_dtypes(include=['object']).columns
                st.metric("Categorical Columns", len(categorical_cols))
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Additional info
            if 'team' in st.session_state.data.columns:
                st.write(f"**Teams:** {st.session_state.data['team'].nunique()}")
            
            # Fix date handling
            if 'date' in st.session_state.data.columns:
                try:
                    # Convert to datetime if it's not already
                    if not pd.api.types.is_datetime64_any_dtype(st.session_state.data['date']):
                        st.session_state.data['date'] = pd.to_datetime(st.session_state.data['date'])
                    
                    date_min = st.session_state.data['date'].min()
                    date_max = st.session_state.data['date'].max()
                    st.write(f"**Date Range:** {date_min.strftime('%Y-%m-%d')} to {date_max.strftime('%Y-%m-%d')}")
                except Exception as e:
                    st.write("**Date Column:** Present but format unclear")
            
            with st.expander("View Data"):
                st.dataframe(st.session_state.data.head(10))
    
    elif st.session_state.active_tab == 'chat':
        if not st.session_state.data_processed:
            st.markdown('<div class="info">', unsafe_allow_html=True)
            st.markdown('''
            **📋 Please upload your data first!**<br>
            To start chatting with the AI agent, you need to upload a data file or generate sample data. 
            Once your data is loaded, you'll be able to ask questions and get instant insights with visualizations.
            ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.header("💬 Chat with AI Agent")
            
            # Display chat history
            if st.session_state.chat_history:
                for i, chat in enumerate(st.session_state.chat_history):
                    # User message
                    st.markdown(f'<div class="user-msg"><strong>You:</strong> {chat["question"]}</div>', unsafe_allow_html=True)
                    
                    # AI response
                    st.markdown(f'<div class="ai-msg"><strong>AI Agent:</strong> {chat["response"]}</div>', unsafe_allow_html=True)
                    
                    # Chart if available
                    if chat.get("chart"):
                        st.plotly_chart(chat["chart"], use_container_width=True)
                        if chat.get("chart_explanation"):
                            st.caption(f"📊 {chat['chart_explanation']}")
            
            # Input area
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
    
    elif st.session_state.active_tab == 'analytics':
        if not st.session_state.data_processed:
            st.markdown('<div class="info">', unsafe_allow_html=True)
            st.markdown('''
            **📋 Please upload your data first!**<br>
            To view analytics, you need to upload a data file or generate sample data.
            ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.header("📈 Data Analytics")
            
            # Quick analytics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Quick Stats")
                numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    for col in numeric_cols[:3]:
                        st.metric(
                            f"Average {col}",
                            f"{st.session_state.data[col].mean():.2f}",
                            f"±{st.session_state.data[col].std():.2f}"
                        )
            
            with col2:
                st.subheader("📋 Data Summary")
                st.write(f"**Total Records:** {len(st.session_state.data)}")
                
                # Fix date handling
                if 'date' in st.session_state.data.columns:
                    try:
                        # Convert to datetime if it's not already
                        if not pd.api.types.is_datetime64_any_dtype(st.session_state.data['date']):
                            st.session_state.data['date'] = pd.to_datetime(st.session_state.data['date'])
                        
                        date_min = st.session_state.data['date'].min()
                        date_max = st.session_state.data['date'].max()
                        st.write(f"**Date Range:** {date_min.strftime('%Y-%m-%d')} to {date_max.strftime('%Y-%m-%d')}")
                    except Exception as e:
                        st.write("**Date Column:** Present but format unclear")
                
                if 'team' in st.session_state.data.columns:
                    st.write(f"**Teams:** {', '.join(st.session_state.data['team'].unique())}")
    
    elif st.session_state.active_tab == 'settings':
        st.header("⚙️ Settings")
        
        # AI Status
        st.markdown('<div class="info">', unsafe_allow_html=True)
        if st.session_state.gemini_model:
            st.success("✅ AI Agent Ready")
        else:
            st.error("❌ AI Agent Not Available")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # App Info
        st.subheader("📱 App Information")
        st.write("**Version:** 1.0.0")
        st.write("**Theme:** Professional Dark")
        st.write("**AI Model:** Gemini 2.0 Flash")
        
        # Clear data button
        if st.button("🗑️ Clear All Data", type="secondary"):
            st.session_state.data = None
            st.session_state.chat_history = []
            st.session_state.data_processed = False
            st.rerun()

if __name__ == "__main__":
    main()
