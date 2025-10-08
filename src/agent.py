"""
Performance Analytics AI Agent
Core agent implementation with memory, planning, and reasoning capabilities
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import json
import google.generativeai as genai
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceAnalyticsAgent:
    """
    AI Agent for Performance Analytics with memory, planning, and reasoning capabilities.
    
    This agent demonstrates key concepts of Agentic AI:
    - Memory: Persistent storage of interactions and analysis history
    - Planning: Step-by-step task decomposition
    - Reasoning: Logical analysis and inference
    - Tool Usage: Modular capability system
    - Learning: Improvement through interactions
    """
    
    def __init__(self, config):
        """Initialize the AI Agent with configuration"""
        self.config = config
        self.memory = []
        self.analysis_history = []
        self.current_data = None
        self.insights = []
        self.plans = []
        
        # Initialize AI model
        self.model = None
        if self.config.ai.enable_ai and self.config.ai.gemini_api_key:
            try:
                genai.configure(api_key=self.config.ai.gemini_api_key)
                self.model = genai.GenerativeModel(self.config.ai.gemini_model)
                logger.info("AI model initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize AI model: {e}")
                self.model = None
        else:
            logger.warning("AI model not configured - limited functionality")
    
    def add_to_memory(self, message: str, role: str = "user", metadata: Dict = None) -> None:
        """Add interaction to agent memory"""
        memory_entry = {
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "message": message,
            "metadata": metadata or {}
        }
        
        self.memory.append(memory_entry)
        
        # Limit memory size
        if len(self.memory) > self.config.agent.memory_limit:
            self.memory = self.memory[-self.config.agent.memory_limit:]
        
        logger.info(f"Added to memory: {role} - {message[:50]}...")
    
    def get_memory_context(self, limit: int = 10) -> str:
        """Get recent memory context for AI processing"""
        recent_memory = self.memory[-limit:] if self.memory else []
        
        context = "Recent interactions:\n"
        for entry in recent_memory:
            context += f"{entry['role'].title()}: {entry['message']}\n"
        
        return context
    
    def plan_task(self, task: str, data: pd.DataFrame = None) -> List[str]:
        """Create a step-by-step plan for completing a task"""
        if not self.config.agent.enable_planning:
            return ["Planning disabled in configuration"]
        
        if not self.model:
            return ["AI model not available for planning"]
        
        try:
            data_context = ""
            if data is not None:
                data_context = f"""
                Data context:
                - Shape: {data.shape}
                - Columns: {list(data.columns)}
                - Data types: {dict(data.dtypes)}
                """
            
            memory_context = self.get_memory_context(5)
            
            prompt = f"""
            As an AI Performance Analytics Agent, create a detailed step-by-step plan to complete this task.
            
            Task: {task}
            {data_context}
            {memory_context}
            
            Create a numbered list of specific, actionable steps. Each step should be clear and executable.
            Focus on data analysis, insight generation, and actionable recommendations.
            
            Format as a numbered list.
            """
            
            response = self.model.generate_content(prompt)
            plan_steps = [step.strip() for step in response.text.split('\n') if step.strip() and step.strip()[0].isdigit()]
            
            plan_entry = {
                "task": task,
                "plan": plan_steps,
                "timestamp": datetime.now().isoformat(),
                "data_shape": data.shape if data is not None else None
            }
            
            self.plans.append(plan_entry)
            self.add_to_memory(f"Created plan for: {task}", "assistant", {"plan_steps": len(plan_steps)})
            
            return plan_steps
            
        except Exception as e:
            error_msg = f"Planning failed: {str(e)}"
            logger.error(error_msg)
            return [error_msg]
    
    def reason_about_data(self, data: pd.DataFrame, question: str) -> str:
        """Use AI reasoning to answer questions about data"""
        if not self.config.agent.enable_reasoning:
            return "Reasoning disabled in configuration"
        
        if not self.model:
            return "AI model not available for reasoning"
        
        try:
            # Prepare comprehensive data summary
            data_summary = self._prepare_data_summary(data)
            memory_context = self.get_memory_context(5)
            
            prompt = f"""
            As an AI Performance Analytics Agent with advanced reasoning capabilities, analyze this data and answer the question.
            
            {data_summary}
            
            Question: {question}
            {memory_context}
            
            Please provide a detailed reasoning process:
            1. What patterns and trends do you observe in the data?
            2. What relationships exist between different variables?
            3. What statistical insights can you derive?
            4. What conclusions can you draw from the analysis?
            5. What are the business implications?
            6. What actionable recommendations would you make?
            
            Be thorough, logical, and evidence-based in your reasoning.
            """
            
            response = self.model.generate_content(prompt)
            reasoning = response.text
            
            self.add_to_memory(f"Reasoned about: {question}", "assistant", {"reasoning_length": len(reasoning)})
            
            return reasoning
            
        except Exception as e:
            error_msg = f"Reasoning failed: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def analyze_data_with_ai(self, data: pd.DataFrame, question: str = None) -> Dict[str, Any]:
        """Use AI to analyze data and generate comprehensive insights"""
        if not self.model:
            return {"error": "AI model not available", "fallback": True}
        
        try:
            data_summary = self._prepare_data_summary(data)
            memory_context = self.get_memory_context(5)
            
            if question:
                prompt = f"""
                As an AI Performance Analytics Agent, analyze this data and answer the user's specific question.
                
                {data_summary}
                {memory_context}
                
                User Question: {question}
                
                Please provide:
                1. Direct answer to the question with supporting evidence
                2. Key insights and patterns from the data
                3. Statistical analysis and trends
                4. Anomalies or outliers detected
                5. Business implications and recommendations
                6. Action items for improvement
                
                Format your response in a clear, professional manner suitable for business stakeholders.
                """
            else:
                prompt = f"""
                As an AI Performance Analytics Agent, perform a comprehensive analysis of this dataset.
                
                {data_summary}
                {memory_context}
                
                Please provide:
                1. Executive summary of key findings
                2. Performance trends and patterns analysis
                3. Statistical insights and correlations
                4. Anomaly detection and outlier analysis
                5. Risk factors and areas of concern
                6. Actionable recommendations for improvement
                7. Future predictions and forecasting insights
                
                Format your response in a clear, professional manner suitable for business stakeholders.
                """
            
            response = self.model.generate_content(prompt)
            
            analysis_result = {
                "analysis": response.text,
                "timestamp": datetime.now().isoformat(),
                "data_shape": data.shape,
                "question": question,
                "model_used": self.config.ai.gemini_model
            }
            
            self.analysis_history.append(analysis_result)
            self.add_to_memory(f"Analyzed data: {question or 'General analysis'}", "assistant", {"analysis_length": len(response.text)})
            
            return analysis_result
            
        except Exception as e:
            error_msg = f"AI analysis failed: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def generate_insights(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate automated insights from data using statistical analysis"""
        insights = []
        
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            categorical_cols = data.select_dtypes(include=['object']).columns
            
            # Statistical insights
            for col in numeric_cols:
                if col in data.columns:
                    stats = data[col].describe()
                    mean_val = stats['mean']
                    std_val = stats['std']
                    median_val = stats['50%']
                    
                    insights.append({
                        "type": "statistical",
                        "metric": col,
                        "insight": f"Average {col}: {mean_val:.2f} (Median: {median_val:.2f}, Std: {std_val:.2f})",
                        "severity": "info",
                        "value": mean_val
                    })
                    
                    # Detect outliers using IQR method
                    Q1 = stats['25%']
                    Q3 = stats['75%']
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
                    if len(outliers) > 0:
                        insights.append({
                            "type": "anomaly",
                            "metric": col,
                            "insight": f"Found {len(outliers)} outliers in {col} ({len(outliers)/len(data)*100:.1f}% of data)",
                            "severity": "warning",
                            "value": len(outliers)
                        })
            
            # Trend analysis for time-based data
            date_cols = [col for col in data.columns if any(keyword in col.lower() for keyword in ['date', 'time', 'period'])]
            if date_cols and numeric_cols:
                date_col = date_cols[0]
                metric_col = numeric_cols[0]
                
                if date_col in data.columns and metric_col in data.columns:
                    try:
                        data_sorted = data.sort_values(date_col)
                        if len(data_sorted) > 1:
                            first_val = data_sorted[metric_col].iloc[0]
                            last_val = data_sorted[metric_col].iloc[-1]
                            trend = "increasing" if last_val > first_val else "decreasing"
                            change_pct = ((last_val - first_val) / first_val * 100) if first_val != 0 else 0
                            
                            insights.append({
                                "type": "trend",
                                "metric": f"{metric_col} over time",
                                "insight": f"Overall {trend} trend ({change_pct:+.1f}% change)",
                                "severity": "success" if trend == "increasing" else "warning",
                                "value": change_pct
                            })
                    except Exception as e:
                        logger.warning(f"Trend analysis failed: {e}")
            
            # Correlation analysis
            if len(numeric_cols) > 1:
                corr_matrix = data[numeric_cols].corr()
                high_corr_pairs = []
                
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:  # Strong correlation
                            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
                
                for col1, col2, corr_val in high_corr_pairs:
                    insights.append({
                        "type": "correlation",
                        "metric": f"{col1} vs {col2}",
                        "insight": f"Strong correlation: {corr_val:.2f}",
                        "severity": "info",
                        "value": corr_val
                    })
            
            # Categorical insights
            for col in categorical_cols:
                if col in data.columns:
                    value_counts = data[col].value_counts()
                    if len(value_counts) > 0:
                        top_category = value_counts.index[0]
                        top_count = value_counts.iloc[0]
                        percentage = (top_count / len(data)) * 100
                        
                        insights.append({
                            "type": "categorical",
                            "metric": col,
                            "insight": f"Most common: {top_category} ({percentage:.1f}% of data)",
                            "severity": "info",
                            "value": percentage
                        })
            
        except Exception as e:
            insights.append({
                "type": "error",
                "metric": "Analysis",
                "insight": f"Error generating insights: {str(e)}",
                "severity": "error",
                "value": 0
            })
            logger.error(f"Insight generation failed: {e}")
        
        return insights
    
    def _prepare_data_summary(self, data: pd.DataFrame) -> str:
        """Prepare comprehensive data summary for AI analysis"""
        try:
            summary = f"""
            Dataset Overview:
            - Shape: {data.shape[0]} rows, {data.shape[1]} columns
            - Columns: {list(data.columns)}
            - Data types: {dict(data.dtypes)}
            - Memory usage: {data.memory_usage(deep=True).sum() / 1024:.1f} KB
            
            Sample Data (first 3 rows):
            {data.head(3).to_string()}
            
            Statistical Summary:
            {data.describe().to_string()}
            
            Missing Values:
            {data.isnull().sum().to_string()}
            """
            
            # Add correlation matrix for numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                summary += f"""
                
            Correlation Matrix:
            {data[numeric_cols].corr().to_string()}
            """
            
            return summary
            
        except Exception as e:
            logger.error(f"Data summary preparation failed: {e}")
            return f"Data shape: {data.shape}, Columns: {list(data.columns)}"
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status and capabilities"""
        return {
            "memory_entries": len(self.memory),
            "analysis_count": len(self.analysis_history),
            "plans_created": len(self.plans),
            "ai_enabled": self.model is not None,
            "current_data_shape": self.current_data.shape if self.current_data is not None else None,
            "capabilities": {
                "memory": True,
                "planning": self.config.agent.enable_planning,
                "reasoning": self.config.agent.enable_reasoning,
                "tools": self.config.agent.enable_tools,
                "auto_insights": self.config.agent.auto_insights
            },
            "config": {
                "memory_limit": self.config.agent.memory_limit,
                "ai_model": self.config.ai.gemini_model if self.model else None,
                "temperature": self.config.ai.temperature
            }
        }
    
    def clear_memory(self) -> None:
        """Clear agent memory"""
        self.memory = []
        self.analysis_history = []
        self.plans = []
        logger.info("Agent memory cleared")
    
    def export_memory(self, filepath: Path) -> None:
        """Export agent memory to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump({
                    "memory": self.memory,
                    "analysis_history": self.analysis_history,
                    "plans": self.plans,
                    "export_timestamp": datetime.now().isoformat()
                }, f, indent=2)
            logger.info(f"Memory exported to {filepath}")
        except Exception as e:
            logger.error(f"Memory export failed: {e}")
    
    def load_memory(self, filepath: Path) -> bool:
        """Load agent memory from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.memory = data.get("memory", [])
                self.analysis_history = data.get("analysis_history", [])
                self.plans = data.get("plans", [])
            logger.info(f"Memory loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Memory load failed: {e}")
            return False
