"""
Data Visualization Module for the Performance Analytics AI Agent
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class DataVisualizer:
    """
    Advanced data visualization class for creating interactive charts and graphs.
    Integrates with the AI Agent to provide intelligent visualization recommendations.
    """
    
    def __init__(self, config):
        """Initialize the visualizer with configuration"""
        self.config = config
        self.color_palette = config.viz.color_palette
        self.chart_height = config.viz.chart_height
        self.enable_animations = config.viz.enable_animations
    
    def create_visualizations(self, data: pd.DataFrame, insights: List[Dict] = None) -> Dict[str, go.Figure]:
        """Create comprehensive visualizations based on data and insights"""
        figures = {}
        
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            categorical_cols = data.select_dtypes(include=['object']).columns
            date_cols = [col for col in data.columns if any(keyword in col.lower() for keyword in ['date', 'time', 'period'])]
            
            # Time series visualization
            if date_cols and numeric_cols:
                fig = self._create_time_series(data, date_cols[0], numeric_cols[0])
                if fig:
                    figures["time_series"] = fig
            
            # Distribution plots
            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                fig = self._create_distribution(data, col)
                if fig:
                    figures[f"distribution_{col}"] = fig
            
            # Correlation heatmap
            if len(numeric_cols) > 1:
                fig = self._create_correlation_heatmap(data, numeric_cols)
                if fig:
                    figures["correlation"] = fig
            
            # Categorical analysis
            if categorical_cols and numeric_cols:
                fig = self._create_categorical_analysis(data, categorical_cols[0], numeric_cols[0])
                if fig:
                    figures["categorical"] = fig
            
            # Box plots for outlier detection
            if numeric_cols:
                fig = self._create_box_plots(data, numeric_cols[:4])
                if fig:
                    figures["box_plots"] = fig
            
            # Scatter plot matrix
            if len(numeric_cols) >= 2:
                fig = self._create_scatter_matrix(data, numeric_cols[:4])
                if fig:
                    figures["scatter_matrix"] = fig
            
            # Performance metrics dashboard
            if insights:
                fig = self._create_insights_dashboard(insights)
                if fig:
                    figures["insights_dashboard"] = fig
            
            logger.info(f"Created {len(figures)} visualizations")
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
        
        return figures
    
    def _create_time_series(self, data: pd.DataFrame, date_col: str, metric_col: str) -> Optional[go.Figure]:
        """Create time series visualization"""
        try:
            # Prepare data
            df_plot = data.copy()
            df_plot[date_col] = pd.to_datetime(df_plot[date_col], errors='coerce')
            df_plot = df_plot.dropna(subset=[date_col, metric_col])
            df_plot = df_plot.sort_values(date_col)
            
            if len(df_plot) == 0:
                return None
            
            # Create line plot
            fig = px.line(
                df_plot, 
                x=date_col, 
                y=metric_col,
                title=f"{metric_col} Over Time",
                markers=True,
                color_discrete_sequence=[self.color_palette[0]]
            )
            
            # Add trend line
            z = np.polyfit(range(len(df_plot)), df_plot[metric_col], 1)
            p = np.poly1d(z)
            fig.add_scatter(
                x=df_plot[date_col],
                y=p(range(len(df_plot))),
                mode='lines',
                name='Trend',
                line=dict(color=self.color_palette[1], dash='dash')
            )
            
            fig.update_layout(
                height=self.chart_height,
                template="plotly_white",
                xaxis_title=date_col,
                yaxis_title=metric_col,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Time series creation failed: {e}")
            return None
    
    def _create_distribution(self, data: pd.DataFrame, column: str) -> Optional[go.Figure]:
        """Create distribution histogram"""
        try:
            fig = px.histogram(
                data, 
                x=column,
                title=f"Distribution of {column}",
                nbins=30,
                color_discrete_sequence=[self.color_palette[2]]
            )
            
            # Add statistics
            mean_val = data[column].mean()
            median_val = data[column].median()
            
            fig.add_vline(
                x=mean_val, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Mean: {mean_val:.2f}"
            )
            fig.add_vline(
                x=median_val, 
                line_dash="dash", 
                line_color="blue",
                annotation_text=f"Median: {median_val:.2f}"
            )
            
            fig.update_layout(
                height=self.chart_height,
                template="plotly_white",
                xaxis_title=column,
                yaxis_title="Frequency"
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Distribution creation failed: {e}")
            return None
    
    def _create_correlation_heatmap(self, data: pd.DataFrame, numeric_cols: List[str]) -> Optional[go.Figure]:
        """Create correlation heatmap"""
        try:
            corr_matrix = data[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                title="Correlation Matrix",
                color_continuous_scale="RdBu",
                aspect="auto",
                text_auto=True
            )
            
            fig.update_layout(
                height=self.chart_height,
                template="plotly_white"
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Correlation heatmap creation failed: {e}")
            return None
    
    def _create_categorical_analysis(self, data: pd.DataFrame, cat_col: str, num_col: str) -> Optional[go.Figure]:
        """Create categorical analysis visualization"""
        try:
            # Group by categorical column and calculate mean
            grouped_data = data.groupby(cat_col)[num_col].agg(['mean', 'count', 'std']).reset_index()
            grouped_data = grouped_data.sort_values('mean', ascending=False)
            
            fig = px.bar(
                grouped_data,
                x=cat_col,
                y='mean',
                title=f"Average {num_col} by {cat_col}",
                color='mean',
                color_continuous_scale="Viridis",
                text='mean'
            )
            
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(
                height=self.chart_height,
                template="plotly_white",
                xaxis_title=cat_col,
                yaxis_title=f"Average {num_col}"
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Categorical analysis creation failed: {e}")
            return None
    
    def _create_box_plots(self, data: pd.DataFrame, numeric_cols: List[str]) -> Optional[go.Figure]:
        """Create box plots for outlier detection"""
        try:
            fig = go.Figure()
            
            for i, col in enumerate(numeric_cols):
                fig.add_trace(go.Box(
                    y=data[col],
                    name=col,
                    boxpoints='outliers',
                    marker_color=self.color_palette[i % len(self.color_palette)]
                ))
            
            fig.update_layout(
                title="Box Plots - Outlier Detection",
                height=self.chart_height,
                template="plotly_white",
                yaxis_title="Value"
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Box plots creation failed: {e}")
            return None
    
    def _create_scatter_matrix(self, data: pd.DataFrame, numeric_cols: List[str]) -> Optional[go.Figure]:
        """Create scatter plot matrix"""
        try:
            fig = px.scatter_matrix(
                data,
                dimensions=numeric_cols,
                title="Scatter Plot Matrix",
                color_discrete_sequence=self.color_palette
            )
            
            fig.update_layout(
                height=self.chart_height + 200,
                template="plotly_white"
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Scatter matrix creation failed: {e}")
            return None
    
    def _create_insights_dashboard(self, insights: List[Dict]) -> Optional[go.Figure]:
        """Create insights dashboard visualization"""
        try:
            # Categorize insights by type
            insight_types = {}
            for insight in insights:
                insight_type = insight.get('type', 'other')
                if insight_type not in insight_types:
                    insight_types[insight_type] = []
                insight_types[insight_type].append(insight)
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Insights by Type', 'Severity Distribution', 'Top Metrics', 'Insight Timeline'],
                specs=[[{"type": "pie"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "scatter"}]]
            )
            
            # Pie chart of insight types
            if insight_types:
                types = list(insight_types.keys())
                counts = [len(insight_types[t]) for t in types]
                fig.add_trace(
                    go.Pie(labels=types, values=counts, name="Types"),
                    row=1, col=1
                )
            
            # Severity distribution
            severities = [insight.get('severity', 'unknown') for insight in insights]
            severity_counts = pd.Series(severities).value_counts()
            fig.add_trace(
                go.Bar(x=severity_counts.index, y=severity_counts.values, name="Severity"),
                row=1, col=2
            )
            
            # Top metrics (by value)
            metric_insights = [insight for insight in insights if 'value' in insight and insight['value'] != 0]
            if metric_insights:
                metric_insights.sort(key=lambda x: abs(x['value']), reverse=True)
                top_metrics = metric_insights[:5]
                
                fig.add_trace(
                    go.Bar(
                        x=[insight['metric'] for insight in top_metrics],
                        y=[insight['value'] for insight in top_metrics],
                        name="Top Metrics"
                    ),
                    row=2, col=1
                )
            
            fig.update_layout(
                height=self.chart_height + 200,
                title="Insights Dashboard",
                template="plotly_white",
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Insights dashboard creation failed: {e}")
            return None
    
    def create_custom_chart(self, chart_type: str, data: pd.DataFrame, x_col: str, y_col: str, **kwargs) -> Optional[go.Figure]:
        """Create custom chart based on type"""
        try:
            if chart_type == "line":
                fig = px.line(data, x=x_col, y=y_col, **kwargs)
            elif chart_type == "bar":
                fig = px.bar(data, x=x_col, y=y_col, **kwargs)
            elif chart_type == "scatter":
                fig = px.scatter(data, x=x_col, y=y_col, **kwargs)
            elif chart_type == "area":
                fig = px.area(data, x=x_col, y=y_col, **kwargs)
            else:
                logger.warning(f"Unknown chart type: {chart_type}")
                return None
            
            fig.update_layout(
                height=self.chart_height,
                template="plotly_white"
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Custom chart creation failed: {e}")
            return None
    
    def get_visualization_recommendations(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get AI-powered visualization recommendations"""
        recommendations = []
        
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            categorical_cols = data.select_dtypes(include=['object']).columns
            date_cols = [col for col in data.columns if any(keyword in col.lower() for keyword in ['date', 'time', 'period'])]
            
            # Time series recommendation
            if date_cols and numeric_cols:
                recommendations.append({
                    "type": "time_series",
                    "title": f"Time Series: {numeric_cols[0]} over {date_cols[0]}",
                    "description": "Shows trends and patterns over time",
                    "priority": "high"
                })
            
            # Distribution recommendation
            if numeric_cols:
                recommendations.append({
                    "type": "distribution",
                    "title": f"Distribution: {numeric_cols[0]}",
                    "description": "Shows data distribution and outliers",
                    "priority": "medium"
                })
            
            # Correlation recommendation
            if len(numeric_cols) > 1:
                recommendations.append({
                    "type": "correlation",
                    "title": "Correlation Matrix",
                    "description": "Shows relationships between numeric variables",
                    "priority": "medium"
                })
            
            # Categorical analysis recommendation
            if categorical_cols and numeric_cols:
                recommendations.append({
                    "type": "categorical",
                    "title": f"Analysis: {numeric_cols[0]} by {categorical_cols[0]}",
                    "description": "Compares performance across categories",
                    "priority": "high"
                })
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
        
        return recommendations
