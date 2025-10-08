"""
Configuration settings for the Performance Analytics AI Agent
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
try:
    result = load_dotenv()
    print(f"Debug - .env file loaded: {result}")
    print(f"Debug - GEMINI_API_KEY from env: {os.getenv('GEMINI_API_KEY', 'Not found')}")
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")
    print("Using system environment variables or defaults")

@dataclass
class AIConfig:
    """AI Model Configuration"""
    gemini_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", "AIzaSyDQMTPH6kMlgJa2WnzkwiYib5qzLLC-CVs"))
    gemini_model: str = "gemini-2.0-flash"
    max_tokens: int = 2048
    temperature: float = 0.7
    enable_ai: bool = True

@dataclass
class UIConfig:
    """User Interface Configuration"""
    page_title: str = "Performance Analytics AI Agent"
    page_icon: str = "🤖"
    layout: str = "wide"
    theme: str = "dark"
    sidebar_state: str = "expanded"
    max_file_size: int = 200  # MB

@dataclass
class DataConfig:
    """Data Processing Configuration"""
    supported_formats: list = field(default_factory=lambda: ['.csv', '.xlsx', '.xls'])
    max_rows: int = 100000
    auto_clean: bool = True
    encoding: str = "utf-8"
    sample_size: int = 1000

@dataclass
class AgentConfig:
    """AI Agent Configuration"""
    memory_limit: int = 100  # Number of interactions to remember
    enable_planning: bool = True
    enable_reasoning: bool = True
    enable_tools: bool = True
    auto_insights: bool = True
    confidence_threshold: float = 0.7

@dataclass
class VisualizationConfig:
    """Visualization Configuration"""
    default_chart_type: str = "line"
    color_palette: list = field(default_factory=lambda: [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ])
    chart_height: int = 400
    enable_animations: bool = True

@dataclass
class AppConfig:
    """Main Application Configuration"""
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    log_level: str = "INFO"
    data_dir: Path = field(default_factory=lambda: Path("data"))
    reports_dir: Path = field(default_factory=lambda: Path("reports"))
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    
    # Sub-configurations
    ai: AIConfig = field(default_factory=AIConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    data: DataConfig = field(default_factory=DataConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    viz: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    def __post_init__(self):
        """Initialize directories after object creation"""
        self.data_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "debug": self.debug,
            "log_level": self.log_level,
            "data_dir": str(self.data_dir),
            "reports_dir": str(self.reports_dir),
            "cache_dir": str(self.cache_dir),
            "ai": {
                "gemini_model": self.ai.gemini_model,
                "max_tokens": self.ai.max_tokens,
                "temperature": self.ai.temperature,
                "enable_ai": self.ai.enable_ai
            },
            "ui": {
                "page_title": self.ui.page_title,
                "page_icon": self.ui.page_icon,
                "layout": self.ui.layout,
                "theme": self.ui.theme,
                "max_file_size": self.ui.max_file_size
            },
            "data": {
                "supported_formats": self.data.supported_formats,
                "max_rows": self.data.max_rows,
                "auto_clean": self.data.auto_clean,
                "encoding": self.data.encoding
            },
            "agent": {
                "memory_limit": self.agent.memory_limit,
                "enable_planning": self.agent.enable_planning,
                "enable_reasoning": self.agent.enable_reasoning,
                "enable_tools": self.agent.enable_tools,
                "auto_insights": self.agent.auto_insights
            },
            "visualization": {
                "default_chart_type": self.viz.default_chart_type,
                "color_palette": self.viz.color_palette,
                "chart_height": self.viz.chart_height,
                "enable_animations": self.viz.enable_animations
            }
        }

# Global configuration instance
config = AppConfig()

# Configuration validation
def validate_config() -> bool:
    """Validate configuration settings"""
    errors = []
    
    # Debug: Print current configuration
    print(f"Debug - AI enabled: {config.ai.enable_ai}")
    print(f"Debug - API key present: {bool(config.ai.gemini_api_key)}")
    if config.ai.gemini_api_key:
        print(f"Debug - API key value: {config.ai.gemini_api_key[:10]}...")
    
    # Check data directory
    if not config.data_dir.exists():
        try:
            config.data_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created data directory: {config.data_dir}")
        except Exception as e:
            errors.append(f"Cannot create data directory: {e}")
    
    # Check file size limit
    if config.ui.max_file_size <= 0:
        errors.append("Max file size must be positive")
    
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("Configuration validation passed!")
    return True

# Export configuration
__all__ = ["config", "validate_config", "AppConfig"]
