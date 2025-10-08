"""
AI Performance Analytics Agent - Main Entry Point
Professional AI Agent for Performance Analytics with Streamlit Dashboard
"""

import streamlit as st
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import the professional dashboard
from dashboards.professional_dashboard import main as dashboard_main

def main():
    """Main entry point for the AI Performance Analytics Agent"""
    dashboard_main()

if __name__ == "__main__":
    main()