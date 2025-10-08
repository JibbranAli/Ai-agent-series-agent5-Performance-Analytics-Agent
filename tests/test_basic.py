"""
Basic tests for AI Performance Analytics Agent
"""

import unittest
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

class TestBasicFunctionality(unittest.TestCase):
    """Basic functionality tests"""
    
    def test_imports(self):
        """Test that all modules can be imported"""
        try:
            from src.agent import PerformanceAnalyticsAgent
            from src.data_processor import DataProcessor
            from src.visualizer import DataVisualizer
            self.assertTrue(True, "All modules imported successfully")
        except ImportError as e:
            self.fail(f"Import error: {e}")
    
    def test_config_loading(self):
        """Test configuration loading"""
        try:
            from config.settings import config, validate_config
            self.assertTrue(hasattr(config, 'ai'))
            self.assertTrue(hasattr(config, 'ui'))
            self.assertTrue(hasattr(config, 'data'))
        except ImportError as e:
            self.fail(f"Config import error: {e}")
    
    def test_sample_data_creation(self):
        """Test sample data creation"""
        try:
            import pandas as pd
            import numpy as np
            
            # Create sample data similar to the dashboard
            np.random.seed(42)
            dates = pd.date_range('2024-01-01', periods=10, freq='D')
            teams = np.random.choice(['Team A', 'Team B'], 10)
            sales = np.random.normal(1000, 200, 10)
            
            data = pd.DataFrame({
                'date': dates,
                'team': teams,
                'sales': sales
            })
            
            self.assertEqual(len(data), 10)
            self.assertEqual(len(data.columns), 3)
            self.assertTrue('date' in data.columns)
            self.assertTrue('team' in data.columns)
            self.assertTrue('sales' in data.columns)
            
        except Exception as e:
            self.fail(f"Sample data creation error: {e}")

if __name__ == '__main__':
    unittest.main()
