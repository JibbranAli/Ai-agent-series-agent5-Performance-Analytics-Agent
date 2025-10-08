"""
Data Processing Module for the Performance Analytics AI Agent
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import openpyxl

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Advanced data processing class for handling file uploads, cleaning, and validation.
    Integrates with the AI Agent to provide intelligent data processing capabilities.
    """
    
    def __init__(self, config):
        """Initialize the data processor with configuration"""
        self.config = config
        self.supported_formats = config.data.supported_formats
        self.max_rows = config.data.max_rows
        self.auto_clean = config.data.auto_clean
        self.encoding = config.data.encoding
    
    def process_uploaded_file(self, file_path: Path) -> Dict[str, Any]:
        """Process uploaded file and return processed data with metadata"""
        try:
            # Validate file format
            file_extension = file_path.suffix.lower()
            if file_extension not in self.supported_formats:
                return {
                    "success": False,
                    "error": f"Unsupported file format: {file_extension}. Supported formats: {self.supported_formats}"
                }
            
            # Load data based on file type
            if file_extension == '.csv':
                data = self._load_csv(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                data = self._load_excel(file_path)
            else:
                return {"success": False, "error": "Unknown file format"}
            
            if data is None:
                return {"success": False, "error": "Failed to load data"}
            
            # Validate data size
            if len(data) > self.max_rows:
                return {
                    "success": False,
                    "error": f"Data too large: {len(data)} rows. Maximum allowed: {self.max_rows}"
                }
            
            # Process and clean data
            processed_data = self._process_data(data)
            
            # Generate metadata
            metadata = self._generate_metadata(processed_data, file_path)
            
            logger.info(f"Successfully processed file: {file_path.name}")
            
            return {
                "success": True,
                "data": processed_data,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"File processing failed: {e}")
            return {
                "success": False,
                "error": f"Processing failed: {str(e)}"
            }
    
    def _load_csv(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load CSV file with error handling"""
        try:
            # Try different encodings
            encodings = [self.encoding, 'utf-8', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    data = pd.read_csv(file_path, encoding=encoding)
                    logger.info(f"CSV loaded successfully with encoding: {encoding}")
                    return data
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, try with error handling
            data = pd.read_csv(file_path, encoding='utf-8', errors='replace')
            logger.warning("CSV loaded with encoding errors replaced")
            return data
            
        except Exception as e:
            logger.error(f"CSV loading failed: {e}")
            return None
    
    def _load_excel(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load Excel file with error handling"""
        try:
            # Try to read the first sheet
            data = pd.read_excel(file_path, engine='openpyxl')
            logger.info("Excel file loaded successfully")
            return data
            
        except Exception as e:
            logger.error(f"Excel loading failed: {e}")
            return None
    
    def _process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process and clean the data"""
        processed_data = data.copy()
        
        if not self.auto_clean:
            return processed_data
        
        try:
            # Remove completely empty rows and columns
            processed_data = processed_data.dropna(how='all')
            processed_data = processed_data.dropna(axis=1, how='all')
            
            # Clean column names
            processed_data.columns = processed_data.columns.str.strip()
            processed_data.columns = processed_data.columns.str.replace(' ', '_')
            processed_data.columns = processed_data.columns.str.lower()
            
            # Convert date columns
            processed_data = self._convert_date_columns(processed_data)
            
            # Handle numeric columns
            processed_data = self._convert_numeric_columns(processed_data)
            
            # Remove duplicate rows
            initial_rows = len(processed_data)
            processed_data = processed_data.drop_duplicates()
            removed_duplicates = initial_rows - len(processed_data)
            
            if removed_duplicates > 0:
                logger.info(f"Removed {removed_duplicates} duplicate rows")
            
            logger.info("Data processing completed successfully")
            
        except Exception as e:
            logger.error(f"Data processing failed: {e}")
        
        return processed_data
    
    def _convert_date_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert potential date columns to datetime"""
        date_keywords = ['date', 'time', 'period', 'created', 'updated', 'timestamp']
        
        for col in data.columns:
            if any(keyword in col.lower() for keyword in date_keywords):
                try:
                    data[col] = pd.to_datetime(data[col], errors='coerce')
                    logger.info(f"Converted column '{col}' to datetime")
                except Exception as e:
                    logger.warning(f"Failed to convert column '{col}' to datetime: {e}")
        
        return data
    
    def _convert_numeric_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert potential numeric columns to numeric type"""
        for col in data.columns:
            if data[col].dtype == 'object':
                try:
                    # Try to convert to numeric
                    numeric_data = pd.to_numeric(data[col], errors='coerce')
                    
                    # If more than 80% of values are numeric, convert the column
                    if numeric_data.notna().sum() / len(data) > 0.8:
                        data[col] = numeric_data
                        logger.info(f"Converted column '{col}' to numeric")
                        
                except Exception as e:
                    logger.warning(f"Failed to convert column '{col}' to numeric: {e}")
        
        return data
    
    def _generate_metadata(self, data: pd.DataFrame, file_path: Path) -> Dict[str, Any]:
        """Generate comprehensive metadata about the processed data"""
        try:
            metadata = {
                "file_info": {
                    "filename": file_path.name,
                    "file_size": file_path.stat().st_size,
                    "file_extension": file_path.suffix,
                    "processed_at": datetime.now().isoformat()
                },
                "data_info": {
                    "shape": data.shape,
                    "rows": len(data),
                    "columns": len(data.columns),
                    "memory_usage": data.memory_usage(deep=True).sum(),
                    "dtypes": dict(data.dtypes)
                },
                "column_info": {
                    "numeric_columns": list(data.select_dtypes(include=[np.number]).columns),
                    "categorical_columns": list(data.select_dtypes(include=['object']).columns),
                    "datetime_columns": list(data.select_dtypes(include=['datetime']).columns),
                    "boolean_columns": list(data.select_dtypes(include=['bool']).columns)
                },
                "data_quality": {
                    "missing_values": data.isnull().sum().to_dict(),
                    "missing_percentage": (data.isnull().sum() / len(data) * 100).to_dict(),
                    "duplicate_rows": data.duplicated().sum(),
                    "empty_columns": data.isnull().all().sum()
                },
                "statistical_summary": {}
            }
            
            # Add statistical summary for numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                metadata["statistical_summary"] = data[numeric_cols].describe().to_dict()
            
            # Add categorical summary
            categorical_cols = data.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                metadata["categorical_summary"] = {}
                for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
                    metadata["categorical_summary"][col] = {
                        "unique_values": data[col].nunique(),
                        "top_values": data[col].value_counts().head().to_dict()
                    }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Metadata generation failed: {e}")
            return {"error": str(e)}
    
    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality and provide recommendations"""
        validation_results = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        try:
            # Check for empty dataset
            if len(data) == 0:
                validation_results["is_valid"] = False
                validation_results["errors"].append("Dataset is empty")
                return validation_results
            
            # Check for missing columns
            if len(data.columns) == 0:
                validation_results["is_valid"] = False
                validation_results["errors"].append("Dataset has no columns")
                return validation_results
            
            # Check for high missing values
            missing_percentage = (data.isnull().sum() / len(data) * 100)
            high_missing_cols = missing_percentage[missing_percentage > 50].index.tolist()
            
            if high_missing_cols:
                validation_results["warnings"].append(
                    f"Columns with >50% missing values: {high_missing_cols}"
                )
            
            # Check for duplicate rows
            duplicate_count = data.duplicated().sum()
            if duplicate_count > 0:
                validation_results["warnings"].append(
                    f"Found {duplicate_count} duplicate rows"
                )
            
            # Check for columns with all same values
            constant_cols = []
            for col in data.columns:
                if data[col].nunique() == 1:
                    constant_cols.append(col)
            
            if constant_cols:
                validation_results["warnings"].append(
                    f"Columns with constant values: {constant_cols}"
                )
            
            # Check for potential date columns
            date_keywords = ['date', 'time', 'period']
            potential_date_cols = [
                col for col in data.columns 
                if any(keyword in col.lower() for keyword in date_keywords)
                and data[col].dtype == 'object'
            ]
            
            if potential_date_cols:
                validation_results["recommendations"].append(
                    f"Consider converting to datetime: {potential_date_cols}"
                )
            
            # Check for potential numeric columns
            potential_numeric_cols = []
            for col in data.select_dtypes(include=['object']).columns:
                try:
                    numeric_data = pd.to_numeric(data[col], errors='coerce')
                    if numeric_data.notna().sum() / len(data) > 0.8:
                        potential_numeric_cols.append(col)
                except:
                    pass
            
            if potential_numeric_cols:
                validation_results["recommendations"].append(
                    f"Consider converting to numeric: {potential_numeric_cols}"
                )
            
            # Check data size
            if len(data) > 10000:
                validation_results["recommendations"].append(
                    "Large dataset - consider sampling for faster analysis"
                )
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            validation_results["errors"].append(f"Validation failed: {str(e)}")
        
        return validation_results
    
    def create_sample_data(self, n_rows: int = 100) -> pd.DataFrame:
        """Create sample data for testing and demonstration"""
        try:
            np.random.seed(42)  # For reproducible results
            
            # Generate sample data
            dates = pd.date_range('2024-01-01', periods=n_rows, freq='D')
            teams = np.random.choice(['Team A', 'Team B', 'Team C', 'Team D'], n_rows)
            sales = np.random.normal(1000, 200, n_rows)
            performance_scores = np.random.uniform(60, 100, n_rows)
            customer_satisfaction = np.random.uniform(3, 5, n_rows)
            categories = np.random.choice(['Q1', 'Q2', 'Q3', 'Q4'], n_rows)
            
            sample_data = pd.DataFrame({
                'date': dates,
                'team': teams,
                'sales': sales,
                'performance_score': performance_scores,
                'customer_satisfaction': customer_satisfaction,
                'category': categories
            })
            
            # Add some missing values for realism
            missing_indices = np.random.choice(n_rows, size=int(n_rows * 0.05), replace=False)
            sample_data.loc[missing_indices, 'customer_satisfaction'] = np.nan
            
            logger.info(f"Created sample data with {n_rows} rows")
            return sample_data
            
        except Exception as e:
            logger.error(f"Sample data creation failed: {e}")
            return pd.DataFrame()
    
    def export_processed_data(self, data: pd.DataFrame, file_path: Path, format: str = 'csv') -> bool:
        """Export processed data to file"""
        try:
            if format.lower() == 'csv':
                data.to_csv(file_path, index=False)
            elif format.lower() == 'excel':
                data.to_excel(file_path, index=False)
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Data exported successfully to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Data export failed: {e}")
            return False
