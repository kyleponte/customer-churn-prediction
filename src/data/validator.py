from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from pydantic import BaseModel, validator
from datetime import datetime

class DataValidationReport(BaseModel):
    timestamp: str
    total_records: int
    missing_values: Dict[str, int]
    categorical_distributions: Dict[str, Dict[str, float]]
    numerical_statistics: Dict[str, Dict[str, float]]
    validation_errors: List[str]
    is_valid: bool

class DataValidator:
    """Validates data quality and generates reports"""
    
    def __init__(self):
        self.required_columns = [
            'tenure', 'monthly_charges', 'total_charges',
            'contract_type', 'payment_method', 'online_security',
            'tech_support', 'internet_service'
        ]
        
        self.categorical_columns = [
            'contract_type', 'payment_method', 'online_security',
            'tech_support', 'internet_service'
        ]
        
        self.numerical_columns = [
            'tenure', 'monthly_charges', 'total_charges'
        ]
    
    def validate_data(self, df: pd.DataFrame) -> DataValidationReport:
        """Validate data and generate report"""
        errors = []
        
        # Check required columns
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Generate report
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_records": len(df),
            "missing_values": self._check_missing_values(df),
            "categorical_distributions": self._check_categorical_distributions(df),
            "numerical_statistics": self._check_numerical_statistics(df),
            "validation_errors": errors,
            "is_valid": len(errors) == 0
        }
        
        return DataValidationReport(**report)
    
    def _check_missing_values(self, df: pd.DataFrame) -> Dict[str, int]:
        """Check for missing values in each column"""
        return df[self.required_columns].isnull().sum().to_dict()
    
    def _check_categorical_distributions(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate distribution of categorical variables"""
        distributions = {}
        for col in self.categorical_columns:
            if col in df.columns:
                dist = df[col].value_counts(normalize=True).to_dict()
                distributions[col] = dist
        return distributions
    
    def _check_numerical_statistics(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for numerical variables"""
        stats = {}
        for col in self.numerical_columns:
            if col in df.columns:
                stats[col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max())
                }
        return stats 