import pandas as pd
from typing import Optional, Literal
from pydantic import BaseModel
import os
import logging
from sklearn.preprocessing import StandardScaler
import numpy as np

logger = logging.getLogger(__name__)

class CustomerData(BaseModel):
    """Schema for customer data validation"""
    customer_id: str
    tenure: int
    monthly_charges: float
    total_charges: float
    contract_type: Literal["Basic", "Premium", "Standard"]
    tech_support: Literal["Yes", "No"]
    internet_service: Literal["Fiber optic", "DSL", "No"]
    churn: int
    Age: int
    Gender: Literal["Male", "Female"]
    Payment_Delay: int  # Keep underscore for API requests
    
    class Config:
        # Allow 'Payment Delay' to be mapped to Payment_Delay
        allow_population_by_field_name = True
        fields = {
            'Payment_Delay': {'alias': 'Payment Delay'}
        }

class DataIngestion:
    """Handles data loading and preprocessing operations"""
    
    def __init__(self):
        self.categorical_columns = [
            'contract_type', 'tech_support', 
            'internet_service', 'Gender',
            'Contract Length'  # Added back as categorical
        ]
        
        self.numeric_columns = [
            'tenure', 'monthly_charges', 'total_charges',
            'Age', 'Payment Delay'
        ]
        
        self.scaler = StandardScaler()
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file"""
        try:
            logger.info(f"Attempting to read CSV from: {filepath}")
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")
            
            # Read the CSV file
            df = pd.read_csv(filepath)
            
            # Log the actual columns for debugging
            logger.info(f"Actual columns in CSV: {df.columns.tolist()}")
            
            # Map Kaggle column names to our format
            column_mapping = {
                'CustomerID': 'customer_id',
                'Tenure': 'tenure',
                'Total Spend': 'monthly_charges',
                'Subscription Type': 'contract_type',
                'Support Calls': 'tech_support',
                'Usage Frequency': 'internet_service',
                'Churn': 'churn',
                'Gender': 'Gender',
                'Age': 'Age',
                'Payment Delay': 'Payment Delay',
                'Contract Length': 'Contract Length'
            }
            
            # Check which columns are actually present
            available_columns = {
                orig: new for orig, new in column_mapping.items()
                if orig in df.columns
            }
            
            # Rename columns
            df = df.rename(columns=available_columns)
            
            # Convert categorical values
            contract_type_map = {
                'Basic': 'Basic',
                'Premium': 'Premium',
                'Standard': 'Standard'
            }
            
            # Map contract types
            if 'contract_type' in df.columns:
                df['contract_type'] = df['contract_type'].map(contract_type_map)
            
            # Convert support calls to Yes/No
            if 'tech_support' in df.columns:
                df['tech_support'] = df['tech_support'].apply(lambda x: 'Yes' if x > 2 else 'No')
            
            # Convert usage frequency to service types
            if 'internet_service' in df.columns:
                df['internet_service'] = df['internet_service'].apply(
                    lambda x: 'Fiber optic' if x > 5 else 'DSL' if x > 2 else 'No'
                )
            
            # Convert numeric columns
            if 'monthly_charges' in df.columns:
                df['monthly_charges'] = pd.to_numeric(df['monthly_charges'], errors='coerce')
                df['total_charges'] = df['monthly_charges'] * df['tenure']
            
            if 'churn' in df.columns:
                df['churn'] = df['churn'].astype(int)
            
            # Log transformed data sample
            logger.info("\nSample of transformed data:")
            logger.info(df[['contract_type', 'tech_support', 'internet_service', 'Payment Delay']].head())
            
            return df.dropna()
            
        except Exception as e:
            logger.error(f"Error loading data from {filepath}: {str(e)}")
            raise Exception(f"Error loading data: {str(e)}")
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate risk score for churn prediction"""
        df = df.copy()
        
        # Handle missing values
        df = df.fillna({
            'tenure': 0,
            'monthly_charges': 0,
            'total_charges': 0,
            'Age': df['Age'].median() if 'Age' in df else 0,
            'Payment Delay': 0,
            'contract_type': 'Basic',
            'tech_support': 'No',
            'internet_service': 'No',
            'Gender': 'Male'
        })
        
        # Calculate risk score (0-100)
        risk_score = np.zeros(len(df))
        
        # Tenure risk (0-30 points)
        risk_score += 30 * np.exp(-df['tenure'] / 3)
        
        # Payment delay risk (0-25 points)
        risk_score += np.minimum(25, df['Payment Delay'] * 0.8)
        
        # Contract risk (0-15 points)
        risk_score += df['contract_type'].map({
            'Basic': 15,
            'Standard': 7,
            'Premium': 0
        }).fillna(15)
        
        # Service risk (0-15 points)
        risk_score += df['tech_support'].map({
            'No': 15,
            'Yes': 0
        }).fillna(15)
        
        # Cost risk (0-10 points)
        median_charge = df['monthly_charges'].median()
        risk_score += np.where(
            df['monthly_charges'] > median_charge * 1.5,
            10, 0
        )
        
        # Age risk (0-5 points)
        risk_score += np.where(df['Age'] < 25, 5, 0)
        
        # Create output DataFrame with just risk score
        result = pd.DataFrame({
            'risk_score': risk_score
        })
        
        if 'churn' in df.columns:
            result['churn'] = df['churn']
        
        return result
    
    def validate_customer_data(self, data: dict) -> CustomerData:
        """Validate incoming customer data"""
        return CustomerData(**data)