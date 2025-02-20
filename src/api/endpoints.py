from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from typing import List
import logging
import time
import numpy as np

from src.data.ingestion import DataIngestion, CustomerData
from src.models.trainer import ModelTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router (changed from FastAPI app)
router = APIRouter()

# Initialize components
data_ingestion = DataIngestion()
model = None

class RiskFactors(BaseModel):
    """Risk factors that contributed to the prediction"""
    tenure_risk: float = Field(..., description="Risk from customer tenure")
    payment_risk: float = Field(..., description="Risk from payment delays")
    contract_risk: float = Field(..., description="Risk from contract type")
    service_risk: float = Field(..., description="Risk from service configuration")
    cost_risk: float = Field(..., description="Risk from monthly charges")
    age_risk: float = Field(..., description="Risk from customer age")

class PredictionResponse(BaseModel):
    """Response model for churn predictions"""
    customer_id: str = Field(..., description="Unique identifier for the customer")
    churn_probability: float = Field(..., 
        description="Probability of customer churning (0-1)",
        ge=0.0,
        le=1.0
    )
    risk_factors: RiskFactors = Field(..., description="Breakdown of risk factors")
    risk_level: str = Field(..., description="Overall risk level (Low/Medium/High)")
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CUST123",
                "churn_probability": 0.75,
                "risk_factors": {
                    "tenure_risk": 0.8,
                    "payment_risk": 0.6,
                    "contract_risk": 0.7,
                    "service_risk": 0.4,
                    "cost_risk": 0.3,
                    "age_risk": 0.2
                },
                "risk_level": "High"
            }
        }

@router.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model
    try:
        model = ModelTrainer.load_model("models/churn_model.pkl")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

@router.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerData):
    """Predict the probability of customer churn"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        customer_dict = customer.dict()
        customer_id = customer_dict.pop('customer_id')
        
        # Convert Payment_Delay to Payment Delay
        if 'Payment_Delay' in customer_dict:
            customer_dict['Payment Delay'] = customer_dict.pop('Payment_Delay')
        
        df = pd.DataFrame([customer_dict])
        
        # Log the input data for debugging
        logger.info(f"Input data: {df.to_dict()}")
        
        # Preprocess data
        df_processed = data_ingestion.preprocess_data(df)
        logger.info(f"Processed columns: {df_processed.columns.tolist()}")
        
        # Ensure all feature columns from training are present
        missing_cols = set(model.feature_columns) - set(df_processed.columns)
        for col in missing_cols:
            df_processed[col] = 0
            
        # Reorder columns to match training data
        df_processed = df_processed[model.feature_columns]
        logger.info(f"Final columns: {df_processed.columns.tolist()}")
        
        # Make prediction
        prediction = model.predict(df_processed.values)
        churn_prob = float(prediction[0])
        
        # Calculate individual risk factors
        # Tenure risk (0-30%)
        tenure_risk = 30 * np.exp(-float(df['tenure'].iloc[0]) / 3) / 100
        
        # Payment risk (0-25%)
        payment_risk = np.minimum(25, float(df['Payment Delay'].iloc[0]) * 0.8) / 100
        
        # Contract risk (0-15%)
        contract_risk = {
            'Basic': 0.15,
            'Standard': 0.07,
            'Premium': 0
        }[customer.contract_type]
        
        # Service risk (0-15%)
        service_risk = 0
        if customer.tech_support == 'No':
            service_risk += 0.10
        if customer.internet_service == 'Fiber optic':
            service_risk += 0.05  # Higher risk for premium service
        elif customer.internet_service == 'No':
            service_risk -= 0.05  # Lower risk for no internet
        
        # Cost risk (0-10%)
        median_charge = 100
        cost_multiplier = float(df['monthly_charges'].iloc[0]) / median_charge
        cost_risk = min(0.10, max(0, (cost_multiplier - 1) * 0.05))
        
        # Age risk (0-5%)
        age_risk = max(0, (30 - float(df['Age'].iloc[0])) * 0.002)  # Higher risk for younger customers
        
        # Gender-based risk (based on historical data patterns)
        gender_factor = 1.1 if customer.Gender == 'Male' else 0.9
        
        # Calculate total risk score
        total_risk = (
            tenure_risk +
            payment_risk +
            contract_risk +
            service_risk +
            cost_risk +
            age_risk
        ) * gender_factor
        
        # Normalize to 0-1 range
        churn_prob = min(1.0, max(0.0, total_risk))
        
        # Get risk level
        risk_level = "High" if churn_prob > 0.6 else "Medium" if churn_prob > 0.2 else "Low"
        
        return {
            "customer_id": customer_id,
            "churn_probability": churn_prob,
            "risk_factors": {
                "tenure_risk": float(tenure_risk),
                "payment_risk": float(payment_risk),
                "contract_risk": float(contract_risk),
                "service_risk": float(service_risk),
                "cost_risk": float(cost_risk),
                "age_risk": float(age_risk)
            },
            "risk_level": risk_level
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.exception("Full traceback:")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    } 