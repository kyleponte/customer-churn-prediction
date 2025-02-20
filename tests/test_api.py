import pytest
from fastapi.testclient import TestClient
import pandas as pd
from unittest.mock import patch, MagicMock

from src.api.endpoints import app
from src.models.trainer import ModelTrainer

client = TestClient(app)

# Mock customer data for testing
VALID_CUSTOMER_DATA = {
    "customer_id": "CUST123",
    "tenure": 24,
    "monthly_charges": 65.0,
    "total_charges": 1560.0,
    "contract_type": "Month-to-month",
    "payment_method": "Electronic check",
    "online_security": "No",
    "tech_support": "No",
    "internet_service": "Fiber optic"
}

@pytest.fixture
def mock_model():
    """Fixture to create a mock model"""
    with patch('src.api.endpoints.model') as mock:
        mock.feature_columns = [
            'tenure', 'monthly_charges', 'total_charges',
            'contract_type_Month-to-month', 'payment_method_Electronic check',
            'online_security_No', 'tech_support_No', 'internet_service_Fiber optic'
        ]
        mock.predict.return_value = [0.75]  # Mock prediction value
        yield mock

def test_health_check_no_model():
    """Test health check endpoint when model is not loaded"""
    with patch('src.api.endpoints.model', None):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "healthy",
            "model_loaded": False
        }

def test_health_check_with_model(mock_model):
    """Test health check endpoint when model is loaded"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model_loaded": True
    }

def test_predict_no_model():
    """Test prediction endpoint when model is not loaded"""
    with patch('src.api.endpoints.model', None):
        response = client.post("/predict", json=VALID_CUSTOMER_DATA)
        assert response.status_code == 500
        assert response.json()["detail"] == "Model not loaded"

def test_predict_valid_data(mock_model):
    """Test prediction endpoint with valid customer data"""
    response = client.post("/predict", json=VALID_CUSTOMER_DATA)
    assert response.status_code == 200
    result = response.json()
    assert result["customer_id"] == VALID_CUSTOMER_DATA["customer_id"]
    assert isinstance(result["churn_probability"], float)
    assert 0 <= result["churn_probability"] <= 1

def test_predict_invalid_data():
    """Test prediction endpoint with invalid customer data"""
    invalid_data = {
        "customer_id": "CUST123",
        "tenure": "invalid",  # Should be int
        "monthly_charges": 65.0
    }
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422  # Validation error

def test_predict_missing_fields():
    """Test prediction endpoint with missing required fields"""
    incomplete_data = {
        "customer_id": "CUST123",
        "tenure": 24
    }
    response = client.post("/predict", json=incomplete_data)
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_startup_event_success():
    """Test successful model loading during startup"""
    with patch('src.models.trainer.ModelTrainer.load_model') as mock_load:
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        
        from src.api.endpoints import startup_event
        await startup_event()
        
        mock_load.assert_called_once_with("models/churn_model.pkl")

@pytest.mark.asyncio
async def test_startup_event_failure():
    """Test failed model loading during startup"""
    with patch('src.models.trainer.ModelTrainer.load_model') as mock_load:
        mock_load.side_effect = Exception("Failed to load model")
        
        from src.api.endpoints import startup_event
        with pytest.raises(Exception):
            await startup_event()