# Customer Churn Predictor

A web application that predicts customer churn probability based on various risk factors.

## Setup

[Access Website](https://customer-churn-prediction-app-wn4q.onrender.com/)

## Features
- Customer churn prediction
- Risk factor analysis
- Interactive web interface

## API Endpoints

- `/predict` - Make single prediction
- `/predict/batch` - Make multiple predictions
- `/model/info` - Get current model info
- `/model/retrain` - Retrain model with new data
- `/monitoring/performance` - Get performance metrics
- `/health` - Check service health

View the complete API documentation at: http://localhost:8000/docs

## Project Structure

```
customer_churn_predictor/
├── src/
│   ├── data/          # Data handling and preprocessing
│   ├── models/        # ML model training and management
│   ├── api/           # API endpoints
│   ├── monitoring/    # Performance monitoring
│   └── main.py        # Application entry point
├── data/              # Data files
├── models/            # Saved models
├── logs/              # Application logs
├── tests/             # Test files
└── requirements.txt   # Project dependencies
```

## Features

- Data validation and preprocessing
- Model training and evaluation
- Real-time predictions via API
- Model version control
- Performance monitoring
- Batch prediction support
