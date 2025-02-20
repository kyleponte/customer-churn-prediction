# Customer Churn Predictor

A web application that predicts customer churn probability based on various risk factors.

## Setup

1. Clone the repository
```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```

2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the application
```bash
uvicorn src.main:app --reload
```

5. Visit http://localhost:8000 in your browser

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