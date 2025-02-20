# Product Requirement: Monolithic Customer Churn Prediction Application

## Overview
A **monolithic Customer Churn Prediction application** is one in which all the functionalities—data ingestion, preprocessing, model training, and prediction—are unified in a single codebase. This means everything from how you gather and clean the data, to how the model is trained and finally served via an API, exists under one project structure. Developers can easily trace data flow and logic end-to-end in one place, which helps reduce complexity when you’re starting out or working with a small team.

## Technology Stack
In this monolithic architecture, **Python** is typically the primary language because of its rich data science ecosystem. Libraries such as:
- **pandas** and **NumPy** for data manipulation
- **scikit-learn** or **XGBoost** for the machine learning pipeline
- A lightweight Python web framework like **FastAPI** or **Flask** for exposing the model via an API

FastAPI, in particular, offers:
- Modern Python features
- Built-in data validation using **Pydantic**
- Automatic documentation generation

All the core components—data handling, model logic, and web endpoints—are packaged together.

## Development Workflow
1. **Data Analysis & Feature Engineering**  
   Data scientists perform exploratory data analysis (EDA) and feature engineering (e.g., handling missing values, encoding categorical variables) in a Jupyter Notebook.  
   
2. **Model Training**  
   - Algorithms like **Logistic Regression**, **Random Forest**, or **Gradient-Boosted Trees** are tested.  
   - Evaluation metrics such as **accuracy**, **F1 score**, **precision**, and **recall** guide model selection.  
   - The final model is serialized (e.g., using **joblib** to generate a `.pkl` file).  

3. **Integration & Deployment**  
   - The application code loads the serialized model at startup.  
   - A single command (e.g., `uvicorn src.main:app --reload`) can spin up the entire app locally.  
   - Every component—from data cleansing scripts to the model scoring endpoint—lives under one codebase.

## API Endpoints
Once deployed, the API can receive new customer data through a simple HTTP request. The server:
1. Loads the model
2. Processes incoming data
3. Returns a probability of churn or a binary prediction (churn vs. no churn)

## Maintenance & Retraining
In a monolithic setup, retraining and updating the model occur in the **same environment**. After improvements:
- The old model file is overwritten by the new one.  
- Any code updates require redeploying the entire application.

## Pros & Cons

### Advantages
- **Simplicity**: Easy for small teams to develop, test, and deploy.  
- **Single Deployment**: Fewer moving parts to manage or orchestrate.  
- **Unified Codebase**: Straightforward to trace and debug end-to-end data flow.  

### Limitations
- **Scalability**: As application complexity grows, redeploying the entire system for small changes can be inefficient.  
- **Coupling**: All services are tightly coupled; changes in one area may affect others.  

## When to Use Monolithic Architecture
A monolithic design is best for:
- **Smaller Teams**: Quick, easy, and all-in-one approach to build, test, and deploy.  
- **Manageable Scale**: Complexity remains low enough that a single codebase is not a bottleneck.  
- **Rapid Iteration**: Fast development cycles without overhead from multiple microservices.

---
**End of Document**
