from data.ingestion import DataIngestion
from models.trainer import ModelTrainer
import logging
import os
import numpy as np
import kagglehub

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_dataset() -> str:
    """Download the dataset from Kaggle"""
    logger.info("Downloading dataset from Kaggle...")
    try:
        path = kagglehub.dataset_download(
            "muhammadshahidazeem/customer-churn-dataset"
        )
        # Find the CSV file in the downloaded directory
        if os.path.isdir(path):
            for file in os.listdir(path):
                if file.endswith('.csv'):
                    csv_path = os.path.join(path, file)
                    logger.info(f"Found CSV file at: {csv_path}")
                    return csv_path
        else:
            return path
            
        raise FileNotFoundError("No CSV file found in the downloaded dataset")
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        raise

def train_churn_model(
    model_save_path: str = 'models/churn_model.pkl'
) -> None:
    """Train and save the churn prediction model"""
    try:
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        # Download dataset
        data_path = download_dataset()
        
        # Initialize components
        data_ingestion = DataIngestion()
        model_trainer = ModelTrainer()
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        df = data_ingestion.load_data(data_path)
        
        # Log data info
        logger.info(f"Columns in data: {df.columns.tolist()}")
        logger.info(f"Number of samples: {len(df)}")
        logger.info(f"Churn distribution: \n{df['churn'].value_counts()}")
        
        # Log sample data before preprocessing
        logger.info("\nSample data before preprocessing:")
        logger.info(df[['contract_type', 'tech_support', 'internet_service', 'Payment Delay', 'tenure', 'monthly_charges']].head())
        
        df_processed = data_ingestion.preprocess_data(df)
        
        # Log processed data statistics
        logger.info("\nProcessed data statistics:")
        logger.info(df_processed.describe())
        
        # Log correlation with churn
        correlations = df_processed.corr()['churn'].sort_values(ascending=False)
        logger.info("\nFeature correlations with churn:")
        logger.info(correlations)
        
        # Prepare and train
        X, y = model_trainer.prepare_data(df_processed)
        metrics = model_trainer.train(X, y)
        
        # Log feature importance
        logger.info("\nTop 10 Most Important Features:")
        sorted_features = sorted(
            metrics['feature_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for feature, importance in sorted_features[:10]:
            logger.info(f"{feature:30} {importance:.4f}")
        
        # Save model
        model_trainer.save_model(model_save_path)
        logger.info(f"Model saved to {model_save_path}")
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    train_churn_model() 