from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pydantic import BaseModel
import time
import json
import os

class PredictionLog(BaseModel):
    timestamp: str
    customer_id: str
    prediction: float
    response_time: float
    features: Dict

class PerformanceMetrics(BaseModel):
    avg_response_time: float
    p95_response_time: float
    requests_per_minute: float
    error_rate: float
    prediction_distribution: Dict[str, float]

class ModelMonitor:
    """Monitors model performance and prediction patterns"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.prediction_log_path = os.path.join(log_dir, "predictions.json")
        self.error_log_path = os.path.join(log_dir, "errors.json")
        os.makedirs(log_dir, exist_ok=True)
    
    def log_prediction(
        self,
        customer_id: str,
        prediction: float,
        features: Dict,
        response_time: float
    ):
        """Log a single prediction"""
        log_entry = PredictionLog(
            timestamp=datetime.now().isoformat(),
            customer_id=customer_id,
            prediction=prediction,
            response_time=response_time,
            features=features
        )
        
        self._append_to_log(self.prediction_log_path, log_entry.dict())
    
    def log_error(self, error: Exception, context: Dict):
        """Log an error"""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context
        }
        
        self._append_to_log(self.error_log_path, error_entry)
    
    def get_performance_metrics(
        self,
        time_window: timedelta = timedelta(hours=1)
    ) -> PerformanceMetrics:
        """Calculate performance metrics for recent predictions"""
        recent_predictions = self._load_recent_logs(
            self.prediction_log_path,
            time_window
        )
        recent_errors = self._load_recent_logs(
            self.error_log_path,
            time_window
        )
        
        if not recent_predictions:
            return None
        
        df_predictions = pd.DataFrame(recent_predictions)
        
        # Calculate metrics
        total_requests = len(recent_predictions)
        time_span = (
            pd.to_datetime(df_predictions['timestamp'].max()) -
            pd.to_datetime(df_predictions['timestamp'].min())
        ).total_seconds() / 60  # in minutes
        
        metrics = PerformanceMetrics(
            avg_response_time=df_predictions['response_time'].mean(),
            p95_response_time=df_predictions['response_time'].quantile(0.95),
            requests_per_minute=total_requests / time_span if time_span > 0 else 0,
            error_rate=len(recent_errors) / total_requests if total_requests > 0 else 0,
            prediction_distribution=self._calculate_prediction_distribution(df_predictions)
        )
        
        return metrics
    
    def _append_to_log(self, log_path: str, entry: Dict):
        """Append an entry to a log file"""
        try:
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(entry)
            
            with open(log_path, 'w') as f:
                json.dump(logs, f)
        except Exception as e:
            print(f"Error logging to {log_path}: {str(e)}")
    
    def _load_recent_logs(
        self,
        log_path: str,
        time_window: timedelta
    ) -> List[Dict]:
        """Load logs within the specified time window"""
        if not os.path.exists(log_path):
            return []
        
        with open(log_path, 'r') as f:
            logs = json.load(f)
        
        cutoff_time = datetime.now() - time_window
        recent_logs = [
            log for log in logs
            if datetime.fromisoformat(log['timestamp']) > cutoff_time
        ]
        
        return recent_logs
    
    def _calculate_prediction_distribution(
        self,
        df: pd.DataFrame,
        bins: int = 10
    ) -> Dict[str, float]:
        """Calculate distribution of prediction values"""
        hist, bin_edges = np.histogram(
            df['prediction'],
            bins=bins,
            range=(0, 1),
            density=True
        )
        
        return {
            f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}": float(hist[i])
            for i in range(len(hist))
        } 