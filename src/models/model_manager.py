from datetime import datetime
from typing import Dict, Optional
import joblib
import json
import os

class ModelManager:
    """Manages model versions and metrics"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.current_version = None
        self.model_info = None
        os.makedirs(models_dir, exist_ok=True)
    
    def save_model(self, model_data: Dict, metrics: Dict) -> str:
        """Save model with version info and metrics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"v_{timestamp}"
        
        # Save model info
        model_info = {
            "version": version,
            "timestamp": timestamp,
            "metrics": metrics,
            "feature_columns": model_data["feature_columns"]
        }
        
        # Save model and info
        model_path = os.path.join(self.models_dir, f"model_{version}.pkl")
        info_path = os.path.join(self.models_dir, f"info_{version}.json")
        
        joblib.dump(model_data, model_path)
        with open(info_path, 'w') as f:
            json.dump(model_info, f)
        
        self.current_version = version
        self.model_info = model_info
        
        return version
    
    def load_model(self, version: Optional[str] = None) -> Dict:
        """Load a specific model version or the latest one"""
        if version is None:
            # Get latest version
            versions = [f for f in os.listdir(self.models_dir) 
                       if f.startswith("model_v_")]
            if not versions:
                raise ValueError("No models found")
            version = versions[-1].replace("model_", "").replace(".pkl", "")
        
        model_path = os.path.join(self.models_dir, f"model_{version}.pkl")
        info_path = os.path.join(self.models_dir, f"info_{version}.json")
        
        if not (os.path.exists(model_path) and os.path.exists(info_path)):
            raise ValueError(f"Model version {version} not found")
        
        with open(info_path, 'r') as f:
            self.model_info = json.load(f)
        
        self.current_version = version
        return joblib.load(model_path)
    
    def get_model_info(self) -> Dict:
        """Get current model information"""
        if self.model_info is None:
            raise ValueError("No model currently loaded")
        return self.model_info 