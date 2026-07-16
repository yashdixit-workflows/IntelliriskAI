import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Tuple, Optional
from tools.preprocessing import DataPreprocessor
from tools.predictor import (
    RandomForestPredictor, 
    LogisticRegressionPredictor, 
    DecisionTreePredictor, 
    XGBoostPredictor, 
    LightGBMPredictor,
    BasePredictor
)

class PredictionManager:
    """
    Prediction Manager: Checks compatibility, selects models, coordinates training and evaluation.
    """
    def __init__(self, semantic_schema: Dict[str, Any]):
        self.schema = semantic_schema
        self.target_col = semantic_schema.get("target_column")
        self.identifier_col = semantic_schema.get("identifier_column")
        self.models_dir = r"x:\creditguard-ai\models"
        os.makedirs(self.models_dir, exist_ok=True)
        
    def check_compatibility(self, df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """
        Validates if the dataset is compatible for training a prediction model.
        """
        if not self.target_col:
            return False, "Target variable has not been identified. Prediction requires a target column."
            
        if self.target_col not in df.columns:
            return False, f"Target column '{self.target_col}' not found in the dataset."
            
        if len(df) < 50:
            return False, f"Dataset size is too small ({len(df)} rows). Machine learning models require at least 50 samples."
            
        # Check if features exist
        features = [c for c in df.columns if c not in [self.target_col, self.identifier_col]]
        if not features:
            return False, "No valid predictor features found. Dataset must contain columns other than the target and identifier."
            
        return True, None

    def select_predictor(self, model_type: str = "Random Forest") -> BasePredictor:
        """
        Instantiates the requested predictor class.
        """
        model_type_clean = str(model_type).strip().lower()
        
        if "random forest" in model_type_clean:
            return RandomForestPredictor()
        elif "logistic" in model_type_clean:
            return LogisticRegressionPredictor()
        elif "decision tree" in model_type_clean:
            return DecisionTreePredictor()
        elif "xgboost" in model_type_clean:
            try:
                return XGBoostPredictor()
            except ImportError:
                print("XGBoost failed import, falling back to Random Forest.")
                return RandomForestPredictor()
        elif "lightgbm" in model_type_clean:
            try:
                return LightGBMPredictor()
            except ImportError:
                print("LightGBM failed import, falling back to Random Forest.")
                return RandomForestPredictor()
        else:
            return RandomForestPredictor()

    def train_and_save_model(self, df: pd.DataFrame, model_type: str = "Random Forest") -> Dict[str, Any]:
        """
        Preprocesses, trains, evaluates, and saves the predictive model and preprocessor.
        """
        is_compatible, reason = self.check_compatibility(df)
        if not is_compatible:
            raise ValueError(f"Dataset is not compatible for prediction training: {reason}")
            
        # 1. Instantiate Preprocessor and fit it
        preprocessor = DataPreprocessor(self.schema)
        
        # 2. Extract features and target
        X_raw = df.copy()
        y_raw, _ = preprocessor.preprocess_target(X_raw)
        
        # Fit preprocessor on full data (or split first)
        # Following ML Best Practices: Split data BEFORE fitting preprocessing scaled pipelines
        # So we split first, then fit on train, transform train and test.
        # Let's perform train-test split:
        train_df, test_df = train_test_split(df, test_size=0.25, random_state=42, stratify=y_raw)
        
        # Fit preprocessor on train set
        preprocessor.fit(train_df)
        
        X_train, feature_names = preprocessor.transform(train_df)
        X_test, _ = preprocessor.transform(test_df)
        
        y_train = y_raw[train_df.index]
        y_test = y_raw[test_df.index]
        
        # 3. Get model and fit
        predictor = self.select_predictor(model_type)
        predictor.fit(X_train, y_train)
        
        # 4. Evaluate on test set
        evaluation_results = predictor.evaluate(X_test, y_test)
        evaluation_results["feature_importances"] = predictor.get_feature_importances(feature_names)
        
        # 5. Save model and preprocessor to disk
        model_path = os.path.join(self.models_dir, "delinquency_model.pkl")
        preprocessor_path = os.path.join(self.models_dir, "preprocessor.pkl")
        
        with open(model_path, "wb") as f:
            pickle.dump(predictor, f)
            
        with open(preprocessor_path, "wb") as f:
            pickle.dump(preprocessor, f)
            
        return {
            "model_path": model_path,
            "preprocessor_path": preprocessor_path,
            "metrics": evaluation_results,
            "feature_names": feature_names
        }
