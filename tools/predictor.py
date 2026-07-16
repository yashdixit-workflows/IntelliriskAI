import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from typing import Dict, Any, List, Optional
import os

class BasePredictor:
    """
    Interface for ML Predictors.
    """
    def __init__(self):
        self.model = None
        self.model_name = "Base"

    def fit(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_feature_importances(self, feature_names: List[str]) -> Dict[str, float]:
        raise NotImplementedError

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model on test dataset.
        """
        preds = self.predict(X)
        probs = self.predict_proba(X)
        
        # Binary Classification Metrics
        acc = accuracy_score(y, preds)
        prec = precision_score(y, preds, zero_division=0)
        rec = recall_score(y, preds, zero_division=0)
        f1 = f1_score(y, preds, zero_division=0)
        
        # ROC AUC
        try:
            auc = roc_auc_score(y, probs)
        except Exception:
            auc = 0.5
            
        cm = confusion_matrix(y, preds).tolist()
        
        return {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(auc),
            "confusion_matrix": cm,
            "model_name": self.model_name
        }


class RandomForestPredictor(BasePredictor):
    def __init__(self, n_estimators=100, max_depth=8, random_state=42):
        super().__init__()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=random_state
        )
        self.model_name = "Random Forest"

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Return probability of positive class
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        return self.predict(X).astype(float)

    def get_feature_importances(self, feature_names: List[str]) -> Dict[str, float]:
        importances = self.model.feature_importances_
        return {name: float(imp) for name, imp in zip(feature_names, importances)}


class LogisticRegressionPredictor(BasePredictor):
    def __init__(self, max_iter=1000, random_state=42):
        super().__init__()
        self.model = LogisticRegression(max_iter=max_iter, random_state=random_state)
        self.model_name = "Logistic Regression"

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        return self.predict(X).astype(float)

    def get_feature_importances(self, feature_names: List[str]) -> Dict[str, float]:
        # Return normalized absolute values of coefficients
        coefs = np.abs(self.model.coef_[0])
        total = np.sum(coefs) if np.sum(coefs) > 0 else 1.0
        normalized_coefs = coefs / total
        return {name: float(imp) for name, imp in zip(feature_names, normalized_coefs)}


class DecisionTreePredictor(BasePredictor):
    def __init__(self, max_depth=6, random_state=42):
        super().__init__()
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        self.model_name = "Decision Tree"

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        return self.predict(X).astype(float)

    def get_feature_importances(self, feature_names: List[str]) -> Dict[str, float]:
        importances = self.model.feature_importances_
        return {name: float(imp) for name, imp in zip(feature_names, importances)}


class XGBoostPredictor(BasePredictor):
    def __init__(self, n_estimators=100, max_depth=5, random_state=42):
        super().__init__()
        try:
            from xgboost import XGBClassifier
            self.model = XGBClassifier(
                n_estimators=n_estimators, 
                max_depth=max_depth, 
                random_state=random_state,
                eval_metric="logloss"
            )
            self.model_name = "XGBoost"
        except ImportError:
            self.model = None
            self.model_name = "XGBoost (Not Available)"

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.model is None:
            raise ImportError("XGBoost is not installed on this system.")
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ImportError("XGBoost is not installed on this system.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ImportError("XGBoost is not installed on this system.")
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importances(self, feature_names: List[str]) -> Dict[str, float]:
        if self.model is None:
            raise ImportError("XGBoost is not installed.")
        importances = self.model.feature_importances_
        return {name: float(imp) for name, imp in zip(feature_names, importances)}


class LightGBMPredictor(BasePredictor):
    def __init__(self, n_estimators=100, max_depth=5, random_state=42):
        super().__init__()
        try:
            from lightgbm import LGBMClassifier
            self.model = LGBMClassifier(
                n_estimators=n_estimators, 
                max_depth=max_depth, 
                random_state=random_state,
                verbosity=-1
            )
            self.model_name = "LightGBM"
        except ImportError:
            self.model = None
            self.model_name = "LightGBM (Not Available)"

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.model is None:
            raise ImportError("LightGBM is not installed on this system.")
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ImportError("LightGBM is not installed on this system.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ImportError("LightGBM is not installed on this system.")
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importances(self, feature_names: List[str]) -> Dict[str, float]:
        if self.model is None:
            raise ImportError("LightGBM is not installed.")
        importances = self.model.feature_importances_
        total = np.sum(importances) if np.sum(importances) > 0 else 1.0
        normalized = importances / total
        return {name: float(imp) for name, imp in zip(feature_names, normalized)}
