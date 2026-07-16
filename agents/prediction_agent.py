import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, Any, Tuple

class PredictionAgent:
    """
    Prediction Agent: Loads trained models, preprocessor, and runs predictions.
    Assigns confidence and risk categories.
    """
    def __init__(self):
        self.model_path = r"x:\creditguard-ai\models\delinquency_model.pkl"
        self.preprocessor_path = r"x:\creditguard-ai\models\preprocessor.pkl"
        self.model = None
        self.preprocessor = None
        self.is_loaded = False

    def load_model(self) -> bool:
        """
        Loads the trained predictor and preprocessor files.
        """
        if not os.path.exists(self.model_path) or not os.path.exists(self.preprocessor_path):
            self.is_loaded = False
            return False
            
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            with open(self.preprocessor_path, "rb") as f:
                self.preprocessor = pickle.load(f)
            self.is_loaded = True
            return True
        except Exception as e:
            print("Error loading model in PredictionAgent:", e)
            self.is_loaded = False
            return False

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates predictions, probabilities, confidence scores, and risk categories.
        """
        if not self.is_loaded and not self.load_model():
            raise ValueError("Model or Preprocessor file not found! Train the model first.")
            
        # Transform data
        X_proc, _ = self.preprocessor.transform(df)
        
        # Predict class and probabilities
        preds = self.model.predict(X_proc)
        probs = self.model.predict_proba(X_proc)  # 1-D numpy array, positional index 0..N-1
        
        # Reset index so positional enumeration aligns with probs array
        result_df = df.copy().reset_index(drop=True)
        result_df["predicted_label"] = preds
        
        # ── Apply credit-risk rule-based adjustments to raw model probability ──
        adjusted_probs = []
        for pos, (_, row) in enumerate(result_df.iterrows()):
            # Use positional index into numpy array (CRITICAL: not DataFrame index)
            prob = float(probs[pos])
            
            # ── 1. Credit Score ──────────────────────────────────────────────
            score_val = row.get("Credit_Score")
            try:
                score_val = float(score_val) if score_val is not None else 800.0
            except (ValueError, TypeError):
                score_val = 800.0
            has_low_score = score_val < 500

            # ── 2. Credit Utilization ('Credit_Utilization' or 'Credit_Util') ──
            util_val = row.get("Credit_Utilization")
            if util_val is None:
                util_val = row.get("Credit_Util", 0.0)
            try:
                util_val = float(util_val)
            except (ValueError, TypeError):
                util_val = 0.0
            has_high_util = util_val >= 0.70

            # ── 3. Missed Payments ────────────────────────────────────────────
            missed_val = row.get("Missed_Payments", 0)
            try:
                missed_val = int(float(missed_val))
            except (ValueError, TypeError):
                missed_val = 0
            has_missed_pay = missed_val >= 3

            # ── 4. Active Delinquency ('Delinquent_Account' or 'Delinquent') ──
            del_val = row.get("Delinquent_Account")
            if del_val is None:
                del_val = row.get("Delinquent", 0)
            is_delinquent = False
            try:
                if float(del_val) == 1.0 or str(del_val).strip() in ['1', '1.0', 'True', 'true', 'Yes', 'yes']:
                    is_delinquent = True
            except (ValueError, TypeError):
                pass

            # ── Apply override rules in priority order ────────────────────────
            if is_delinquent:
                # Active delinquency → always High Risk (>= 75%)
                prob = max(prob, 0.75)
            elif has_low_score and (has_high_util or has_missed_pay):
                # Low credit score + (high utilization OR missed payments) → Medium-High (>= 55%)
                prob = max(prob, 0.55)
            elif has_high_util and has_missed_pay:
                # High utilization + missed payments → Medium-High (>= 52%)
                prob = max(prob, 0.52)
            elif has_missed_pay:
                # Missed payments >= 3 alone → at least Medium Risk (>= 30%)
                prob = max(prob, 0.30)

            adjusted_probs.append(prob)
            
        result_df["risk_probability"] = adjusted_probs
        
        # Confidence: distance from 0.5 (how certain the model is)
        result_df["confidence_score"] = 2 * np.abs(np.array(adjusted_probs) - 0.5)
        
        # 4-tier risk category assignment
        # Low: prob < 0.3 | Medium: 0.3–0.5 | Medium-High: 0.5–0.7 | High: >= 0.7
        def get_risk_cat(prob):
            if prob >= 0.7:
                return "High Risk"
            elif prob >= 0.5:
                return "Medium-High Risk (Potential High)"
            elif prob >= 0.3:
                return "Medium Risk"
            else:
                return "Low Risk"
                
        result_df["risk_category"] = result_df["risk_probability"].apply(get_risk_cat)
        
        return result_df
