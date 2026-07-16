import pandas as pd
import numpy as np
from typing import Dict, Any, List
from agents import get_gemini_client

class ExplainabilityAgent:
    """
    Explainability Agent: Translates model outputs into clear, business-focused explanations
    at global (feature importance) and local (individual account risk) levels.
    """
    def __init__(self, semantic_schema: Dict[str, Any]):
        self.schema = semantic_schema
        self.client = get_gemini_client()

    def explain_global_importance(self, importance_dict: Dict[str, float]) -> str:
        """
        Explains feature importances in plain business terms.
        """
        top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Rule-based description
        rule_desc = "The predictive model relies on the following key metrics to evaluate portfolio risks:\n"
        for idx, (f, imp) in enumerate(top_features):
            rule_desc += f"{idx+1}. '{f}' (Impact: {imp*100:.1f}%)\n"
            
        prompt = f"""
Write a 1-paragraph business description summarizing these machine learning model feature importances:
{top_features}

Explain what this means for credit risk monitoring and which metrics managers should focus on.
"""
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            return response.text.strip()
        except Exception:
            return rule_desc

    def explain_local_prediction(self, row: pd.Series, risk_prob: float, risk_cat: str) -> str:
        """
        Explains why a specific customer has their risk category based on key data triggers.
        """
        # Find features that stand out
        highlights = []
        
        # Standard indicators if present
        if "Credit_Score" in row:
            val = float(row["Credit_Score"])
            if val < 580:
                highlights.append(f"Low Credit Score: {val} (High Risk threshold is <580)")
            elif val > 700:
                highlights.append(f"Excellent Credit Score: {val} (Positive factor)")
                
        if "Missed_Payments" in row:
            val = int(row["Missed_Payments"])
            if val > 2:
                highlights.append(f"Frequent Missed Payments: {val} times (Strong delinquency indicator)")
                
        if "Credit_Utilization" in row:
            val = float(row["Credit_Utilization"])
            if val > 0.6:
                highlights.append(f"High Credit Utilization: {val*100:.1f}% (Indicates high credit usage)")
                
        if "Debt_to_Income_Ratio" in row:
            val = float(row["Debt_to_Income_Ratio"])
            if val > 0.45:
                highlights.append(f"Elevated Debt-to-Income: {val*100:.1f}% (Financial strain marker)")

        # Fallback text
        default_explanation = (
            f"Account is classified as {risk_cat} with a probability score of {risk_prob*100:.1f}%. "
            f"Key risk triggers identified in their profile include:\n" + 
            "\n".join([f" - {h}" for h in highlights]) if highlights else " - Profile indicators are within standard limits."
        )

        prompt = f"""
Provide a concise, 2-3 sentence personalized business explanation of why this customer was classified as '{risk_cat}' with a risk score of {risk_prob*100:.1f}%.
Translate their profile features into clear credit risk causes.

Customer Profile:
{row.to_dict()}

Risk Highlights Identified:
{highlights}
"""
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            return response.text.strip()
        except Exception:
            return default_explanation
