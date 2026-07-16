import pandas as pd
from typing import Dict, Any, List
from agents import get_gemini_client

class RecommendationAgent:
    """
    Recommendation Agent: Translates risk predictions and explanations into actionable business recommendations.
    """
    def __init__(self, semantic_schema: Dict[str, Any]):
        self.schema = semantic_schema
        self.client = get_gemini_client()

    def generate_recommendations(self, row: pd.Series, risk_cat: str, risk_prob: float) -> str:
        """
        Generates business recommendations based on risk classification.
        """
        # Rule-based defaults
        if risk_cat == "High Risk":
            default_rec = (
                "1. Immediate Outreach: Initiate phone/email contact to discuss account status.\n"
                "2. Restructuring Offer: Propose a temporary payment extension or interest restructuring to avoid default.\n"
                "3. Monitoring Increase: Lock credit utilization limits and flag the account for daily automated checks."
            )
        elif risk_cat == "Medium-High Risk (Potential High)":
            default_rec = (
                "1. Pre-emptive Outreach: Send customized warnings and contact within 24 hours of any missed payment.\n"
                "2. Limit Freeze: Freeze credit card limits and restrict cash advances.\n"
                "3. Auto-Pay Drive: Incentivize migration to automated ACH or recurring debit transactions."
            )
        elif risk_cat == "Medium Risk":
            default_rec = (
                "1. Reminder Alert: Send email/SMS reminder alerts 3 days prior to the next payment cycle.\n"
                "2. Personalized Plan: Present standard monthly payment plans or automated draft setups.\n"
                "3. Balance Cap: Cap credit card limit increases temporarily."
            )
        else:
            default_rec = (
                "1. Regular Tracking: Continue standard monthly reporting.\n"
                "2. Marketing Opportunities: High score, low risk. Suitable for limit increases or cross-selling promotions."
            )

        prompt = f"""
Given the customer profile below, who is classified as '{risk_cat}' with a risk score of {risk_prob*100:.1f}%,
write 3 specific, highly actionable, personalized business recommendation bullet points for a credit risk analyst.

Customer Profile:
{row.to_dict()}
"""
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            return response.text.strip()
        except Exception:
            return default_rec
        
    def generate_bulk_recommendations(self, predictions_df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Generates macro business action recommendations based on the whole portfolio prediction statistics.
        """
        high_risk_count = (predictions_df["risk_category"] == "High Risk").sum()
        med_high_count = (predictions_df["risk_category"] == "Medium-High Risk (Potential High)").sum()
        med_risk_count = (predictions_df["risk_category"] == "Medium Risk").sum()
        low_risk_count = (predictions_df["risk_category"] == "Low Risk").sum()
        
        default_bulk_recs = [
            f"Set up an active outreach task force focusing on the {high_risk_count} High Risk accounts representing immediate exposure.",
            f"Closely monitor and preemptively restrict limits on the {med_high_count} Medium-High Risk (Potential High) accounts to prevent defaults.",
            f"Launch automatic payment reminders for the {med_risk_count} Medium Risk accounts to mitigate transition to High Risk.",
            "Promote premium credit line limits for the top 10% lowest risk accounts to increase portfolio yield safely."
        ]
        
        return {
            "portfolio_actions": default_bulk_recs
        }
