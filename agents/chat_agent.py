import pandas as pd
import json
import re
from typing import Dict, Any, List
from agents import get_gemini_client

class ChatAgent:
    """
    Chat Agent: Enables conversational Q&A against dataset schema and predictions.
    """
    def __init__(self, semantic_schema: Dict[str, Any]):
        self.schema = semantic_schema
        self.client = get_gemini_client()
        self.id_col = semantic_schema.get("identifier_column", "Customer_ID")

    def answer_query(self, user_query: str, state: Dict[str, Any]) -> str:
        """
        Parses the query and returns a natural language response.
        """
        query_clean = user_query.strip().lower()
        
        # Load dataset state
        df = state.get("cleaned_df")
        pred_df = state.get("predictions_df")
        kpis = state.get("kpis", {})
        
        # Compile contextual dataset details for Gemini
        context = {
            "schema": self.schema,
            "kpis": kpis,
            "row_count": len(df) if df is not None else 0
        }
        
        # 1. Direct Pattern Matching (Fast / Fallback)
        # Check if user is asking about a specific customer ID
        cust_match = re.search(r'cust\d+', query_clean)
        if cust_match:
            cust_id = cust_match.group(0).upper()
            if pred_df is not None and self.id_col in pred_df.columns:
                cust_row = pred_df[pred_df[self.id_col].astype(str).str.upper() == cust_id]
                if not cust_row.empty:
                    row_data = cust_row.iloc[0].to_dict()
                    prob = row_data.get("risk_probability", 0.0)
                    cat = row_data.get("risk_category", "Unknown")
                    score = row_data.get("Credit_Score", "N/A")
                    rec = row_data.get("personal_recommendation", "Immediate contact")
                    
                    return (
                        f"### Profile for Customer **{cust_id}**\n"
                        f"- **Risk Classification:** `{cat}` (Probability: **{prob*100:.1f}%**)\n"
                        f"- **Credit Score:** `{score}`\n"
                        f"- **Income:** `${row_data.get('Income', 0):,.2f}`\n"
                        f"- **Credit Utilization:** `{row_data.get('Credit_Utilization', 0)*100:.1f}%`\n"
                        f"- **Missed Payments:** `{row_data.get('Missed_Payments', 0)}` times\n\n"
                        f"**Action Recommended:**\n"
                        f"> {rec}"
                    )
                else:
                    return f"Customer ID **{cust_id}** was not found in the loaded dataset registry."

        # Check if user asks for "high risk" accounts
        if "high risk" in query_clean:
            if pred_df is not None:
                high_risk = pred_df[pred_df["risk_category"] == "High Risk"].sort_values(by="risk_probability", ascending=False)
                if not high_risk.empty:
                    top_ids = high_risk[self.id_col].head(8).tolist()
                    response = (
                        f"There are **{len(high_risk)}** accounts classified as **High Risk** (representing "
                        f"{(len(high_risk)/len(pred_df))*100:.1f}% of the portfolio).\n\n"
                        f"Here are the top high-risk accounts:\n"
                    )
                    for idx, row in high_risk.head(8).iterrows():
                        response += f"- **{row[self.id_col]}** (Probability: {row['risk_probability']*100:.1f}%)\n"
                    return response
                else:
                    return "There are no accounts flagged as High Risk in the portfolio."

        # Check if user asks for general summary
        if "summary" in query_clean or "overview" in query_clean or "kpi" in query_clean:
            summary_lines = ["### Portfolio Metrics Summary:"]
            for k, v in kpis.items():
                summary_lines.append(f"- **{k.replace('_', ' ').title()}:** {v}")
            return "\n".join(summary_lines)

        # 2. Gemini fallback
        # Enrich prompt with a sample of predictions
        sample_rows = []
        if pred_df is not None:
            sample_rows = pred_df.head(10).to_dict(orient="records")

        prompt = f"""
You are an expert Credit risk Analyst and Platform Guide for InsightPilot AI.
Answer the user's natural language question about the dataset and model.
Keep your response concise, professional, and directly useful for decision makers.

Platform State Context:
{json.dumps(context, indent=2)}

Prediction Data Samples (Top 10):
{json.dumps(sample_rows, indent=2)}

User Question:
"{user_query}"
"""
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            return response.text.strip()
        except Exception:
            return (
                "I'm currently running in Rule-Based Mode because the Gemini API is offline.\n"
                "You can query: \n"
                "1. Specific customer records (e.g., *'Show details for CUST0001'*)\n"
                "2. High risk lists (e.g., *'Who are the highest risk customers?'*)\n"
                "3. Portfolio indicators (e.g., *'Give me an overview summary'*)"
            )
