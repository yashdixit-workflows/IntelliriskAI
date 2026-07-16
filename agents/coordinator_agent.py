import pandas as pd
from typing import Dict, Any, Optional, Tuple
import os
from tools.data_loader import DataLoader
from agents.schema_agent import SchemaUnderstandingAgent
from agents.data_agent import DataQualityAgent
from agents.eda_agent import EDAAgent
from agents.prediction_manager import PredictionManager
from agents.prediction_agent import PredictionAgent
from agents.explainability_agent import ExplainabilityAgent
from agents.recommendation_agent import RecommendationAgent
from agents.report_agent import ReportAgent
from agents.chat_agent import ChatAgent

class CoordinatorAgent:
    """
    Coordinator Agent: Root agent orchestrating the multi-agent execution pipeline.
    """
    def __init__(self):
        self.state = {
            "file_path": None,
            "raw_df": None,
            "cleaned_df": None,
            "schema": None,
            "data_quality_report": None,
            "eda_report": None,
            "model_metrics": None,
            "predictions_df": None,
            "bulk_recs": None,
            "report_paths": None,
            "active_model_name": None,
            "narrative_insights": "",
            "kpis": {}
        }
        self.schema_agent = None
        self.data_agent = None
        self.eda_agent = None
        self.prediction_manager = None
        self.prediction_agent = PredictionAgent()
        self.explainability_agent = None
        self.recommendation_agent = None
        self.report_agent = None
        self.chat_agent = None

    def initialize_pipeline(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Loads the dataset and starts orchestration.
        """
        self.state["file_path"] = file_path
        
        # 1. Load Data
        df, err = DataLoader.load_dataset(file_path)
        if err:
            return False, err
            
        self.state["raw_df"] = df
        
        # 2. Schema Understanding
        self.schema_agent = SchemaUnderstandingAgent()
        schema = self.schema_agent.infer_schema(df)
        self.state["schema"] = schema
        
        # 3. Data Quality Profiling & Cleaning
        self.data_agent = DataQualityAgent(schema)
        dq_report = self.data_agent.profile_data_quality(df)
        self.state["data_quality_report"] = dq_report
        
        cleaned_df = self.data_agent.clean_dataset(df)
        self.state["cleaned_df"] = cleaned_df
        
        # 4. EDA Agent
        self.eda_agent = EDAAgent(schema)
        eda_report = self.eda_agent.perform_eda(cleaned_df)
        self.state["eda_report"] = eda_report
        self.state["narrative_insights"] = eda_report.get("narrative_insights", "")
        
        # Initialize other agents with schema
        self.explainability_agent = ExplainabilityAgent(schema)
        self.recommendation_agent = RecommendationAgent(schema)
        self.report_agent = ReportAgent(schema)
        self.chat_agent = ChatAgent(schema)
        
        # 5. Initialize KPI metrics
        self._calculate_kpis(cleaned_df)
        
        # 6. Auto-run predictions if a saved model already exists on disk
        # This ensures new risk rules always apply on every pipeline load
        if self.prediction_agent.load_model():
            try:
                self.run_predictions()
            except Exception as e:
                print(f"Auto-prediction skipped: {e}")
        
        return True, None

    def _calculate_kpis(self, df: pd.DataFrame):
        """
        Calculates standard metrics for the executive overview.
        """
        kpis = {
            "total_accounts": len(df),
        }
        
        # Attempt to get Credit Score averages
        if "Credit_Score" in df.columns:
            kpis["average_credit_score"] = round(float(df["Credit_Score"].mean()), 1)
            
        # Income averages
        if "Income" in df.columns:
            kpis["average_income"] = f"${round(float(df["Income"].mean()), 2):,}"
            
        # Delinquency rate
        target = self.state["schema"].get("target_column")
        if target and target in df.columns:
            del_count = (df[target] == 1).sum()
            rate = (del_count / len(df)) * 100
            kpis["historical_delinquency_rate"] = f"{rate:.1f}%"
            
        self.state["kpis"] = kpis

    def train_predictive_model(self, model_type: str = "Random Forest") -> Dict[str, Any]:
        """
        Invokes Prediction Manager to train a predictive model.
        """
        cleaned_df = self.state["cleaned_df"]
        if cleaned_df is None:
            raise ValueError("Dataset has not been loaded. Load data first.")
            
        self.prediction_manager = PredictionManager(self.state["schema"])
        train_results = self.prediction_manager.train_and_save_model(cleaned_df, model_type)
        
        self.state["model_metrics"] = train_results["metrics"]
        self.state["active_model_name"] = model_type
        
        # Run prediction on the portfolio
        self.run_predictions()
        
        return train_results

    def run_predictions(self):
        """
        Infers risk categories and confidences on the active portfolio.
        """
        cleaned_df = self.state["cleaned_df"]
        self.prediction_agent.load_model()
        
        pred_df = self.prediction_agent.predict(cleaned_df)
        
        # Batch assign rule-based explanations and recommendations to prevent rate limit issues
        explanations = []
        recommendations = []
        for idx, row in pred_df.iterrows():
            prob = row["risk_probability"]
            cat = row["risk_category"]
            
            # Simple rule mapping for faster bulk tables
            if cat == "High Risk":
                exp = f"Risk is High ({prob*100:.1f}%) due to multiple delinquency indicators."
                rec = "Immediate outreach & payment restructuring."
            elif cat == "Medium-High Risk (Potential High)":
                exp = f"Risk is Medium-High ({prob*100:.1f}%). Elevated risk indicators."
                rec = "Restrict credit limits & increase monitoring frequency."
            elif cat == "Medium Risk":
                exp = f"Risk is Medium ({prob*100:.1f}%). Moderate risk profile."
                rec = "Send payment reminders & monitor."
            else:
                exp = f"Risk is Low ({prob*100:.1f}%). Solid credit indicators."
                rec = "Maintain standard tracking."
                
            explanations.append(exp)
            recommendations.append(rec)
            
        pred_df["personal_explanation"] = explanations
        pred_df["personal_recommendation"] = recommendations
        
        self.state["predictions_df"] = pred_df
        
        # Generate bulk macro actions
        self.state["bulk_recs"] = self.recommendation_agent.generate_bulk_recommendations(pred_df)
        
        # Update overview KPIs with predictions
        self._update_kpis_with_predictions(pred_df)

    def _update_kpis_with_predictions(self, pred_df: pd.DataFrame):
        """
        Enriches overview KPIs with model output statistics.
        """
        high_risk_count = (pred_df["risk_category"].isin(["High Risk", "Medium-High Risk (Potential High)"])).sum()
        high_risk_pct = (high_risk_count / len(pred_df)) * 100
        
        self.state["kpis"]["predicted_high_risk_accounts"] = f"{high_risk_count} ({high_risk_pct:.1f}%)"
        self.state["kpis"]["average_predicted_risk"] = f"{pred_df['risk_probability'].mean() * 100:.1f}%"

    def explain_individual(self, customer_id: str) -> Dict[str, str]:
        """
        Generates personalized LLM explanation and recommendation for a specific account.
        """
        pred_df = self.state["predictions_df"]
        id_col = self.state["schema"].get("identifier_column", "Customer_ID")
        
        if pred_df is None or id_col not in pred_df.columns:
            return {"error": "Predictions have not been run yet."}
            
        row = pred_df[pred_df[id_col].astype(str) == str(customer_id)]
        if row.empty:
            return {"error": f"Customer ID {customer_id} not found."}
            
        row_series = row.iloc[0]
        prob = row_series["risk_probability"]
        cat = row_series["risk_category"]
        
        explanation = self.explainability_agent.explain_local_prediction(row_series, prob, cat)
        recommendation = self.recommendation_agent.generate_recommendations(row_series, cat, prob)
        
        # Update specific rows in prediction df for display consistency
        pred_df.loc[pred_df[id_col].astype(str) == str(customer_id), "personal_explanation"] = explanation
        pred_df.loc[pred_df[id_col].astype(str) == str(customer_id), "personal_recommendation"] = recommendation
        
        return {
            "customer_id": customer_id,
            "risk_category": cat,
            "risk_probability": f"{prob*100:.1f}%",
            "explanation": explanation,
            "recommendation": recommendation
        }

    def generate_reports(self) -> Dict[str, str]:
        """
        Triggers Report Agent to generate output files.
        """
        paths = self.report_agent.generate_reports(self.state)
        self.state["report_paths"] = paths
        return paths

    def chat_query(self, user_query: str) -> str:
        """
        Dispatches user conversational queries to Chat Agent.
        """
        if self.chat_agent is None:
            return "Platform has not been initialized with a dataset yet. Please upload a dataset first."
        return self.chat_agent.answer_query(user_query, self.state)
