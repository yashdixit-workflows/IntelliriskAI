import pandas as pd
import numpy as np
from typing import Dict, Any, List
import os
import json
from tools.visualizations import Visualizer
from agents import get_gemini_client

class EDAAgent:
    """
    EDA Agent: Performs automated Exploratory Data Analysis, creates visualizations,
    and produces business narrative insights.
    """
    def __init__(self, semantic_schema: Dict[str, Any]):
        self.schema = semantic_schema
        self.numerical_cols = semantic_schema.get("numerical_features", [])
        self.categorical_cols = semantic_schema.get("categorical_features", [])
        self.temporal_cols = semantic_schema.get("temporal_features", [])
        self.target_col = semantic_schema.get("target_column")
        self.visualizer = Visualizer()
        self.client = get_gemini_client()

    def perform_eda(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Executes stats, generates charts, and returns insights.
        """
        # 1. Summary Statistics
        stats_summary = {}
        for col in self.numerical_cols:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(series) > 0:
                    stats_summary[col] = {
                        "mean": float(series.mean()),
                        "median": float(series.median()),
                        "std": float(series.std()),
                        "min": float(series.min()),
                        "max": float(series.max())
                    }

        # 2. Correlations
        corr_matrix = {}
        valid_numeric = [c for c in self.numerical_cols if c in df.columns]
        if len(valid_numeric) >= 2:
            corr = df[valid_numeric].corr()
            corr_matrix = corr.to_dict()

        # 3. Generate Visualizations (and save to disk)
        charts = {}
        heatmap_path = self.visualizer.plot_correlation_heatmap(df, valid_numeric)
        if heatmap_path:
            charts["correlation_heatmap"] = heatmap_path
            
        trend_path = self.visualizer.plot_delinquency_trend(df, self.temporal_cols)
        if trend_path:
            charts["delinquency_trend"] = trend_path

        # Generate a couple of distribution charts for primary variables
        dist_charts = []
        for col in valid_numeric[:2]:
            path = self.visualizer.plot_numerical_distribution(df, col)
            dist_charts.append({"column": col, "path": path})
        charts["distributions"] = dist_charts

        # Generate a categorical bar chart
        freq_charts = []
        for col in [c for c in self.categorical_cols if c in df.columns][:1]:
            path = self.visualizer.plot_categorical_frequency(df, col)
            freq_charts.append({"column": col, "path": path})
        charts["frequencies"] = freq_charts

        # 4. Generate Narrative Insights (dual-mode)
        insights = self._generate_narrative_insights(df, stats_summary, corr_matrix)

        return {
            "summary_statistics": stats_summary,
            "correlations": corr_matrix,
            "generated_charts": charts,
            "narrative_insights": insights
        }

    def _generate_narrative_insights(self, df: pd.DataFrame, stats: dict, corr: dict) -> str:
        """
        Creates analytical takeaways from correlations and stats. Falls back to rule-based generation.
        """
        # Find high correlation pairs
        high_corr_pairs = []
        if corr:
            for col1 in corr:
                for col2 in corr[col1]:
                    if col1 != col2 and abs(corr[col1][col2]) > 0.4:
                        # Avoid duplicates
                        pair = sorted([col1, col2])
                        if pair not in high_corr_pairs:
                            high_corr_pairs.append(pair)

        # 1. Rule-based explanation
        rule_insights = []
        rule_insights.append(f"The dataset contains {len(df)} records across {len(df.columns)} columns.")
        
        if self.target_col and self.target_col in df.columns:
            target_series = df[self.target_col].astype(str).str.strip().str.lower()
            # If target has delinquency representations
            delinq_indicators = ["1", "1.0", "yes", "true", "delinquent"]
            delinq_count = target_series.isin(delinq_indicators).sum()
            delinq_pct = (delinq_count / len(df)) * 100
            rule_insights.append(f"The target column '{self.target_col}' shows a delinquency rate of {delinq_pct:.1f}% ({delinq_count} delinquent accounts).")
            
        if high_corr_pairs:
            rule_insights.append("Significant feature correlation findings:")
            for p in high_corr_pairs[:3]:
                c_val = corr[p[0]][p[1]]
                relation = "positive" if c_val > 0 else "inverse"
                rule_insights.append(f"  - '{p[0]}' and '{p[1]}' share a strong {relation} correlation of {c_val:.2f}.")

        # Check key metrics for credit delinquency if available
        if "Credit_Score" in df.columns and "Delinquent_Account" in df.columns:
            avg_score_del = df[df["Delinquent_Account"] == 1]["Credit_Score"].mean()
            avg_score_ok = df[df["Delinquent_Account"] == 0]["Credit_Score"].mean()
            if not pd.isna(avg_score_del) and not pd.isna(avg_score_ok):
                rule_insights.append(f"The average credit score of delinquent accounts is {avg_score_del:.1f}, compared to {avg_score_ok:.1f} for non-delinquent accounts.")

        default_text = "\n".join(rule_insights)

        # 2. LLM Synthesis
        prompt = f"""
Write a professional, 3-paragraph executive summary analyzing the exploratory data analysis findings of this dataset.
Focus on business risks, portfolio implications, correlations, and key takeaways.
Keep the style formal, professional, and insight-driven.

Data Profile Summary:
- Record Count: {len(df)}
- Columns: {list(df.columns)}
- Numerical Stats: {json.dumps(stats)}
- Highly Correlated Features: {high_corr_pairs}
- Rule-based summary: {default_text}
"""
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            return response.text.strip()
        except Exception:
            return default_text
