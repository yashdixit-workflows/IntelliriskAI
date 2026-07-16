import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any, Optional

class Visualizer:
    """
    Generates data charts and saves them to outputs/plots/.
    """
    def __init__(self, output_dir: str = r"x:\creditguard-ai\outputs\plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Apply clean, premium styling
        sns.set_theme(style="white")
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
            'text.color': '#333333',
            'axes.labelcolor': '#333333',
            'xtick.color': '#333333',
            'ytick.color': '#333333',
            'figure.autolayout': True
        })

    def plot_correlation_heatmap(self, df: pd.DataFrame, numeric_cols: List[str]) -> Optional[str]:
        """
        Generates correlation heatmap of numeric columns and returns output path.
        """
        if len(numeric_cols) < 2:
            return None
        
        plt.figure(figsize=(8, 6))
        corr = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Sleek color map matching premium aesthetics
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .8}, annot=True, fmt=".2f")
        
        plt.title("Feature Correlation Matrix", fontsize=12, fontweight='bold', pad=15)
        path = os.path.join(self.output_dir, "correlation_heatmap.png")
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        return path

    def plot_risk_distribution(self, risk_categories: pd.Series) -> str:
        """
        Generates bar plot of risk distribution.
        """
        plt.figure(figsize=(6, 4))
        counts = risk_categories.value_counts()
        
        # Color codes matching Zinc Theme: Low (green), Medium (amber), High (red)
        color_map = {
            'Low Risk': '#16a34a',
            'Medium Risk': '#d97706',
            'High Risk': '#dc2626'
        }
        colors = [color_map.get(cat, '#71717a') for cat in counts.index]
        
        sns.barplot(x=counts.index, y=counts.values, palette=colors)
        plt.title("Portfolio Risk Profile", fontsize=12, fontweight='bold', pad=15)
        plt.ylabel("Number of Accounts")
        plt.xlabel("Risk Categories")
        
        # Despine
        sns.despine()
        
        path = os.path.join(self.output_dir, "risk_distribution.png")
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        return path

    def plot_feature_importance(self, importance_dict: Dict[str, float]) -> str:
        """
        Generates feature importance bar chart.
        """
        plt.figure(figsize=(8, 5))
        
        # Sort importances
        sorted_imp = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        features, scores = zip(*sorted_imp)
        
        # Generate chart
        sns.barplot(x=list(scores), y=list(features), color="#2563eb") # Slate blue accent
        plt.title("Top Feature Importances", fontsize=12, fontweight='bold', pad=15)
        plt.xlabel("Relative Contribution Score")
        plt.ylabel("Dataset Features")
        
        sns.despine()
        
        path = os.path.join(self.output_dir, "feature_importance.png")
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        return path

    def plot_numerical_distribution(self, df: pd.DataFrame, col: str) -> str:
        """
        Generates histogram distribution plot of a numerical column.
        """
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True, color="#2563eb", bins=20)
        plt.title(f"Distribution of {col}", fontsize=11, fontweight='bold', pad=12)
        plt.xlabel(col)
        plt.ylabel("Frequency")
        
        sns.despine()
        
        clean_col_name = "".join(x for x in col if x.isalnum())
        path = os.path.join(self.output_dir, f"distribution_{clean_col_name}.png")
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        return path

    def plot_categorical_frequency(self, df: pd.DataFrame, col: str) -> str:
        """
        Generates horizontal bar chart of a categorical feature frequency.
        """
        plt.figure(figsize=(7, 4))
        counts = df[col].dropna().value_counts()
        sns.barplot(x=counts.values, y=counts.index, color="#2563eb")
        plt.title(f"Frequency of {col}", fontsize=11, fontweight='bold', pad=12)
        plt.xlabel("Counts")
        plt.ylabel(col)
        
        sns.despine()
        
        clean_col_name = "".join(x for x in col if x.isalnum())
        path = os.path.join(self.output_dir, f"frequency_{clean_col_name}.png")
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        return path

    def plot_delinquency_trend(self, df: pd.DataFrame, temporal_cols: List[str]) -> Optional[str]:
        """
        Plots trend of delinquency or late payments across temporal columns (e.g. Month_1 to Month_6).
        """
        if not temporal_cols:
            return None
            
        # Count delinquency-like entries per month
        late_indicators = ["late", "missed", "delinquent"]
        
        rates = []
        for col in temporal_cols:
            if col in df.columns:
                series = df[col].astype(str).str.strip().str.lower()
                is_delinquent = series.isin(late_indicators).sum()
                rate = (is_delinquent / len(df)) * 100
                rates.append(rate)
                
        if not rates:
            return None
            
        plt.figure(figsize=(8, 4))
        plt.plot(temporal_cols, rates, marker='o', color='#dc2626', linewidth=2, markersize=6)
        plt.title("Delinquency Rate Trend Over Time", fontsize=12, fontweight='bold', pad=15)
        plt.ylabel("Delinquency Rate (%)")
        plt.xlabel("Time Sequence Period")
        plt.ylim(0, max(rates) + 5 if rates else 100)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        sns.despine(left=True, bottom=True)
        
        path = os.path.join(self.output_dir, "delinquency_trend.png")
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        return path
