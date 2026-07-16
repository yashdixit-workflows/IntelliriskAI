import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import json
from agents import get_gemini_client

class DataQualityAgent:
    """
    Data Quality Agent: Detects missing values, duplicates, outliers, invalid values,
    and performs automated cleaning.
    """
    def __init__(self, semantic_schema: Dict[str, Any]):
        self.schema = semantic_schema
        self.numerical_cols = semantic_schema.get("numerical_features", [])
        self.categorical_cols = semantic_schema.get("categorical_features", [])
        self.client = get_gemini_client()

    def profile_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs automated checks and returns a data quality report.
        """
        total_rows = len(df)
        
        # 1. Missing Values
        missing_report = {}
        for col in df.columns:
            null_count = int(df[col].isnull().sum())
            if null_count > 0:
                missing_report[col] = {
                    "count": null_count,
                    "percentage": float((null_count / total_rows) * 100)
                }

        # 2. Duplicate rows
        duplicate_count = int(df.duplicated().sum())

        # 3. Outliers (using IQR)
        outlier_report = {}
        for col in self.numerical_cols:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(series) > 0:
                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    outliers = series[(series < lower_bound) | (series > upper_bound)]
                    if len(outliers) > 0:
                        outlier_report[col] = {
                            "count": int(len(outliers)),
                            "percentage": float((len(outliers) / total_rows) * 100),
                            "lower_bound": float(lower_bound),
                            "upper_bound": float(upper_bound)
                        }

        # 4. Invalid values (simple logical rules)
        invalid_report = {}
        # Age should be reasonable (e.g. between 18 and 120)
        for col in df.columns:
            col_lower = col.lower()
            if "age" in col_lower and pd.api.types.is_numeric_dtype(df[col]):
                invalid_ages = df[(df[col] < 0) | (df[col] > 120)]
                if len(invalid_ages) > 0:
                    invalid_report[col] = {
                        "count": len(invalid_ages),
                        "reason": "Age is negative or exceeds 120 years."
                    }
            elif "income" in col_lower and pd.api.types.is_numeric_dtype(df[col]):
                invalid_income = df[df[col] < 0]
                if len(invalid_income) > 0:
                    invalid_report[col] = {
                        "count": len(invalid_income),
                        "reason": "Income value is negative."
                    }

        # 5. Generate cleaning recommendations (dual-mode)
        recommendations = self._generate_cleaning_recommendations(
            missing_report, duplicate_count, outlier_report, invalid_report
        )

        return {
            "total_rows": total_rows,
            "missing_values": missing_report,
            "duplicate_rows_count": duplicate_count,
            "outliers": outlier_report,
            "invalid_values": invalid_report,
            "cleaning_recommendations": recommendations
        }

    def _generate_cleaning_recommendations(
        self, missing: dict, duplicates: int, outliers: dict, invalid: dict
    ) -> List[str]:
        """
        Creates recommendations based on data profile. Falls back to rules if Gemini is unavailable.
        """
        recs = []
        if duplicates > 0:
            recs.append(f"Remove {duplicates} duplicate rows from the dataset.")
            
        for col, details in missing.items():
            if details["percentage"] > 50.0:
                recs.append(f"Column '{col}' has {details['percentage']:.1f}% missing values. Suggest dropping this column.")
            else:
                recs.append(f"Column '{col}' has {details['count']} missing values. Suggest imputing with median/mode.")
                
        for col, details in outliers.items():
            recs.append(f"Column '{col}' has {details['count']} outliers ({details['percentage']:.1f}%). Suggest checking for validity or using robust scaling.")

        for col, details in invalid.items():
            recs.append(f"Column '{col}' contains {details['count']} invalid values. Suggest replacing with column average or removing.")

        # Try to use LLM to summarize and enrich recommendations
        prompt = f"""
Given the following data quality issues, write a list of 3-4 professional, actionable data cleaning recommendations for a data engineer.
Keep your response as a simple JSON array of strings.

Issues Profile:
- Duplicates: {duplicates}
- Missing: {json.dumps(missing)}
- Outliers: {json.dumps(outliers)}
- Invalid values: {json.dumps(invalid)}
"""
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            response_text = response.text.strip()
            if response_text.startswith("```"):
                response_text = re.sub(r"^```(?:json)?\n", "", response_text)
                response_text = re.sub(r"\n```$", "", response_text)
            llm_recs = json.loads(response_text)
            if isinstance(llm_recs, list) and len(llm_recs) > 0:
                return llm_recs
        except Exception:
            pass # Fall back to rule-based list
            
        return recs

    # ------------------------------------------------------------------
    # Internal helper: select the best grouping columns for a target col
    # ------------------------------------------------------------------
    def _get_group_cols(self, df: pd.DataFrame, target_col: str) -> List[str]:
        """
        Pick categorical columns most likely to correlate with target_col
        for grouped imputation. Prefer domain-relevant columns; fall back
        to any low-cardinality categorical present.
        """
        col_lower = target_col.lower()

        # Domain priority mappings: if the target column name contains any of
        # these keywords, prefer the listed grouping columns (if they exist).
        domain_hints = {
            ("loan", "balance", "amount", "debt"):
                ["Loan_Type", "Employment_Status", "Credit_Score_Category",
                 "Credit_Card_Type", "Region"],
            ("income", "salary"):
                ["Employment_Status", "Region", "Credit_Card_Type"],
            ("credit", "score"):
                ["Employment_Status", "Region", "Credit_Card_Type"],
            ("age",):
                ["Employment_Status", "Region"],
        }

        preferred = []
        for keywords, candidates in domain_hints.items():
            if any(kw in col_lower for kw in keywords):
                preferred = [c for c in candidates if c in df.columns]
                break

        if not preferred:
            # Generic fallback: pick low-cardinality categoricals (≤15 unique values)
            preferred = [
                c for c in self.categorical_cols
                if c in df.columns and df[c].nunique(dropna=True) <= 15
            ]

        return preferred[:2]  # Use at most 2 grouping columns to avoid over-segmentation

    def _grouped_median_impute(self, df: pd.DataFrame, col: str) -> pd.Series:
        """
        Smart imputation for a single numerical column using grouped medians.

        Strategy (in order of preference):
          1. Group by 2 related categorical columns → group median
          2. Group by 1 related categorical column → group median
          3. Global median
          4. Global mean (if median is also NaN)
          5. 0 (last resort)
        """
        series = df[col].copy()
        missing_mask = series.isna()

        if not missing_mask.any():
            return series  # Nothing to impute

        group_cols = self._get_group_cols(df, col)

        # Try progressively coarser groupings
        grouping_levels = []
        if len(group_cols) >= 2:
            grouping_levels.append(group_cols[:2])
        if len(group_cols) >= 1:
            grouping_levels.append(group_cols[:1])

        for g_cols in grouping_levels:
            # Compute per-group median from non-missing rows
            group_medians = (
                df.loc[~missing_mask, g_cols + [col]]
                .groupby(g_cols)[col]
                .median()
            )

            def fill_from_group(row):
                if not pd.isna(row[col]):
                    return row[col]
                key = tuple(row[c] for c in g_cols) if len(g_cols) > 1 else row[g_cols[0]]
                try:
                    val = group_medians.loc[key]
                    return val if not pd.isna(val) else np.nan
                except KeyError:
                    return np.nan

            filled = df.apply(fill_from_group, axis=1)
            # Only adopt this level's result where it succeeded
            series = series.where(~missing_mask, filled)
            missing_mask = series.isna()
            if not missing_mask.any():
                return series  # All filled

        # Global median fallback
        global_median = series.dropna().median()
        if pd.isna(global_median):
            global_median = series.dropna().mean()
        if pd.isna(global_median):
            global_median = 0.0

        series = series.fillna(global_median)
        return series

    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes data cleaning: removes duplicates, imputes missing values
        using smart grouped-median imputation (not a flat global median),
        and standardises bounds.
        """
        cleaned_df = df.copy()

        # 1. Drop duplicates
        cleaned_df = cleaned_df.drop_duplicates()

        # 2. Handle categorical columns FIRST so they are available as grouping keys
        for col in self.categorical_cols:
            if col in cleaned_df.columns:
                if not cleaned_df[col].mode().empty:
                    mode_val = cleaned_df[col].mode()[0]
                else:
                    mode_val = "Unknown"
                cleaned_df[col] = cleaned_df[col].fillna(mode_val).astype(str)

        # 3. Impute numerical columns using grouped-median strategy
        for col in self.numerical_cols:
            if col not in cleaned_df.columns:
                continue

            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")

            # Domain-level bound enforcement → nullify implausible values
            col_lower = col.lower()
            if "age" in col_lower:
                cleaned_df.loc[
                    (cleaned_df[col] < 0) | (cleaned_df[col] > 120), col
                ] = np.nan
            elif "income" in col_lower:
                cleaned_df.loc[cleaned_df[col] < 0, col] = np.nan
            elif any(kw in col_lower for kw in ("balance", "amount", "loan", "debt")):
                # Loan balances cannot be negative
                cleaned_df.loc[cleaned_df[col] < 0, col] = np.nan

            if cleaned_df[col].isna().any():
                cleaned_df[col] = self._grouped_median_impute(cleaned_df, col)

        return cleaned_df

