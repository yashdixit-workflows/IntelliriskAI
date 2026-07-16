import pandas as pd
import json
import re
from typing import Dict, Any
from agents import get_gemini_client

class SchemaUnderstandingAgent:
    """
    Schema Understanding Agent: Automatically detects semantic schema and metadata of unknown datasets.
    """
    def __init__(self):
        self.client = get_gemini_client()

    def infer_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Infers schema by creating metadata profiles and using Gemini for semantic classification.
        """
        # 1. Gather basic metadata
        metadata = {}
        for col in df.columns:
            series = df[col]
            sample_vals = series.dropna().head(4).tolist()
            unique_count = int(series.nunique())
            null_count = int(series.isnull().sum())
            null_rate = float(null_count / len(df))
            dtype = str(series.dtype)
            
            # Simple stats
            stats = {}
            if pd.api.types.is_numeric_dtype(series):
                stats = {
                    "min": float(series.min()) if not pd.isna(series.min()) else None,
                    "max": float(series.max()) if not pd.isna(series.max()) else None,
                    "mean": float(series.mean()) if not pd.isna(series.mean()) else None
                }
                
            metadata[col] = {
                "data_type": dtype,
                "unique_values_count": unique_count,
                "null_count": null_count,
                "null_rate": null_rate,
                "sample_values": sample_vals,
                "statistics": stats
            }

        # 2. Call Gemini to determine semantic meanings and types
        prompt = f"""
Analyze the following dataset metadata profile and infer the semantic details for every column.
Your response MUST be a valid JSON object matching the schema below.
Do not return any surrounding markdown text except optionally a ```json code block.

Metadata Profile:
{json.dumps(metadata, indent=2)}

Required Output JSON Schema:
{{
  "columns": {{
    "<column_name>": {{
      "semantic_type": "identifier" | "numerical" | "categorical" | "date" | "temporal" | "target",
      "business_meaning": "brief description of what this column represents",
      "is_pii": true | false,
      "pii_type": "name" | "email" | "phone" | "address" | "ssn" | "other" | null
    }}
  }},
  "identifier_column": "<name_of_id_column_or_null>",
  "target_column": "<name_of_target_prediction_variable_or_null>",
  "numerical_features": ["list", "of", "numerical", "columns"],
  "categorical_features": ["list", "of", "categorical", "columns"],
  "date_columns": ["list", "of", "date", "columns"],
  "temporal_features": ["list", "of", "time_series_or_sequence_columns"]
}}

Rules:
1. "temporal" features refer to sequence data such as periodic payments (e.g. Month_1, Month_2) or time series measurements.
2. Identify target variables (e.g. delinquent account indicator, churn status, customer score) and set "target_column".
3. Return ONLY a valid JSON object.
"""
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            
            response_text = response.text.strip()
            # Clean markdown code blocks if present
            if response_text.startswith("```"):
                response_text = re.sub(r"^```(?:json)?\n", "", response_text)
                response_text = re.sub(r"\n```$", "", response_text)
                
            schema_json = json.loads(response_text)
            return schema_json
            
        except Exception as e:
            print("Error in Schema Agent Gemini call, falling back to rule-based:", e)
            return self._rule_based_inference(df)

    def _rule_based_inference(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Fallback rule-based schema understanding in case API call fails.
        """
        columns = {}
        identifier_column = None
        target_column = None
        numerical_features = []
        categorical_features = []
        date_columns = []
        temporal_features = []
        
        for col in df.columns:
            series = df[col]
            unique_count = series.nunique()
            dtype = str(series.dtype)
            col_lower = col.lower()
            
            # 1. Identify Target
            if "delinquent" in col_lower or "target" in col_lower or "label" in col_lower:
                target_column = col
                sem_type = "target"
            # 2. Identify Identifier
            elif "id" in col_lower or col_lower == "customer_id" or col_lower == "account_number":
                if unique_count > len(df) * 0.8:
                    identifier_column = col
                    sem_type = "identifier"
                else:
                    categorical_features.append(col)
                    sem_type = "categorical"
            # 3. Identify Date
            elif "date" in col_lower or "time" in col_lower:
                date_columns.append(col)
                sem_type = "date"
            # 4. Identify Temporal
            elif "month_" in col_lower or "week_" in col_lower or "year_" in col_lower:
                temporal_features.append(col)
                sem_type = "temporal"
            # 5. Numerical features
            elif pd.api.types.is_numeric_dtype(series):
                numerical_features.append(col)
                sem_type = "numerical"
            # 6. Categorical features
            else:
                categorical_features.append(col)
                sem_type = "categorical"
                
            columns[col] = {
                "semantic_type": sem_type,
                "business_meaning": f"Auto-inferred {sem_type} variable.",
                "is_pii": "id" in col_lower or "name" in col_lower or "phone" in col_lower,
                "pii_type": "other" if ("id" in col_lower or "name" in col_lower) else None
            }
            
        return {
            "columns": columns,
            "identifier_column": identifier_column,
            "target_column": target_column,
            "numerical_features": numerical_features,
            "categorical_features": categorical_features,
            "date_columns": date_columns,
            "temporal_features": temporal_features
        }
