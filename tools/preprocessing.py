import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import Dict, List, Tuple, Any, Optional

class DataPreprocessor:
    """
    Standard preprocessor for cleaning, imputing, scaling, and encoding datasets.
    """
    def __init__(self, semantic_schema: Dict[str, Any]):
        self.schema = semantic_schema
        self.numerical_cols = semantic_schema.get("numerical_features", [])
        self.categorical_cols = semantic_schema.get("categorical_features", [])
        self.temporal_cols = semantic_schema.get("temporal_features", [])
        self.target_col = semantic_schema.get("target_column")
        self.identifier_col = semantic_schema.get("identifier_column")
        
        # Imputers and preprocessors
        self.num_imputer = SimpleImputer(strategy="median")
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        
        # Store categories and fitted state
        self.is_fitted = False
        self.encoded_cat_names = []
        self.processed_feature_names = []
        
        # Hardcoded sequence mappings for known delinquency patterns
        self.seq_mapping = {
            "on-time": 0,
            "ontime": 0,
            "late": 1,
            "missed": 2,
            "delinquent": 2,
            "active": 0,
            "closed": 1
        }

    def _map_sequence_value(self, val: Any) -> float:
        if pd.isna(val):
            return np.nan
        val_str = str(val).strip().lower()
        return self.seq_mapping.get(val_str, 0) # Default to 0 if unknown

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """
        Fits the preprocessing pipeline on the input DataFrame.
        """
        # 1. Fit numerical features
        num_features = self.numerical_cols.copy()
        
        # Convert temporal columns into numerical values by mapping them ordinally
        temp_features = []
        for col in self.temporal_cols:
            if col in df.columns:
                mapped_series = df[col].apply(self._map_sequence_value)
                # If mostly valid numerical mappings, treat as numeric
                temp_features.append(col)
        
        all_numeric_cols = num_features + temp_features
        
        if all_numeric_cols:
            numeric_data = df[num_features].copy() if num_features else pd.DataFrame(index=df.index)
            for col in temp_features:
                numeric_data[col] = df[col].apply(self._map_sequence_value)
            # Convert to float and fit
            for col in all_numeric_cols:
                numeric_data[col] = pd.to_numeric(numeric_data[col], errors="coerce")
            self.num_imputer.fit(numeric_data)
            imputed_num = self.num_imputer.transform(numeric_data)
            self.scaler.fit(imputed_num)
            
        # 2. Fit categorical features
        valid_cat_cols = [c for c in self.categorical_cols if c in df.columns]
        if valid_cat_cols:
            cat_data = df[valid_cat_cols].fillna("Missing").astype(str)
            self.cat_imputer.fit(cat_data)
            imputed_cat = self.cat_imputer.transform(cat_data)
            self.encoder.fit(imputed_cat)
            self.encoded_cat_names = list(self.encoder.get_feature_names_out(valid_cat_cols))
            
        # 3. Assemble final features list
        self.processed_feature_names = all_numeric_cols + self.encoded_cat_names
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Transforms the input DataFrame based on fitted states.
        Returns:
            Tuple[X_preprocessed (numpy array), feature_names (list)]
        """
        if not self.is_fitted:
            raise ValueError("DataPreprocessor has not been fitted yet!")

        # 1. Process numerical & temporal features
        num_features = self.numerical_cols.copy()
        temp_features = []
        
        temp_data_dict = {}
        for col in self.temporal_cols:
            if col in df.columns:
                temp_data_dict[col] = df[col].apply(self._map_sequence_value)
                temp_features.append(col)
                
        all_numeric_cols = num_features + temp_features
        
        X_num = np.empty((len(df), 0))
        if all_numeric_cols:
            numeric_data = df[num_features].copy() if num_features else pd.DataFrame(index=df.index)
            for col in temp_features:
                numeric_data[col] = temp_data_dict[col]
            for col in all_numeric_cols:
                numeric_data[col] = pd.to_numeric(numeric_data[col], errors="coerce")
                
            imputed_num = self.num_imputer.transform(numeric_data)
            scaled_num = self.scaler.transform(imputed_num)
            X_num = scaled_num

        # 2. Process categorical features
        X_cat = np.empty((len(df), 0))
        valid_cat_cols = [c for c in self.categorical_cols if c in df.columns]
        if valid_cat_cols:
            cat_data = df[valid_cat_cols].fillna("Missing").astype(str)
            imputed_cat = self.cat_imputer.transform(cat_data)
            encoded_cat = self.encoder.transform(imputed_cat)
            X_cat = encoded_cat

        # 3. Combine processed features
        if X_num.shape[1] > 0 and X_cat.shape[1] > 0:
            X_out = np.hstack((X_num, X_cat))
        elif X_num.shape[1] > 0:
            X_out = X_num
        else:
            X_out = X_cat
            
        return X_out, self.processed_feature_names

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        return self.fit(df).transform(df)

    def preprocess_target(self, df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[Dict[str, int]]]:
        """
        Extracts and converts target variable to numeric.
        """
        if not self.target_col or self.target_col not in df.columns:
            return None, None
        
        target = df[self.target_col].copy()
        
        # If target is categorical, map it
        if target.dtype == object or str(target.dtype) == 'category':
            unique_vals = sorted(target.dropna().unique())
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            mapped_target = target.map(mapping).fillna(-1).astype(int).values
            return mapped_target, mapping
        else:
            # Drop NaN rows or fill for training target
            mapped_target = target.fillna(0).astype(int).values
            return mapped_target, None
