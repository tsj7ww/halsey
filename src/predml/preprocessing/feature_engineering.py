from typing import List, Optional, Union, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


class FeatureEngineer:
    """Feature engineering pipeline for preprocessing data."""

    def __init__(
        self,
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        scaling_method: str = "standard",
        handle_missing: bool = True
    ):
        """Initialize the feature engineering pipeline.

        Args:
            numeric_features: List of numeric feature names
            categorical_features: List of categorical feature names
            scaling_method: Method for scaling numeric features ('standard' or 'minmax')
            handle_missing: Whether to handle missing values
        """
        self.numeric_features = numeric_features or []
        self.categorical_features = categorical_features or []
        self.scaling_method = scaling_method
        self.handle_missing = handle_missing
        
        self.numeric_scaler = None
        self.categorical_encoder = None
        self.numeric_imputer = None
        self.categorical_imputer = None

    def fit(self, df: pd.DataFrame) -> None:
        """Fit the feature engineering pipeline.

        Args:
            df: Input DataFrame
        """
        # Initialize imputers if needed
        if self.handle_missing:
            self.numeric_imputer = SimpleImputer(strategy="mean")
            self.categorical_imputer = SimpleImputer(strategy="most_frequent")
            
            if self.numeric_features:
                self.numeric_imputer.fit(df[self.numeric_features])
            if self.categorical_features:
                self.categorical_imputer.fit(df[self.categorical_features])

        # Initialize and fit scalers/encoders
        if self.numeric_features:
            if self.scaling_method == "standard":
                self.numeric_scaler = StandardScaler()
            else:
                self.numeric_scaler = MinMaxScaler()
            
            numeric_data = df[self.numeric_features]
            if self.handle_missing:
                numeric_data = pd.DataFrame(
                    self.numeric_imputer.transform(numeric_data),
                    columns=self.numeric_features
                )
            self.numeric_scaler.fit(numeric_data)

        if self.categorical_features:
            self.categorical_encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
            categorical_data = df[self.categorical_features]
            if self.handle_missing:
                categorical_data = pd.DataFrame(
                    self.categorical_imputer.transform(categorical_data),
                    columns=self.categorical_features
                )
            self.categorical_encoder.fit(categorical_data)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform the input data using the fitted pipeline.

        Args:
            df: Input DataFrame

        Returns:
            Transformed features as numpy array
        """
        transformed_features = []

        # Transform numeric features
        if self.numeric_features:
            numeric_data = df[self.numeric_features]
            if self.handle_missing and self.numeric_imputer is not None:
                numeric_data = pd.DataFrame(
                    self.numeric_imputer.transform(numeric_data),
                    columns=self.numeric_features
                )
            if self.numeric_scaler is not None:
                numeric_data = self.numeric_scaler.transform(numeric_data)
            transformed_features.append(numeric_data)

        # Transform categorical features
        if self.categorical_features:
            categorical_data = df[self.categorical_features]
            if self.handle_missing and self.categorical_imputer is not None:
                categorical_data = pd.DataFrame(
                    self.categorical_imputer.transform(categorical_data),
                    columns=self.categorical_features
                )
            if self.categorical_encoder is not None:
                categorical_data = self.categorical_encoder.transform(categorical_data)
            transformed_features.append(categorical_data)

        return np.hstack(transformed_features) if transformed_features else np.array([])

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform the input data.

        Args:
            df: Input DataFrame

        Returns:
            Transformed features as numpy array
        """
        self.fit(df)
        return self.transform(df)

    def get_feature_names(self) -> List[str]:
        """Get the names of the transformed features.

        Returns:
            List of feature names
        """
        feature_names = []
        
        if self.numeric_features:
            feature_names.extend(self.numeric_features)
            
        if self.categorical_features and self.categorical_encoder is not None:
            categorical_names = self.categorical_encoder.get_feature_names_out(
                self.categorical_features
            )
            feature_names.extend(categorical_names)
            
        return feature_names