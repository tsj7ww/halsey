"""Data cleaning utilities for preprocessing."""

from typing import List, Dict, Optional, Union, Any
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from datetime import datetime


class DataCleaner:
    """Utility class for cleaning and preprocessing data."""

    def __init__(
        self,
        remove_duplicates: bool = True,
        handle_missing: bool = True,
        drop_threshold: float = 0.9,
        date_columns: Optional[List[str]] = None
    ):
        """Initialize the data cleaner.

        Args:
            remove_duplicates: Whether to remove duplicate rows
            handle_missing: Whether to handle missing values
            drop_threshold: Drop columns with missing values above this threshold
            date_columns: List of columns to parse as dates
        """
        self.remove_duplicates = remove_duplicates
        self.handle_missing = handle_missing
        self.drop_threshold = drop_threshold
        self.date_columns = date_columns or []
        
        self.numeric_imputer = SimpleImputer(strategy="mean")
        self.categorical_imputer = SimpleImputer(strategy="most_frequent")
        self.columns_dropped: List[str] = []
        self.duplicates_removed: int = 0

    def clean(
        self,
        df: pd.DataFrame,
        numeric_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Clean the input DataFrame.

        Args:
            df: Input DataFrame
            numeric_columns: List of numeric column names
            categorical_columns: List of categorical column names

        Returns:
            Cleaned DataFrame
        """
        # Create a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Identify column types if not provided
        if numeric_columns is None:
            numeric_columns = cleaned_df.select_dtypes(
                include=['int64', 'float64']
            ).columns.tolist()
        
        if categorical_columns is None:
            categorical_columns = cleaned_df.select_dtypes(
                include=['object', 'category']
            ).columns.tolist()

        # Remove duplicates if requested
        if self.remove_duplicates:
            initial_rows = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            self.duplicates_removed = initial_rows - len(cleaned_df)

        # Handle missing values
        if self.handle_missing:
            # Drop columns with too many missing values
            missing_ratios = cleaned_df.isnull().sum() / len(cleaned_df)
            cols_to_drop = missing_ratios[missing_ratios > self.drop_threshold].index
            cleaned_df = cleaned_df.drop(columns=cols_to_drop)
            self.columns_dropped.extend(cols_to_drop)

            # Impute remaining missing values
            for col in numeric_columns:
                if col in cleaned_df.columns and cleaned_df[col].isnull().any():
                    cleaned_df[col] = self.numeric_imputer.fit_transform(
                        cleaned_df[[col]]
                    )

            for col in categorical_columns:
                if col in cleaned_df.columns and cleaned_df[col].isnull().any():
                    cleaned_df[col] = self.categorical_imputer.fit_transform(
                        cleaned_df[[col]]
                    )

        # Parse date columns
        for col in self.date_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = pd.to_datetime(
                    cleaned_df[col],
                    errors='coerce'
                )

        return cleaned_df

    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = "zscore",
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """Remove outliers from specified columns.

        Args:
            df: Input DataFrame
            columns: Columns to check for outliers
            method: Method to use ('zscore' or 'iqr')
            threshold: Threshold for outlier detection

        Returns:
            DataFrame with outliers removed
        """
        cleaned_df = df.copy()
        
        for col in columns:
            if col not in cleaned_df.columns:
                continue
                
            if method == "zscore":
                z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
                cleaned_df = cleaned_df[z_scores < threshold]
            
            elif method == "iqr":
                Q1 = cleaned_df[col].quantile(0.25)
                Q3 = cleaned_df[col].quantile(0.75)
                IQR = Q3 - Q1
                cleaned_df = cleaned_df[
                    (cleaned_df[col] >= Q1 - threshold * IQR) &
                    (cleaned_df[col] <= Q3 + threshold * IQR)
                ]

        return cleaned_df

    def get_cleaning_report(self) -> Dict[str, Any]:
        """Get a report of the cleaning operations performed.

        Returns:
            Dictionary containing cleaning statistics
        """
        return {
            "duplicates_removed": self.duplicates_removed,
            "columns_dropped": self.columns_dropped,
            "timestamp": datetime.now().isoformat()
        }