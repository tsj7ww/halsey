from typing import Dict, Any, Optional, Union, List
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError


class DataValidator:
    """Utility for validating input data and parameters."""

    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
        numeric_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None
    ) -> None:
        """Validate DataFrame structure and contents.

        Args:
            df: Input DataFrame
            required_columns: List of required column names
            numeric_columns: List of columns that should be numeric
            categorical_columns: List of columns that should be categorical

        Raises:
            ValueError: If validation fails
        """
        if df.empty:
            raise ValueError("DataFrame is empty")

        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

        if numeric_columns:
            non_numeric = [col for col in numeric_columns if not pd.api.types.is_numeric_dtype(df[col])]
            if non_numeric:
                raise ValueError(f"Columns must be numeric: {non_numeric}")

        if categorical_columns:
            non_categorical = [
                col for col in categorical_columns
                if not pd.api.types.is_categorical_dtype(df[col]) and not pd.api.types.is_object_dtype(df[col])
            ]
            if non_categorical:
                raise ValueError(f"Columns must be categorical: {non_categorical}")

    @staticmethod
    def validate_model_params(
        params: Dict[str, Any],
        required_params: Optional[List[str]] = None,
        param_types: Optional[Dict[str, type]] = None
    ) -> None:
        """Validate model parameters.

        Args:
            params: Model parameters
            required_params: List of required parameter names
            param_types: Dictionary of parameter names and their expected types

        Raises:
            ValueError: If validation fails
        """
        if required_params:
            missing_params = set(required_params) - set(params.keys())
            if missing_params:
                raise ValueError(f"Missing required parameters: {missing_params}")

        if param_types:
            for param_name, expected_type in param_types.items():
                if param_name in params:
                    if not isinstance(params[param_name], expected_type):
                        raise ValueError(
                            f"Parameter {param_name} must be of type {expected_type}"
                        )

    @staticmethod
    def validate_feature_names(
        feature_names: List[str],
        expected_names: List[str]
    ) -> None:
        """Validate feature names match expected names.

        Args:
            feature_names: List of feature names
            expected_names: List of expected feature names

        Raises:
            ValueError: If validation fails
        """
        if set(feature_names) != set(expected_names):
            raise ValueError(
                f"Feature names mismatch. Expected: {expected_names}, Got: {feature_names}"
            )

    @staticmethod
    def validate_predictions(
        predictions: np.ndarray,
        expected_shape: Optional[tuple] = None,
        valid_classes: Optional[List[Any]] = None
    ) -> None:
        """Validate model predictions.

        Args:
            predictions: Model predictions
            expected_shape: Expected shape of predictions
            valid_classes: List of valid class labels for classification

        Raises:
            ValueError: If validation fails
        """
        if expected_shape and predictions.shape != expected_shape:
            raise ValueError(
                f"Predictions shape mismatch. Expected: {expected_shape}, Got: {predictions.shape}"
            )

        if valid_classes is not None:
            invalid_classes = set(np.unique(predictions)) - set(valid_classes)
            if invalid_classes:
                raise ValueError(f"Invalid prediction classes found: {invalid_classes}")