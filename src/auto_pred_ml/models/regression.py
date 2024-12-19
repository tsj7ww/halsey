from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from .base import BaseModel


class RegressionModel(BaseModel):
    """Regression model implementation supporting multiple algorithms."""

    SUPPORTED_MODELS = {
        "random_forest": RandomForestRegressor,
        "gradient_boosting": GradientBoostingRegressor,
        "linear": LinearRegression,
        "ridge": Ridge,
        "lasso": Lasso
    }

    def __init__(
        self,
        model_type: str = "random_forest",
        model_params: Optional[Dict[str, Any]] = None
    ):
        """Initialize the regression model.

        Args:
            model_type: Type of model to use (from SUPPORTED_MODELS)
            model_params: Parameters for the selected model
        """
        super().__init__(model_params)
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.model_type = model_type

    def build(self) -> None:
        """Build the regression model."""
        model_class = self.SUPPORTED_MODELS[self.model_type]
        self.model = model_class(**self.model_params)

    def train(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        validation_data: Optional[tuple] = None
    ) -> Dict[str, float]:
        """Train the regression model.

        Args:
            X: Training features
            y: Training targets
            validation_data: Optional tuple of (X_val, y_val)

        Returns:
            Dictionary of training metrics
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        
        if self.model is None:
            self.build()

        self.model.fit(X, y)
        
        metrics = {}
        y_pred = self.model.predict(X)
        metrics["train_mse"] = mean_squared_error(y, y_pred)
        metrics["train_rmse"] = np.sqrt(metrics["train_mse"])
        metrics["train_mae"] = mean_absolute_error(y, y_pred)
        metrics["train_r2"] = r2_score(y, y_pred)
        
        if validation_data:
            X_val, y_val = validation_data
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            y_val_pred = self.model.predict(X_val)
            metrics["val_mse"] = mean_squared_error(y_val, y_val_pred)
            metrics["val_rmse"] = np.sqrt(metrics["val_mse"])
            metrics["val_mae"] = mean_absolute_error(y_val, y_val_pred)
            metrics["val_r2"] = r2_score(y_val, y_val_pred)
        
        return metrics

    def predict(
        self, X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Generate predictions for input data.

        Args:
            X: Input features

        Returns:
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return self.model.predict(X)