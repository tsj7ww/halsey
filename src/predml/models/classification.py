from typing import Any, Dict, Optional, Union, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .base import BaseModel


class ClassificationModel(BaseModel):
    """Classification model implementation supporting multiple algorithms."""

    SUPPORTED_MODELS = {
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "logistic_regression": LogisticRegression
    }

    def __init__(
        self,
        model_type: str = "random_forest",
        model_params: Optional[Dict[str, Any]] = None,
        class_weights: Optional[Dict[int, float]] = None
    ):
        """Initialize the classification model.

        Args:
            model_type: Type of model to use (from SUPPORTED_MODELS)
            model_params: Parameters for the selected model
            class_weights: Optional weights for imbalanced classes
        """
        super().__init__(model_params)
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.model_type = model_type
        self.class_weights = class_weights

    def build(self) -> None:
        """Build the classification model."""
        model_class = self.SUPPORTED_MODELS[self.model_type]
        params = self.model_params.copy()
        
        if self.class_weights:
            params["class_weight"] = self.class_weights
            
        self.model = model_class(**params)

    def train(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        validation_data: Optional[tuple] = None
    ) -> Dict[str, float]:
        """Train the classification model.

        Args:
            X: Training features
            y: Training labels
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
        metrics["train_accuracy"] = accuracy_score(y, y_pred)
        metrics["train_f1"] = f1_score(y, y_pred, average="weighted")
        
        if validation_data:
            X_val, y_val = validation_data
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            y_val_pred = self.model.predict(X_val)
            metrics["val_accuracy"] = accuracy_score(y_val, y_val_pred)
            metrics["val_f1"] = f1_score(y_val, y_val_pred, average="weighted")
        
        return metrics

    def predict(
        self, X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Generate predictions for input data.

        Args:
            X: Input features

        Returns:
            Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return self.model.predict(X)

    def predict_proba(
        self, X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Generate class probabilities for input data.

        Args:
            X: Input features

        Returns:
            Class probabilities
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return self.model.predict_proba(X)