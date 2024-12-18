from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class BaseModel(ABC):
    """Base class for all prediction models."""

    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """Initialize the model with parameters.

        Args:
            model_params: Dictionary of model parameters
        """
        self.model_params = model_params or {}
        self.model: Optional[BaseEstimator] = None
        self.feature_names: Optional[list] = None

    @abstractmethod
    def build(self) -> None:
        """Build the underlying model architecture."""
        pass

    @abstractmethod
    def train(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        validation_data: Optional[tuple] = None,
    ) -> Dict[str, float]:
        """Train the model on the provided data.

        Args:
            X: Training features
            y: Training targets
            validation_data: Optional tuple of (X_val, y_val)

        Returns:
            Dictionary of training metrics
        """
        pass

    @abstractmethod
    def predict(
        self, X: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.Series]:
        """Generate predictions for the input data.

        Args:
            X: Input features

        Returns:
            Model predictions
        """
        pass

    def save(self, path: str) -> None:
        """Save the model to disk.

        Args:
            path: Path to save the model
        """
        import joblib
        
        if self.model is None:
            raise ValueError("Model has not been built yet")
        
        model_data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "model_params": self.model_params
        }
        joblib.dump(model_data, path)

    def load(self, path: str) -> None:
        """Load the model from disk.

        Args:
            path: Path to load the model from
        """
        import joblib
        
        model_data = joblib.load(path)
        self.model = model_data["model"]
        self.feature_names = model_data["feature_names"]
        self.model_params = model_data["model_params"]