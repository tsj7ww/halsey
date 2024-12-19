"""
Base Model Module
===============

This module defines the abstract base class for all models in the package.
It establishes the common interface and shared functionality for model training,
prediction, and evaluation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
import joblib
import json
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseModel(ABC, BaseEstimator):
    """Abstract base class for all models.
    
    This class defines the interface that all models must implement
    and provides common utility methods for model operations.
    """
    
    def __init__(self,
                 name: str,
                 problem_type: str,
                 params: Optional[Dict] = None,
                 use_gpu: bool = False,
                 random_state: int = 42):
        """Initialize the model.
        
        Args:
            name: Model name
            problem_type: Type of problem ('classification' or 'regression')
            params: Model parameters
            use_gpu: Whether to use GPU acceleration
            random_state: Random state for reproducibility
        """
        self.name = name
        self.problem_type = problem_type
        self.params = params or {}
        self.use_gpu = use_gpu
        self.random_state = random_state
        
        # State tracking
        self.is_fitted = False
        self.feature_names_: Optional[List[str]] = None
        self.feature_importance_: Optional[Dict[str, float]] = None
        self.training_history_: List[Dict] = []
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate initialization parameters."""
        valid_problems = {'classification', 'regression'}
        if self.problem_type not in valid_problems:
            raise ValueError(f"problem_type must be one of: {valid_problems}")
    
    @abstractmethod
    def _create_model(self) -> Any:
        """Create the underlying model instance.
        
        Returns:
            Model instance
        """
        pass
    
    @abstractmethod
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass
    
    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
            **kwargs) -> 'BaseModel':
        """Fit the model to the data.
        
        Args:
            X: Training features
            y: Target variable
            validation_data: Optional tuple of (X_val, y_val)
            **kwargs: Additional fitting parameters
            
        Returns:
            self: Fitted model
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        
        # Store feature names
        self.feature_names_ = list(X.columns)
        
        # Track training start
        start_time = datetime.now()
        
        try:
            self._fit_implementation(X, y, validation_data, **kwargs)
            
            # Calculate feature importance
            self.feature_importance_ = self._get_feature_importance()
            
            # Record training metadata
            self.training_history_.append({
                'timestamp': start_time.isoformat(),
                'duration': (datetime.now() - start_time).total_seconds(),
                'n_samples': len(X),
                'n_features': len(self.feature_names_),
                'parameters': self.get_params()
            })
            
            self.is_fitted = True
            
        except Exception as e:
            logger.error(f"Error fitting model: {str(e)}")
            raise
        
        return self
    
    @abstractmethod
    def _fit_implementation(self,
                          X: pd.DataFrame,
                          y: pd.Series,
                          validation_data: Optional[Tuple[pd.DataFrame, pd.Series]],
                          **kwargs):
        """Implementation of model fitting.
        
        Args:
            X: Training features
            y: Target variable
            validation_data: Optional tuple of (X_val, y_val)
            **kwargs: Additional fitting parameters
        """
        pass
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for given data.
        
        Args:
            X: Input features
            
        Returns:
            Model predictions
        """
        self._validate_prediction_input(X)
        return self._predict_implementation(X)
    
    @abstractmethod
    def _predict_implementation(self, X: pd.DataFrame) -> np.ndarray:
        """Implementation of prediction logic.
        
        Args:
            X: Input features
            
        Returns:
            Model predictions
        """
        pass
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate probability predictions for classification.
        
        Args:
            X: Input features
            
        Returns:
            Probability predictions
        """
        if self.problem_type != 'classification':
            raise ValueError("predict_proba is only available for classification")
            
        self._validate_prediction_input(X)
        return self._predict_proba_implementation(X)
    
    @abstractmethod
    def _predict_proba_implementation(self, X: pd.DataFrame) -> np.ndarray:
        """Implementation of probability prediction logic.
        
        Args:
            X: Input features
            
        Returns:
            Probability predictions
        """
        pass
    
    def _validate_prediction_input(self, X: pd.DataFrame):
        """Validate input data for prediction.
        
        Args:
            X: Input features to validate
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
            
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
            
        missing_features = set(self.feature_names_) - set(X.columns)
        if missing_features:
            raise ValueError(f"Features missing from input: {missing_features}")
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the fitted model to disk.
        
        Args:
            path: Path to save the model
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model")
            
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save model and metadata separately
            model_path = path.with_suffix('.joblib')
            meta_path = path.with_suffix('.json')
            
            # Save model
            joblib.dump(self, model_path)
            
            # Save metadata
            metadata = {
                'name': self.name,
                'problem_type': self.problem_type,
                'feature_names': self.feature_names_,
                'feature_importance': self.feature_importance_,
                'training_history': self.training_history_,
                'parameters': self.get_params()
            }
            
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=4)
                
            logger.info(f"Saved model to {model_path} and metadata to {meta_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'BaseModel':
        """Load a fitted model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded model
        """
        try:
            model_path = Path(path).with_suffix('.joblib')
            model = joblib.load(model_path)
            
            if not isinstance(model, cls):
                raise TypeError(f"Loaded object is not a {cls.__name__}")
                
            logger.info(f"Loaded model from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters.
        
        Args:
            deep: Whether to return parameters of nested objects
            
        Returns:
            Dictionary of parameters
        """
        params = {
            'name': self.name,
            'problem_type': self.problem_type,
            'params': self.params.copy(),
            'use_gpu': self.use_gpu,
            'random_state': self.random_state
        }
        return params
    
    def set_params(self, **params) -> 'BaseModel':
        """Set model parameters.
        
        Args:
            **params: Parameters to set
            
        Returns:
            self: Model with updated parameters
        """
        if self.is_fitted:
            raise RuntimeError("Cannot set parameters on fitted model")
            
        valid_params = self.get_params()
        
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(f"Invalid parameter {key}")
            setattr(self, key, value)
            
        self._validate_parameters()
        return self
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata.
        
        Returns:
            Dictionary containing model metadata
        """
        return {
            'name': self.name,
            'problem_type': self.problem_type,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names_,
            'feature_importance': self.feature_importance_,
            'training_history': self.training_history_,
            'parameters': self.get_params()
        }