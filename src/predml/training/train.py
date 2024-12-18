from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold

from ..models.base import BaseModel
from ..preprocessing.feature_engineering import FeatureEngineer


class ModelTrainer:
    """Utility class for training and evaluating models."""

    def __init__(
        self,
        model: BaseModel,
        feature_engineer: Optional[FeatureEngineer] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """Initialize the model trainer.

        Args:
            model: Model instance to train
            feature_engineer: Optional feature engineering pipeline
            test_size: Proportion of data to use for validation
            random_state: Random seed for reproducibility
        """
        self.model = model
        self.feature_engineer = feature_engineer
        self.test_size = test_size
        self.random_state = random_state
        self.feature_names = None

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare the data for training.

        Args:
            df: Input DataFrame
            target_column: Name of the target column

        Returns:
            Tuple of (features, target)
        """
        X = df.drop(columns=[target_column])
        y = df[target_column]

        if self.feature_engineer is not None:
            X = self.feature_engineer.fit_transform(X)
            self.feature_names = self.feature_engineer.get_feature_names()
        else:
            self.feature_names = X.columns.tolist()
        
        return X, y

    def train(
        self,
        df: pd.DataFrame,
        target_column: str,
        stratify: Optional[bool] = None,
        cv_folds: Optional[int] = None
    ) -> Dict[str, Any]:
        """Train the model and evaluate performance.

        Args:
            df: Input DataFrame
            target_column: Name of the target column
            stratify: Whether to use stratified splitting for classification
            cv_folds: Number of cross-validation folds (if None, uses train/test split)

        Returns:
            Dictionary of training metrics and model info
        """
        X, y = self.prepare_data(df, target_column)
        
        if cv_folds is not None:
            return self._train_with_cv(X, y, cv_folds, stratify)
        else:
            return self._train_with_split(X, y, stratify)

    def _train_with_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        stratify: Optional[bool]
    ) -> Dict[str, Any]:
        """Train using train/test split.

        Args:
            X: Feature matrix
            y: Target vector
            stratify: Whether to use stratified splitting

        Returns:
            Training metrics and model info
        """
        stratify_param = y if stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )

        metrics = self.model.train(X_train, y_train, validation_data=(X_val, y_val))
        
        return {
            "metrics": metrics,
            "feature_names": self.feature_names,
            "model_type": self.model.__class__.__name__,
            "model_params": self.model.model_params
        }

    def _train_with_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int,
        stratify: Optional[bool]
    ) -> Dict[str, Any]:
        """Train using cross-validation.

        Args:
            X: Feature matrix
            y: Target vector
            cv_folds: Number of CV folds
            stratify: Whether to use stratified CV

        Returns:
            Training metrics and model info
        """
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state) if stratify else KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        fold_metrics = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            fold_result = self.model.train(X_train, y_train, validation_data=(X_val, y_val))
            fold_metrics.append(fold_result)

        # Aggregate CV metrics
        cv_metrics = {}
        for metric in fold_metrics[0].keys():
            values = [fold[metric] for fold in fold_metrics]
            cv_metrics[f"{metric}_mean"] = np.mean(values)
            cv_metrics[f"{metric}_std"] = np.std(values)

        return {
            "metrics": cv_metrics,
            "fold_metrics": fold_metrics,
            "feature_names": self.feature_names,
            "model_type": self.model.__class__.__name__,
            "model_params": self.model.model_params
        }