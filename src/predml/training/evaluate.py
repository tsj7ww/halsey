from typing import Dict, List, Optional, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error,
    roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class ClassificationMetrics:
    """Container for classification metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: Optional[float] = None
    confusion_matrix: Optional[np.ndarray] = None


@dataclass
class RegressionMetrics:
    """Container for regression metrics."""
    mse: float
    rmse: float
    mae: float
    r2: float
    explained_variance: float


class ModelEvaluator:
    """Utility class for model evaluation."""

    def __init__(self, task_type: str = "classification"):
        """Initialize the evaluator.

        Args:
            task_type: Type of task ("classification" or "regression")
        """
        if task_type not in ["classification", "regression"]:
            raise ValueError("Task type must be 'classification' or 'regression'")
        self.task_type = task_type

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Union[ClassificationMetrics, RegressionMetrics]:
        """Evaluate model predictions.

        Args:
            y_true: True labels/values
            y_pred: Predicted labels/values
            y_prob: Predicted probabilities (for classification)

        Returns:
            Metrics container with computed metrics
        """
        if self.task_type == "classification":
            return self._evaluate_classification(y_true, y_pred, y_prob)
        else:
            return self._evaluate_regression(y_true, y_pred)

    def _evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> ClassificationMetrics:
        """Compute classification metrics."""
        metrics = ClassificationMetrics(
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, average='weighted'),
            recall=recall_score(y_true, y_pred, average='weighted'),
            f1=f1_score(y_true, y_pred, average='weighted'),
            confusion_matrix=confusion_matrix(y_true, y_pred)
        )
        
        if y_prob is not None:
            if y_prob.shape[1] == 2:  # Binary classification
                metrics.auc_roc = roc_auc_score(y_true, y_prob[:, 1])
            else:  # Multi-class
                metrics.auc_roc = roc_auc_score(y_true, y_prob, multi_class='ovr')
        
        return metrics

    def _evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> RegressionMetrics:
        """Compute regression metrics."""
        mse = mean_squared_error(y_true, y_pred)
        return RegressionMetrics(
            mse=mse,
            rmse=np.sqrt(mse),
            mae=mean_absolute_error(y_true, y_pred),
            r2=r2_score(y_true, y_pred),
            explained_variance=1 - (np.var(y_true - y_pred) / np.var(y_true))
        )

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None
    ) -> None:
        """Plot confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Optional label names
        """
        if self.task_type != "classification":
            raise ValueError("Confusion matrix is only for classification tasks")
            
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> None:
        """Plot residuals for regression.

        Args:
            y_true: True values
            y_pred: Predicted values
        """
        if self.task_type != "regression":
            raise ValueError("Residual plot is only for regression tasks")
            
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Residual Plot')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.show()