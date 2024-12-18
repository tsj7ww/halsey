import logging
import sys
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime


class ModelLogger:
    """Logging utility for model operations."""

    def __init__(
        self,
        name: str = "predml",
        log_file: Optional[str] = None,
        level: int = logging.INFO
    ):
        """Initialize the logger.

        Args:
            name: Logger name
            log_file: Optional path to log file
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []  # Remove any existing handlers
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        
        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Add file handler if log_file is specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    def log_model_training(
        self,
        model_type: str,
        params: Dict[str, Any],
        metrics: Dict[str, float]
    ) -> None:
        """Log model training information.

        Args:
            model_type: Type of model being trained
            params: Model parameters
            metrics: Training metrics
        """
        self.logger.info(f"Training {model_type} model")
        self.logger.info(f"Parameters: {params}")
        self.logger.info(f"Metrics: {metrics}")

    def log_prediction(
        self,
        model_type: str,
        input_shape: tuple,
        prediction_time: float
    ) -> None:
        """Log prediction information.

        Args:
            model_type: Type of model used
            input_shape: Shape of input data
            prediction_time: Time taken for prediction
        """
        self.logger.info(
            f"Made predictions using {model_type} model on data with shape {input_shape}"
        )
        self.logger.info(f"Prediction time: {prediction_time:.4f} seconds")

    def log_error(
        self,
        error: Exception,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log error information.

        Args:
            error: Exception that occurred
            additional_info: Additional context about the error
        """
        self.logger.error(f"Error occurred: {str(error)}")
        if additional_info:
            self.logger.error(f"Additional context: {additional_info}")

    def log_feature_engineering(
        self,
        n_features_in: int,
        n_features_out: int,
        transformations: Dict[str, Any]
    ) -> None:
        """Log feature engineering information.

        Args:
            n_features_in: Number of input features
            n_features_out: Number of output features
            transformations: Applied transformations
        """
        self.logger.info(
            f"Feature engineering: {n_features_in} input features -> {n_features_out} output features"
        )
        self.logger.info(f"Applied transformations: {transformations}")