import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
import yaml


@dataclass
class ModelConfig:
    """Model configuration container."""
    model_type: str
    model_params: Dict[str, Any]
    feature_engineering: Dict[str, Any]
    training: Dict[str, Any]


class ConfigManager:
    """Configuration management utility."""

    @staticmethod
    def load_config(config_path: str) -> ModelConfig:
        """Load configuration from YAML file.

        Args:
            config_path: Path to config file

        Returns:
            ModelConfig instance
        """
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        return ModelConfig(
            model_type=config_dict['model_type'],
            model_params=config_dict.get('model_params', {}),
            feature_engineering=config_dict.get('feature_engineering', {}),
            training=config_dict.get('training', {})
        )

    @staticmethod
    def save_config(config: ModelConfig, config_path: str) -> None:
        """Save configuration to YAML file.

        Args:
            config: ModelConfig instance
            config_path: Path to save config
        """
        config_dict = {
            'model_type': config.model_type,
            'model_params': config.model_params,
            'feature_engineering': config.feature_engineering,
            'training': config.training
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


# Default configurations
DEFAULT_CLASSIFICATION_CONFIG = ModelConfig(
    model_type="random_forest",
    model_params={
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2
    },
    feature_engineering={
        "scaling_method": "standard",
        "handle_missing": True
    },
    training={
        "test_size": 0.2,
        "random_state": 42,
        "cv_folds": 5
    }
)

DEFAULT_REGRESSION_CONFIG = ModelConfig(
    model_type="random_forest",
    model_params={
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2
    },
    feature_engineering={
        "scaling_method": "standard",
        "handle_missing": True
    },
    training={
        "test_size": 0.2,
        "random_state": 42,
        "cv_folds": 5
    }
)