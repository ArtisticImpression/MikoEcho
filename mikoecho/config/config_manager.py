"""
Configuration Manager for MikoEcho

Handles loading and validation of YAML configuration files.
"""

import os
from typing import Any, Dict, Optional
from pathlib import Path
import yaml
from omegaconf import OmegaConf, DictConfig


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to YAML config file. If None, uses default.
        """
        if config_path is None:
            # Use default config
            config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> DictConfig:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert to OmegaConf for dot notation access
        config = OmegaConf.create(config_dict)
        
        # Validate config
        self._validate_config(config)
        
        return config
    
    def _validate_config(self, config: DictConfig) -> None:
        """Validate configuration structure and values."""
        required_sections = ['model', 'training', 'data', 'inference']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
        
        # Validate model config
        if config.model.vocoder.sample_rate not in [16000, 22050, 24000, 44100]:
            raise ValueError(f"Invalid sample rate: {config.model.vocoder.sample_rate}")
        
        # Validate training config
        if config.training.batch_size < 1:
            raise ValueError("Batch size must be >= 1")
        
        if config.training.num_epochs < 1:
            raise ValueError("Number of epochs must be >= 1")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'model.vocoder.sample_rate')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        try:
            return OmegaConf.select(self.config, key, default=default)
        except Exception:
            return default
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates
        """
        self.config = OmegaConf.merge(self.config, updates)
    
    def save(self, output_path: Optional[str] = None) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            output_path: Output path. If None, overwrites original.
        """
        if output_path is None:
            output_path = self.config_path
        
        with open(output_path, 'w') as f:
            yaml.dump(OmegaConf.to_container(self.config), f, default_flow_style=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return OmegaConf.to_container(self.config, resolve=True)
    
    @property
    def model_config(self) -> DictConfig:
        """Get model configuration."""
        return self.config.model
    
    @property
    def training_config(self) -> DictConfig:
        """Get training configuration."""
        return self.config.training
    
    @property
    def data_config(self) -> DictConfig:
        """Get data configuration."""
        return self.config.data
    
    @property
    def inference_config(self) -> DictConfig:
        """Get inference configuration."""
        return self.config.inference
