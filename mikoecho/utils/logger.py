"""
Logging Utilities

Provides logging with TensorBoard integration.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Logger with TensorBoard support."""
    
    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: str = "mikoecho",
        use_tensorboard: bool = True,
        log_level: int = logging.INFO,
    ):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory for logs
            experiment_name: Name of experiment
            use_tensorboard: Whether to use TensorBoard
            log_level: Logging level
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.use_tensorboard = use_tensorboard
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Python logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(log_level)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(self.log_dir / f"{experiment_name}.log")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # TensorBoard writer
        if use_tensorboard:
            tb_dir = self.log_dir / "tensorboard" / experiment_name
            self.writer = SummaryWriter(log_dir=str(tb_dir))
        else:
            self.writer = None
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = "",
    ):
        """
        Log metrics to TensorBoard.
        
        Args:
            metrics: Dictionary of metrics
            step: Training step
            prefix: Prefix for metric names
        """
        if self.writer is None:
            return
        
        for name, value in metrics.items():
            tag = f"{prefix}/{name}" if prefix else name
            self.writer.add_scalar(tag, value, step)
    
    def log_audio(
        self,
        audio: Any,
        step: int,
        tag: str = "audio",
        sample_rate: int = 22050,
    ):
        """
        Log audio to TensorBoard.
        
        Args:
            audio: Audio tensor
            step: Training step
            tag: Tag for audio
            sample_rate: Sample rate
        """
        if self.writer is None:
            return
        
        self.writer.add_audio(tag, audio, step, sample_rate=sample_rate)
    
    def log_image(
        self,
        image: Any,
        step: int,
        tag: str = "image",
    ):
        """
        Log image to TensorBoard.
        
        Args:
            image: Image tensor
            step: Training step
            tag: Tag for image
        """
        if self.writer is None:
            return
        
        self.writer.add_image(tag, image, step)
    
    def close(self):
        """Close logger and TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()
