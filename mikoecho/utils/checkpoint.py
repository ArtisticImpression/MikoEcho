"""
Checkpoint Utilities

Handles model checkpointing and loading.
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional
import shutil


class CheckpointManager:
    """Manages model checkpoints."""
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        keep_last_n: int = 3,
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoints
            keep_last_n: Number of recent checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_last_n = keep_last_n
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Track checkpoints
        self.checkpoints = []
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
    ) -> Path:
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            step: Current step
            metrics: Optional metrics dictionary
            is_best: Whether this is the best checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics or {},
        }
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch{epoch}_step{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Track checkpoint
        self.checkpoints.append(checkpoint_path)
        
        # Save as best if specified
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            shutil.copy(checkpoint_path, best_path)
        
        # Save as latest
        latest_path = self.checkpoint_dir / "latest_model.pt"
        shutil.copy(checkpoint_path, latest_path)
        
        # Clean old checkpoints
        self._cleanup_checkpoints()
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cpu',
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            model: Model to load weights into
            optimizer: Optional optimizer to load state into
            device: Device to load on
            
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint
    
    def load_latest(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cpu',
    ) -> Optional[Dict[str, Any]]:
        """
        Load latest checkpoint.
        
        Args:
            model: Model to load weights into
            optimizer: Optional optimizer
            device: Device to load on
            
        Returns:
            Checkpoint dictionary or None if no checkpoint exists
        """
        latest_path = self.checkpoint_dir / "latest_model.pt"
        
        if not latest_path.exists():
            return None
        
        return self.load_checkpoint(str(latest_path), model, optimizer, device)
    
    def load_best(
        self,
        model: torch.nn.Module,
        device: str = 'cpu',
    ) -> Optional[Dict[str, Any]]:
        """
        Load best checkpoint.
        
        Args:
            model: Model to load weights into
            device: Device to load on
            
        Returns:
            Checkpoint dictionary or None if no checkpoint exists
        """
        best_path = self.checkpoint_dir / "best_model.pt"
        
        if not best_path.exists():
            return None
        
        return self.load_checkpoint(str(best_path), model, None, device)
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints keeping only last N."""
        if len(self.checkpoints) <= self.keep_last_n:
            return
        
        # Sort by modification time
        self.checkpoints.sort(key=lambda p: p.stat().st_mtime)
        
        # Remove oldest checkpoints
        to_remove = self.checkpoints[:-self.keep_last_n]
        for checkpoint_path in to_remove:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
        
        # Update tracking
        self.checkpoints = self.checkpoints[-self.keep_last_n:]
