"""
Content Disentangler Module

Separates content from speaker identity to enable clean voice conversion.
"""

import torch
import torch.nn as nn
from typing import Optional


class ContentDisentangler(nn.Module):
    """
    Neural module that disentangles content from speaker identity.
    
    Uses adversarial training and information bottleneck to ensure
    content features are speaker-agnostic.
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize content disentangler.
        
        Args:
            input_dim: Input feature dimension (from speech encoder)
            hidden_dim: Hidden layer dimension
            num_layers: Number of transformer layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Transformer encoder for content refinement
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Speaker classifier (for adversarial training)
        self.speaker_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 256),  # 256 speaker classes
        )
    
    def forward(
        self,
        content_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Disentangle content from speaker identity.
        
        Args:
            content_features: Content features from speech encoder [batch, seq_len, input_dim]
            attention_mask: Attention mask [batch, seq_len]
            
        Returns:
            disentangled_content: Speaker-agnostic content [batch, seq_len, hidden_dim]
        """
        # Project to hidden dimension
        x = self.input_proj(content_features)
        
        # Create attention mask for transformer (True = masked)
        if attention_mask is not None:
            # Convert from [batch, seq_len] to [batch, seq_len] boolean
            # where True means "ignore this position"
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        # Apply transformer
        x = self.transformer(
            x,
            src_key_padding_mask=src_key_padding_mask,
        )
        
        # Layer normalization
        disentangled_content = self.layer_norm(x)
        
        return disentangled_content
    
    def classify_speaker(
        self,
        content_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Classify speaker from content features (for adversarial training).
        
        This is used during training to ensure content features don't
        contain speaker information.
        
        Args:
            content_features: Content features [batch, seq_len, hidden_dim]
            
        Returns:
            speaker_logits: Speaker classification logits [batch, num_speakers]
        """
        # Pool over sequence dimension
        pooled = content_features.mean(dim=1)
        
        # Classify speaker
        speaker_logits = self.speaker_classifier(pooled)
        
        return speaker_logits
    
    def compute_information_bottleneck_loss(
        self,
        content_features: torch.Tensor,
        speaker_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute information bottleneck loss to minimize speaker info in content.
        
        Args:
            content_features: Content features [batch, seq_len, hidden_dim]
            speaker_embeddings: Speaker embeddings [batch, embedding_dim]
            
        Returns:
            loss: Information bottleneck loss
        """
        # Pool content features
        pooled_content = content_features.mean(dim=1)  # [batch, hidden_dim]
        
        # Normalize
        pooled_content = nn.functional.normalize(pooled_content, p=2, dim=-1)
        speaker_embeddings = nn.functional.normalize(speaker_embeddings, p=2, dim=-1)
        
        # Compute similarity (should be minimized)
        similarity = torch.sum(pooled_content * speaker_embeddings, dim=-1)
        
        # Loss is the absolute similarity (we want it close to 0)
        loss = torch.abs(similarity).mean()
        
        return loss
