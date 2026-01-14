"""
Style Encoder Module

Encodes emotion and style information for expressive voice synthesis.
"""

import torch
import torch.nn as nn
from typing import Optional, List


class StyleEncoder(nn.Module):
    """
    Emotion and style embedding layer.
    
    Supports multiple emotions: neutral, calm, excited, sad, energetic.
    """
    
    # Emotion mapping
    EMOTIONS = {
        'neutral': 0,
        'calm': 1,
        'excited': 2,
        'sad': 3,
        'energetic': 4,
    }
    
    def __init__(
        self,
        emotion_dim: int = 64,
        num_emotions: int = 5,
        hidden_dim: int = 256,
        use_reference_audio: bool = True,
    ):
        """
        Initialize style encoder.
        
        Args:
            emotion_dim: Dimension of emotion embeddings
            num_emotions: Number of emotion categories
            hidden_dim: Hidden layer dimension for reference audio encoding
            use_reference_audio: Whether to extract style from reference audio
        """
        super().__init__()
        
        self.emotion_dim = emotion_dim
        self.num_emotions = num_emotions
        self.hidden_dim = hidden_dim
        self.use_reference_audio = use_reference_audio
        
        # Learnable emotion embeddings
        self.emotion_embeddings = nn.Embedding(num_emotions, emotion_dim)
        
        # Reference audio style encoder (optional)
        if use_reference_audio:
            self.reference_encoder = nn.Sequential(
                nn.Conv1d(80, hidden_dim, kernel_size=3, padding=1),  # 80 = n_mels
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.reference_proj = nn.Linear(hidden_dim, emotion_dim)
        
        # Style fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(emotion_dim, emotion_dim),
            nn.LayerNorm(emotion_dim),
            nn.Tanh(),
        )
    
    def forward(
        self,
        emotion: Optional[torch.Tensor] = None,
        reference_mel: Optional[torch.Tensor] = None,
        emotion_strength: float = 1.0,
    ) -> torch.Tensor:
        """
        Encode style and emotion.
        
        Args:
            emotion: Emotion indices [batch] or emotion names
            reference_mel: Reference mel-spectrogram [batch, n_mels, time]
            emotion_strength: Strength of emotion (0-1)
            
        Returns:
            style_embedding: Style embedding [batch, emotion_dim]
        """
        batch_size = 1
        
        # Get emotion embedding
        if emotion is not None:
            if isinstance(emotion, (list, tuple)):
                # Convert emotion names to indices
                emotion_indices = torch.tensor(
                    [self.EMOTIONS[e] for e in emotion],
                    dtype=torch.long,
                    device=self.emotion_embeddings.weight.device,
                )
            else:
                emotion_indices = emotion
            
            batch_size = emotion_indices.shape[0]
            emotion_emb = self.emotion_embeddings(emotion_indices)
            emotion_emb = emotion_emb * emotion_strength
        else:
            # Default to neutral
            device = self.emotion_embeddings.weight.device
            if reference_mel is not None:
                batch_size = reference_mel.shape[0]
            emotion_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
            emotion_emb = self.emotion_embeddings(emotion_indices)
        
        # Get reference style embedding
        if reference_mel is not None and self.use_reference_audio:
            # Encode reference mel-spectrogram
            ref_features = self.reference_encoder(reference_mel)  # [batch, hidden_dim, 1]
            ref_features = ref_features.squeeze(-1)  # [batch, hidden_dim]
            ref_emb = self.reference_proj(ref_features)  # [batch, emotion_dim]
            
            # Fuse emotion and reference style
            style_embedding = emotion_emb + ref_emb
        else:
            style_embedding = emotion_emb
        
        # Apply fusion layer
        style_embedding = self.fusion(style_embedding)
        
        return style_embedding
    
    def get_emotion_embedding(self, emotion_name: str) -> torch.Tensor:
        """
        Get embedding for a specific emotion.
        
        Args:
            emotion_name: Name of emotion ('neutral', 'calm', etc.)
            
        Returns:
            Emotion embedding [emotion_dim]
        """
        if emotion_name not in self.EMOTIONS:
            raise ValueError(f"Unknown emotion: {emotion_name}. "
                           f"Available: {list(self.EMOTIONS.keys())}")
        
        emotion_idx = torch.tensor(
            [self.EMOTIONS[emotion_name]],
            dtype=torch.long,
            device=self.emotion_embeddings.weight.device,
        )
        
        embedding = self.emotion_embeddings(emotion_idx)
        return embedding.squeeze(0)
    
    def interpolate_emotions(
        self,
        emotion1: str,
        emotion2: str,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        """
        Interpolate between two emotions.
        
        Args:
            emotion1: First emotion name
            emotion2: Second emotion name
            alpha: Interpolation factor (0 = emotion1, 1 = emotion2)
            
        Returns:
            Interpolated emotion embedding [emotion_dim]
        """
        emb1 = self.get_emotion_embedding(emotion1)
        emb2 = self.get_emotion_embedding(emotion2)
        
        interpolated = (1 - alpha) * emb1 + alpha * emb2
        interpolated = self.fusion(interpolated.unsqueeze(0))
        
        return interpolated.squeeze(0)
    
    @staticmethod
    def get_available_emotions() -> List[str]:
        """Get list of available emotions."""
        return list(StyleEncoder.EMOTIONS.keys())
