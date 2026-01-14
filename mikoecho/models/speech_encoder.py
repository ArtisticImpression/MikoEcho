"""
Speech Encoder Module

Uses HuBERT for self-supervised speech encoding to extract content representations
while removing speaker identity information.
"""

import torch
import torch.nn as nn
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from typing import Optional, Tuple


class SpeechEncoder(nn.Module):
    """
    Self-supervised speech encoder using HuBERT.
    
    Extracts content representations from audio while being speaker-agnostic.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/hubert-large-ls960-ft",
        freeze: bool = True,
        output_dim: int = 1024,
    ):
        """
        Initialize speech encoder.
        
        Args:
            model_name: HuBERT model name from HuggingFace
            freeze: Whether to freeze HuBERT weights
            output_dim: Output dimension for content features
        """
        super().__init__()
        
        self.model_name = model_name
        self.freeze = freeze
        self.output_dim = output_dim
        
        # Load pre-trained HuBERT
        self.hubert = HubertModel.from_pretrained(model_name)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        
        # Freeze HuBERT if specified
        if freeze:
            for param in self.hubert.parameters():
                param.requires_grad = False
        
        # Get HuBERT output dimension
        hubert_dim = self.hubert.config.hidden_size
        
        # Projection layer to desired output dimension
        self.projection = nn.Sequential(
            nn.Linear(hubert_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )
    
    def forward(
        self,
        audio: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract content features from audio.
        
        Args:
            audio: Input audio tensor [batch, time]
            attention_mask: Optional attention mask [batch, time]
            
        Returns:
            content_features: Content representations [batch, seq_len, output_dim]
            attention_mask: Attention mask for features [batch, seq_len]
        """
        # Extract features with HuBERT
        with torch.set_grad_enabled(not self.freeze):
            outputs = self.hubert(
                audio,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        
        # Use last hidden state
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hubert_dim]
        
        # Project to output dimension
        content_features = self.projection(hidden_states)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(
                content_features.shape[:2],
                dtype=torch.long,
                device=content_features.device,
            )
        
        return content_features, attention_mask
    
    def extract_features(self, audio_path: str) -> torch.Tensor:
        """
        Extract features from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Content features tensor
        """
        import torchaudio
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample if needed
        if sample_rate != self.feature_extractor.sampling_rate:
            resampler = torchaudio.transforms.Resample(
                sample_rate,
                self.feature_extractor.sampling_rate,
            )
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Add batch dimension
        waveform = waveform.squeeze(0).unsqueeze(0)
        
        # Extract features
        with torch.no_grad():
            features, _ = self.forward(waveform)
        
        return features
    
    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self.parameters()).device
