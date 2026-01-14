"""
Speaker Encoder Module

Uses ECAPA-TDNN for speaker identity encoding to extract speaker embeddings
for voice cloning.
"""

import torch
import torch.nn as nn
import torchaudio
from typing import Optional
from speechbrain.pretrained import EncoderClassifier


class SpeakerEncoder(nn.Module):
    """
    Speaker identity encoder using ECAPA-TDNN.
    
    Extracts speaker embeddings from reference audio for voice cloning.
    """
    
    def __init__(
        self,
        model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
        embedding_dim: int = 192,
        freeze: bool = False,
    ):
        """
        Initialize speaker encoder.
        
        Args:
            model_name: SpeechBrain model name
            embedding_dim: Dimension of speaker embeddings
            freeze: Whether to freeze encoder weights
        """
        super().__init__()
        
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.freeze = freeze
        
        # Load pre-trained ECAPA-TDNN from SpeechBrain
        self.encoder = EncoderClassifier.from_hparams(
            source=model_name,
            savedir=f"pretrained_models/{model_name.split('/')[-1]}",
        )
        
        # Freeze if specified
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # The ECAPA-TDNN outputs 192-dim embeddings by default
        # Add projection if different dimension needed
        ecapa_dim = 192
        if embedding_dim != ecapa_dim:
            self.projection = nn.Linear(ecapa_dim, embedding_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(
        self,
        audio: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Extract speaker embeddings from audio.
        
        Args:
            audio: Input audio tensor [batch, time] or [batch, 1, time]
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            speaker_embeddings: Speaker embeddings [batch, embedding_dim]
        """
        # Ensure correct shape [batch, time]
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        
        # Extract embeddings with ECAPA-TDNN
        with torch.set_grad_enabled(not self.freeze):
            # SpeechBrain expects [batch, time]
            embeddings = self.encoder.encode_batch(audio)
            
            # Remove extra dimensions if present
            if embeddings.dim() == 3:
                embeddings = embeddings.squeeze(1)
        
        # Project to desired dimension
        embeddings = self.projection(embeddings)
        
        # L2 normalization
        if normalize:
            embeddings = nn.functional.normalize(embeddings, p=2, dim=-1)
        
        return embeddings
    
    def extract_speaker_embedding(
        self,
        audio_path: str,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Extract speaker embedding from audio file.
        
        Args:
            audio_path: Path to audio file
            normalize: Whether to L2-normalize embedding
            
        Returns:
            Speaker embedding tensor [embedding_dim]
        """
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample to 16kHz (ECAPA-TDNN expects 16kHz)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Remove channel dimension and add batch dimension
        waveform = waveform.squeeze(0).unsqueeze(0)
        
        # Move to same device as model
        waveform = waveform.to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.forward(waveform, normalize=normalize)
        
        # Remove batch dimension
        return embedding.squeeze(0)
    
    def compute_similarity(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cosine similarity between two speaker embeddings.
        
        Args:
            embedding1: First embedding [embedding_dim] or [batch, embedding_dim]
            embedding2: Second embedding [embedding_dim] or [batch, embedding_dim]
            
        Returns:
            Similarity score(s) in range [-1, 1]
        """
        # Ensure embeddings are normalized
        embedding1 = nn.functional.normalize(embedding1, p=2, dim=-1)
        embedding2 = nn.functional.normalize(embedding2, p=2, dim=-1)
        
        # Compute cosine similarity
        similarity = torch.sum(embedding1 * embedding2, dim=-1)
        
        return similarity
    
    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self.parameters()).device
