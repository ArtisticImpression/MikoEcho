"""
Voice Cloner

Extracts speaker embeddings for voice cloning from reference audio.
"""

import torch
import torchaudio
from pathlib import Path
from typing import Union, Optional
import numpy as np

from mikoecho.models.speaker_encoder import SpeakerEncoder
from mikoecho.data.audio_processor import AudioProcessor


class VoiceCloner:
    """
    Voice cloning interface for extracting speaker embeddings.
    
    Usage:
        cloner = VoiceCloner()
        speaker_embedding = cloner.clone_voice("reference.wav")
    """
    
    def __init__(
        self,
        speaker_encoder: Optional[SpeakerEncoder] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize voice cloner.
        
        Args:
            speaker_encoder: Pre-trained speaker encoder (creates new if None)
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device
        
        # Initialize speaker encoder
        if speaker_encoder is None:
            self.speaker_encoder = SpeakerEncoder()
        else:
            self.speaker_encoder = speaker_encoder
        
        self.speaker_encoder.to(device)
        self.speaker_encoder.eval()
        
        # Audio processor
        self.audio_processor = AudioProcessor(sample_rate=16000)  # ECAPA-TDNN uses 16kHz
    
    def clone_voice(
        self,
        reference_audio_path: Union[str, Path],
        duration: Optional[float] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Clone voice from reference audio.
        
        Args:
            reference_audio_path: Path to reference audio file
            duration: Duration to use (None = use full audio)
            normalize: Whether to normalize embedding
            
        Returns:
            Speaker embedding tensor [embedding_dim]
        """
        # Load and preprocess audio
        waveform, sample_rate = self.audio_processor.load_audio(
            str(reference_audio_path),
            duration=duration,
        )
        
        waveform = self.audio_processor.preprocess_audio(
            waveform,
            sample_rate,
            normalize=True,
            to_mono=True,
        )
        
        # Move to device
        waveform = waveform.to(self.device)
        
        # Extract speaker embedding
        with torch.no_grad():
            speaker_embedding = self.speaker_encoder(
                waveform,
                normalize=normalize,
            )
        
        # Remove batch dimension
        if speaker_embedding.dim() > 1:
            speaker_embedding = speaker_embedding.squeeze(0)
        
        return speaker_embedding
    
    def clone_voice_from_multiple(
        self,
        reference_audio_paths: list,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Clone voice from multiple reference audios (averaged).
        
        This can improve robustness by averaging embeddings from multiple samples.
        
        Args:
            reference_audio_paths: List of paths to reference audio files
            normalize: Whether to normalize final embedding
            
        Returns:
            Averaged speaker embedding [embedding_dim]
        """
        embeddings = []
        
        for audio_path in reference_audio_paths:
            embedding = self.clone_voice(audio_path, normalize=False)
            embeddings.append(embedding)
        
        # Average embeddings
        avg_embedding = torch.stack(embeddings).mean(dim=0)
        
        # Normalize if requested
        if normalize:
            avg_embedding = torch.nn.functional.normalize(avg_embedding, p=2, dim=-1)
        
        return avg_embedding
    
    def compute_similarity(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
    ) -> float:
        """
        Compute similarity between two speaker embeddings.
        
        Args:
            embedding1: First speaker embedding
            embedding2: Second speaker embedding
            
        Returns:
            Similarity score in [0, 1]
        """
        similarity = self.speaker_encoder.compute_similarity(embedding1, embedding2)
        
        # Convert to [0, 1] range
        similarity = (similarity + 1) / 2
        
        return similarity.item()
    
    def save_embedding(
        self,
        embedding: torch.Tensor,
        output_path: Union[str, Path],
    ):
        """
        Save speaker embedding to file.
        
        Args:
            embedding: Speaker embedding tensor
            output_path: Output file path (.pt or .npy)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.pt':
            torch.save(embedding, output_path)
        elif output_path.suffix == '.npy':
            np.save(output_path, embedding.cpu().numpy())
        else:
            raise ValueError(f"Unsupported file format: {output_path.suffix}")
    
    def load_embedding(
        self,
        embedding_path: Union[str, Path],
    ) -> torch.Tensor:
        """
        Load speaker embedding from file.
        
        Args:
            embedding_path: Path to embedding file (.pt or .npy)
            
        Returns:
            Speaker embedding tensor
        """
        embedding_path = Path(embedding_path)
        
        if embedding_path.suffix == '.pt':
            embedding = torch.load(embedding_path, map_location=self.device)
        elif embedding_path.suffix == '.npy':
            embedding = torch.from_numpy(np.load(embedding_path)).to(self.device)
        else:
            raise ValueError(f"Unsupported file format: {embedding_path.suffix}")
        
        return embedding
