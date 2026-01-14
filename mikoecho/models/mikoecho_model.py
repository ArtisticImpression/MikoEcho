"""
Main MikoEcho Model

Orchestrates all components for voice cloning and voice-to-voice conversion.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import torchaudio

from mikoecho.models.speech_encoder import SpeechEncoder
from mikoecho.models.speaker_encoder import SpeakerEncoder
from mikoecho.models.content_disentangler import ContentDisentangler
from mikoecho.models.style_encoder import StyleEncoder
from mikoecho.models.vocoder import Vocoder


class MikoEchoModel(nn.Module):
    """
    Main MikoEcho model for voice cloning and conversion.
    
    Architecture:
        Input Audio → Speech Encoder → Content Disentangler → 
        + Speaker Embedding + Style Embedding → Vocoder → Output Audio
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MikoEcho model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        
        self.config = config
        
        # Initialize components
        self.speech_encoder = SpeechEncoder(
            model_name=config['speech_encoder']['model_name'],
            freeze=config['speech_encoder']['freeze'],
            output_dim=config['speech_encoder']['output_dim'],
        )
        
        self.speaker_encoder = SpeakerEncoder(
            model_name=config['speaker_encoder']['model_name'],
            embedding_dim=config['speaker_encoder']['embedding_dim'],
            freeze=config['speaker_encoder']['freeze'],
        )
        
        self.content_disentangler = ContentDisentangler(
            input_dim=config['speech_encoder']['output_dim'],
            hidden_dim=config['content_disentangler']['hidden_dim'],
            num_layers=config['content_disentangler']['num_layers'],
            dropout=config['content_disentangler']['dropout'],
        )
        
        self.style_encoder = StyleEncoder(
            emotion_dim=config['style_encoder']['emotion_dim'],
            num_emotions=config['style_encoder']['num_emotions'],
            hidden_dim=config['style_encoder']['hidden_dim'],
        )
        
        self.vocoder = Vocoder(
            n_mels=config['vocoder']['n_mels'],
            sample_rate=config['vocoder']['sample_rate'],
            hop_length=config['vocoder']['hop_length'],
            win_length=config['vocoder']['win_length'],
            n_fft=config['vocoder']['n_fft'],
        )
        
        # Mel-spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config['vocoder']['sample_rate'],
            n_fft=config['vocoder']['n_fft'],
            win_length=config['vocoder']['win_length'],
            hop_length=config['vocoder']['hop_length'],
            n_mels=config['vocoder']['n_mels'],
        )
        
        # Fusion layer to combine content, speaker, and style
        fusion_input_dim = (
            config['content_disentangler']['hidden_dim'] +
            config['speaker_encoder']['embedding_dim'] +
            config['style_encoder']['emotion_dim']
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, config['vocoder']['n_mels']),
            nn.LayerNorm(config['vocoder']['n_mels']),
            nn.ReLU(),
        )
    
    def forward(
        self,
        source_audio: torch.Tensor,
        reference_audio: torch.Tensor,
        emotion: Optional[torch.Tensor] = None,
        emotion_strength: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for voice conversion.
        
        Args:
            source_audio: Source audio to convert [batch, time]
            reference_audio: Reference audio for target speaker [batch, time]
            emotion: Emotion indices or names [batch]
            emotion_strength: Strength of emotion (0-1)
            
        Returns:
            Dictionary containing:
                - output_audio: Converted audio [batch, time]
                - mel_spectrogram: Generated mel-spectrogram [batch, n_mels, time]
                - speaker_embedding: Speaker embedding [batch, embedding_dim]
                - content_features: Content features [batch, seq_len, hidden_dim]
                - style_embedding: Style embedding [batch, emotion_dim]
        """
        # Extract content from source audio
        content_features, attention_mask = self.speech_encoder(source_audio)
        
        # Disentangle content from speaker identity
        disentangled_content = self.content_disentangler(
            content_features,
            attention_mask,
        )
        
        # Extract speaker embedding from reference audio
        speaker_embedding = self.speaker_encoder(reference_audio)
        
        # Extract reference mel for style encoding
        reference_mel = self.mel_transform(reference_audio)
        
        # Encode style and emotion
        style_embedding = self.style_encoder(
            emotion=emotion,
            reference_mel=reference_mel,
            emotion_strength=emotion_strength,
        )
        
        # Expand speaker and style embeddings to match sequence length
        seq_len = disentangled_content.shape[1]
        speaker_expanded = speaker_embedding.unsqueeze(1).expand(-1, seq_len, -1)
        style_expanded = style_embedding.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Fuse content, speaker, and style
        fused_features = torch.cat([
            disentangled_content,
            speaker_expanded,
            style_expanded,
        ], dim=-1)
        
        # Project to mel-spectrogram dimension
        mel_features = self.fusion_layer(fused_features)
        
        # Transpose for vocoder [batch, n_mels, time]
        mel_spectrogram = mel_features.transpose(1, 2)
        
        # Generate waveform
        output_audio = self.vocoder(mel_spectrogram)
        
        return {
            'output_audio': output_audio,
            'mel_spectrogram': mel_spectrogram,
            'speaker_embedding': speaker_embedding,
            'content_features': disentangled_content,
            'style_embedding': style_embedding,
        }
    
    def clone_voice(
        self,
        reference_audio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract speaker embedding for voice cloning.
        
        Args:
            reference_audio: Reference audio [batch, time]
            
        Returns:
            Speaker embedding [batch, embedding_dim]
        """
        with torch.no_grad():
            speaker_embedding = self.speaker_encoder(reference_audio)
        return speaker_embedding
    
    def convert_voice(
        self,
        source_audio: torch.Tensor,
        speaker_embedding: torch.Tensor,
        emotion: str = 'neutral',
        emotion_strength: float = 1.0,
    ) -> torch.Tensor:
        """
        Convert source audio to target speaker voice.
        
        Args:
            source_audio: Source audio [batch, time]
            speaker_embedding: Target speaker embedding [batch, embedding_dim]
            emotion: Emotion name
            emotion_strength: Strength of emotion (0-1)
            
        Returns:
            Converted audio [batch, time]
        """
        with torch.no_grad():
            # Extract content
            content_features, attention_mask = self.speech_encoder(source_audio)
            disentangled_content = self.content_disentangler(
                content_features,
                attention_mask,
            )
            
            # Encode style
            style_embedding = self.style_encoder(
                emotion=[emotion] * source_audio.shape[0],
                emotion_strength=emotion_strength,
            )
            
            # Expand embeddings
            seq_len = disentangled_content.shape[1]
            speaker_expanded = speaker_embedding.unsqueeze(1).expand(-1, seq_len, -1)
            style_expanded = style_embedding.unsqueeze(1).expand(-1, seq_len, -1)
            
            # Fuse and generate
            fused_features = torch.cat([
                disentangled_content,
                speaker_expanded,
                style_expanded,
            ], dim=-1)
            
            mel_features = self.fusion_layer(fused_features)
            mel_spectrogram = mel_features.transpose(1, 2)
            output_audio = self.vocoder(mel_spectrogram)
        
        return output_audio
    
    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self.parameters()).device
