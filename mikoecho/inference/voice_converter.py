"""
Voice Converter

Converts source audio to target speaker voice with emotion control.
"""

import torch
import torchaudio
from pathlib import Path
from typing import Union, Optional, List

from mikoecho.models.mikoecho_model import MikoEchoModel
from mikoecho.data.audio_processor import AudioProcessor
from mikoecho.inference.voice_cloner import VoiceCloner


class VoiceConverter:
    """
    Voice-to-voice conversion system.
    
    Converts source speech to target speaker voice while preserving content.
    
    Usage:
        converter = VoiceConverter(model)
        output = converter.convert(
            "source.wav",
            speaker_embedding,
            emotion="excited"
        )
    """
    
    def __init__(
        self,
        model: MikoEchoModel,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        sample_rate: int = 22050,
    ):
        """
        Initialize voice converter.
        
        Args:
            model: Trained MikoEcho model
            device: Device to run on
            sample_rate: Audio sample rate
        """
        self.device = device
        self.sample_rate = sample_rate
        
        # Model
        self.model = model.to(device)
        self.model.eval()
        
        # Audio processor
        self.audio_processor = AudioProcessor(sample_rate=sample_rate)
        
        # Voice cloner for extracting speaker embeddings
        self.voice_cloner = VoiceCloner(
            speaker_encoder=model.speaker_encoder,
            device=device,
        )
        
        # Available emotions
        self.emotions = ['neutral', 'calm', 'excited', 'sad', 'energetic']
    
    def convert(
        self,
        source_audio_path: Union[str, Path],
        speaker_embedding: torch.Tensor,
        emotion: str = 'neutral',
        emotion_strength: float = 1.0,
        output_path: Optional[Union[str, Path]] = None,
    ) -> torch.Tensor:
        """
        Convert source audio to target speaker voice.
        
        Args:
            source_audio_path: Path to source audio file
            speaker_embedding: Target speaker embedding
            emotion: Emotion name ('neutral', 'calm', 'excited', 'sad', 'energetic')
            emotion_strength: Emotion strength (0-1)
            output_path: Optional path to save output audio
            
        Returns:
            Converted audio waveform [time]
        """
        # Validate emotion
        if emotion not in self.emotions:
            raise ValueError(f"Unknown emotion: {emotion}. Available: {self.emotions}")
        
        # Load and preprocess source audio
        waveform, sample_rate = self.audio_processor.load_audio(str(source_audio_path))
        waveform = self.audio_processor.preprocess_audio(
            waveform,
            sample_rate,
            normalize=True,
            to_mono=True,
        )
        
        # Add batch dimension
        waveform = waveform.unsqueeze(0).to(self.device)
        
        # Ensure speaker embedding has batch dimension
        if speaker_embedding.dim() == 1:
            speaker_embedding = speaker_embedding.unsqueeze(0)
        speaker_embedding = speaker_embedding.to(self.device)
        
        # Convert voice
        with torch.no_grad():
            output_audio = self.model.convert_voice(
                waveform,
                speaker_embedding,
                emotion=emotion,
                emotion_strength=emotion_strength,
            )
        
        # Remove batch dimension
        output_audio = output_audio.squeeze(0)
        
        # Save if output path provided
        if output_path is not None:
            self.audio_processor.save_audio(
                output_audio.cpu(),
                str(output_path),
                sample_rate=self.sample_rate,
            )
        
        return output_audio.cpu()
    
    def convert_with_reference(
        self,
        source_audio_path: Union[str, Path],
        reference_audio_path: Union[str, Path],
        emotion: str = 'neutral',
        emotion_strength: float = 1.0,
        output_path: Optional[Union[str, Path]] = None,
    ) -> torch.Tensor:
        """
        Convert using reference audio to extract speaker embedding.
        
        Args:
            source_audio_path: Path to source audio
            reference_audio_path: Path to reference audio for target speaker
            emotion: Emotion name
            emotion_strength: Emotion strength (0-1)
            output_path: Optional output path
            
        Returns:
            Converted audio waveform
        """
        # Extract speaker embedding from reference
        speaker_embedding = self.voice_cloner.clone_voice(reference_audio_path)
        
        # Convert
        return self.convert(
            source_audio_path,
            speaker_embedding,
            emotion=emotion,
            emotion_strength=emotion_strength,
            output_path=output_path,
        )
    
    def batch_convert(
        self,
        source_audio_paths: List[Union[str, Path]],
        speaker_embedding: torch.Tensor,
        emotion: str = 'neutral',
        emotion_strength: float = 1.0,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> List[torch.Tensor]:
        """
        Convert multiple audio files.
        
        Args:
            source_audio_paths: List of source audio paths
            speaker_embedding: Target speaker embedding
            emotion: Emotion name
            emotion_strength: Emotion strength
            output_dir: Optional output directory
            
        Returns:
            List of converted audio waveforms
        """
        outputs = []
        
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, source_path in enumerate(source_audio_paths):
            # Determine output path
            if output_dir is not None:
                source_name = Path(source_path).stem
                output_path = output_dir / f"{source_name}_converted.wav"
            else:
                output_path = None
            
            # Convert
            output = self.convert(
                source_path,
                speaker_embedding,
                emotion=emotion,
                emotion_strength=emotion_strength,
                output_path=output_path,
            )
            
            outputs.append(output)
        
        return outputs
    
    def get_available_emotions(self) -> List[str]:
        """Get list of available emotions."""
        return self.emotions.copy()
