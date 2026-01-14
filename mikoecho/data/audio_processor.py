"""
Audio Processing Utilities

Handles audio preprocessing, mel-spectrogram extraction, and transformations.
"""

import torch
import torchaudio
import numpy as np
from typing import Tuple, Optional
from pathlib import Path


class AudioProcessor:
    """Audio preprocessing and feature extraction."""
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
    ):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Target sample rate
            n_fft: FFT size
            hop_length: Hop length for STFT
            win_length: Window length
            n_mels: Number of mel bands
            f_min: Minimum frequency
            f_max: Maximum frequency (None = sample_rate / 2)
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate / 2
        
        # Mel-spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=self.f_max,
            n_mels=n_mels,
            power=1.0,  # Magnitude spectrogram
        )
        
        # Inverse mel transform
        self.inverse_mel = torchaudio.transforms.InverseMelScale(
            n_stft=n_fft // 2 + 1,
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=self.f_max,
        )
    
    def load_audio(
        self,
        audio_path: str,
        duration: Optional[float] = None,
        offset: float = 0.0,
    ) -> Tuple[torch.Tensor, int]:
        """
        Load audio file.
        
        Args:
            audio_path: Path to audio file
            duration: Duration to load (seconds)
            offset: Start offset (seconds)
            
        Returns:
            waveform: Audio tensor [channels, time]
            sample_rate: Original sample rate
        """
        # Calculate frames
        num_frames = -1
        if duration is not None:
            # Load with duration info to get sample rate first
            info = torchaudio.info(audio_path)
            num_frames = int(duration * info.sample_rate)
        
        frame_offset = 0
        if offset > 0:
            info = torchaudio.info(audio_path)
            frame_offset = int(offset * info.sample_rate)
        
        # Load audio
        waveform, sample_rate = torchaudio.load(
            audio_path,
            frame_offset=frame_offset,
            num_frames=num_frames,
        )
        
        return waveform, sample_rate
    
    def preprocess_audio(
        self,
        waveform: torch.Tensor,
        source_sample_rate: int,
        normalize: bool = True,
        to_mono: bool = True,
    ) -> torch.Tensor:
        """
        Preprocess audio waveform.
        
        Args:
            waveform: Input waveform [channels, time]
            source_sample_rate: Source sample rate
            normalize: Whether to normalize amplitude
            to_mono: Whether to convert to mono
            
        Returns:
            Preprocessed waveform [time] or [1, time]
        """
        # Resample if needed
        if source_sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                source_sample_rate,
                self.sample_rate,
            )
            waveform = resampler(waveform)
        
        # Convert to mono
        if to_mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Normalize
        if normalize:
            waveform = waveform / (torch.abs(waveform).max() + 1e-8)
        
        return waveform
    
    def extract_mel_spectrogram(
        self,
        waveform: torch.Tensor,
        log_scale: bool = True,
    ) -> torch.Tensor:
        """
        Extract mel-spectrogram from waveform.
        
        Args:
            waveform: Input waveform [batch, time] or [time]
            log_scale: Whether to apply log scaling
            
        Returns:
            Mel-spectrogram [batch, n_mels, time] or [n_mels, time]
        """
        # Ensure batch dimension
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False
        
        # Extract mel-spectrogram
        mel = self.mel_transform(waveform)
        
        # Apply log scaling
        if log_scale:
            mel = torch.log(torch.clamp(mel, min=1e-5))
        
        # Remove batch dimension if added
        if squeeze_batch:
            mel = mel.squeeze(0)
        
        return mel
    
    def pitch_shift(
        self,
        waveform: torch.Tensor,
        n_steps: int,
    ) -> torch.Tensor:
        """
        Shift pitch by n semitones.
        
        Args:
            waveform: Input waveform [channels, time]
            n_steps: Number of semitones to shift (positive or negative)
            
        Returns:
            Pitch-shifted waveform
        """
        if n_steps == 0:
            return waveform
        
        # Use torchaudio pitch shift
        pitch_shifter = torchaudio.transforms.PitchShift(
            sample_rate=self.sample_rate,
            n_steps=n_steps,
        )
        
        return pitch_shifter(waveform)
    
    def time_stretch(
        self,
        waveform: torch.Tensor,
        rate: float,
    ) -> torch.Tensor:
        """
        Stretch time by rate.
        
        Args:
            waveform: Input waveform [channels, time]
            rate: Stretch rate (> 1 = faster, < 1 = slower)
            
        Returns:
            Time-stretched waveform
        """
        if rate == 1.0:
            return waveform
        
        # Use torchaudio time stretch
        time_stretcher = torchaudio.transforms.TimeStretch(
            hop_length=self.hop_length,
            n_freq=self.n_fft // 2 + 1,
        )
        
        # Convert to spectrogram
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=True,
        )
        
        # Stretch
        stretched_spec = time_stretcher(spec, rate)
        
        # Convert back to waveform
        stretched_waveform = torch.istft(
            stretched_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        
        return stretched_waveform
    
    def save_audio(
        self,
        waveform: torch.Tensor,
        output_path: str,
        sample_rate: Optional[int] = None,
    ) -> None:
        """
        Save audio to file.
        
        Args:
            waveform: Audio waveform [channels, time] or [time]
            output_path: Output file path
            sample_rate: Sample rate (uses self.sample_rate if None)
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        # Ensure 2D tensor
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Ensure on CPU
        waveform = waveform.cpu()
        
        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save
        torchaudio.save(output_path, waveform, sample_rate)
