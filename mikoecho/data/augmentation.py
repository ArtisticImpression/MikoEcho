"""
Audio Augmentation Pipeline

Applies various augmentations for robust training.
"""

import torch
import torchaudio
import random
from typing import Optional


class AudioAugmentation:
    """Audio augmentation for voice cloning training."""
    
    def __init__(
        self,
        sample_rate: int = 22050,
        noise_prob: float = 0.3,
        reverb_prob: float = 0.2,
        pitch_shift_range: tuple = (-2, 2),
        speed_range: tuple = (0.9, 1.1),
    ):
        """
        Initialize augmentation pipeline.
        
        Args:
            sample_rate: Audio sample rate
            noise_prob: Probability of adding noise
            reverb_prob: Probability of adding reverb
            pitch_shift_range: Range of pitch shift in semitones
            speed_range: Range of speed change
        """
        self.sample_rate = sample_rate
        self.noise_prob = noise_prob
        self.reverb_prob = reverb_prob
        self.pitch_shift_range = pitch_shift_range
        self.speed_range = speed_range
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations to waveform.
        
        Args:
            waveform: Input waveform [channels, time]
            
        Returns:
            Augmented waveform
        """
        # Add noise
        if random.random() < self.noise_prob:
            waveform = self.add_noise(waveform)
        
        # Add reverb
        if random.random() < self.reverb_prob:
            waveform = self.add_reverb(waveform)
        
        # Pitch shift
        if random.random() < 0.3:
            waveform = self.pitch_shift(waveform)
        
        # Speed change
        if random.random() < 0.3:
            waveform = self.speed_change(waveform)
        
        return waveform
    
    def add_noise(
        self,
        waveform: torch.Tensor,
        snr_db: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Add Gaussian noise to waveform.
        
        Args:
            waveform: Input waveform
            snr_db: Signal-to-noise ratio in dB (random if None)
            
        Returns:
            Noisy waveform
        """
        if snr_db is None:
            snr_db = random.uniform(10, 30)
        
        # Calculate noise power
        signal_power = waveform.pow(2).mean()
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # Generate and add noise
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        noisy_waveform = waveform + noise
        
        return noisy_waveform
    
    def add_reverb(
        self,
        waveform: torch.Tensor,
        room_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Add reverb effect.
        
        Args:
            waveform: Input waveform
            room_scale: Room size scale (random if None)
            
        Returns:
            Reverb waveform
        """
        if room_scale is None:
            room_scale = random.uniform(0.1, 0.5)
        
        # Simple reverb using delayed copies
        delay_samples = int(room_scale * self.sample_rate * 0.05)  # 50ms max
        decay = 0.3
        
        reverb = waveform.clone()
        for i in range(1, 4):
            delay = delay_samples * i
            if delay < waveform.shape[-1]:
                delayed = torch.nn.functional.pad(
                    waveform[..., :-delay],
                    (delay, 0)
                ) * (decay ** i)
                reverb = reverb + delayed
        
        # Normalize
        reverb = reverb / reverb.abs().max()
        
        return reverb
    
    def pitch_shift(
        self,
        waveform: torch.Tensor,
        n_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Shift pitch.
        
        Args:
            waveform: Input waveform
            n_steps: Number of semitones (random if None)
            
        Returns:
            Pitch-shifted waveform
        """
        if n_steps is None:
            n_steps = random.randint(*self.pitch_shift_range)
        
        if n_steps == 0:
            return waveform
        
        pitch_shifter = torchaudio.transforms.PitchShift(
            sample_rate=self.sample_rate,
            n_steps=n_steps,
        )
        
        return pitch_shifter(waveform)
    
    def speed_change(
        self,
        waveform: torch.Tensor,
        speed: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Change playback speed.
        
        Args:
            waveform: Input waveform
            speed: Speed factor (random if None)
            
        Returns:
            Speed-changed waveform
        """
        if speed is None:
            speed = random.uniform(*self.speed_range)
        
        if speed == 1.0:
            return waveform
        
        # Resample to change speed
        new_sample_rate = int(self.sample_rate * speed)
        resampler = torchaudio.transforms.Resample(
            self.sample_rate,
            new_sample_rate,
        )
        
        # Resample and then resample back
        resampled = resampler(waveform)
        resampler_back = torchaudio.transforms.Resample(
            new_sample_rate,
            self.sample_rate,
        )
        
        return resampler_back(resampled)
