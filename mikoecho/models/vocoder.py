"""
Neural Vocoder Module

HiFi-GAN vocoder for high-fidelity waveform generation from mel-spectrograms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class ResBlock(nn.Module):
    """Residual block for HiFi-GAN generator."""
    
    def __init__(self, channels: int, kernel_size: int = 3, dilation: tuple = (1, 3, 5)):
        super().__init__()
        
        self.convs1 = nn.ModuleList([
            nn.Conv1d(
                channels, channels, kernel_size,
                dilation=d, padding=self._get_padding(kernel_size, d)
            )
            for d in dilation
        ])
        
        self.convs2 = nn.ModuleList([
            nn.Conv1d(
                channels, channels, kernel_size,
                dilation=1, padding=self._get_padding(kernel_size, 1)
            )
            for _ in dilation
        ])
    
    @staticmethod
    def _get_padding(kernel_size: int, dilation: int) -> int:
        return int((kernel_size * dilation - dilation) / 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = conv1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = conv2(xt)
            x = xt + x
        return x


class HiFiGANGenerator(nn.Module):
    """HiFi-GAN Generator for mel-to-waveform synthesis."""
    
    def __init__(
        self,
        n_mels: int = 80,
        upsample_rates: List[int] = [8, 8, 2, 2],
        upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_initial_channel: int = 512,
    ):
        super().__init__()
        
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        
        # Initial convolution
        self.conv_pre = nn.Conv1d(n_mels, upsample_initial_channel, 7, padding=3)
        
        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(
                    upsample_initial_channel // (2 ** i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    k, u, padding=(k - u) // 2
                )
            )
        
        # Residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d))
        
        # Output convolution
        self.conv_post = nn.Conv1d(ch, 1, 7, padding=3)
    
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Generate waveform from mel-spectrogram.
        
        Args:
            mel: Mel-spectrogram [batch, n_mels, time]
            
        Returns:
            waveform: Generated waveform [batch, 1, time * hop_length]
        """
        x = self.conv_pre(mel)
        
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x


class Vocoder(nn.Module):
    """
    HiFi-GAN vocoder wrapper for MikoEcho.
    """
    
    def __init__(
        self,
        n_mels: int = 80,
        sample_rate: int = 22050,
        hop_length: int = 256,
        win_length: int = 1024,
        n_fft: int = 1024,
    ):
        """
        Initialize vocoder.
        
        Args:
            n_mels: Number of mel bands
            sample_rate: Audio sample rate
            hop_length: Hop length for STFT
            win_length: Window length for STFT
            n_fft: FFT size
        """
        super().__init__()
        
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        
        # HiFi-GAN generator
        self.generator = HiFiGANGenerator(n_mels=n_mels)
    
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Generate waveform from mel-spectrogram.
        
        Args:
            mel: Mel-spectrogram [batch, n_mels, time]
            
        Returns:
            waveform: Generated waveform [batch, time]
        """
        # Generate waveform
        waveform = self.generator(mel)
        
        # Remove channel dimension
        waveform = waveform.squeeze(1)
        
        return waveform
    
    def inference(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Inference mode for waveform generation.
        
        Args:
            mel: Mel-spectrogram [batch, n_mels, time]
            
        Returns:
            waveform: Generated waveform [batch, time]
        """
        with torch.no_grad():
            return self.forward(mel)
    
    @staticmethod
    def load_pretrained(checkpoint_path: str, device: str = 'cpu') -> 'Vocoder':
        """
        Load pre-trained vocoder from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load model on
            
        Returns:
            Loaded vocoder model
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Create model
        vocoder = Vocoder(
            n_mels=checkpoint.get('n_mels', 80),
            sample_rate=checkpoint.get('sample_rate', 22050),
            hop_length=checkpoint.get('hop_length', 256),
        )
        
        # Load weights
        vocoder.load_state_dict(checkpoint['model_state_dict'])
        vocoder.eval()
        
        return vocoder.to(device)
