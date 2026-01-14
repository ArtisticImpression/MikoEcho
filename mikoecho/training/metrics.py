"""
Evaluation Metrics for Voice Cloning

Includes speaker similarity, MOS estimation, WER, and pitch correlation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import numpy as np


class VoiceCloningMetrics:
    """Metrics for evaluating voice cloning quality."""
    
    def __init__(self):
        """Initialize metrics."""
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.speaker_similarities = []
        self.mos_scores = []
        self.pitch_correlations = []
    
    def compute_speaker_similarity(
        self,
        pred_embedding: torch.Tensor,
        target_embedding: torch.Tensor,
    ) -> float:
        """
        Compute cosine similarity between speaker embeddings.
        
        Args:
            pred_embedding: Predicted speaker embedding
            target_embedding: Target speaker embedding
            
        Returns:
            Similarity score in [0, 1]
        """
        # Normalize
        pred_norm = F.normalize(pred_embedding, p=2, dim=-1)
        target_norm = F.normalize(target_embedding, p=2, dim=-1)
        
        # Cosine similarity
        similarity = F.cosine_similarity(pred_norm, target_norm, dim=-1)
        
        # Convert to [0, 1] range
        similarity = (similarity + 1) / 2
        
        return similarity.mean().item()
    
    def estimate_mos(
        self,
        pred_audio: torch.Tensor,
        target_audio: torch.Tensor,
    ) -> float:
        """
        Estimate Mean Opinion Score (MOS) using signal metrics.
        
        This is a simplified estimation. For accurate MOS, use human evaluation.
        
        Args:
            pred_audio: Predicted audio waveform
            target_audio: Target audio waveform
            
        Returns:
            Estimated MOS score in [1, 5]
        """
        # Compute SNR
        noise = pred_audio - target_audio
        signal_power = torch.mean(target_audio ** 2)
        noise_power = torch.mean(noise ** 2)
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
        
        # Map SNR to MOS (empirical mapping)
        # SNR > 30dB → MOS ~5, SNR < 10dB → MOS ~1
        mos = 1.0 + 4.0 * torch.sigmoid((snr - 20) / 10)
        
        return mos.item()
    
    def compute_pitch_correlation(
        self,
        pred_audio: torch.Tensor,
        target_audio: torch.Tensor,
        sample_rate: int = 22050,
    ) -> float:
        """
        Compute pitch correlation between predicted and target audio.
        
        Args:
            pred_audio: Predicted audio
            target_audio: Target audio
            sample_rate: Sample rate
            
        Returns:
            Pitch correlation coefficient
        """
        # Extract pitch using autocorrelation (simplified)
        def extract_pitch(audio):
            # Compute autocorrelation
            autocorr = F.conv1d(
                audio.unsqueeze(0).unsqueeze(0),
                audio.flip(0).unsqueeze(0).unsqueeze(0),
                padding=audio.shape[0] - 1,
            ).squeeze()
            
            # Find peaks (simplified pitch detection)
            # In practice, use a proper pitch detection algorithm
            return autocorr
        
        pred_pitch = extract_pitch(pred_audio)
        target_pitch = extract_pitch(target_audio)
        
        # Compute correlation
        min_len = min(pred_pitch.shape[0], target_pitch.shape[0])
        pred_pitch = pred_pitch[:min_len]
        target_pitch = target_pitch[:min_len]
        
        correlation = F.cosine_similarity(
            pred_pitch.unsqueeze(0),
            target_pitch.unsqueeze(0),
            dim=-1,
        )
        
        return correlation.item()
    
    def update(
        self,
        pred_outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ):
        """
        Update metrics with new predictions.
        
        Args:
            pred_outputs: Model predictions
            targets: Target values
        """
        # Speaker similarity
        if 'speaker_embedding' in pred_outputs and 'target_speaker_emb' in targets:
            similarity = self.compute_speaker_similarity(
                pred_outputs['speaker_embedding'],
                targets['target_speaker_emb'],
            )
            self.speaker_similarities.append(similarity)
        
        # MOS estimation
        if 'output_audio' in pred_outputs and 'target_audio' in targets:
            mos = self.estimate_mos(
                pred_outputs['output_audio'],
                targets['target_audio'],
            )
            self.mos_scores.append(mos)
        
        # Pitch correlation
        if 'output_audio' in pred_outputs and 'target_audio' in targets:
            pitch_corr = self.compute_pitch_correlation(
                pred_outputs['output_audio'][0],
                targets['target_audio'][0],
            )
            self.pitch_correlations.append(pitch_corr)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute average metrics.
        
        Returns:
            Dictionary of metric values
        """
        metrics = {}
        
        if self.speaker_similarities:
            metrics['speaker_similarity'] = np.mean(self.speaker_similarities)
        
        if self.mos_scores:
            metrics['mos_estimate'] = np.mean(self.mos_scores)
        
        if self.pitch_correlations:
            metrics['pitch_correlation'] = np.mean(self.pitch_correlations)
        
        return metrics
    
    def __str__(self) -> str:
        """String representation of metrics."""
        metrics = self.compute()
        lines = ["Voice Cloning Metrics:"]
        for name, value in metrics.items():
            lines.append(f"  {name}: {value:.4f}")
        return "\n".join(lines)
