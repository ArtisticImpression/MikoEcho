"""
Loss Functions for MikoEcho Training

Includes reconstruction, speaker similarity, content preservation, and adversarial losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class MikoEchoLoss(nn.Module):
    """Combined loss function for MikoEcho training."""
    
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        speaker_similarity_weight: float = 0.5,
        content_preservation_weight: float = 0.3,
        adversarial_weight: float = 0.1,
        perceptual_weight: float = 0.2,
    ):
        """
        Initialize loss function.
        
        Args:
            reconstruction_weight: Weight for mel reconstruction loss
            speaker_similarity_weight: Weight for speaker similarity loss
            content_preservation_weight: Weight for content preservation loss
            adversarial_weight: Weight for adversarial loss
            perceptual_weight: Weight for perceptual loss
        """
        super().__init__()
        
        self.reconstruction_weight = reconstruction_weight
        self.speaker_similarity_weight = speaker_similarity_weight
        self.content_preservation_weight = content_preservation_weight
        self.adversarial_weight = adversarial_weight
        self.perceptual_weight = perceptual_weight
        
        # Loss components
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.cosine_embedding_loss = nn.CosineEmbeddingLoss()
    
    def reconstruction_loss(
        self,
        pred_mel: torch.Tensor,
        target_mel: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mel-spectrogram reconstruction loss.
        
        Args:
            pred_mel: Predicted mel-spectrogram
            target_mel: Target mel-spectrogram
            
        Returns:
            Reconstruction loss
        """
        # L1 loss for mel-spectrogram
        l1 = self.l1_loss(pred_mel, target_mel)
        
        # L2 loss for additional smoothness
        l2 = self.l2_loss(pred_mel, target_mel)
        
        return l1 + 0.5 * l2
    
    def speaker_similarity_loss(
        self,
        pred_speaker_emb: torch.Tensor,
        target_speaker_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Speaker similarity loss using cosine embedding.
        
        Args:
            pred_speaker_emb: Predicted speaker embedding
            target_speaker_emb: Target speaker embedding
            
        Returns:
            Speaker similarity loss
        """
        # Normalize embeddings
        pred_norm = F.normalize(pred_speaker_emb, p=2, dim=-1)
        target_norm = F.normalize(target_speaker_emb, p=2, dim=-1)
        
        # Cosine similarity (we want high similarity, so minimize negative)
        similarity = F.cosine_similarity(pred_norm, target_norm, dim=-1)
        loss = 1 - similarity.mean()
        
        return loss
    
    def content_preservation_loss(
        self,
        source_content: torch.Tensor,
        output_content: torch.Tensor,
    ) -> torch.Tensor:
        """
        Content preservation loss to ensure content is maintained.
        
        Args:
            source_content: Content features from source audio
            output_content: Content features from output audio
            
        Returns:
            Content preservation loss
        """
        # L2 loss between content features
        return self.l2_loss(source_content, output_content)
    
    def perceptual_loss(
        self,
        pred_mel: torch.Tensor,
        target_mel: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perceptual loss using mel-spectrogram differences.
        
        Args:
            pred_mel: Predicted mel-spectrogram
            target_mel: Target mel-spectrogram
            
        Returns:
            Perceptual loss
        """
        # Compute differences at multiple scales
        loss = 0.0
        
        # Full resolution
        loss += self.l1_loss(pred_mel, target_mel)
        
        # Downsampled (2x)
        pred_down2 = F.avg_pool1d(pred_mel, kernel_size=2, stride=2)
        target_down2 = F.avg_pool1d(target_mel, kernel_size=2, stride=2)
        loss += self.l1_loss(pred_down2, target_down2)
        
        # Downsampled (4x)
        pred_down4 = F.avg_pool1d(pred_mel, kernel_size=4, stride=4)
        target_down4 = F.avg_pool1d(target_mel, kernel_size=4, stride=4)
        loss += self.l1_loss(pred_down4, target_down4)
        
        return loss / 3.0
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.
        
        Args:
            outputs: Model outputs dictionary
            targets: Target values dictionary
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Reconstruction loss
        if 'mel_spectrogram' in outputs and 'target_mel' in targets:
            losses['reconstruction'] = self.reconstruction_loss(
                outputs['mel_spectrogram'],
                targets['target_mel'],
            )
        
        # Speaker similarity loss
        if 'speaker_embedding' in outputs and 'target_speaker_emb' in targets:
            losses['speaker_similarity'] = self.speaker_similarity_loss(
                outputs['speaker_embedding'],
                targets['target_speaker_emb'],
            )
        
        # Content preservation loss
        if 'content_features' in outputs and 'source_content' in targets:
            losses['content_preservation'] = self.content_preservation_loss(
                targets['source_content'],
                outputs['content_features'],
            )
        
        # Perceptual loss
        if 'mel_spectrogram' in outputs and 'target_mel' in targets:
            losses['perceptual'] = self.perceptual_loss(
                outputs['mel_spectrogram'],
                targets['target_mel'],
            )
        
        # Total loss
        total_loss = 0.0
        if 'reconstruction' in losses:
            total_loss += self.reconstruction_weight * losses['reconstruction']
        if 'speaker_similarity' in losses:
            total_loss += self.speaker_similarity_weight * losses['speaker_similarity']
        if 'content_preservation' in losses:
            total_loss += self.content_preservation_weight * losses['content_preservation']
        if 'perceptual' in losses:
            total_loss += self.perceptual_weight * losses['perceptual']
        
        losses['total'] = total_loss
        
        return losses
