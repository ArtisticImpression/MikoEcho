"""
Dataset Module

PyTorch datasets for multi-speaker voice cloning training.
"""

import torch
from torch.utils.data import Dataset
import torchaudio
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import random

from mikoecho.data.audio_processor import AudioProcessor


class VoiceCloningDataset(Dataset):
    """
    Multi-speaker dataset for voice cloning training.
    
    Supports LibriSpeech, VCTK, CREMA-D, and custom datasets.
    """
    
    def __init__(
        self,
        data_dir: str,
        dataset_name: str = "librispeech",
        split: str = "train",
        sample_rate: int = 22050,
        max_duration: float = 10.0,
        min_duration: float = 1.0,
        augment: bool = False,
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Root directory of dataset
            dataset_name: Dataset name ('librispeech', 'vctk', 'crema-d', 'custom')
            split: Dataset split ('train', 'val', 'test')
            sample_rate: Target sample rate
            max_duration: Maximum audio duration (seconds)
            min_duration: Minimum audio duration (seconds)
            augment: Whether to apply augmentation
        """
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name.lower()
        self.split = split
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.augment = augment
        
        # Audio processor
        self.audio_processor = AudioProcessor(sample_rate=sample_rate)
        
        # Load dataset metadata
        self.samples = self._load_dataset()
        self.speaker_to_idx = self._build_speaker_mapping()
        
        print(f"Loaded {len(self.samples)} samples from {dataset_name} ({split})")
        print(f"Number of speakers: {len(self.speaker_to_idx)}")
    
    def _load_dataset(self) -> List[Dict]:
        """Load dataset metadata."""
        if self.dataset_name == "librispeech":
            return self._load_librispeech()
        elif self.dataset_name == "vctk":
            return self._load_vctk()
        elif self.dataset_name == "crema-d":
            return self._load_crema_d()
        elif self.dataset_name == "custom":
            return self._load_custom()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _load_librispeech(self) -> List[Dict]:
        """Load LibriSpeech dataset."""
        samples = []
        
        # LibriSpeech structure: speaker_id/chapter_id/audio_file.flac
        audio_files = list(self.data_dir.rglob("*.flac"))
        
        for audio_path in audio_files:
            # Get speaker ID from path
            speaker_id = audio_path.parent.parent.name
            
            # Get audio info
            info = torchaudio.info(str(audio_path))
            duration = info.num_frames / info.sample_rate
            
            # Filter by duration
            if self.min_duration <= duration <= self.max_duration:
                samples.append({
                    'audio_path': str(audio_path),
                    'speaker_id': speaker_id,
                    'duration': duration,
                })
        
        return samples
    
    def _load_vctk(self) -> List[Dict]:
        """Load VCTK dataset."""
        samples = []
        
        # VCTK structure: wav48/speaker_id/audio_file.wav
        wav_dir = self.data_dir / "wav48"
        if not wav_dir.exists():
            wav_dir = self.data_dir / "wav48_silence_trimmed"
        
        audio_files = list(wav_dir.rglob("*.wav"))
        
        for audio_path in audio_files:
            speaker_id = audio_path.parent.name
            
            info = torchaudio.info(str(audio_path))
            duration = info.num_frames / info.sample_rate
            
            if self.min_duration <= duration <= self.max_duration:
                samples.append({
                    'audio_path': str(audio_path),
                    'speaker_id': speaker_id,
                    'duration': duration,
                })
        
        return samples
    
    def _load_crema_d(self) -> List[Dict]:
        """Load CREMA-D emotional speech dataset."""
        samples = []
        
        # CREMA-D structure: AudioWAV/speaker_sentence_emotion_intensity.wav
        audio_dir = self.data_dir / "AudioWAV"
        audio_files = list(audio_dir.glob("*.wav"))
        
        for audio_path in audio_files:
            # Parse filename: 1001_DFA_ANG_XX.wav
            parts = audio_path.stem.split('_')
            speaker_id = parts[0]
            emotion = parts[2] if len(parts) > 2 else 'NEU'
            
            info = torchaudio.info(str(audio_path))
            duration = info.num_frames / info.sample_rate
            
            if self.min_duration <= duration <= self.max_duration:
                samples.append({
                    'audio_path': str(audio_path),
                    'speaker_id': speaker_id,
                    'emotion': emotion,
                    'duration': duration,
                })
        
        return samples
    
    def _load_custom(self) -> List[Dict]:
        """Load custom dataset from metadata JSON."""
        metadata_path = self.data_dir / f"{self.split}_metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            samples = json.load(f)
        
        return samples
    
    def _build_speaker_mapping(self) -> Dict[str, int]:
        """Build speaker ID to index mapping."""
        speaker_ids = sorted(set(s['speaker_id'] for s in self.samples))
        return {spk: idx for idx, spk in enumerate(speaker_ids)}
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            Dictionary containing:
                - source_audio: Source audio tensor
                - reference_audio: Reference audio from same speaker
                - speaker_id: Speaker index
                - emotion: Emotion label (if available)
        """
        sample = self.samples[idx]
        
        # Load source audio
        waveform, sr = self.audio_processor.load_audio(sample['audio_path'])
        source_audio = self.audio_processor.preprocess_audio(waveform, sr)
        
        # Get reference audio from same speaker
        speaker_samples = [s for s in self.samples 
                          if s['speaker_id'] == sample['speaker_id'] 
                          and s['audio_path'] != sample['audio_path']]
        
        if speaker_samples:
            ref_sample = random.choice(speaker_samples)
            ref_waveform, ref_sr = self.audio_processor.load_audio(ref_sample['audio_path'])
            reference_audio = self.audio_processor.preprocess_audio(ref_waveform, ref_sr)
        else:
            # Use same audio as reference if no other samples
            reference_audio = source_audio.clone()
        
        # Get speaker index
        speaker_idx = self.speaker_to_idx[sample['speaker_id']]
        
        # Prepare output
        output = {
            'source_audio': source_audio.squeeze(0),
            'reference_audio': reference_audio.squeeze(0),
            'speaker_id': torch.tensor(speaker_idx, dtype=torch.long),
        }
        
        # Add emotion if available
        if 'emotion' in sample:
            output['emotion'] = sample['emotion']
        
        return output


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching variable-length audio.
    
    Pads audio to same length within batch.
    """
    # Find max length
    max_source_len = max(item['source_audio'].shape[0] for item in batch)
    max_ref_len = max(item['reference_audio'].shape[0] for item in batch)
    
    # Pad and stack
    source_audios = []
    reference_audios = []
    speaker_ids = []
    
    for item in batch:
        # Pad source audio
        source = item['source_audio']
        if source.shape[0] < max_source_len:
            source = torch.nn.functional.pad(source, (0, max_source_len - source.shape[0]))
        source_audios.append(source)
        
        # Pad reference audio
        reference = item['reference_audio']
        if reference.shape[0] < max_ref_len:
            reference = torch.nn.functional.pad(reference, (0, max_ref_len - reference.shape[0]))
        reference_audios.append(reference)
        
        speaker_ids.append(item['speaker_id'])
    
    return {
        'source_audio': torch.stack(source_audios),
        'reference_audio': torch.stack(reference_audios),
        'speaker_id': torch.stack(speaker_ids),
    }
