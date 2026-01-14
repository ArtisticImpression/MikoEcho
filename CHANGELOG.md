# Changelog

All notable changes to MikoEcho will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-14

### Added
- Initial release of MikoEcho
- Core voice cloning architecture with HuBERT, ECAPA-TDNN, and HiFi-GAN
- Voice-to-voice conversion with emotion control
- Support for 5 emotions: neutral, calm, excited, sad, energetic
- CLI tools for inference (`mikoecho-infer`)
- Dataset support for LibriSpeech, VCTK, and CREMA-D
- Audio augmentation pipeline (noise, reverb, pitch shift, speed change)
- Comprehensive training system with multi-GPU support
- Loss functions: reconstruction, speaker similarity, content preservation
- Evaluation metrics: speaker similarity, MOS estimation, pitch correlation
- TensorBoard integration for training monitoring
- Checkpoint management with automatic cleanup
- Configuration system using YAML
- Complete documentation (README, CONTRIBUTING, ETHICS, MODEL_CARD)
- Example scripts for basic usage and emotion interpolation
- MIT License with ethical use addendum

### Features
- Clone voices from 3-30 seconds of reference audio
- High-fidelity audio generation (22050 Hz)
- Emotion strength control (0-1 scale)
- Batch conversion support
- Speaker embedding save/load functionality
- Offline, local processing (no external APIs)

### Documentation
- Comprehensive README with installation and usage
- Ethics and safety guidelines
- Contributing guide for developers
- Model card with performance metrics and limitations
- Code examples and tutorials

### Dependencies
- PyTorch 2.0+
- torchaudio
- transformers (HuBERT)
- speechbrain (ECAPA-TDNN)
- librosa, soundfile
- tensorboard, wandb
- fastapi, uvicorn

## [Unreleased]

### Planned
- Real-time streaming inference
- ONNX export for deployment
- Multi-language support
- Additional emotion categories
- Improved speaker verification
- Unit and integration tests
- Audio watermarking
- Web-based demo interface
