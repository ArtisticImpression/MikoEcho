"""
MikoEcho Inference CLI

Command-line interface for voice cloning and conversion.
"""

import argparse
import torch
from pathlib import Path

from mikoecho.models.mikoecho_model import MikoEchoModel
from mikoecho.inference.voice_cloner import VoiceCloner
from mikoecho.inference.voice_converter import VoiceConverter
from mikoecho.config.config_manager import ConfigManager
from mikoecho.utils.checkpoint import CheckpointManager


def main():
    parser = argparse.ArgumentParser(description="MikoEcho Voice Cloning & Conversion")
    
    # Mode
    parser.add_argument(
        "mode",
        choices=["clone", "convert"],
        help="Operation mode: clone (extract speaker embedding) or convert (voice conversion)"
    )
    
    # Common arguments
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device to use")
    
    # Clone mode arguments
    parser.add_argument("--reference", type=str,
                       help="Path to reference audio for voice cloning")
    parser.add_argument("--output-embedding", type=str,
                       help="Path to save speaker embedding (.pt or .npy)")
    
    # Convert mode arguments
    parser.add_argument("--source", type=str,
                       help="Path to source audio to convert")
    parser.add_argument("--speaker-embedding", type=str,
                       help="Path to speaker embedding file")
    parser.add_argument("--emotion", type=str, default="neutral",
                       choices=["neutral", "calm", "excited", "sad", "energetic"],
                       help="Emotion for conversion")
    parser.add_argument("--emotion-strength", type=float, default=1.0,
                       help="Emotion strength (0-1)")
    parser.add_argument("--output", type=str,
                       help="Path to save converted audio")
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load config
    config_manager = ConfigManager(args.config)
    model_config = config_manager.model_config
    
    # Initialize model
    print("Loading model...")
    model = MikoEchoModel(model_config)
    
    # Load checkpoint
    checkpoint_manager = CheckpointManager()
    checkpoint = checkpoint_manager.load_checkpoint(
        args.checkpoint,
        model,
        device=device,
    )
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, step {checkpoint['step']}")
    
    model.to(device)
    model.eval()
    
    # Execute mode
    if args.mode == "clone":
        if not args.reference:
            parser.error("--reference is required for clone mode")
        if not args.output_embedding:
            parser.error("--output-embedding is required for clone mode")
        
        print(f"Cloning voice from: {args.reference}")
        
        # Initialize voice cloner
        cloner = VoiceCloner(model.speaker_encoder, device=device)
        
        # Clone voice
        speaker_embedding = cloner.clone_voice(args.reference)
        
        # Save embedding
        cloner.save_embedding(speaker_embedding, args.output_embedding)
        
        print(f"Speaker embedding saved to: {args.output_embedding}")
    
    elif args.mode == "convert":
        if not args.source:
            parser.error("--source is required for convert mode")
        if not args.output:
            parser.error("--output is required for convert mode")
        
        # Initialize voice converter
        converter = VoiceConverter(model, device=device)
        
        # Load or extract speaker embedding
        if args.speaker_embedding:
            print(f"Loading speaker embedding from: {args.speaker_embedding}")
            cloner = VoiceCloner(model.speaker_encoder, device=device)
            speaker_embedding = cloner.load_embedding(args.speaker_embedding)
        elif args.reference:
            print(f"Extracting speaker embedding from: {args.reference}")
            cloner = VoiceCloner(model.speaker_encoder, device=device)
            speaker_embedding = cloner.clone_voice(args.reference)
        else:
            parser.error("Either --speaker-embedding or --reference is required for convert mode")
        
        print(f"Converting: {args.source}")
        print(f"Emotion: {args.emotion} (strength: {args.emotion_strength})")
        
        # Convert voice
        output_audio = converter.convert(
            args.source,
            speaker_embedding,
            emotion=args.emotion,
            emotion_strength=args.emotion_strength,
            output_path=args.output,
        )
        
        print(f"Converted audio saved to: {args.output}")


if __name__ == "__main__":
    main()
