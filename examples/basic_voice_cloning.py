"""
Example: Basic Voice Cloning

This example demonstrates how to clone a voice and use it for conversion.
"""

from mikoecho import VoiceCloner, VoiceConverter, MikoEchoModel
from mikoecho.config import ConfigManager

def main():
    # Load configuration
    config = ConfigManager("configs/config.yaml")
    
    # Initialize model
    print("Loading MikoEcho model...")
    model = MikoEchoModel(config.model_config)
    
    # Load checkpoint (you need a trained model)
    # from mikoecho.utils import CheckpointManager
    # checkpoint_manager = CheckpointManager()
    # checkpoint_manager.load_best(model, device="cuda")
    
    # Initialize voice cloner
    print("Initializing voice cloner...")
    cloner = VoiceCloner(model.speaker_encoder, device="cuda")
    
    # Clone voice from reference audio
    print("Cloning voice from reference audio...")
    reference_audio = "examples/reference_speaker.wav"
    speaker_embedding = cloner.clone_voice(reference_audio)
    
    # Save speaker embedding for later use
    cloner.save_embedding(speaker_embedding, "speaker_embedding.pt")
    print("Speaker embedding saved!")
    
    # Initialize voice converter
    print("Initializing voice converter...")
    converter = VoiceConverter(model, device="cuda")
    
    # Convert your voice to the target speaker
    print("Converting voice...")
    source_audio = "examples/my_voice.wav"
    output_audio = converter.convert(
        source_audio_path=source_audio,
        speaker_embedding=speaker_embedding,
        emotion="neutral",
        emotion_strength=1.0,
        output_path="output_converted.wav"
    )
    
    print("Voice conversion complete! Output saved to: output_converted.wav")
    
    # Try different emotions
    emotions = ["calm", "excited", "sad", "energetic"]
    for emotion in emotions:
        print(f"Converting with emotion: {emotion}")
        converter.convert(
            source_audio_path=source_audio,
            speaker_embedding=speaker_embedding,
            emotion=emotion,
            emotion_strength=0.8,
            output_path=f"output_{emotion}.wav"
        )
    
    print("All conversions complete!")

if __name__ == "__main__":
    main()
