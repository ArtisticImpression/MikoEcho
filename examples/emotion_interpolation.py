"""
Example: Emotion Interpolation

Demonstrates how to interpolate between different emotions.
"""

from mikoecho import VoiceConverter, MikoEchoModel
from mikoecho.config import ConfigManager
from mikoecho.models.style_encoder import StyleEncoder

def main():
    # Load model
    config = ConfigManager("configs/config.yaml")
    model = MikoEchoModel(config.model_config)
    
    # Initialize converter
    converter = VoiceConverter(model, device="cuda")
    
    # Load speaker embedding
    from mikoecho import VoiceCloner
    cloner = VoiceCloner(model.speaker_encoder, device="cuda")
    speaker_embedding = cloner.load_embedding("speaker_embedding.pt")
    
    # Interpolate between calm and excited
    print("Creating emotion interpolation...")
    
    source_audio = "examples/my_voice.wav"
    
    # Generate samples with different interpolation values
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        # 0.0 = calm, 1.0 = excited
        emotion = "calm" if alpha < 0.5 else "excited"
        strength = abs(alpha - 0.5) * 2  # Interpolation strength
        
        print(f"Alpha: {alpha:.2f}, Emotion: {emotion}, Strength: {strength:.2f}")
        
        converter.convert(
            source_audio_path=source_audio,
            speaker_embedding=speaker_embedding,
            emotion=emotion,
            emotion_strength=strength,
            output_path=f"output_interp_{alpha:.2f}.wav"
        )
    
    print("Emotion interpolation complete!")

if __name__ == "__main__":
    main()
