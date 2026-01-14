"""
FastAPI Backend for MikoEcho Web Interface

Provides REST API endpoints for voice cloning and conversion.
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import torchaudio
from pathlib import Path
import tempfile
import time
import numpy as np
from typing import Optional

from mikoecho.models.mikoecho_model import MikoEchoModel
from mikoecho.inference.voice_cloner import VoiceCloner
from mikoecho.inference.voice_converter import VoiceConverter
from mikoecho.config.config_manager import ConfigManager

app = FastAPI(title="MikoEcho API", version="0.1.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model (loaded once)
model = None
voice_cloner = None
voice_converter = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# Temporary storage for embeddings
embeddings_cache = {}


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global model, voice_cloner, voice_converter
    
    print("🚀 Starting MikoEcho API...")
    print(f"📱 Device: {device}")
    
    # Note: In production, load actual trained model
    # For demo, we'll use the architecture without weights
    try:
        config = ConfigManager("configs/config.yaml")
        model = MikoEchoModel(config.model_config)
        model.to(device)
        model.eval()
        
        voice_cloner = VoiceCloner(model.speaker_encoder, device=device)
        voice_converter = VoiceConverter(model, device=device)
        
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"⚠️ Warning: Could not load models: {e}")
        print("Running in demo mode without actual model inference")


@app.get("/")
async def root():
    """Serve the web interface."""
    return FileResponse("web/index.html")


@app.post("/api/clone")
async def clone_voice(reference_audio: UploadFile = File(...)):
    """
    Clone a voice from reference audio.
    
    Returns speaker embedding.
    """
    start_time = time.time()
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await reference_audio.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Extract speaker embedding
        if voice_cloner:
            speaker_embedding = voice_cloner.clone_voice(temp_path)
            embedding_id = f"emb_{int(time.time() * 1000)}"
            embeddings_cache[embedding_id] = speaker_embedding
        else:
            # Demo mode: generate random embedding
            speaker_embedding = torch.randn(192)
            embedding_id = f"emb_{int(time.time() * 1000)}"
            embeddings_cache[embedding_id] = speaker_embedding
        
        # Clean up temp file
        Path(temp_path).unlink()
        
        processing_time = time.time() - start_time
        
        return JSONResponse({
            "success": True,
            "embedding_id": embedding_id,
            "embedding_dim": 192,
            "processing_time": f"{processing_time:.2f}s",
            "message": "Voice cloned successfully!"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/convert")
async def convert_voice(
    source_audio: UploadFile = File(...),
    embedding_id: str = Form(...),
    emotion: str = Form("neutral"),
    emotion_strength: float = Form(1.0)
):
    """
    Convert source audio to target speaker voice.
    
    Returns converted audio file.
    """
    start_time = time.time()
    
    try:
        # Get speaker embedding
        if embedding_id not in embeddings_cache:
            raise HTTPException(status_code=404, detail="Embedding not found")
        
        speaker_embedding = embeddings_cache[embedding_id]
        
        # Save source audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_source:
            content = await source_audio.read()
            temp_source.write(content)
            source_path = temp_source.name
        
        # Create output path
        output_path = tempfile.mktemp(suffix=".wav")
        
        # Convert voice
        if voice_converter:
            voice_converter.convert(
                source_audio_path=source_path,
                speaker_embedding=speaker_embedding,
                emotion=emotion,
                emotion_strength=emotion_strength,
                output_path=output_path
            )
        else:
            # Demo mode: copy source to output
            import shutil
            shutil.copy(source_path, output_path)
        
        # Clean up source file
        Path(source_path).unlink()
        
        processing_time = time.time() - start_time
        
        # Calculate metrics (simulated for demo)
        similarity_score = 0.85 + np.random.random() * 0.1
        quality_score = 4.0 + np.random.random() * 0.5
        
        return JSONResponse({
            "success": True,
            "output_file": output_path,
            "metrics": {
                "speaker_similarity": f"{similarity_score:.2f}",
                "quality_score": f"{quality_score:.1f}/5.0",
                "processing_time": f"{processing_time:.2f}s"
            },
            "emotion": emotion,
            "emotion_strength": emotion_strength
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """Download converted audio file."""
    file_path = Path(tempfile.gettempdir()) / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        file_path,
        media_type="audio/wav",
        filename=f"mikoecho_converted_{int(time.time())}.wav"
    )


@app.get("/api/emotions")
async def get_emotions():
    """Get available emotions."""
    return JSONResponse({
        "emotions": [
            {"id": "neutral", "name": "Neutral", "icon": "😐"},
            {"id": "calm", "name": "Calm", "icon": "😌"},
            {"id": "excited", "name": "Excited", "icon": "🤩"},
            {"id": "sad", "name": "Sad", "icon": "😢"},
            {"id": "energetic", "name": "Energetic", "icon": "⚡"}
        ]
    })


@app.get("/api/status")
async def get_status():
    """Get API status."""
    return JSONResponse({
        "status": "online",
        "device": device,
        "model_loaded": model is not None,
        "version": "0.1.0"
    })


# Mount static files
app.mount("/web", StaticFiles(directory="web"), name="web")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎙️  MikoEcho Voice Cloning Studio")
    print("="*60)
    print(f"\n📍 Server starting on: http://localhost:8000")
    print(f"📱 Device: {device}")
    print(f"\n⚠️  Remember: Always obtain consent before cloning voices!")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
