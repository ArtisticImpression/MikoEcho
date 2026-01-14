# MikoEcho Web Interface

Beautiful, interactive web interface for MikoEcho voice cloning system.

## Features

- 🎨 **Stunning UI** - Modern gradient design with smooth animations
- 📊 **Real-time Visualizations** - Waveform and spectrogram displays
- 🎭 **Emotion Control** - Interactive emotion selection with strength slider
- 📈 **Live Metrics** - Speaker similarity, quality scores, processing time
- 🔄 **Drag & Drop** - Easy file uploads with preview
- 📱 **Responsive** - Works on desktop and mobile

## Quick Start

### 1. Install Dependencies

```bash
pip install fastapi uvicorn python-multipart
```

### 2. Start the Server

```bash
# From MikoEcho root directory
python scripts/api.py
```

### 3. Open in Browser

Navigate to: http://localhost:8000

## Usage

### Step 1: Clone a Voice
1. Upload or drag & drop reference audio (3-30 seconds)
2. Preview the audio and waveform
3. Click "Clone Voice" to extract speaker embedding
4. Wait for processing to complete

### Step 2: Convert Your Voice
1. Upload your audio file
2. Select desired emotion (Neutral, Calm, Excited, Sad, Energetic)
3. Adjust emotion strength (0-100%)
4. Click "Convert Voice"

### Step 3: Review Results
- Compare original vs converted audio
- View spectrograms side-by-side
- Check quality metrics
- Download the result

## Architecture

```
web/
├── index.html      # Main interface
├── style.css       # Styling with gradients & animations
└── app.js          # Interactive visualizations

scripts/
└── api.py          # FastAPI backend
```

## API Endpoints

- `POST /api/clone` - Clone voice from reference audio
- `POST /api/convert` - Convert voice with emotion control
- `GET /api/emotions` - Get available emotions
- `GET /api/status` - Check API status
- `GET /api/download/{filename}` - Download converted audio

## Visualizations

### Waveform Display
- Real-time audio waveform rendering
- Amplitude visualization
- Interactive canvas

### Spectrogram
- Frequency-time representation
- Color-coded intensity
- Beautiful gradient effects

### Embedding Visualization
- Speaker embedding bar chart
- Animated appearance
- 192-dimensional vector display

## Customization

### Colors
Edit `style.css` to change color scheme:
```css
:root {
    --primary: #6366f1;
    --secondary: #8b5cf6;
    --success: #10b981;
}
```

### Emotions
Add new emotions in `api.py`:
```python
emotions = [
    {"id": "custom", "name": "Custom", "icon": "🎯"}
]
```

## Production Deployment

### With Trained Model

1. Train MikoEcho model
2. Place checkpoint in `checkpoints/best_model.pt`
3. Update `api.py` to load checkpoint:

```python
from mikoecho.utils import CheckpointManager

checkpoint_manager = CheckpointManager()
checkpoint_manager.load_best(model, device=device)
```

### Docker Deployment

```dockerfile
FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "scripts/api.py"]
```

### Nginx Reverse Proxy

```nginx
location / {
    proxy_pass http://localhost:8000;
    proxy_set_header Host $host;
}
```

## Browser Compatibility

- ✅ Chrome/Edge (recommended)
- ✅ Firefox
- ✅ Safari
- ⚠️ IE11 (limited support)

## Performance

- **Waveform rendering**: <100ms
- **Spectrogram generation**: <200ms
- **File upload**: Instant preview
- **Smooth animations**: 60fps

## Security

- CORS enabled for development
- File type validation
- Size limits enforced
- Temporary file cleanup

## Troubleshooting

### Port Already in Use
```bash
# Change port in api.py
uvicorn.run(app, port=8001)
```

### Audio Not Playing
- Check browser audio permissions
- Verify file format (WAV, MP3, FLAC)
- Check console for errors

### Visualizations Not Showing
- Enable JavaScript
- Check canvas support
- Update browser

## Credits

Built with:
- FastAPI
- HTML5 Canvas
- Web Audio API
- CSS Gradients & Animations

---

**Artistic Impression** | MikoEcho v0.1.0
