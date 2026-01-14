// MikoEcho Web Interface - Interactive Audio Visualizations
// This demo uses simulated processing - connect to actual backend for production

class MikoEchoApp {
    constructor() {
        this.referenceFile = null;
        this.sourceFile = null;
        this.speakerEmbedding = null;
        this.selectedEmotion = 'neutral';
        this.emotionStrength = 1.0;
        this.ttsEmotion = 'neutral';
        this.ttsEmotionStrength = 1.0;
        this.speechSpeed = 1.0;
        this.generatedSpeechAudio = null;

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupAudioContext();
        this.setupTTSListeners();
    }

    setupEventListeners() {
        // Reference upload
        const referenceUpload = document.getElementById('reference-upload');
        const referenceFile = document.getElementById('reference-file');

        referenceUpload.addEventListener('click', () => referenceFile.click());
        referenceUpload.addEventListener('dragover', (e) => {
            e.preventDefault();
            referenceUpload.style.borderColor = 'var(--primary-dark)';
        });
        referenceUpload.addEventListener('dragleave', () => {
            referenceUpload.style.borderColor = 'var(--primary)';
        });
        referenceUpload.addEventListener('drop', (e) => {
            e.preventDefault();
            this.handleReferenceFile(e.dataTransfer.files[0]);
        });
        referenceFile.addEventListener('change', (e) => {
            this.handleReferenceFile(e.target.files[0]);
        });

        // Source upload
        const sourceUpload = document.getElementById('source-upload');
        const sourceFile = document.getElementById('source-file');

        sourceUpload.addEventListener('click', () => sourceFile.click());
        sourceUpload.addEventListener('dragover', (e) => {
            e.preventDefault();
            sourceUpload.style.borderColor = 'var(--primary-dark)';
        });
        sourceUpload.addEventListener('dragleave', () => {
            sourceUpload.style.borderColor = 'var(--primary)';
        });
        sourceUpload.addEventListener('drop', (e) => {
            e.preventDefault();
            this.handleSourceFile(e.dataTransfer.files[0]);
        });
        sourceFile.addEventListener('change', (e) => {
            this.handleSourceFile(e.target.files[0]);
        });

        // Clone button
        document.getElementById('clone-btn').addEventListener('click', () => {
            this.cloneVoice();
        });

        // Convert button
        document.getElementById('convert-btn').addEventListener('click', () => {
            this.convertVoice();
        });

        // Emotion buttons
        document.querySelectorAll('.emotion-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.emotion-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.selectedEmotion = btn.dataset.emotion;
            });
        });

        // Emotion strength slider
        const strengthSlider = document.getElementById('emotion-strength');
        const strengthValue = document.getElementById('emotion-value');
        strengthSlider.addEventListener('input', (e) => {
            this.emotionStrength = e.target.value / 100;
            strengthValue.textContent = e.target.value + '%';
        });

        // Download button
        document.getElementById('download-btn').addEventListener('click', () => {
            this.downloadResult();
        });

        // New conversion button
        document.getElementById('new-conversion-btn').addEventListener('click', () => {
            this.resetApp();
        });
    }

    setupAudioContext() {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }

    async handleReferenceFile(file) {
        if (!file || !file.type.startsWith('audio/')) {
            alert('Please select a valid audio file');
            return;
        }

        this.referenceFile = file;

        // Show preview
        const preview = document.getElementById('reference-preview');
        const audio = document.getElementById('reference-audio');
        const fileName = document.getElementById('reference-name');
        const duration = document.getElementById('reference-duration');

        const url = URL.createObjectURL(file);
        audio.src = url;
        fileName.textContent = file.name;

        audio.addEventListener('loadedmetadata', () => {
            const mins = Math.floor(audio.duration / 60);
            const secs = Math.floor(audio.duration % 60);
            duration.textContent = `${mins}:${secs.toString().padStart(2, '0')}`;

            // Draw waveform
            this.drawWaveform(file, 'reference-waveform');
        });

        preview.style.display = 'block';
        document.getElementById('clone-btn').disabled = false;
    }

    async handleSourceFile(file) {
        if (!file || !file.type.startsWith('audio/')) {
            alert('Please select a valid audio file');
            return;
        }

        this.sourceFile = file;

        // Show preview
        const preview = document.getElementById('source-preview');
        const audio = document.getElementById('source-audio');
        const fileName = document.getElementById('source-name');
        const duration = document.getElementById('source-duration');

        const url = URL.createObjectURL(file);
        audio.src = url;
        fileName.textContent = file.name;

        audio.addEventListener('loadedmetadata', () => {
            const mins = Math.floor(audio.duration / 60);
            const secs = Math.floor(audio.duration % 60);
            duration.textContent = `${mins}:${secs.toString().padStart(2, '0')}`;

            // Draw waveform
            this.drawWaveform(file, 'source-waveform');
        });

        preview.style.display = 'block';
        document.getElementById('emotion-controls').style.display = 'block';

        if (this.speakerEmbedding) {
            document.getElementById('convert-btn').disabled = false;
        }
    }

    async drawWaveform(file, canvasId) {
        const canvas = document.getElementById(canvasId);
        const ctx = canvas.getContext('2d');
        const width = canvas.width = canvas.offsetWidth * 2;
        const height = canvas.height = canvas.offsetHeight * 2;

        try {
            const arrayBuffer = await file.arrayBuffer();
            const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
            const data = audioBuffer.getChannelData(0);
            const step = Math.ceil(data.length / width);
            const amp = height / 2;

            // Clear canvas
            ctx.clearRect(0, 0, width, height);

            // Draw waveform
            ctx.fillStyle = 'rgba(99, 102, 241, 0.3)';
            ctx.strokeStyle = 'rgb(99, 102, 241)';
            ctx.lineWidth = 2;

            ctx.beginPath();
            ctx.moveTo(0, amp);

            for (let i = 0; i < width; i++) {
                const min = Math.min(...data.slice(i * step, (i + 1) * step));
                const max = Math.max(...data.slice(i * step, (i + 1) * step));

                ctx.lineTo(i, (1 + min) * amp);
            }

            for (let i = width - 1; i >= 0; i--) {
                const min = Math.min(...data.slice(i * step, (i + 1) * step));
                const max = Math.max(...data.slice(i * step, (i + 1) * step));

                ctx.lineTo(i, (1 + max) * amp);
            }

            ctx.closePath();
            ctx.fill();
            ctx.stroke();

        } catch (error) {
            console.error('Error drawing waveform:', error);
        }
    }

    async cloneVoice() {
        const progressContainer = document.getElementById('clone-progress');
        const progressFill = document.getElementById('clone-progress-fill');
        const progressText = document.getElementById('clone-progress-text');
        const resultCard = document.getElementById('clone-result');

        progressContainer.style.display = 'block';

        // Simulate processing
        const steps = [
            { progress: 20, text: 'Loading audio file...' },
            { progress: 40, text: 'Extracting features with HuBERT...' },
            { progress: 60, text: 'Computing speaker embedding...' },
            { progress: 80, text: 'Normalizing embedding...' },
            { progress: 100, text: 'Voice cloned successfully!' }
        ];

        for (const step of steps) {
            await this.sleep(800);
            progressFill.style.width = step.progress + '%';
            progressText.textContent = step.text;
        }

        // Simulate speaker embedding
        this.speakerEmbedding = new Array(192).fill(0).map(() => Math.random() * 2 - 1);

        // Show result
        await this.sleep(500);
        progressContainer.style.display = 'none';
        resultCard.style.display = 'block';

        // Draw embedding visualization
        this.drawEmbeddingViz();

        // Enable convert button if source is loaded
        if (this.sourceFile) {
            document.getElementById('convert-btn').disabled = false;
        }
    }

    drawEmbeddingViz() {
        const container = document.getElementById('embedding-viz');
        container.innerHTML = '';

        // Create bar chart visualization
        const bars = 50;
        const values = this.speakerEmbedding.slice(0, bars);

        for (let i = 0; i < bars; i++) {
            const bar = document.createElement('div');
            bar.style.cssText = `
                display: inline-block;
                width: ${100 / bars}%;
                height: ${Math.abs(values[i]) * 50 + 20}px;
                background: linear-gradient(135deg, #667eea, #764ba2);
                margin: 0 1px;
                vertical-align: bottom;
                border-radius: 2px;
                animation: fadeInUp 0.3s ease-out ${i * 0.01}s both;
            `;
            container.appendChild(bar);
        }
    }

    async convertVoice() {
        const progressContainer = document.getElementById('convert-progress');
        const progressFill = document.getElementById('convert-progress-fill');
        const progressText = document.getElementById('convert-progress-text');

        progressContainer.style.display = 'block';

        // Simulate processing
        const steps = [
            { progress: 15, text: 'Loading source audio...' },
            { progress: 30, text: 'Extracting content features...' },
            { progress: 45, text: 'Disentangling speaker identity...' },
            { progress: 60, text: `Applying ${this.selectedEmotion} emotion...` },
            { progress: 75, text: 'Fusing features...' },
            { progress: 90, text: 'Generating waveform with HiFi-GAN...' },
            { progress: 100, text: 'Conversion complete!' }
        ];

        for (const step of steps) {
            await this.sleep(1000);
            progressFill.style.width = step.progress + '%';
            progressText.textContent = step.text;
        }

        // Show results
        await this.sleep(500);
        this.showResults();
    }

    async showResults() {
        const resultsSection = document.getElementById('results-section');
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });

        // Set audio sources
        const originalAudio = document.getElementById('original-comparison');
        const convertedAudio = document.getElementById('converted-audio');

        originalAudio.src = URL.createObjectURL(this.sourceFile);
        convertedAudio.src = URL.createObjectURL(this.sourceFile); // Demo: same file

        // Draw spectrograms
        this.drawSpectrogram('original-spectrogram');
        this.drawSpectrogram('converted-spectrogram');

        // Show metrics
        document.getElementById('similarity-score').textContent = '0.89';
        document.getElementById('quality-score').textContent = '4.3/5.0';
        document.getElementById('processing-time').textContent = '2.4s';

        // Animate metrics
        this.animateMetrics();
    }

    drawSpectrogram(canvasId) {
        const canvas = document.getElementById(canvasId);
        const ctx = canvas.getContext('2d');
        const width = canvas.width = canvas.offsetWidth * 2;
        const height = canvas.height = canvas.offsetHeight * 2;

        // Create gradient spectrogram effect
        const gradient = ctx.createLinearGradient(0, 0, 0, height);
        gradient.addColorStop(0, '#667eea');
        gradient.addColorStop(0.5, '#764ba2');
        gradient.addColorStop(1, '#f093fb');

        ctx.fillStyle = '#1e293b';
        ctx.fillRect(0, 0, width, height);

        // Draw frequency bands
        for (let x = 0; x < width; x += 4) {
            for (let y = 0; y < height; y += 4) {
                const intensity = Math.random() * 0.8 + 0.2;
                const alpha = intensity * (1 - y / height);
                ctx.fillStyle = `rgba(102, 126, 234, ${alpha})`;
                ctx.fillRect(x, y, 3, 3);
            }
        }
    }

    animateMetrics() {
        const metrics = document.querySelectorAll('.metric-value');
        metrics.forEach((metric, index) => {
            metric.style.animation = `fadeInUp 0.6s ease-out ${index * 0.2}s both`;
        });
    }

    downloadResult() {
        // In production, this would download the actual converted audio
        alert('Download functionality will be connected to backend API');
    }

    resetApp() {
        location.reload();
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new MikoEchoApp();
});
