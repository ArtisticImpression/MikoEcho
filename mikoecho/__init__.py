"""
MikoEcho - Production-Grade Voice Cloning & Voice-to-Voice Transformation

A neural voice system for voice cloning and real-time voice conversion.
Developed by Artistic Impression.

Copyright (c) 2026 Artistic Impression
"""

__version__ = "0.1.0"
__author__ = "Artistic Impression"
__license__ = "MIT"

from mikoecho.models.mikoecho_model import MikoEchoModel
from mikoecho.inference.voice_cloner import VoiceCloner
from mikoecho.inference.voice_converter import VoiceConverter

__all__ = [
    "MikoEchoModel",
    "VoiceCloner",
    "VoiceConverter",
]
