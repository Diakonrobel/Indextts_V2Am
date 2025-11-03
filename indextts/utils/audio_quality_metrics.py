import torch
import torchaudio
import numpy as np
from typing import Dict


def calculate_audio_quality_metrics(audio_path: str) -> Dict:
    try:
        audio, sr = torchaudio.load(audio_path)
        
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        audio_np = audio.numpy().flatten()
        
        # Basic metrics
        rms = float(np.sqrt(np.mean(audio_np ** 2)))
        peak = float(np.max(np.abs(audio_np)))
        duration = len(audio_np) / sr
        
        # Zero Crossing Rate
        zcr = float(np.sum(np.diff(np.sign(audio_np)) != 0) / len(audio_np))
        
        # Spectral features
        mel_spec = torchaudio.transforms.MelSpectrogram(sr)(audio)
        spectral_centroid = float(torch.mean(torch.argmax(mel_spec, dim=1)))
        
        # Quality score (0-10)
        quality_score = calculate_quality_score(rms, peak, zcr)
        
        return {
            'rms_energy': round(rms, 4),
            'peak_level': round(peak, 4),
            'duration_seconds': round(duration, 2),
            'zero_crossing_rate': round(zcr, 4),
            'spectral_centroid': round(spectral_centroid, 2),
            'quality_score': round(quality_score, 2),
            'sample_rate': sr,
            'is_clipping': peak > 0.99,
            'is_too_quiet': rms < 0.01,
            'is_noisy': zcr > 0.5
        }
    
    except Exception as e:
        return {'error': str(e)}


def calculate_quality_score(rms: float, peak: float, zcr: float) -> float:
    score = 10.0
    
    # Penalize clipping
    if peak > 0.99:
        score -= 3.0
    
    # Penalize too quiet
    if rms < 0.01:
        score -= 2.0
    
    # Penalize high noise
    if zcr > 0.5:
        score -= 2.0
    
    # Reward good energy level
    if 0.1 < rms < 0.7:
        score += 1.0
    
    return max(0.0, min(10.0, score))
