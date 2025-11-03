"""Simplified mel spectrogram quantization for training

This provides a quick quantization method that doesn't require a trained DAC model.
Quality: Acceptable for initial training and testing.
For production: Use proper DAC encoding.
"""
import torch
import torch.nn.functional as F
from typing import Tuple


def simple_mel_quantization(mel_spectrogram: torch.Tensor, n_codes: int = 8194) -> torch.Tensor:
    """
    Simple uniform quantization of mel spectrograms to discrete codes
    
    Args:
        mel_spectrogram: Mel spectrogram tensor [batch, n_mels, time] or [n_mels, time]
        n_codes: Number of quantization bins (default 8194 for IndexTTS2)
        
    Returns:
        Quantized codes [batch, time] or [time]
    """
    original_shape = mel_spectrogram.shape
    is_batched = len(original_shape) == 3
    
    if not is_batched:
        mel_spectrogram = mel_spectrogram.unsqueeze(0)
    
    # Take mean across mel bins to get time-series
    # Shape: [batch, n_mels, time] -> [batch, time]
    mel_compressed = mel_spectrogram.mean(dim=1)
    
    # Normalize to [0, 1] per sample
    batch_size, time_len = mel_compressed.shape
    codes_list = []
    
    for i in range(batch_size):
        mel_sample = mel_compressed[i]
        
        # Min-max normalization
        min_val = mel_sample.min()
        max_val = mel_sample.max()
        
        if max_val - min_val > 1e-6:
            mel_norm = (mel_sample - min_val) / (max_val - min_val)
        else:
            mel_norm = torch.zeros_like(mel_sample)
        
        # Quantize to discrete bins
        codes = (mel_norm * (n_codes - 1)).long()
        codes = codes.clamp(0, n_codes - 1)
        
        codes_list.append(codes)
    
    codes = torch.stack(codes_list)
    
    if not is_batched:
        codes = codes.squeeze(0)
    
    return codes


def kmeans_mel_quantization(
    mel_spectrogram: torch.Tensor,
    n_codes: int = 8194,
    n_iter: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    K-means based quantization (better quality than uniform)
    
    Args:
        mel_spectrogram: Mel spectrogram [batch, n_mels, time]
        n_codes: Number of clusters
        n_iter: K-means iterations
        
    Returns:
        Tuple of (codes, codebook)
    """
    original_shape = mel_spectrogram.shape
    is_batched = len(original_shape) == 3
    
    if not is_batched:
        mel_spectrogram = mel_spectrogram.unsqueeze(0)
    
    batch_size, n_mels, time_len = mel_spectrogram.shape
    
    # Reshape to [batch * time, n_mels]
    mel_flat = mel_spectrogram.transpose(1, 2).reshape(-1, n_mels)
    
    # Initialize centroids randomly
    indices = torch.randperm(mel_flat.shape[0])[:n_codes]
    centroids = mel_flat[indices].clone()
    
    # K-means iterations
    for _ in range(n_iter):
        # Assign to nearest centroid
        distances = torch.cdist(mel_flat, centroids)
        assignments = distances.argmin(dim=1)
        
        # Update centroids
        for k in range(n_codes):
            mask = assignments == k
            if mask.sum() > 0:
                centroids[k] = mel_flat[mask].mean(dim=0)
    
    # Final assignment
    distances = torch.cdist(mel_flat, centroids)
    codes = distances.argmin(dim=1)
    
    # Reshape back
    codes = codes.reshape(batch_size, time_len)
    
    if not is_batched:
        codes = codes.squeeze(0)
    
    return codes, centroids


def quantize_mel_batch(mel_spectrograms: torch.Tensor, method: str = 'simple', **kwargs) -> torch.Tensor:
    """
    Batch quantization with choice of method
    
    Args:
        mel_spectrograms: Batch of mel spectrograms [batch, n_mels, time]
        method: 'simple' or 'kmeans'
        **kwargs: Additional arguments for quantization method
        
    Returns:
        Quantized codes [batch, time]
    """
    if method == 'simple':
        return simple_mel_quantization(mel_spectrograms, **kwargs)
    elif method == 'kmeans':
        codes, _ = kmeans_mel_quantization(mel_spectrograms, **kwargs)
        return codes
    else:
        raise ValueError(f"Unknown quantization method: {method}")
