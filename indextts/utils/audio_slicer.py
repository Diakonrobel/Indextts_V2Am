"""Intelligent audio slicer using Voice Activity Detection (VAD)"""
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False


@dataclass
class AudioSlice:
    """Represents a sliced audio segment"""
    start_sample: int
    end_sample: int
    start_time: float
    end_time: float
    duration: float
    audio_data: torch.Tensor
    sample_rate: int
    rms_energy: float
    peak_level: float
    audio_path: Optional[str] = None
    

class IntelligentAudioSlicer:
    """VAD-based audio slicing with quality validation"""
    
    def __init__(
        self,
        sample_rate: int = 24000,
        min_duration: float = 1.0,
        max_duration: float = 10.0,
        target_duration: float = 3.0,
        vad_threshold: float = 0.5,
        min_silence_duration: float = 0.3,
        denoise: bool = False
    ):
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.target_duration = target_duration
        self.vad_threshold = vad_threshold
        self.min_silence_duration = min_silence_duration
        self.denoise = denoise
        self.logger = logging.getLogger(__name__)
        
        # Try to load Silero VAD
        self.vad_model = self._load_vad_model()
    
    def _load_vad_model(self):
        """Load Silero VAD model"""
        try:
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self.logger.info("✅ Loaded Silero VAD model")
            return model
        except Exception as e:
            self.logger.warning(f"Failed to load Silero VAD: {e}. Using energy-based fallback.")
            return None
    
    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """Load and preprocess audio file"""
        audio, sr = torchaudio.load(audio_path)
        
        # Convert to mono
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
        
        # Denoise if requested
        if self.denoise and NOISEREDUCE_AVAILABLE:
            audio_np = audio.squeeze().numpy()
            audio_np = nr.reduce_noise(y=audio_np, sr=self.sample_rate)
            audio = torch.from_numpy(audio_np).unsqueeze(0)
        
        # Normalize
        audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
        
        return audio.squeeze(0), self.sample_rate
    
    def detect_speech_segments(
        self,
        audio: torch.Tensor,
        sample_rate: int
    ) -> List[Tuple[int, int]]:
        """Detect speech segments using VAD"""
        if self.vad_model is not None:
            return self._vad_silero(audio, sample_rate)
        else:
            return self._vad_energy_based(audio, sample_rate)
    
    def _vad_silero(self, audio: torch.Tensor, sample_rate: int) -> List[Tuple[int, int]]:
        """Use Silero VAD for speech detection"""
        try:
            # Silero VAD expects 16kHz
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                audio_16k = resampler(audio.unsqueeze(0)).squeeze(0)
            else:
                audio_16k = audio
            
            # Get speech timestamps
            speech_timestamps = self.vad_model(
                audio_16k.unsqueeze(0),
                16000
            )
            
            # Convert back to original sample rate
            segments = []
            for ts in speech_timestamps:
                start_sample = int(ts['start'] * sample_rate / 16000)
                end_sample = int(ts['end'] * sample_rate / 16000)
                segments.append((start_sample, end_sample))
            
            return segments
            
        except Exception as e:
            self.logger.error(f"Silero VAD failed: {e}. Using energy fallback.")
            return self._vad_energy_based(audio, sample_rate)
    
    def _vad_energy_based(self, audio: torch.Tensor, sample_rate: int) -> List[Tuple[int, int]]:
        """Energy-based VAD fallback"""
        # Calculate short-term energy
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        hop_length = int(0.010 * sample_rate)    # 10ms hop
        
        # Pad audio
        audio_padded = torch.nn.functional.pad(audio, (0, frame_length))
        
        # Calculate energy per frame
        frames = audio_padded.unfold(0, frame_length, hop_length)
        energy = torch.mean(frames ** 2, dim=1)
        
        # Normalize energy
        energy = energy / (torch.max(energy) + 1e-8)
        
        # Threshold
        speech_frames = energy > self.vad_threshold
        
        # Convert to segments
        segments = []
        in_speech = False
        start_frame = 0
        
        min_silence_frames = int(self.min_silence_duration * sample_rate / hop_length)
        silence_counter = 0
        
        for i, is_speech in enumerate(speech_frames):
            if is_speech:
                if not in_speech:
                    start_frame = i
                    in_speech = True
                silence_counter = 0
            else:
                if in_speech:
                    silence_counter += 1
                    if silence_counter >= min_silence_frames:
                        # End of speech segment
                        start_sample = start_frame * hop_length
                        end_sample = (i - min_silence_frames) * hop_length
                        segments.append((start_sample, end_sample))
                        in_speech = False
                        silence_counter = 0
        
        # Handle last segment
        if in_speech:
            start_sample = start_frame * hop_length
            end_sample = len(audio)
            segments.append((start_sample, end_sample))
        
        return segments
    
    def slice_audio(
        self,
        audio_path: str,
        output_dir: Optional[str] = None
    ) -> List[AudioSlice]:
        """Slice audio into speech segments"""
        # Load audio
        audio, sr = self.load_audio(audio_path)
        
        # Detect speech segments
        segments = self.detect_speech_segments(audio, sr)
        
        # Process segments
        slices = []
        valid_segment_idx = 0
        
        for start_sample, end_sample in segments:
            duration = (end_sample - start_sample) / sr
            
            # Skip too short or too long segments
            if duration < self.min_duration or duration > self.max_duration:
                continue
            
            # Extract audio slice
            audio_slice = audio[start_sample:end_sample]
            
            # Calculate quality metrics
            rms = float(torch.sqrt(torch.mean(audio_slice ** 2)))
            peak = float(torch.max(torch.abs(audio_slice)))
            
            # Create slice object
            slice_obj = AudioSlice(
                start_sample=start_sample,
                end_sample=end_sample,
                start_time=start_sample / sr,
                end_time=end_sample / sr,
                duration=duration,
                audio_data=audio_slice,
                sample_rate=sr,
                rms_energy=rms,
                peak_level=peak
            )
            
            # Save if output_dir provided
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                output_file = output_path / f"slice_{valid_segment_idx:05d}.wav"
                torchaudio.save(
                    str(output_file),
                    audio_slice.unsqueeze(0),
                    sr
                )
                slice_obj.audio_path = str(output_file)
            
            slices.append(slice_obj)
            valid_segment_idx += 1
        
        self.logger.info(f"Sliced audio into {len(slices)} segments")
        return slices
    
    def validate_slice(self, slice_obj: AudioSlice) -> Tuple[bool, List[str]]:
        """Validate audio slice quality"""
        issues = []
        
        # Check duration
        if slice_obj.duration < self.min_duration:
            issues.append(f"Too short: {slice_obj.duration:.2f}s")
        if slice_obj.duration > self.max_duration:
            issues.append(f"Too long: {slice_obj.duration:.2f}s")
        
        # Check energy
        if slice_obj.rms_energy < 0.01:
            issues.append(f"Too quiet: RMS={slice_obj.rms_energy:.4f}")
        
        # Check clipping
        if slice_obj.peak_level > 0.99:
            issues.append(f"Clipping detected: peak={slice_obj.peak_level:.4f}")
        
        return len(issues) == 0, issues
