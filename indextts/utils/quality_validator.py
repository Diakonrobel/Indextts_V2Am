"""Comprehensive dataset quality validation with multiple quality checks"""
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass, asdict
import json

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


@dataclass
class QualityMetrics:
    """Quality metrics for an audio sample"""
    # Audio quality
    snr_db: float
    rms_energy: float
    peak_level: float
    zero_crossing_rate: float
    spectral_centroid: float
    duration: float
    sample_rate: int
    
    # Flags
    is_clipping: bool
    is_too_quiet: bool
    is_noisy: bool
    is_valid_duration: bool
    
    # Transcription (if available)
    transcription_wer: Optional[float] = None
    transcription_match: Optional[bool] = None
    
    # Overall score
    quality_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


class DatasetQualityValidator:
    """Validate dataset quality with comprehensive checks"""
    
    def __init__(
        self,
        min_snr_db: float = 20.0,
        min_duration: float = 1.0,
        max_duration: float = 10.0,
        min_rms: float = 0.01,
        max_peak: float = 0.99,
        max_zcr: float = 0.5,
        use_whisper: bool = False,
        whisper_model: str = "base"
    ):
        self.min_snr_db = min_snr_db
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_rms = min_rms
        self.max_peak = max_peak
        self.max_zcr = max_zcr
        self.use_whisper = use_whisper and WHISPER_AVAILABLE
        self.logger = logging.getLogger(__name__)
        
        # Load Whisper if requested
        self.whisper_model = None
        if self.use_whisper:
            try:
                self.whisper_model = whisper.load_model(whisper_model)
                self.logger.info(f"✅ Loaded Whisper model: {whisper_model}")
            except Exception as e:
                self.logger.error(f"Failed to load Whisper: {e}")
    
    def calculate_snr(self, audio: torch.Tensor) -> float:
        """Calculate Signal-to-Noise Ratio in dB"""
        # Simple SNR estimation using signal energy vs noise floor
        audio_np = audio.numpy()
        
        # Estimate noise from quietest 10% of samples
        sorted_samples = np.sort(np.abs(audio_np))
        noise_samples = sorted_samples[:len(sorted_samples) // 10]
        noise_power = np.mean(noise_samples ** 2)
        
        # Signal power
        signal_power = np.mean(audio_np ** 2)
        
        # SNR in dB
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = 100.0  # Very high SNR
        
        return float(snr)
    
    def calculate_spectral_centroid(self, audio: torch.Tensor, sr: int) -> float:
        """Calculate spectral centroid"""
        # Compute spectrogram
        spec = torch.stft(
            audio,
            n_fft=1024,
            hop_length=512,
            return_complex=True
        )
        magnitude = torch.abs(spec)
        
        # Calculate weighted frequency
        freqs = torch.fft.rfftfreq(1024, 1/sr)
        centroid = torch.sum(magnitude * freqs.unsqueeze(1), dim=0) / (torch.sum(magnitude, dim=0) + 1e-8)
        
        return float(torch.mean(centroid))
    
    def check_audio_quality(
        self,
        audio_path: str,
        expected_text: Optional[str] = None
    ) -> QualityMetrics:
        """Comprehensive audio quality check"""
        # Load audio
        audio, sr = torchaudio.load(audio_path)
        
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        audio = audio.squeeze()
        
        # Calculate metrics
        duration = len(audio) / sr
        rms = float(torch.sqrt(torch.mean(audio ** 2)))
        peak = float(torch.max(torch.abs(audio)))
        
        # Zero crossing rate
        zcr = float(torch.sum(torch.abs(torch.diff(torch.sign(audio)))) / (2 * len(audio)))
        
        # SNR
        snr = self.calculate_snr(audio)
        
        # Spectral centroid
        spectral_centroid = self.calculate_spectral_centroid(audio, sr)
        
        # Quality flags
        is_clipping = peak > self.max_peak
        is_too_quiet = rms < self.min_rms
        is_noisy = zcr > self.max_zcr or snr < self.min_snr_db
        is_valid_duration = self.min_duration <= duration <= self.max_duration
        
        # Transcription validation if Whisper available and text provided
        transcription_wer = None
        transcription_match = None
        
        if self.whisper_model and expected_text:
            try:
                result = self.whisper_model.transcribe(audio_path)
                transcribed_text = result['text'].strip()
                
                # Simple word error rate
                expected_words = expected_text.lower().split()
                transcribed_words = transcribed_text.lower().split()
                
                # Levenshtein distance approximation
                matches = sum(1 for w in expected_words if w in transcribed_words)
                transcription_wer = 1.0 - (matches / max(len(expected_words), len(transcribed_words)))
                transcription_match = transcription_wer < 0.3  # < 30% error
                
            except Exception as e:
                self.logger.error(f"Whisper transcription failed: {e}")
        
        # Calculate overall quality score (0-10)
        quality_score = self._calculate_quality_score(
            snr, rms, peak, zcr, duration,
            is_clipping, is_too_quiet, is_noisy, is_valid_duration,
            transcription_match
        )
        
        return QualityMetrics(
            snr_db=snr,
            rms_energy=rms,
            peak_level=peak,
            zero_crossing_rate=zcr,
            spectral_centroid=spectral_centroid,
            duration=duration,
            sample_rate=sr,
            is_clipping=is_clipping,
            is_too_quiet=is_too_quiet,
            is_noisy=is_noisy,
            is_valid_duration=is_valid_duration,
            transcription_wer=transcription_wer,
            transcription_match=transcription_match,
            quality_score=quality_score
        )
    
    def _calculate_quality_score(
        self,
        snr: float,
        rms: float,
        peak: float,
        zcr: float,
        duration: float,
        is_clipping: bool,
        is_too_quiet: bool,
        is_noisy: bool,
        is_valid_duration: bool,
        transcription_match: Optional[bool]
    ) -> float:
        """Calculate overall quality score 0-10"""
        score = 10.0
        
        # Penalize issues
        if is_clipping:
            score -= 3.0
        if is_too_quiet:
            score -= 2.0
        if is_noisy:
            score -= 2.0
        if not is_valid_duration:
            score -= 2.0
        if transcription_match is not None and not transcription_match:
            score -= 1.5
        
        # Reward good metrics
        if snr > self.min_snr_db:
            score += min(1.0, (snr - self.min_snr_db) / 10.0)
        if self.min_rms < rms < 0.7:
            score += 0.5
        
        return max(0.0, min(10.0, score))
    
    def validate_dataset(
        self,
        dataset_manifest: str,
        output_report: Optional[str] = None
    ) -> Dict:
        """Validate entire dataset from manifest"""
        # Load manifest
        samples = []
        with open(dataset_manifest, 'r', encoding='utf-8') as f:
            for line in f:
                samples.append(json.loads(line))
        
        self.logger.info(f"Validating {len(samples)} samples...")
        
        # Validate each sample
        results = []
        passed = 0
        failed = 0
        
        for i, sample in enumerate(samples):
            audio_path = sample.get('audio_path')
            text = sample.get('text')
            
            if not audio_path or not Path(audio_path).exists():
                self.logger.warning(f"Sample {i}: Audio file not found")
                failed += 1
                continue
            
            try:
                metrics = self.check_audio_quality(audio_path, text)
                
                result = {
                    'index': i,
                    'audio_path': audio_path,
                    'metrics': metrics.to_dict(),
                    'passed': metrics.quality_score >= 6.0
                }
                
                results.append(result)
                
                if result['passed']:
                    passed += 1
                else:
                    failed += 1
                    
            except Exception as e:
                self.logger.error(f"Sample {i} validation failed: {e}")
                failed += 1
        
        # Generate summary
        summary = {
            'total_samples': len(samples),
            'validated': len(results),
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / len(results) if results else 0.0,
            'metrics_summary': self._summarize_metrics(results)
        }
        
        # Save report
        if output_report:
            report_data = {
                'summary': summary,
                'details': results
            }
            
            with open(output_report, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2)
            
            self.logger.info(f"Quality report saved: {output_report}")
        
        return summary
    
    def _summarize_metrics(self, results: List[Dict]) -> Dict:
        """Calculate aggregate statistics"""
        if not results:
            return {}
        
        metrics = [r['metrics'] for r in results]
        
        return {
            'avg_snr_db': np.mean([m['snr_db'] for m in metrics]),
            'avg_quality_score': np.mean([m['quality_score'] for m in metrics]),
            'avg_duration': np.mean([m['duration'] for m in metrics]),
            'clipping_rate': np.mean([m['is_clipping'] for m in metrics]),
            'too_quiet_rate': np.mean([m['is_too_quiet'] for m in metrics]),
            'noisy_rate': np.mean([m['is_noisy'] for m in metrics])
        }
    
    def generate_quality_report(self, validation_results: Dict, output_path: str):
        """Generate human-readable quality report"""
        summary = validation_results
        
        report = f"""
# Dataset Quality Report

## Summary
- Total Samples: {summary['total_samples']}
- Validated: {summary['validated']}
- Passed: {summary['passed']}
- Failed: {summary['failed']}
- Pass Rate: {summary['pass_rate']:.1%}

## Metrics Summary
- Average SNR: {summary['metrics_summary'].get('avg_snr_db', 0):.1f} dB
- Average Quality Score: {summary['metrics_summary'].get('avg_quality_score', 0):.1f}/10
- Average Duration: {summary['metrics_summary'].get('avg_duration', 0):.2f}s
- Clipping Rate: {summary['metrics_summary'].get('clipping_rate', 0):.1%}
- Too Quiet Rate: {summary['metrics_summary'].get('too_quiet_rate', 0):.1%}
- Noisy Rate: {summary['metrics_summary'].get('noisy_rate', 0):.1%}

## Recommendations
"""        
        # Add recommendations
        if summary['metrics_summary'].get('avg_snr_db', 0) < self.min_snr_db:
            report += "- ⚠️ Low SNR detected. Consider denoising audio files.\n"
        if summary['metrics_summary'].get('clipping_rate', 0) > 0.05:
            report += "- ⚠️ High clipping rate. Check recording levels.\n"
        if summary['pass_rate'] < 0.95:
            report += "- ⚠️ Low pass rate. Review failed samples and improve data quality.\n"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"Quality report generated: {output_path}")
