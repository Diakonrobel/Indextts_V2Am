"""Comprehensive dataset processor integrating all components"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import random

from .subtitle_parser import SubtitleParser, SubtitleSegment
from .web_downloader import WebMediaDownloader, DownloadResult
from .audio_slicer import IntelligentAudioSlicer, AudioSlice
from .quality_validator import DatasetQualityValidator, QualityMetrics

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


@dataclass
class ProcessingStats:
    """Statistics for dataset processing"""
    total_inputs: int = 0
    processed: int = 0
    failed: int = 0
    total_duration: float = 0.0
    avg_quality_score: float = 0.0
    pass_rate: float = 0.0


class ComprehensiveDatasetProcessor:
    """All-in-one dataset processor for IndexTTS v2"""
    
    def __init__(
        self,
        output_dir: str = "processed_dataset",
        sample_rate: int = 24000,
        min_duration: float = 1.0,
        max_duration: float = 10.0,
        min_snr_db: float = 20.0,
        denoise: bool = False,
        use_vad: bool = True,
        validate_quality: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.audio_dir = self.output_dir / "audio"
        self.manifests_dir = self.output_dir / "manifests"
        self.reports_dir = self.output_dir / "reports"
        
        for dir_path in [self.audio_dir, self.manifests_dir, self.reports_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Initialize components
        self.subtitle_parser = SubtitleParser()
        self.web_downloader = WebMediaDownloader(
            output_dir=str(self.output_dir / "downloads")
        )
        self.audio_slicer = IntelligentAudioSlicer(
            sample_rate=sample_rate,
            min_duration=min_duration,
            max_duration=max_duration,
            denoise=denoise
        )
        self.quality_validator = DatasetQualityValidator(
            min_snr_db=min_snr_db,
            min_duration=min_duration,
            max_duration=max_duration
        )
        
        self.sample_rate = sample_rate
        self.validate_quality = validate_quality
        self.logger = logging.getLogger(__name__)
        
        if RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None
    
    def process_from_srt_vtt(
        self,
        media_path: str,
        subtitle_path: str,
        dataset_name: str = "dataset"
    ) -> Tuple[List[Dict], ProcessingStats]:
        """Process media file with SRT/VTT subtitles"""
        self._log(f"Processing {media_path} with subtitles {subtitle_path}")
        
        # Parse subtitles
        segments = self.subtitle_parser.parse_file(subtitle_path)
        self._log(f"Parsed {len(segments)} subtitle segments")
        
        # Validate timing
        is_valid, issues = self.subtitle_parser.validate_timing(segments)
        if not is_valid:
            self._log(f"⚠️ Timing issues found: {len(issues)} warnings", level="warning")
        
        # Extract audio segments
        extracted = self.subtitle_parser.extract_audio_segments(
            audio_path=media_path,
            segments=segments,
            output_dir=str(self.audio_dir / dataset_name),
            format='wav'
        )
        
        # Validate quality if requested
        if self.validate_quality:
            extracted = self._filter_by_quality(extracted)
        
        # Create manifest
        manifest_path = self.manifests_dir / f"{dataset_name}_manifest.jsonl"
        self._save_manifest(extracted, manifest_path)
        
        # Calculate stats
        stats = self._calculate_stats(extracted, len(segments))
        
        self._log(f"✅ Processed {len(extracted)}/{len(segments)} segments")
        return extracted, stats
    
    def process_from_urls(
        self,
        urls: List[str],
        dataset_name: str = "dataset",
        max_workers: int = 4,
        extract_subtitles: bool = True
    ) -> Tuple[List[Dict], ProcessingStats]:
        """Process multiple web URLs"""
        self._log(f"Downloading {len(urls)} URLs...")
        
        # Download media and subtitles
        download_results = self.web_downloader.batch_download(
            urls=urls,
            max_workers=max_workers,
            extract_audio=True,
            extract_subtitles=extract_subtitles,
            audio_format='wav',
            sample_rate=self.sample_rate
        )
        
        # Process each successful download
        all_samples = []
        
        for i, result in enumerate(download_results):
            if not result.success:
                continue
            
            if result.subtitle_path and extract_subtitles:
                # Process with subtitles
                try:
                    samples, _ = self.process_from_srt_vtt(
                        media_path=result.audio_path,
                        subtitle_path=result.subtitle_path,
                        dataset_name=f"{dataset_name}_url{i}"
                    )
                    all_samples.extend(samples)
                except Exception as e:
                    self._log(f"Failed to process URL {i}: {e}", level="error")
            
            elif result.audio_path:
                # Process audio only with VAD slicing
                try:
                    samples = self._process_audio_only(
                        audio_path=result.audio_path,
                        dataset_name=f"{dataset_name}_url{i}"
                    )
                    all_samples.extend(samples)
                except Exception as e:
                    self._log(f"Failed to process audio {i}: {e}", level="error")
        
        # Save combined manifest
        manifest_path = self.manifests_dir / f"{dataset_name}_manifest.jsonl"
        self._save_manifest(all_samples, manifest_path)
        
        stats = self._calculate_stats(all_samples, len(urls))
        
        self._log(f"✅ Processed {len(all_samples)} samples from {len(urls)} URLs")
        return all_samples, stats
    
    def _process_audio_only(
        self,
        audio_path: str,
        dataset_name: str
    ) -> List[Dict]:
        """Process audio file without subtitles using VAD"""
        # Slice audio
        slices = self.audio_slicer.slice_audio(
            audio_path=audio_path,
            output_dir=str(self.audio_dir / dataset_name)
        )
        
        # Convert to manifest format
        samples = []
        for slice_obj in slices:
            sample = {
                'audio_path': getattr(slice_obj, 'audio_path', ''),
                'text': '',  # No text available
                'duration': slice_obj.duration,
                'start_time': slice_obj.start_time,
                'end_time': slice_obj.end_time,
                'rms_energy': slice_obj.rms_energy,
                'peak_level': slice_obj.peak_level
            }
            samples.append(sample)
        
        return samples
    
    def _filter_by_quality(self, samples: List[Dict]) -> List[Dict]:
        """Filter samples by quality metrics"""
        filtered = []
        
        for sample in samples:
            audio_path = sample.get('audio_path')
            text = sample.get('text')
            
            if not audio_path or not Path(audio_path).exists():
                continue
            
            try:
                metrics = self.quality_validator.check_audio_quality(
                    audio_path=audio_path,
                    expected_text=text
                )
                
                # Accept if quality score >= 6.0
                if metrics.quality_score >= 6.0:
                    sample['quality_metrics'] = metrics.to_dict()
                    filtered.append(sample)
                else:
                    self._log(
                        f"Rejected: {audio_path} (score: {metrics.quality_score:.1f})",
                        level="debug"
                    )
                    
            except Exception as e:
                self._log(f"Quality check failed for {audio_path}: {e}", level="error")
                continue
        
        self._log(f"Quality filter: {len(filtered)}/{len(samples)} passed")
        return filtered
    
    def _save_manifest(self, samples: List[Dict], manifest_path: Path):
        """Save samples to JSONL manifest"""
        with open(manifest_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        self._log(f"Manifest saved: {manifest_path}")
    
    def _calculate_stats(self, samples: List[Dict], total_inputs: int) -> ProcessingStats:
        """Calculate processing statistics"""
        if not samples:
            return ProcessingStats(total_inputs=total_inputs)
        
        total_duration = sum(s.get('duration', 0) for s in samples)
        
        # Average quality score
        quality_scores = [
            s.get('quality_metrics', {}).get('quality_score', 0)
            for s in samples
            if 'quality_metrics' in s
        ]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        return ProcessingStats(
            total_inputs=total_inputs,
            processed=len(samples),
            failed=total_inputs - len(samples),
            total_duration=total_duration,
            avg_quality_score=avg_quality,
            pass_rate=len(samples) / total_inputs if total_inputs > 0 else 0.0
        )
    
    def split_dataset(
        self,
        manifest_path: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        random_seed: int = 42
    ) -> Dict[str, str]:
        """Split dataset into train/val/test"""
        # Load manifest
        samples = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                samples.append(json.loads(line))
        
        # Shuffle
        random.seed(random_seed)
        random.shuffle(samples)
        
        # Split
        total = len(samples)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        splits = {
            'train': samples[:train_size],
            'val': samples[train_size:train_size + val_size],
            'test': samples[train_size + val_size:]
        }
        
        # Save splits
        manifest_dir = Path(manifest_path).parent
        split_paths = {}
        
        for split_name, split_samples in splits.items():
            split_path = manifest_dir / f"{split_name}_manifest.jsonl"
            self._save_manifest(split_samples, split_path)
            split_paths[split_name] = str(split_path)
        
        self._log(f"Dataset split: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")
        return split_paths
    
    def _log(self, message: str, level: str = "info"):
        """Log message with optional rich formatting"""
        if self.console:
            if level == "error":
                self.console.print(f"[red]❌ {message}[/red]")
            elif level == "warning":
                self.console.print(f"[yellow]⚠️  {message}[/yellow]")
            else:
                self.console.print(f"[green]✓[/green] {message}")
        
        # Also log to logger
        getattr(self.logger, level)(message)
