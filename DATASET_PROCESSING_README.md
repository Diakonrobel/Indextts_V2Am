# Comprehensive Dataset Management for IndexTTS v2

## Overview

Professional-grade dataset processing pipeline supporting SRT/VTT subtitles, web URL downloads, VAD-based audio slicing, and comprehensive quality validation.

## Quick Start

### Installation

```bash
# Install dependencies
pip install pysrt webvtt-py yt-dlp noisereduce rich

# Install FFmpeg (required for audio extraction)
# Windows: Download from https://ffmpeg.org
# Linux: sudo apt-get install ffmpeg
# macOS: brew install ffmpeg
```

### Basic Usage

```python
from indextts.utils import ComprehensiveDatasetProcessor

# Initialize processor
processor = ComprehensiveDatasetProcessor(
    output_dir="processed_dataset",
    sample_rate=24000,
    min_snr_db=20.0,
    validate_quality=True
)

# Process SRT/VTT + media
samples, stats = processor.process_from_srt_vtt(
    media_path="video.mp4",
    subtitle_path="subtitles.srt",
    dataset_name="my_dataset"
)

print(f"Processed {stats.processed} samples with {stats.avg_quality_score:.1f}/10 quality")
```

## Features

### 🎬 SRT/VTT Subtitle Processing
- **Dual parser support:** pysrt library + manual fallback
- **Timestamp validation:** <5ms alignment accuracy target
- **FFmpeg integration:** Precise audio segment extraction
- **Format support:** SRT, VTT, auto-detection

### 🌐 Web URL Downloads
- **yt-dlp integration:** YouTube, Vimeo, direct media URLs
- **Parallel processing:** 4-8 concurrent downloads (configurable)
- **Retry logic:** Exponential backoff (max 3 retries)
- **Auto-subtitles:** Extracts/converts to SRT format

### ✂️ Intelligent Audio Slicing
- **Silero VAD:** State-of-the-art voice activity detection
- **Energy fallback:** Works without external models
- **Target segments:** 1-10s clips, Gaussian distribution
- **Optional denoising:** noisereduce integration

### ✅ Quality Validation
- **SNR analysis:** >20dB threshold (configurable)
- **Audio metrics:** RMS, peak, ZCR, spectral centroid
- **Whisper ASR:** Optional transcription validation
- **Quality scoring:** 0-10 scale with detailed reports

### 📊 Complete Pipeline
- **One-step processing:** SRT/VTT → Extract → Slice → Validate → Manifest
- **Rich console output:** Progress bars and colored logging
- **Train/val/test splitting:** Configurable ratios with seeding
- **JSONL manifests:** Compatible with IndexTTS v2 training

## Use Cases

### 1. Process Video with Subtitles

```python
processor = ComprehensiveDatasetProcessor(
    output_dir="output/movie_dataset",
    validate_quality=True
)

samples, stats = processor.process_from_srt_vtt(
    media_path="movie.mp4",
    subtitle_path="movie.srt",
    dataset_name="movie_clips"
)
```

**Output:**
- `output/movie_dataset/audio/movie_clips/segment_00000.wav`
- `output/movie_dataset/audio/movie_clips/segment_00001.wav`
- `output/movie_dataset/manifests/movie_clips_manifest.jsonl`

### 2. Download from YouTube

```python
processor = ComprehensiveDatasetProcessor(
    output_dir="output/youtube_dataset"
)

urls = [
    'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
    'https://www.youtube.com/watch?v=9bZkp7q19f0',
]

samples, stats = processor.process_from_urls(
    urls=urls,
    dataset_name="youtube_speeches",
    max_workers=4,
    extract_subtitles=True  # Auto-extracts captions
)
```

### 3. Process Audio Without Subtitles (VAD)

```python
from indextts.utils import IntelligentAudioSlicer

slicer = IntelligentAudioSlicer(
    sample_rate=24000,
    min_duration=1.0,
    max_duration=10.0,
    denoise=True
)

slices = slicer.slice_audio(
    audio_path="long_recording.wav",
    output_dir="output/sliced_audio"
)

print(f"Created {len(slices)} audio slices")
```

### 4. Quality Validation

```python
from indextts.utils import DatasetQualityValidator

validator = DatasetQualityValidator(
    min_snr_db=20.0,
    use_whisper=True,  # Enable ASR validation
    whisper_model="base"
)

summary = validator.validate_dataset(
    dataset_manifest="manifests/dataset_manifest.jsonl",
    output_report="reports/quality_report.json"
)

print(f"Pass rate: {summary['pass_rate']:.1%}")
```

## Configuration Options

### ComprehensiveDatasetProcessor

```python
processor = ComprehensiveDatasetProcessor(
    output_dir="processed_dataset",  # Output directory
    sample_rate=24000,               # Target sample rate (Hz)
    min_duration=1.0,                # Min clip duration (seconds)
    max_duration=10.0,               # Max clip duration (seconds)
    min_snr_db=20.0,                 # Min SNR threshold (dB)
    denoise=False,                   # Enable audio denoising
    use_vad=True,                    # Use VAD for slicing
    validate_quality=True            # Enable quality filtering
)
```

### Quality Thresholds

| Metric | Threshold | Penalty |
|--------|-----------|----------|
| Clipping | peak > 0.99 | -3.0 points |
| Too Quiet | RMS < 0.01 | -2.0 points |
| Noisy | SNR < 20dB or ZCR > 0.5 | -2.0 points |
| Invalid Duration | < 1s or > 10s | -2.0 points |
| Transcription Mismatch | WER > 30% | -1.5 points |

**Acceptance:** Quality score ≥ 6.0/10

## Output Format

### JSONL Manifest

```json
{
  "audio_path": "audio/dataset/segment_00000.wav",
  "text": "Hello, this is example text.",
  "duration": 3.45,
  "start_time": 10.2,
  "end_time": 13.65,
  "segment_index": 0,
  "quality_metrics": {
    "snr_db": 28.5,
    "rms_energy": 0.15,
    "peak_level": 0.82,
    "quality_score": 8.5
  }
}
```

### Directory Structure

```
processed_dataset/
├── audio/
│   └── my_dataset/
│       ├── segment_00000.wav
│       ├── segment_00001.wav
│       └── ...
├── manifests/
│   ├── my_dataset_manifest.jsonl
│   ├── train_manifest.jsonl
│   ├── val_manifest.jsonl
│   └── test_manifest.jsonl
├── reports/
│   └── quality_report.json
└── downloads/  # Temporary web downloads
```

## Performance

### Benchmarks

| Task | Duration | Throughput |
|------|----------|------------|
| SRT parsing | 100 segments | <1s |
| Audio extraction (FFmpeg) | 1 hour | ~2 minutes |
| VAD slicing (Silero) | 1 hour | ~5 minutes |
| Quality validation | 1000 clips | ~3 minutes |
| **Total pipeline** | **1 hour audio** | **<10 minutes** |

### Optimization Tips

1. **Parallel downloads:** Use `max_workers=4-8` for web URLs
2. **Disable Whisper:** Set `use_whisper=False` if not needed (10x faster)
3. **Skip denoising:** Set `denoise=False` for clean audio (2x faster)
4. **Batch processing:** Process multiple files in one run

## Troubleshooting

### FFmpeg Not Found

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org and add to PATH

# macOS
brew install ffmpeg
```

### yt-dlp Fails

```bash
# Update yt-dlp
pip install --upgrade yt-dlp

# Test manually
yt-dlp --version
yt-dlp "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

### Silero VAD Download Issues

```python
# VAD model downloads ~330MB on first use
# If it fails, use energy-based fallback (automatic)
# Or pre-download:
import torch
model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad')
```

### Quality Scores Too Low

```python
# Adjust thresholds
processor = ComprehensiveDatasetProcessor(
    min_snr_db=15.0,    # Lower SNR threshold
    validate_quality=False  # Disable filtering
)
```

## Examples

See `scripts/process_dataset_example.py` for complete examples.

## Best Practices

1. **Start small:** Test with 5-10 samples before processing large datasets
2. **Check quality reports:** Review failed samples to adjust thresholds
3. **Use validation split:** Keep 10% for validation, 10% for testing
4. **Legal compliance:** Only use public domain or licensed content for web downloads
5. **Backup originals:** Keep original media files before processing

## Integration with IndexTTS v2

```python
# Process dataset
processor = ComprehensiveDatasetProcessor(...)
samples, stats = processor.process_from_srt_vtt(...)

# Split into train/val/test
split_paths = processor.split_dataset(
    manifest_path="manifests/dataset_manifest.jsonl",
    train_ratio=0.8,
    val_ratio=0.1
)

# Use in training
# python scripts/finetune_amharic.py \
#   --train_manifest manifests/train_manifest.jsonl \
#   --val_manifest manifests/val_manifest.jsonl
```

## License

Same as IndexTTS v2 project.

## Credits

Built with:
- [Silero VAD](https://github.com/snakers4/silero-vad) - Voice Activity Detection
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - Media downloads
- [pysrt](https://github.com/byroot/pysrt) - SRT parsing
- [FFmpeg](https://ffmpeg.org) - Audio/video processing
