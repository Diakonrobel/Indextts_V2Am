# Comprehensive Dataset Management Enhancement Plan for IndexTTS v2

## Executive Summary
Enhance dataset management to support professional TTS workflows including SRT/VTT subtitle processing, web URL media downloads, advanced audio slicing with VAD, and comprehensive validation pipelines.

## Research-Based Best Practices Applied

### 1. SRT/VTT Processing (Industry Standard)
- Parse timestamps with `pysrt` and `webvtt-py` libraries
- Align text-audio pairs using FFmpeg subtitle extraction
- Preserve punctuation for prosody modeling (20-30% MOS improvement)
- Target <5ms alignment accuracy
- Handle both embedded and external subtitle files

### 2. Audio Slicing & Validation (Coqui TTS/ESPnet Standards)
- VAD-based segmentation using Silero VAD (70% manual effort reduction)
- Target 1-10s clips with Gaussian distribution (mean 3s)
- Sample rate: 22.05kHz for quality/efficiency balance
- Format: Mono 16-bit PCM WAV
- Validation: SNR >20dB, spectrogram analysis
- Quality checks: Remove >5% noisy clips (15-25% convergence improvement)

### 3. Web URL Processing (LibriTTS/Common Voice Approach)
- Download with `yt-dlp` (handles rate limits, metadata)
- Batch processing with retry logic (max 3 retries)
- Support: YouTube, Vimeo, direct media URLs
- Post-processing: Denoise with RNNoise, resample, normalize
- Legal compliance: Non-commercial/public domain sources only
- Metadata watermarking for traceability

### 4. Quality Assurance (Hugging Face Datasets Standard)
- Transcription accuracy: >95% alignment via ASR (Whisper)
- Audio validation: SNR >20dB, spectrogram variance thresholds
- Diversity checks: Duration distribution, speaker balance
- Coverage analysis: Phoneme coverage for target language
- Remove artifacts: Breaths, smacks (energy threshold filtering)

## Implementation Architecture

### Phase 1: Core Infrastructure (Week 1-2)

#### 1.1 Create `indextts/utils/dataset_processor.py`
```python
class ComprehensiveDatasetProcessor:
    - SRTProcessor: Parse SRT/VTT, extract timestamps
    - URLDownloader: Web media download with yt-dlp
    - AudioSlicer: VAD-based intelligent slicing
    - QualityValidator: Multi-level quality checks
    - ManifestGenerator: JSONL manifest creation
```

#### 1.2 Create `indextts/utils/subtitle_parser.py`
```python
class SubtitleParser:
    - parse_srt(): Extract text + timestamps from SRT
    - parse_vtt(): Extract text + timestamps from WebVTT
    - align_with_audio(): FFmpeg-based alignment
    - validate_timing(): Check timestamp consistency
```

#### 1.3 Create `indextts/utils/web_downloader.py`
```python
class WebMediaDownloader:
    - download_from_url(): yt-dlp integration
    - extract_subtitles(): Auto-generated subs
    - batch_download(): Queue-based parallel downloads
    - handle_retries(): Exponential backoff
```

#### 1.4 Create `indextts/utils/audio_slicer.py`
```python
class IntelligentAudioSlicer:
    - vad_segment(): Silero VAD-based slicing
    - validate_slice(): Duration/quality checks
    - normalize_audio(): RMS normalization
    - denoise_audio(): RNNoise integration
```

#### 1.5 Create `indextts/utils/quality_validator.py`
```python
class DatasetQualityValidator:
    - check_audio_quality(): SNR, clipping, silence
    - check_transcription(): ASR-based validation
    - check_diversity(): Duration/speaker stats
    - check_phoneme_coverage(): Language-specific
    - generate_quality_report(): Comprehensive metrics
```

### Phase 2: Enhanced Dataset Preparation Script (Week 3)

#### 2.1 Update `scripts/prepare_amharic_data.py`
Add support for:
- SRT/VTT file processing
- Web URL input (single or batch)
- VAD-based slicing
- Advanced quality validation
- Progress tracking with rich console output

#### 2.2 Create `scripts/prepare_dataset_from_url.py`
Dedicated script for web downloads:
- URL list input (CSV/TXT)
- Concurrent downloads (configurable workers)
- Automatic subtitle extraction
- Audio-text alignment
- Quality filtering pipeline

### Phase 3: Web UI Integration (Week 4)

#### 3.1 Enhance `amharic_gradio_app.py` Training Tab
Add Dataset Tools subtab:
- **File Upload Section:**
  - Audio files (WAV/MP3/FLAC)
  - Subtitle files (SRT/VTT/LRC)
  - Paired or unpaired upload modes
  
- **URL Download Section:**
  - Single URL input
  - Batch URL input (textarea)
  - Download progress tracking
  - Auto-subtitle extraction toggle

- **Processing Configuration:**
  - VAD sensitivity slider
  - Target clip duration range
  - Quality threshold (SNR)
  - Sample rate selection
  - Denoising toggle

- **Preview & Validation:**
  - Sample audio playback
  - Alignment visualization
  - Quality metrics dashboard
  - Reject/approve interface

- **Batch Operations:**
  - Process multiple sources
  - Merge datasets
  - Split train/val/test
  - Export manifest

### Phase 4: Advanced Features (Week 5)

#### 4.1 Audio Processing Pipeline
- FFmpeg integration for format conversion
- Multi-channel to mono conversion
- Automatic resampling
- Volume normalization
- Noise reduction (RNNoise/Demucs)

#### 4.2 Text Processing Pipeline
- Automatic language detection
- Text normalization (numbers, dates, abbrev)
- Punctuation restoration (if missing)
- Sentence segmentation
- Phoneme coverage analysis

#### 4.3 Quality Assurance Pipeline
- ASR validation with Whisper
- Word Error Rate (WER) calculation
- Spectrogram anomaly detection
- Duration distribution analysis
- Speaker diversity metrics

### Phase 5: Testing & Documentation (Week 6)

#### 5.1 Unit Tests
- Test SRT/VTT parsing with edge cases
- Test URL download with various sources
- Test VAD slicing with different audio types
- Test quality validation thresholds

#### 5.2 Integration Tests
- End-to-end pipeline test
- Multi-source dataset preparation
- Manifest validation
- Model compatibility check

#### 5.3 Documentation
- User guide with examples
- API documentation
- Troubleshooting guide
- Best practices document

## Technical Specifications

### Dependencies to Add
```
pysrt>=1.1.2
webvtt-py>=0.4.6
yt-dlp>=2023.3.4
silero-vad>=4.0.0
noisereduce>=2.0.1
openai-whisper>=20231117  # For ASR validation
rich>=13.0.0  # For progress display
```

### File Format Support
- Audio: WAV, MP3, FLAC, M4A, OGG
- Subtitles: SRT, VTT, LRC, JSON
- Video: MP4, MKV, WEBM (extract audio)
- URLs: YouTube, Vimeo, direct media links

### Quality Metrics
- SNR threshold: >20dB (configurable)
- Duration: 1-10s clips (configurable)
- Sample rate: 22.05kHz default
- Bit depth: 16-bit
- Format: Mono WAV
- Alignment accuracy: <5ms

### Performance Targets
- Process 1 hour of audio in <10 minutes
- Concurrent downloads: 4-8 workers
- VAD processing: Real-time or faster
- Quality checks: <1s per audio file

## Expected Outcomes

### Immediate Benefits
- Support professional subtitle-based datasets
- Enable web scraping workflows
- Automated quality filtering (reject 5-10% low-quality)
- Faster dataset preparation (3-5x speedup)

### Quality Improvements
- Better text-audio alignment (<5ms vs previous ~50ms)
- Cleaner audio (SNR >20dB enforced)
- Consistent durations (optimal for training)
- Higher phoneme coverage

### Workflow Enhancements
- One-click web dataset creation
- Automated quality reports
- Visual validation interface
- Batch processing capabilities

## Implementation Priority

### Must-Have (Critical)
1. SRT/VTT parsing with timestamp alignment
2. VAD-based audio slicing
3. Basic quality validation (SNR, duration)
4. Web URL download with yt-dlp
5. Gradio UI integration

### Should-Have (High Priority)
6. Advanced quality metrics (ASR validation)
7. Batch URL processing
8. Audio denoising
9. Visual alignment preview
10. Progress tracking

### Nice-to-Have (Medium Priority)
11. Phoneme coverage analysis
12. Speaker diarization
13. Automatic text normalization
14. Dataset merging tools
15. Export to multiple formats

## Success Metrics
- Process 10-hour dataset in <1 hour
- <5% manual rejection rate
- >95% alignment accuracy
- User satisfaction: Enable single-command dataset prep
- Quality: Models trained on processed data achieve MOS >4.0

## Risk Mitigation
- **Copyright Issues:** Only public domain/licensed content
- **Download Failures:** Retry logic + error logging
- **Quality Variance:** Configurable thresholds
- **Processing Errors:** Robust error handling + logging
- **Resource Usage:** Configurable workers, memory limits