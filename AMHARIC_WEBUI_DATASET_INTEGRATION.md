# Amharic WebUI Dataset Processing Integration Guide

## Overview

The Amharic Gradio WebUI now includes **comprehensive dataset processing** capabilities with full Amharic language support, replacing the previous basic upload/prepare functions.

## What's New

### ✅ Replaced Components

**Old (Basic):**
- Simple file upload (audio + text only)
- Basic preparation script runner
- No subtitle support
- No quality validation
- No web downloads

**New (Comprehensive):**
- Multi-format upload (audio + text + SRT/VTT)
- Web URL processing (YouTube, Vimeo)
- 3 processing modes (traditional, SRT/VTT, VAD-only)
- Automatic quality validation (SNR, duration, metrics)
- Amharic text normalization integrated
- Parallel processing (4-8 workers)

## Features

### 📤 File Upload Section

**Supported Formats:**
- **Audio:** WAV, MP3, FLAC, M4A, OGG
- **Text:** TXT, JSON
- **Subtitles:** SRT, VTT (optional for auto-alignment)

**Usage:**
1. Click "📢 Audio Files" to upload audio
2. Click "📝 Text Files" to upload transcriptions
3. **Optional:** Click "📋 Subtitle Files (SRT/VTT)" for automatic alignment
4. Enter dataset name (e.g., `my_amharic_dataset`)
5. Click "📤 Upload Files"

### 🌐 Web URL Processing

**Supported Sources:**
- YouTube videos (with auto-captions)
- Vimeo videos
- Direct media URLs

**Usage:**
1. Paste URLs (one per line):
   ```
   https://www.youtube.com/watch?v=example1
   https://www.youtube.com/watch?v=example2
   ```
2. Enter dataset name
3. Set parallel downloads (1-8 workers, default 4)
4. Toggle "Extract Subtitles" (enabled by default)
5. Click "🌐 Download & Process URLs"

**Notes:**
- Uses yt-dlp for downloads
- Auto-extracts Amharic, English, German subtitles
- Converts to SRT format
- Takes 10-30 minutes depending on content

### 🔄 Dataset Preparation

**Processing Modes:**

1. **Traditional** (Default)
   - Pairs audio files with text files by filename
   - Uses `AmharicDatasetPreparer` from existing scripts
   - Applies `AmharicTextNormalizer` for text preprocessing
   - Best for: Existing paired datasets

2. **SRT/VTT**
   - Automatically aligns audio with subtitle timestamps
   - Extracts audio segments using FFmpeg
   - <5ms alignment accuracy
   - Best for: Videos with subtitles, movies, lectures

3. **VAD-only**
   - Uses Voice Activity Detection to slice audio
   - No text required (can add transcriptions later)
   - Silero VAD or energy-based fallback
   - Best for: Long recordings, podcasts, raw audio

**Configuration:**

```
⚙️ Processing Configuration:
- Min Duration: 0.5-30s (default 1s)
- Max Duration: 5-300s (default 10s)
- Sample Rate: 16000, 22050, 24000 Hz (default 24000)

🎯 Quality Controls:
- Min SNR: 10-40 dB (default 20 dB)
- Enable Denoising: Toggle (slower but cleaner)
- Use VAD Slicing: Toggle (intelligent speech detection)
```

## Amharic Language Support

### Text Normalization

The processor automatically applies Amharic-specific preprocessing:

- **Number expansion:** "123" → "አንድ መቶ ሀያ ሰለስት"
- **Contraction handling:** Amharic-specific contractions
- **Punctuation preservation:** Keeps እ፡ for prosody
- **Character normalization:** Handles Amharic script variations

This is done through the existing `AmharicTextNormalizer` class, ensuring consistency with training scripts.

### Quality Metrics for Amharic

**Audio Quality:**
- SNR >20dB threshold (adjustable)
- RMS energy: 0.01-0.7 range
- Zero-crossing rate: <0.5 for clean speech
- Duration: 1-10s optimal for TTS

**Text Quality:**
- Amharic character validation
- Minimum length: 5 characters
- Maximum length: 500 characters
- UNK token ratio checks

## Workflow Examples

### Example 1: Traditional Paired Dataset

```
1. Prepare files:
   audio/
     recording_001.wav
     recording_002.wav
   text/
     recording_001.txt (contains: "ሰላም ዓለም")
     recording_002.txt (contains: "እንዴት ነህ")

2. Upload:
   - Select all WAV files
   - Select all TXT files
   - Dataset name: "amharic_recordings"
   - Click Upload

3. Process:
   - Mode: Traditional
   - Min Duration: 1s
   - Max Duration: 10s
   - SNR: 20dB
   - Click "Process Dataset"

4. Output:
   processed_datasets/amharic_recordings/
     audio/segment_00000.wav
     audio/segment_00001.wav
     manifests/amharic_recordings_manifest.jsonl
```

### Example 2: YouTube Video with Captions

```
1. Find Amharic content on YouTube
   Example: Amharic news, speeches, lectures

2. Process:
   - Paste URL in "URLs" field
   - Dataset name: "youtube_amharic_news"
   - Parallel downloads: 4
   - Extract subtitles: ON
   - Click "Download & Process URLs"

3. Processing steps (automatic):
   - Downloads video
   - Extracts audio (24kHz mono WAV)
   - Downloads auto-generated Amharic captions (if available)
   - Converts to SRT format
   - Aligns audio with timestamps
   - Extracts segments
   - Validates quality
   - Creates manifest

4. Output:
   processed_datasets/youtube_amharic_news/
     downloads/  # Original downloads
     audio/youtube_amharic_news_url0/segment_*.wav
     manifests/youtube_amharic_news_manifest.jsonl
```

### Example 3: Movie with SRT Subtitles

```
1. Prepare:
   - Movie file: amharic_movie.mp4
   - Subtitle file: amharic_movie.srt (Amharic text with timestamps)

2. Upload:
   - Audio: amharic_movie.mp4 (or extracted audio)
   - Subtitles: amharic_movie.srt
   - Dataset name: "movie_clips"

3. Process:
   - Mode: SRT/VTT
   - Other settings default
   - Click "Process Dataset"

4. Output:
   - Aligned 1-10s clips
   - Amharic text from subtitles (normalized)
   - Quality validated
   - Ready for training
```

## Output Format

### Manifest (JSONL)

Each line is a JSON object:

```json
{
  "audio_path": "audio/dataset/segment_00000.wav",
  "text": "ሰላም ዓለም፣ እንዴት ነሽ",
  "original_text": "ሰላም ዓለም, እንዴት ነሽ?",
  "duration": 3.45,
  "start_time": 10.2,
  "end_time": 13.65,
  "segment_index": 0,
  "sample_rate": 24000,
  "quality_metrics": {
    "snr_db": 28.5,
    "rms_energy": 0.15,
    "peak_level": 0.82,
    "quality_score": 8.5
  }
}
```

**Fields:**
- `text`: Normalized Amharic text (used for training)
- `original_text`: Original text before normalization
- `quality_metrics`: Automatic quality assessment
- `duration`: Clip length in seconds
- `sample_rate`: Audio sample rate

### Directory Structure

```
processed_datasets/
├── my_amharic_dataset/
│   ├── audio/
│   │   └── my_amharic_dataset/
│   │       ├── segment_00000.wav
│   │       ├── segment_00001.wav
│   │       └── ...
│   ├── manifests/
│   │   └── my_amharic_dataset_manifest.jsonl
│   └── downloads/  # If from web URLs
│       └── temp files
└── reports/
    └── quality_report.json
```

## Quality Control

### Automatic Filtering

The processor automatically rejects:
- ❌ Clips <1s or >10s (configurable)
- ❌ SNR <20dB (too noisy)
- ❌ Peak >0.99 (clipping detected)
- ❌ RMS <0.01 (too quiet)
- ❌ Quality score <6.0/10

### Quality Report

After processing, check:
```
processed_datasets/reports/quality_report.json
```

Contains:
- Total samples processed
- Pass rate
- Average SNR, duration, quality score
- Rejection reasons

## Integration with Training

After dataset processing:

1. **Check manifest:** Review `*_manifest.jsonl`
2. **Verify quality:** Check pass rate in logs
3. **Start training:**
   ```
   python scripts/finetune_amharic.py \
     --train_manifest processed_datasets/my_dataset/manifests/train_manifest.jsonl \
     --val_manifest processed_datasets/my_dataset/manifests/val_manifest.jsonl
   ```

## Troubleshooting

### "FFmpeg not found"
```bash
# Windows: Download from https://ffmpeg.org
# Add to PATH

# Test:
ffmpeg -version
```

### "yt-dlp not found"
```bash
pip install yt-dlp

# Test:
yt-dlp --version
```

### "Low quality scores"

```
1. Check audio files:
   - Are they noisy?
   - Proper sample rate?
   - Good recording quality?

2. Adjust thresholds:
   - Lower Min SNR to 15dB
   - Disable quality validation temporarily

3. Review rejected samples:
   - Check logs for rejection reasons
```

### "No Amharic subtitles found"

```
For YouTube:
- Check if video has Amharic captions
- Try auto-generated captions
- Manually download SRT and upload

For videos:
- Ensure SRT file is UTF-8 encoded
- Check Amharic text renders correctly
```

## Performance Tips

1. **Parallel Downloads:** Use max_workers=4-8 for faster web processing
2. **Disable Denoising:** If audio is clean, disable for 2x speed
3. **Batch Processing:** Upload multiple datasets at once
4. **Local Files First:** Process local files before web URLs
5. **Quality Checks:** Review first batch before processing full dataset

## Best Practices

1. **Test Small First:** Start with 5-10 samples
2. **Check Quality:** Review quality scores before full processing
3. **Backup Originals:** Keep original files safe
4. **Use SRT/VTT:** Best alignment accuracy with subtitle files
5. **Amharic Text:** Verify text displays correctly in manifests
6. **Legal Compliance:** Only use licensed/public domain content for web downloads

## Future Enhancements

Planned features:
- [ ] Automatic Amharic ASR validation (Whisper)
- [ ] Phoneme coverage analysis for Amharic
- [ ] Speaker diarization
- [ ] Data augmentation (speed/pitch)
- [ ] Visual alignment preview
- [ ] Export to multiple manifest formats

## Support

For issues:
1. Check logs: `logs/gradio/gradio_app.log`
2. Review manifest output
3. Test with example data first
4. Verify all dependencies installed

## Credits

Integrated Components:
- Comprehensive Dataset Processor (indextts/utils/dataset_processor.py)
- Amharic Text Normalizer (indextts/utils/amharic_front.py)
- Subtitle Parser (indextts/utils/subtitle_parser.py)
- Web Downloader (indextts/utils/web_downloader.py)
- Audio Slicer (indextts/utils/audio_slicer.py)
- Quality Validator (indextts/utils/quality_validator.py)
