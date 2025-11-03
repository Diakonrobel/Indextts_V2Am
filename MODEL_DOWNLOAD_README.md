# IndexTTS2 Model Download Guide

## ⚠️ Important Notice

This script downloads the **base IndexTTS2 English/Chinese model** for `webui.py`.

**For Amharic (`amharic_gradio_app.py`)**, you need different models:
- `amharic_bpe.model` (Amharic vocabulary)
- `checkpoints/bigvgan_v2_22khz_80band_256x/` (vocoder)
- Amharic fine-tuned checkpoints

These are NOT available for automatic download yet. You must train them yourself or obtain them separately.

---

## For Base IndexTTS2 (webui.py)

This guide helps you download the required pretrained model files for the base IndexTTS2.

## Quick Start

### Option 1: Automatic Download (Recommended)

**For Windows:**
```bash
double-click download_models.bat
```

**For Python:**
```bash
python download_models.py
```

### Option 2: Manual Download

Download these files from Hugging Face and place them in `./checkpoints/`:

1. **bpe.model** (~5 MB)  
   https://huggingface.co/IndexTeam/IndexTTS-2/resolve/main/bpe.model

2. **gpt.pth** (~200 MB)  
   https://huggingface.co/IndexTeam/IndexTTS-2/resolve/main/gpt.pth

3. **config.yaml** (~2 KB)  
   https://huggingface.co/IndexTeam/IndexTTS-2/resolve/main/config.yaml

4. **s2mel.pth** (~50 MB)  
   https://huggingface.co/IndexTeam/IndexTTS-2/resolve/main/s2mel.pth

5. **wav2vec2bert_stats.pt** (~1 MB)  
   https://huggingface.co/IndexTeam/IndexTTS-2/resolve/main/wav2vec2bert_stats.pt

**Total size:** ~256 MB

## Requirements

- Python 3.8 or higher
- Internet connection
- ~300 MB free disk space

The script will automatically install `huggingface-hub` if not already installed.

## Troubleshooting

### "ModuleNotFoundError: No module named 'huggingface_hub'"

Install manually:
```bash
pip install huggingface-hub
```

### Slow Download Speed

- The files are large (~256 MB total)
- Download time depends on your internet speed
- Typical download time: 2-10 minutes

### Download Interrupted

- Simply run the script again
- Already downloaded files will be skipped
- Only missing files will be re-downloaded

### Permission Errors

- Make sure you have write permissions in the project directory
- On Windows, try running as Administrator

## Verify Installation

After downloading, check that all files exist:

```bash
dir checkpoints
```

You should see:
- bpe.model
- config.yaml
- gpt.pth
- s2mel.pth
- wav2vec2bert_stats.pt

## Next Steps

Once all files are downloaded, you can run:

```bash
python webui.py
```

The WebUI should start without errors!

## Source

All models are from the official IndexTeam repository:  
https://huggingface.co/IndexTeam/IndexTTS-2

## License

These models are distributed under the IndexTeam license. Please see the Hugging Face repository for details.
