# Amharic IndexTTS2 WebUI Quick Start Guide

## 🎯 Overview

`amharic_gradio_app.py` is your **PRIMARY** web interface for Amharic TTS training and inference.

✅ **Now Enhanced** with professional UI design inspired by XTTS v2

---

## 🚀 Launch the WebUI

```bash
python amharic_gradio_app.py
```

The interface will be available at: `http://localhost:7860`

---

## 📋 Prerequisites

### Required Models

Before launching, you need these Amharic-specific models:

1. **Amharic Vocabulary**: `amharic_bpe.model`
   - Train using: `python scripts/train_amharic_vocabulary.py`

2. **Vocoder**: `checkpoints/bigvgan_v2_22khz_80band_256x/`
   - Download separately or use pretrained BigVGAN

3. **Amharic Checkpoints**: Fine-tuned model weights
   - Train using training scripts in `scripts/`

### Required Python Packages

```bash
pip install gradio torch torchaudio numpy pyyaml psutil
```

---

## 🎨 UI Features (NEW)

### Professional Design
- **Gradient Header**: Eye-catching purple gradient (#667eea to #764ba2)
- **5 Main Tabs**: Organized workflow from training to deployment
- **Modern Theme**: Gradio Soft theme with custom CSS
- **Emoji Icons**: Visual hierarchy for better UX
- **Responsive Layout**: Works on desktop and mobile

### Tab Organization

#### 1. 🚀 Training Tab
- **Dataset Management**: Upload and prepare Amharic audio/text datasets
- **Training Configuration**: Set epochs, batch size, learning rate, optimizations
- **Live Monitoring**: Real-time loss curves, current metrics, auto-refresh
- **Control Panel**: Start/Stop training with visual feedback

#### 2. 🎵 Inference Tab
- **Model Loading**: Load trained Amharic checkpoints
- **Single Text Generation**: Synthesize Amharic speech
- **Prosody Controls**: Amharic-specific features
  - Gemination emphasis (ሁለት vs ሁሌት)
  - Ejective consonant strength (ጥ, ቅ, ጭ)
  - Syllable duration control
  - Stress pattern (penultimate/final/initial)
- **Quality Metrics**: RMS, peak level, ZCR, quality score
- **Batch Processing**: Generate multiple texts at once

#### 3. 🔬 Model Comparison Tab
- **A/B Testing**: Load two models side-by-side
- **Comparative Analysis**: Generate same text with both models
- **Metrics Comparison**: See which model performs better
- **Winner Determination**: Automated quality-based scoring

#### 4. 📊 System Monitor Tab
- **GPU Status**: Device info, memory usage
- **Checkpoint Browser**: List all available trained models
- **Training History**: View past training runs
- **System Configuration**: PyTorch, CUDA, Gradio versions

#### 5. 📁 Model Management Tab
- **Model Info**: Current model parameters and size
- **Export Models**: Convert to PyTorch/ONNX/TensorRT
- **Model Validation**: Check checkpoint integrity

---

## 💡 Typical Workflow

### For Training
1. Go to **Training Tab**
2. Upload your Amharic audio files and transcriptions
3. Configure training parameters (recommend: 8 epochs, batch size 1, lr 2e-5)
4. Enable optimizations: SDPA ✓, EMA ✓, Mixed Precision ✓
5. Click "🚀 Start Training"
6. Monitor progress in real-time with live loss curves

### For Inference
1. Download models first (or train your own)
2. Go to **Inference Tab**
3. Load your trained Amharic model
4. Enter Amharic text: `ሰላም ዓለም! እንዴት ነሽ?`
5. Adjust prosody controls for natural speech
6. Click "🎙️ Generate Speech"
7. Review quality metrics and listen to output

### For Comparison
1. Go to **Model Comparison Tab**
2. Load two different checkpoints (e.g., epoch 10 vs epoch 20)
3. Enter test text in Amharic
4. Click "⚖️ Generate & Compare"
5. Listen to both outputs and see automated winner

---

## 🎯 Best Practices

### Training
- **Start Small**: Use 1-2 hours of data first to test
- **Monitor Loss**: Should decrease from ~5-10 to <2
- **Save Checkpoints**: Every 1000 steps recommended
- **Use Optimizations**: SDPA + EMA + Mixed Precision saves 50% GPU memory

### Inference
- **Prosody Controls**: Start with defaults (all 1.0) and adjust incrementally
- **Quality Metrics**: Aim for quality score >7/10
- **Batch Processing**: Use for multiple sentences to save time

### System Monitoring
- **GPU Memory**: Keep usage <90% to avoid OOM errors
- **Checkpoints**: Regularly clean old checkpoints to save space
- **Logs**: Check system logs if errors occur

---

## 🐛 Troubleshooting

### "No model loaded"
- Make sure you clicked "🔄 Load Model for Inference" first
- Verify model paths are correct
- Check that `amharic_bpe.model` exists

### "Training won't start"
- Verify dataset paths in Training Configuration
- Check that config YAML file exists
- Ensure pretrained model checkpoint is available

### "Poor audio quality"
- Increase training epochs
- Adjust prosody controls (try gemination=1.2, ejective=1.1)
- Check input audio quality in dataset
- Review quality metrics flags

### "Out of memory"
- Reduce batch size to 1
- Increase gradient accumulation to 16 or 32
- Enable mixed precision
- Close other GPU applications

---

## 📊 UI Enhancements Applied

### Visual Improvements
✅ Modern gradient header (#667eea to #764ba2)  
✅ Enhanced tab navigation with larger buttons  
✅ Professional footer with gradient background  
✅ Consistent spacing and padding  
✅ Box-shadow effects for depth

### UX Improvements
✅ Clear tab names with emoji icons  
✅ Tab IDs for programmatic access  
✅ Larger, more visible buttons  
✅ Feature highlights in header  
✅ Status boxes with color coding

### Code Quality
✅ Organized CSS with semantic class names  
✅ Consistent styling across all tabs  
✅ Responsive design principles  
✅ Compiles without errors

---

## 🎓 Advanced Usage

### Custom Amharic Prosody

Amharic has unique phonetic features:

- **Gemination**: Doubled consonants that change meaning
  - Example: ጠለ (he painted) vs ጠለለ (he shaded)
  - Control: Increase gemination slider to 1.3-1.5 for emphasis

- **Ejective Consonants**: Glottalized sounds (ጥ, ቅ, ጭ, ፅ)
  - More forceful than regular consonants
  - Control: Increase ejective slider to 1.2-1.4 for clarity

- **Syllable-Based Rhythm**: Amharic is syllable-timed
  - Control: Adjust syllable duration (0.7=fast, 1.3=slow)

- **Stress Pattern**: Typically penultimate syllable
  - Most words stress second-to-last syllable
  - Control: Select "penultimate" (default) or override

---

## 📞 Support

For issues:
1. Check `logs/gradio/gradio_app.log`
2. Review system logs in System Monitor tab
3. Consult training scripts documentation

---

## 🎉 Summary

The Amharic IndexTTS2 WebUI now features:
- ✅ Professional XTTS v2-inspired design
- ✅ 5 comprehensive tabs for complete workflow
- ✅ Amharic-specific prosody controls
- ✅ Real-time training monitoring
- ✅ Quality metrics and A/B testing
- ✅ Production-ready interface

**Launch**: `python amharic_gradio_app.py`  
**Access**: http://localhost:7860
