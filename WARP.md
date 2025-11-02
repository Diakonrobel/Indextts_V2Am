# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is an **IndexTTS2 fine-tuning system** specialized for **Amharic language** (ፊደል script). It's a fork/adaptation of the original IndexTTS2 zero-shot TTS model with additional fine-tuning infrastructure for low-resource languages, particularly Amharic. The system uses LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.

## Core Architecture

### Three-Component System
1. **Base Model**: IndexTTS2 UnifiedVoice architecture (24 transformer layers, 1280 hidden dim)
2. **Conditioning Modules**: Conformer-perceiver architecture for speaker and emotion conditioning
3. **Vocoder**: BigVGAN v2 for mel-to-waveform conversion

### Key Design Patterns
- **Multi-stage training**: Three-stage training pipeline (basic alignment → speaker-emotion disentanglement → duration control)
- **Emotion-speaker disentanglement**: Separate conditioning for timbre and emotional expression
- **Duration control**: Novel autoregressive duration adaptation mechanism
- **Cross-lingual transfer**: Leverages pre-trained multilingual knowledge for Amharic

## Common Commands

### Environment Setup
```powershell
# Install dependencies using uv package manager (REQUIRED)
uv sync --all-extras

# For Windows (if DeepSpeed installation fails):
uv sync --extra webui

# Verify GPU acceleration
uv run tools/gpu_check.py
```

### Model Download
```powershell
# Download IndexTTS-2 model from HuggingFace
uv tool install "huggingface-hub[cli,hf_xet]"
hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints

# OR from ModelScope
uv tool install modelscope
modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
```

### Running the WebUI
```powershell
# Basic WebUI
uv run webui.py

# With FP16 (recommended for VRAM efficiency)
uv run webui.py --fp16

# With all optimizations
uv run webui.py --fp16 --cuda_kernel --deepspeed
```

### Amharic Fine-tuning Workflow

#### Step 1: Prepare Vocabulary
```powershell
uv run scripts/train_amharic_vocabulary.py `
    --text_files data/amharic_texts.txt `
    --output_dir models/amharic_vocab `
    --vocab_size 8000
```

#### Step 2: Prepare Dataset
```powershell
uv run scripts/prepare_amharic_data.py `
    --audio_dir data/audio_files `
    --text_dir data/text_files `
    --output_dir data/amharic_dataset `
    --sample_rate 24000
```

#### Step 3: Fine-tune Model (LoRA approach - RECOMMENDED)
```powershell
# LoRA training (parameter-efficient, 70% memory savings)
uv run scripts/finetune_amharic.py `
    --config configs/amharic_config.yaml `
    --model_path checkpoints/gpt.pth `
    --amharic_vocab models/amharic_vocab/amharic_bpe.model `
    --train_manifest data/amharic_dataset/train.jsonl `
    --val_manifest data/amharic_dataset/val.jsonl `
    --use_lora `
    --learning_rate 5e-5
```

#### Step 4: Full Layer Training (for 200+ hour datasets)
```powershell
# Full training (maximum quality, requires more VRAM)
uv run scripts/optimized_full_layer_finetune_amharic.py `
    --config configs/amharic_200hr_full_training_config.yaml `
    --model_path checkpoints/gpt.pth `
    --amharic_vocab models/amharic_vocab/amharic_bpe.model `
    --train_manifest data/amharic_dataset/train.jsonl `
    --val_manifest data/amharic_dataset/val.jsonl
```

#### Step 5: Evaluate Model
```powershell
uv run scripts/evaluate_amharic.py `
    --config configs/amharic_config.yaml `
    --model_path checkpoints/amharic/best_amharic_model.pt `
    --amharic_vocab models/amharic_vocab/amharic_bpe.model `
    --output_dir checkpoints/amharic/evaluation
```

### Testing
```powershell
# Run padding tests
uv run tests/padding_test.py

# Run regression tests
uv run tests/regression_test.py
```

## Directory Structure & Key Files

### Configuration Files
- `configs/amharic_config.yaml` - LoRA-based Amharic fine-tuning (default)
- `configs/amharic_200hr_full_training_config.yaml` - Full-layer training for large datasets
- `configs/german_config.yaml` - German language fine-tuning reference
- `checkpoints/config.yaml` - Base IndexTTS2 model configuration

### Core Modules
- `indextts/gpt/model_v2.py` - UnifiedVoice GPT architecture (24 layers, multi-conditioning)
- `indextts/gpt/conformer_encoder.py` - Conformer-perceiver conditioning module
- `indextts/infer_v2.py` - IndexTTS2 inference engine
- `indextts/utils/amharic_front.py` - Amharic text normalization and tokenization
- `indextts/utils/enhanced_amharic_model.py` - Enhanced UnifiedVoice with three-stage training
- `indextts/adapters/lora.py` - LoRA adapter implementation

### Training Scripts
- `scripts/finetune_amharic.py` - LoRA-based Amharic fine-tuning (531 lines)
- `scripts/optimized_full_layer_finetune_amharic.py` - Full-layer training with memory optimizations
- `scripts/prepare_amharic_data.py` - Dataset preparation (478 lines)
- `scripts/train_amharic_vocabulary.py` - SentencePiece BPE vocabulary training (311 lines)
- `scripts/evaluate_amharic.py` - Model evaluation metrics (445 lines)

### Web Interface
- `webui.py` - Gradio web UI for inference
- `amharic_gradio_app.py` - Amharic-specific Gradio interface
- `launch_gradio.py` - Gradio launcher

## Important Technical Details

### Training Strategy Selection
**LoRA (Default):**
- Use for: 10-50 hour datasets, single GPU training, academic research
- Parameters: ~0.1% trainable (150K / 150M total)
- Memory: 70% reduction vs full training
- Speed: 3-5x faster
- Config: `configs/amharic_config.yaml`

**Full Layer Training:**
- Use for: 200+ hour datasets, commercial applications, maximum quality
- Parameters: 100% trainable
- Memory: Requires 16GB+ VRAM (use FP16 + gradient checkpointing)
- Speed: Baseline (slower but deeper convergence)
- Config: `configs/amharic_200hr_full_training_config.yaml`

### Memory Optimization Techniques
- **Mixed Precision (FP16)**: CRITICAL for T4/consumer GPUs - use `--fp16` or set `mixed_precision: true` in config
- **Gradient Checkpointing**: Trades computation for memory - enabled in configs
- **Activation Checkpointing**: Additional memory savings for full training
- **CPU Offload**: Offload optimizer states to CPU when needed
- **Batch Size**: Start with 1-2 for full training, 4-8 for LoRA

### Amharic Language Specifics
- **Script**: Modern Amharic (ፊደል/Ge'ez script) with 276+ characters
- **Vocabulary Size**: 8000 BPE tokens (sufficient for comprehensive coverage)
- **Text Normalization**: Number expansion (አንድ→1), abbreviation expansion (ዶ/ր→ዶክተር)
- **Character Coverage**: Target 99.9% for Amharic script
- **Tokenization**: SentencePiece BPE optimized for Amharic morphology

### Model Configuration
```yaml
# Core GPT architecture
gpt:
    model_dim: 1280          # Hidden dimension
    layers: 24               # Transformer layers
    heads: 20                # Attention heads
    max_mel_tokens: 1815     # Max mel spectrogram length
    max_text_tokens: 600     # Max text sequence length
    condition_type: "conformer_perceiver"  # Conditioning architecture

# Mel spectrogram
dataset:
    sample_rate: 24000       # Audio sample rate
    mel:
        n_fft: 1024
        hop_length: 256
        n_mels: 100
```

### Anti-Overfitting Strategies (Critical for Small Datasets)
- **Data Augmentation**: Speed (0.9-1.1x), pitch (±0.5 semitones), noise injection
- **Regularization**: Dropout (0.25 full training, 0.1 LoRA), label smoothing (0.2)
- **Early Stopping**: Monitor validation loss/CER with patience=4
- **Validation Split**: 15% for 200hr datasets, 10% for smaller
- **Learning Rate**: 2e-5 (full), 5e-5 (LoRA)
- **Gradient Clipping**: 0.3 (full), 1.0 (LoRA)

## Development Workflows

### Adding New Language Support
1. Create language-specific text normalizer (extend `indextts/utils/amharic_front.py`)
2. Train SentencePiece vocabulary with appropriate `vocab_size`
3. Create language config file (copy from `configs/amharic_config.yaml`)
4. Prepare dataset manifests with `scripts/prepare_amharic_data.py` as template
5. Fine-tune using LoRA or full training depending on dataset size

### Debugging Training Issues
- Check `logs/amharic_*/training.log` for detailed logs
- Use TensorBoard: `tensorboard --logdir runs/`
- Monitor with WandB if enabled (set in config)
- Validate data quality with `scripts/validate_amharic_pipeline.py`
- Check GPU memory with `tools/gpu_check.py`

### Inference Usage Patterns
```python
from indextts.infer_v2 import IndexTTS2

# Initialize model
tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    use_fp16=True,  # Recommended
    use_cuda_kernel=False,
    use_deepspeed=False
)

# Basic synthesis (voice cloning)
tts.infer(
    spk_audio_prompt='examples/voice_01.wav',
    text="Your text here",
    output_path="output.wav",
    verbose=True
)

# With emotion control (separate audio)
tts.infer(
    spk_audio_prompt='examples/voice_07.wav',
    text="Your text here",
    emo_audio_prompt="examples/emo_sad.wav",
    emo_alpha=0.9,  # 0.0-1.0, emotion intensity
    output_path="output.wav"
)

# With emotion vector (8-float array)
# [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
tts.infer(
    spk_audio_prompt='examples/voice_10.wav',
    text="Your text here",
    emo_vector=[0, 0, 0, 0, 0, 0, 0.45, 0],
    use_random=False,
    output_path="output.wav"
)

# With text-based emotion control
tts.infer(
    spk_audio_prompt='examples/voice_12.wav',
    text="Your text here",
    emo_alpha=0.6,  # Lower for text mode
    use_emo_text=True,
    emo_text="Optional separate emotion description",
    use_random=False,
    output_path="output.wav"
)
```

## Common Issues & Solutions

### VRAM Issues
- Enable FP16: `use_fp16=True` or `--fp16`
- Enable gradient checkpointing in config
- Reduce batch size (try 1 for full training)
- Use LoRA instead of full training
- Enable CPU offload in config

### Training Instability
- Reduce learning rate (try 1e-5 for full, 3e-5 for LoRA)
- Increase warmup steps (3000+ for full training)
- Enable label smoothing (0.1-0.2)
- Check data quality (audio clipping, text errors)
- Increase gradient accumulation steps

### Poor Amharic Quality
- Verify 99.9% character coverage with `scripts/verify_amharic_coverage.py`
- Check text normalization (numbers, abbreviations)
- Ensure sufficient training data (50+ hours minimum)
- Validate audio quality (SNR > 10dB)
- Consider full training for datasets > 100 hours

### DeepSpeed Installation Failures (Windows)
- Skip DeepSpeed: `uv sync --extra webui` (omit `--all-extras`)
- DeepSpeed provides marginal benefits on single GPU
- Only useful for multi-GPU setups

## Important Notes

- **Always use `uv run`** to execute Python scripts - never activate venv manually
- **CUDA 12.8+ required** for GPU support
- **Git LFS required** for downloading model checkpoints (`git lfs install`)
- **HuggingFace mirror**: Set `HF_ENDPOINT="https://hf-mirror.com"` if downloads are slow
- **LoRA community consensus**: Full-layer training required for new languages with large datasets (200+ hours)
- **Checkpoint management**: Models saved every 250 steps (full) or 500 steps (LoRA)
- **Resume training**: Use `--resume_from_checkpoint` flag with checkpoint path

## Project Documentation

Key documentation files:
- `README.md` - Official IndexTTS2 documentation
- `FINAL_IMPLEMENTATION_GUIDE.md` - Complete implementation overview
- `AMHARIC_INDEXTTS2_README.md` - Amharic-specific guide
- `TRAINING_STRATEGIES_ANALYSIS.md` - LoRA vs full training comparison
- `docs/AMHARIC_TRAINING_WORKFLOW_GUIDE.md` - Visual workflow guide
- `AMHARIC_200HR_OPTIMIZATION_GUIDE.md` - Large dataset optimization
- `CAPABILITIES_PRESERVATION_GUIDE.md` - Maintaining model capabilities
- `CHECKPOINT_MANAGEMENT_COMPLETE.md` - Checkpoint best practices

## Contact & Support

- QQ Group: 663272642, 1013410623
- Discord: https://discord.gg/uT32E7KDmy
- Email: indexspeech@bilibili.com
- Official Repo: https://github.com/index-tts/index-tts

## License

Custom Bilibili IndexTTS license - see `LICENSE` and `LICENSE_ZH.txt`
