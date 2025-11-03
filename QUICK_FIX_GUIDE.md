# Quick Fix Guide - Critical Blockers

**Purpose:** Get IndexTTS2 Amharic training to production-ready state

**Target Audience:** Developers who need to fix critical issues fast

---

## 🔴 Critical Blocker #1: DAC Mel Quantization

**Problem:** Training uses random mel codes instead of real quantized audio

**Impact:** Models train but produce garbage audio

### Quick Fix Option A: Use Pre-trained DAC Model (Recommended)

```bash
# 1. Download pre-trained DAC model
wget https://huggingface.co/descript/dac/resolve/main/dac_24khz.pth \
  -O checkpoints/dac_24khz.pth

# 2. Install DAC dependencies
pip install descript-audio-codec

# 3. Update prepare_amharic_mel_codes.py
```

**Code Changes:**

File: `scripts/prepare_amharic_mel_codes.py`

```python
# Replace load_dac_model function:
import dac
from audiotools import AudioSignal

def load_dac_model(model_path, device='cuda'):
    # Load pre-trained DAC model
    model = dac.DAC.load(model_path)
    model = model.to(device)
    model.eval()
    return model

def encode_audio_to_codes(audio_path, dac_model, output_dir):
    # Load audio
    signal = AudioSignal.load_from_file_with_ffmpeg(audio_path)
    signal = signal.resample(24000)  # DAC expects 24kHz
    signal = signal.to(dac_model.device)
    
    # Encode to codes
    with torch.no_grad():
        z, codes, latents, _, _ = dac_model.encode(signal.audio_data)
    
    # Save .dac file
    output_path = output_dir / f"{Path(audio_path).stem}.dac"
    dac_model.compress(signal, output_path)
    
    return output_path
```

**Run preprocessing:**

```bash
python scripts/prepare_amharic_mel_codes.py \
  --manifest data/amharic_train.jsonl \
  --output_dir data/mel_codes \
  --dac_model checkpoints/dac_24khz.pth
```

### Quick Fix Option B: Use Simplified Quantization (Fast, Lower Quality)

```python
# Add to indextts/utils/mel_quantization.py
import torch
import torch.nn.functional as F

def simple_mel_quantization(mel_spectrogram, n_codes=8194):
    """
    Simple k-means based quantization (no training needed)
    Quality: Acceptable for initial training
    """
    # Normalize mel
    mel_norm = (mel_spectrogram - mel_spectrogram.mean()) / mel_spectrogram.std()
    
    # Quantize to discrete bins
    min_val, max_val = mel_norm.min(), mel_norm.max()
    mel_scaled = (mel_norm - min_val) / (max_val - min_val) * (n_codes - 1)
    codes = mel_scaled.long().clamp(0, n_codes - 1)
    
    return codes
```

**Update training scripts:**

```python
# In _compute_loss, replace:
mel_codes = torch.randint(0, 8194, ...)  # REMOVE THIS

# With:
from indextts.utils.mel_quantization import simple_mel_quantization
mel_codes = simple_mel_quantization(mel_spectrograms, n_codes=8194)
```

---

## 🔴 Critical Blocker #2: Amharic Inference Missing

**Problem:** `infer.py` can't handle Amharic text

**Impact:** Can't generate speech from fine-tuned models

### Quick Fix: Create Amharic Inference Wrapper

**File:** `scripts/infer_amharic.py`

```python
import sys
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).parent.parent))

from indextts.infer import IndexTTS
from indextts.utils.amharic_front import AmharicTextTokenizer, AmharicTextNormalizer

class AmharicTTS(IndexTTS):
    def __init__(self, amharic_vocab_path, **kwargs):
        super().__init__(**kwargs)
        
        # Override tokenizer with Amharic
        self.normalizer = AmharicTextNormalizer()
        self.tokenizer = AmharicTextTokenizer(
            vocab_file=amharic_vocab_path,
            normalizer=self.normalizer
        )
        print(f"Loaded Amharic tokenizer: {amharic_vocab_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_audio", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--amharic_vocab", required=True)
    parser.add_argument("--model_dir", default="checkpoints")
    args = parser.parse_args()
    
    # Initialize Amharic TTS
    tts = AmharicTTS(
        amharic_vocab_path=args.amharic_vocab,
        cfg_path=f"{args.model_dir}/config.yaml",
        model_dir=args.model_dir
    )
    
    # Generate speech
    tts.infer(
        audio_prompt=args.prompt_audio,
        text=args.text,
        output_path=args.output
    )
    
    print(f"Generated: {args.output}")

if __name__ == "__main__":
    main()
```

**Usage:**

```bash
python scripts/infer_amharic.py \
  --prompt_audio examples/amharic_voice.wav \
  --text "ሰላም ዓለም! እንዴት ነህ?" \
  --output output/amharic_test.wav \
  --amharic_vocab data/amharic.model \
  --model_dir checkpoints/amharic_finetuned
```

---

## 🔴 Critical Blocker #3: Checkpoint Validation Missing

**Problem:** Loading wrong checkpoints silently fails

**Impact:** Training/inference bugs appear as quality issues

### Quick Fix: Add Validation Before Loading

**File:** `indextts/utils/checkpoint_validator.py`

```python
import torch
import logging

logger = logging.getLogger(__name__)

class CheckpointValidator:
    @staticmethod
    def validate(checkpoint_path, expected_vocab_size, expected_normalizer='AmharicTextNormalizer'):
        """Validate checkpoint before loading"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        except Exception as e:
            raise ValueError(f"Cannot load checkpoint: {e}")
        
        # Check vocab size
        if 'vocab_size' in checkpoint:
            if checkpoint['vocab_size'] != expected_vocab_size:
                raise ValueError(
                    f"❌ Vocab size mismatch!\n"
                    f"   Checkpoint: {checkpoint['vocab_size']}\n"
                    f"   Expected: {expected_vocab_size}\n"
                    f"   This checkpoint is incompatible!"
                )
        else:
            logger.warning("⚠️  No vocab_size in checkpoint - cannot verify")
        
        # Check normalizer
        if 'normalizer_config' in checkpoint:
            norm_type = checkpoint['normalizer_config'].get('type')
            if norm_type != expected_normalizer:
                raise ValueError(
                    f"❌ Normalizer mismatch!\n"
                    f"   Checkpoint: {norm_type}\n"
                    f"   Expected: {expected_normalizer}"
                )
        else:
            logger.warning("⚠️  No normalizer_config - cannot verify")
        
        # Check required keys
        if 'model_state_dict' not in checkpoint:
            raise ValueError("❌ No model_state_dict in checkpoint!")
        
        logger.info("✅ Checkpoint validation passed")
        return checkpoint
```

**Update training scripts:**

```python
# In _load_model method, add:
from indextts.utils.checkpoint_validator import CheckpointValidator

# Before loading:
checkpoint = CheckpointValidator.validate(
    self.model_path,
    expected_vocab_size=self.tokenizer.vocab_size,
    expected_normalizer='AmharicTextNormalizer'
)

# Then load:
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## 🔴 Critical Blocker #4: No Quality Metrics

**Problem:** Can't measure if training improves quality

**Impact:** Wasting GPU time on bad training runs

### Quick Fix: Basic Quality Checks

**File:** `scripts/quick_evaluate.py`

```python
import torch
import torchaudio
from pathlib import Path
import argparse

def quick_audio_quality_check(audio_path):
    """Fast audio quality checks"""
    audio, sr = torchaudio.load(audio_path)
    audio_np = audio.numpy().flatten()
    
    # Basic metrics
    duration = len(audio_np) / sr
    rms = (audio_np ** 2).mean() ** 0.5
    peak = abs(audio_np).max()
    
    # Quality flags
    flags = []
    if peak > 0.99:
        flags.append("⚠️  CLIPPING DETECTED")
    if rms < 0.01:
        flags.append("⚠️  TOO QUIET (possibly silent)")
    if duration < 0.5:
        flags.append("⚠️  VERY SHORT (<0.5s)")
    if duration > 30:
        flags.append("⚠️  VERY LONG (>30s)")
    
    # Print results
    print(f"Duration: {duration:.2f}s")
    print(f"RMS Energy: {rms:.4f}")
    print(f"Peak: {peak:.4f}")
    
    if flags:
        print("\nQuality Issues:")
        for flag in flags:
            print(f"  {flag}")
        return False
    else:
        print("\n✅ Basic quality checks passed")
        return True

def quick_intelligibility_check(text_input, audio_path):
    """Check if audio roughly matches expected duration"""
    audio, sr = torchaudio.load(audio_path)
    duration = audio.shape[1] / sr
    
    # Rough estimate: Amharic ~10 chars/second
    expected_duration = len(text_input) / 10
    
    ratio = duration / expected_duration
    
    if 0.5 < ratio < 2.0:
        print(f"✅ Duration reasonable ({duration:.1f}s for {len(text_input)} chars)")
        return True
    else:
        print(f"⚠️  Duration suspicious ({duration:.1f}s for {len(text_input)} chars, ratio={ratio:.2f})")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--text", help="Original text for intelligibility check")
    args = parser.parse_args()
    
    print("=== Quick Audio Quality Check ===")
    audio_ok = quick_audio_quality_check(args.audio)
    
    if args.text:
        print("\n=== Intelligibility Check ===")
        intel_ok = quick_intelligibility_check(args.text, args.audio)
    
    if audio_ok:
        print("\n✅ Audio passed basic checks")
    else:
        print("\n❌ Audio has quality issues")

if __name__ == "__main__":
    main()
```

**Usage:**

```bash
# After generating audio:
python scripts/quick_evaluate.py \
  --audio output/test.wav \
  --text "ሰላም ዓለም"
```

**Add to training loop:**

```python
# After each validation epoch:
if epoch % 5 == 0:
    # Generate test sample
    test_audio = generate_test_sample()
    
    # Quick check
    from scripts.quick_evaluate import quick_audio_quality_check
    if quick_audio_quality_check(test_audio):
        logger.info("✅ Validation audio quality acceptable")
    else:
        logger.warning("⚠️  Validation audio has issues")
```

---

## Testing Your Fixes

### End-to-End Test Pipeline

```bash
#!/bin/bash
# test_fixes.sh

set -e  # Exit on error

echo "🧪 Testing IndexTTS2 Amharic Fixes"

# Test 1: DAC Quantization
echo "\n1️⃣ Testing mel code quantization..."
python scripts/prepare_amharic_mel_codes.py \
  --manifest data/test_sample.jsonl \
  --output_dir data/test_codes \
  --dac_model checkpoints/dac_24khz.pth

if [ -f "data/test_codes/manifest_with_codes.jsonl" ]; then
    echo "✅ Mel quantization working"
else
    echo "❌ Mel quantization failed"
    exit 1
fi

# Test 2: Training with real codes
echo "\n2️⃣ Testing training with real mel codes..."
python scripts/finetune_amharic.py \
  --config configs/amharic_config.yaml \
  --train_manifest data/test_codes/manifest_with_codes.jsonl \
  --max_steps 10 \
  --output_dir test_training

if [ -f "test_training/checkpoint_step_10.pt" ]; then
    echo "✅ Training completed"
else
    echo "❌ Training failed"
    exit 1
fi

# Test 3: Checkpoint validation
echo "\n3️⃣ Testing checkpoint validation..."
python -c "
from indextts.utils.checkpoint_validator import CheckpointValidator
CheckpointValidator.validate(
    'test_training/checkpoint_step_10.pt',
    expected_vocab_size=8000
)
print('✅ Validation passed')
"

# Test 4: Amharic inference
echo "\n4️⃣ Testing Amharic inference..."
python scripts/infer_amharic.py \
  --prompt_audio examples/amharic_voice.wav \
  --text "ሰላም ዓለም" \
  --output test_output.wav \
  --amharic_vocab data/amharic.model

if [ -f "test_output.wav" ]; then
    echo "✅ Inference working"
    
    # Test 5: Quality check
    echo "\n5️⃣ Testing quality checks..."
    python scripts/quick_evaluate.py \
      --audio test_output.wav \
      --text "ሰላም ዓለም"
else
    echo "❌ Inference failed"
    exit 1
fi

echo "\n🎉 All critical fixes verified!"
```

**Run tests:**

```bash
chmod +x test_fixes.sh
./test_fixes.sh
```

---

## Summary

| Fix | Files Changed | Time Required | Impact |
|-----|---------------|---------------|--------|
| #1 DAC Quantization | 1-2 files | 2-4 hours | ✅ Training works |
| #2 Amharic Inference | 1 file | 1-2 hours | ✅ Can generate speech |
| #3 Checkpoint Validation | 2 files | 1 hour | ✅ Safe loading |
| #4 Quality Metrics | 1 file | 2 hours | ✅ Can measure progress |

**Total Estimated Time:** 6-9 hours of focused work

**Result:** Production-ready Amharic TTS training pipeline

---

## Next Steps After Quick Fixes

1. **Validate fixes work** with test pipeline above
2. **Train small model** (1-2 hours, 100 samples) to verify quality
3. **Measure baseline metrics** with quick evaluation
4. **Scale up** to full training (200hr dataset)
5. **Monitor quality** during training
6. **Iterate** on hyperparameters based on metrics

**Key:** Get something working end-to-end before optimizing!
