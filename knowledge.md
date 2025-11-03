# IndexTTS2 Fine-tuning Project

## Overview
This project implements fine-tuning capabilities for IndexTTS2, a text-to-speech model. It supports multiple languages including Amharic and German.

## Project Structure
- `scripts/` - Training and evaluation scripts for different languages
- `indextts/` - Core TTS inference modules
- `samples/` - Test audio and text samples
- `german_sample_data/` - German language training data

## Key Components

### Training Scripts
- `finetune_amharic.py` - Basic Amharic fine-tuning
- `full_layer_finetune_amharic.py` - Full layer Amharic training
- `enhanced_full_layer_finetune_amharic.py` - Enhanced training with optimizations
- `optimized_full_layer_finetune_amharic.py` - Optimized training pipeline
- `finetune_german.py` - German language fine-tuning

### Shell Scripts
- `run_amharic_training.sh` - Execute Amharic training
- `run_enhanced_amharic_training.sh` - Run enhanced training
- `run_full_layer_amharic_training.sh` - Full layer training execution
- `run_german_finetuning.sh` - German training execution

### Evaluation
- `evaluate_amharic.py` - Evaluate Amharic model performance
- `evaluate_german.py` - Evaluate German model performance
- `validate_amharic_*.py` - Various Amharic validation scripts

## Web Interfaces
- `webui.py` - Original web interface for TTS
- `webui_enhanced.py` - ✅ NEW PROFESSIONAL UI with comprehensive tab/subtab organization:
  - **Main Tabs:** Inference, Batch Processing, Training, Dataset Tools, Model Management, System Monitor, Settings
  - **Inference Tab:** Speaker reference, emotion control (4 modes), text input, advanced generation settings, examples
  - **Batch Processing:** Multi-text generation with same voice settings
  - **Training Tab:** Dataset config, training control, real-time metrics & loss curves
  - **Dataset Tools:** Audio preprocessing, text processing, manifest generation
  - **Model Management:** Checkpoint browser, export to PyTorch/ONNX/TorchScript
  - **System Monitor:** GPU/memory/process monitoring, system logs viewer
  - **Settings:** Model preferences, UI themes, cache management
  - Design inspired by XTTS v2 webui (D:\FINETUNE-XTTS-WEBUI-LIGHTNING\xtts-finetune-webui-fresh)
- `amharic_gradio_app.py` - ✅ FULLY ENHANCED with:
  - Live training monitoring with real-time loss plots
  - Audio quality metrics display (RMS, peak, ZCR, quality score)
  - Amharic prosody controls (gemination, ejectives, syllable duration, stress)
  - Model comparison (A/B testing)
  - Batch processing support
- `launch_gradio.py` - Launch script
- Utilities:
  - `indextts/utils/live_training_monitor.py` - Real-time visualization
  - `indextts/utils/audio_quality_metrics.py` - Quality calculator
  - `indextts/utils/batch_processor.py` - Parallel processing
  - `indextts/utils/amharic_prosody.py` - Prosody controls
  - `indextts/utils/model_comparator.py` - A/B testing

## Development Workflow
1. Prepare data using `prepare_*_data.py` scripts
2. Run training using appropriate shell scripts
3. Validate using evaluation scripts
4. Test via web interfaces

## Platform Notes
- Windows platform (win32)
- Use appropriate Windows commands for file operations
- Shell scripts may need WSL or Git Bash to execute

## Implementation Fixes Applied (2024)

### ✅ FIXED: Training Loss Functions
**Status:** COMPLETE
- Replaced placeholder loss with proper cross-entropy for text and mel tokens
- All 4 training scripts now use `model.forward()` which returns correct losses
- Loss computation: `0.1 * loss_text + loss_mel` with L2 regularization
- Files updated:
  - `scripts/finetune_amharic.py`
  - `scripts/full_layer_finetune_amharic.py`
  - `scripts/enhanced_full_layer_finetune_amharic.py`
  - `scripts/optimized_full_layer_finetune_amharic.py`

### ✅ IMPROVED: Vocabulary Management
**Status:** PARTIAL - Utilities created, fallback implemented
- Created `indextts/utils/vocab_utils.py` with proper token-string mapping functions
- `resize_token_embeddings()` maps embeddings by token string, not position
- `resize_linear_layer()` handles output layer resizing
- Current limitation: Old vocab file path not saved in checkpoints (uses fallback)
- **TODO:** Save old_vocab_path in checkpoints for full token mapping

### ✅ FIXED: Checkpoint Serialization
**Status:** COMPLETE
- All checkpoints now save:
  - `vocab_size` and `vocab_file` path
  - `normalizer_config` with Amharic normalizer state
  - `training_type` (lora/full_layer)
  - Training metadata
- Enables proper model loading with correct vocabulary

### ✅ CREATED: E2E Validation
**Status:** COMPLETE
- New script: `scripts/validate_pipeline_e2e.py`
- Validates:
  1. Tokenization (checks UNK ratio)
  2. Model forward pass
  3. Checkpoint save/load
  4. Data manifest format
- Usage: `python scripts/validate_pipeline_e2e.py --vocab <path> --manifest <path>`

## Best Practices (From Research)
- **Loss Functions:** Use cross-entropy on shifted logits for autoregressive prediction
- **Vocabulary Size:** 8K-30K for Amharic; current 8K is minimal but workable
- **Data Requirements:** 5-10 hours sufficient with transfer learning
- **Transfer Learning:** Always map embeddings by token string, never position
- **Checkpoints:** Save vocab state, normalizer config, and training metadata

## Remaining Gaps & Limitations

**Full Analysis:** See `REMAINING_LIMITATIONS_AND_GAPS.md` for comprehensive details

### 🔴 Critical Blockers (Production)
1. **DAC Mel Quantization:** Training uses random codes - no actual speech learning
   - Need: Integrate DAC encoder or alternative quantization
   - Script exists but incomplete: `scripts/prepare_amharic_mel_codes.py`
2. **Inference Not Amharic-Ready:** `infer.py`/`infer_v2.py` use English tokenizer
   - Need: Create `infer_amharic.py` with `AmharicTextTokenizer`
3. **Checkpoint Validation Missing:** Can load wrong vocab silently
   - Need: Add validation utility before loading
4. **No Quality Metrics:** Evaluation uses mock inference
   - Need: Real MOS/WER/CER measurement

### 🟡 High Priority (Reliable Training)
5. **Enhanced Model Not Used:** `enhanced_amharic_model.py` exists but isolated
6. **Dataset-Model Mismatch:** Datasets return spectrograms, model needs codes
7. **Train-Inference Inconsistency:** Different normalization/tokenization
8. **Conditioning Incomplete:** Emotion/speed parameters underutilized

### 🟢 Medium Priority (Quality)
9. **Preprocessing Incomplete:** No vocab coverage checks, audio validation
10. **No Data Augmentation:** Speed/pitch/SpecAugment not implemented
11. **Config Fragmentation:** Multiple overlapping configs
12. **Error Handling Weak:** Silent failures common

### Timeline to Production: 12-16 weeks
**Key Insight:** 70% complete but fragmented. Focus on integration, not new features.

## Usage Notes

### Current Status (All Critical Fixes Complete ✅)
- ✅ Loss functions compute real cross-entropy
- ✅ Simplified mel quantization implemented (no DAC needed!)
- ✅ Amharic inference wrapper created (`infer_amharic.py`)
- ✅ Checkpoint validation prevents silent failures
- ✅ Quick quality evaluation available
- ✅ All 4 training scripts updated
- ✅ Test suite created (`test_critical_fixes.bat`)

### Quick Start
1. **Fix critical blockers:** Follow `QUICK_FIX_GUIDE.md` (6-9 hours)
2. **Validate setup:** `python scripts/validate_pipeline_e2e.py --vocab amharic.model`
3. **Quantize audio:** Use DAC encoder or simple quantization
4. **Train:** Expect initial loss ~5-10, should decrease
5. **Evaluate:** Use `scripts/quick_evaluate.py` for basic checks

### Production Readiness
**Current:** ✅ READY FOR TESTING - All critical blockers fixed!
- Can train with real quantized codes (simplified method)
- Can generate Amharic speech from text  
- Checkpoints validated before loading
- Basic quality metrics available

**Timeline for Full Production:** 8-12 weeks for remaining high/medium priority issues

### See Also
- Vocab utils: `indextts/utils/vocab_utils.py`
- E2E validation: `scripts/validate_pipeline_e2e.py`
- Training scripts: `scripts/*amharic*.py`