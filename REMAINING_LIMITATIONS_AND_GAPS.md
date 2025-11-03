# IndexTTS2 Amharic Fine-tuning - Remaining Limitations & Missing Implementations

**Status:** Post-Critical-Fixes Analysis (January 2025)

## Executive Summary

**Core Finding:** The implementation exists in functional fragments but lacks critical integrations. Training pipeline is operational but cannot produce quality models without mel code quantization. Inference exists but is not Amharic-ready.

**Production Readiness:** 🔴 NOT READY - Multiple blockers prevent production deployment

---

## 🔴 CRITICAL BLOCKERS (Must Fix for Production)

### 1. Mel Code Quantization Pipeline Missing

**Problem:**
- Training expects discrete mel codes but receives random codes as fallback
- No actual speech learning occurs - model trains on noise
- DAC encoder integration incomplete (skeleton script only)

**Impact:** Training runs but produces unusable models

**Files Affected:**
- All training scripts use `torch.randint()` for mel_codes
- `scripts/prepare_amharic_mel_codes.py` - incomplete (no DAC model loading)
- `scripts/amharic_dataset_with_codes.py` - not used in training

**Fix Required:**
```python
# Current (BROKEN):
mel_codes = torch.randint(0, 8194, (batch_size, seq_len))  # Random!

# Needed:
mel_codes = load_precomputed_dac_codes(audio_path)  # Real quantized codes
```

**Implementation Steps:**
1. Obtain/train DAC model for mel quantization
2. Pre-process all training audio to .dac format
3. Update datasets to load .dac files
4. Modify training scripts to use `AmharicTTSDatasetWithCodes`

---

### 2. Inference Pipeline Not Amharic-Ready

**Problem:**
- `infer.py` hardcoded for English/Chinese tokenization
- `infer_v2.py` has no Amharic text normalization
- Amharic punctuation not handled in segment splitting
- Tokenizer mismatch (English BPE vs Amharic SentencePiece)

**Impact:** Cannot generate Amharic speech from fine-tuned models

**Files Affected:**
- `indextts/infer.py` - Lines 132-140 (English tokenizer)
- `indextts/infer_v2.py` - Lines 420-450 (no Amharic support)

**Current Code:**
```python
# infer.py - BROKEN for Amharic
self.tokenizer = TextTokenizer(self.bpe_path, self.normalizer)
# Uses English BPE, not Amharic SentencePiece!
```

**Fix Required:**
```python
# Needed: Amharic-specific inference
from indextts.utils.amharic_front import AmharicTextTokenizer
self.amharic_tokenizer = AmharicTextTokenizer(amharic_vocab_file)
```

**Implementation Steps:**
1. Create `indextts/infer_amharic.py` or modify existing inference
2. Integrate `AmharicTextTokenizer` and `AmharicTextNormalizer`
3. Add Amharic punctuation handling (።፤፣ etc.)
4. Test end-to-end: Amharic text → audio

---

### 3. Checkpoint Validation Unsafe

**Problem:**
- No validation when loading checkpoints
- Vocabulary size mismatches cause silent failures
- Normalizer state not verified
- Can load wrong model without error

**Impact:** Training/inference failures appear as "quality issues" not errors

**Files Affected:**
- All training scripts `_load_model()` methods
- `indextts/utils/checkpoint.py`

**Missing Checks:**
```python
# NO validation currently!
checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model_state_dict'])

# Needed:
assert checkpoint['vocab_size'] == tokenizer.vocab_size
assert checkpoint['normalizer_config']['type'] == 'AmharicTextNormalizer'
assert checkpoint.get('language') == 'amharic'
```

**Fix Required:**
- Add checkpoint validation utility
- Verify vocab compatibility before loading
- Check model architecture matches
- Validate normalizer consistency

---

### 4. No Quality Metrics Implementation

**Problem:**
- `evaluate_amharic.py` uses mock inference
- No actual MOS (Mean Opinion Score) collection
- No WER/CER measurement with Amharic ASR
- No intelligibility or naturalness metrics

**Impact:** Cannot measure if training improves quality

**Files Affected:**
- `scripts/evaluate_amharic.py` - Lines 190-250 (mock results)

**Current Implementation:**
```python
# MOCK INFERENCE - not real!
predicted_mel_length = np.random.randint(100, 1000)
# No actual model.generate() call!
```

**Fix Required:**
1. Integrate real inference in evaluation
2. Add Amharic ASR for WER computation
3. Implement audio quality metrics (PESQ, STOI)
4. Create listening test framework for MOS

---

## 🟡 HIGH PRIORITY (Needed for Reliable Training)

### 5. Enhanced Model Architecture Not Used

**Problem:**
- `indextts/utils/enhanced_amharic_model.py` implements advanced 3-stage training
- Includes gradient reversal, speaker-emotion disentanglement
- But main training scripts don't use it

**Impact:** Missing performance improvements from research

**Gap:**
```python
# Current training scripts use:
model = UnifiedVoice(...)  # Base model

# But enhanced model exists:
model = EnhancedAmharicUnifiedVoice(...)  # 3-stage, disentanglement
# Not integrated!
```

**Fix Required:**
- Integrate `EnhancedAmharicUnifiedVoice` into training pipeline
- Implement 3-stage training orchestration
- Add speaker-emotion disentanglement
- Test gradient reversal effectiveness

---

### 6. Dataset-Model Integration Incomplete

**Problem:**
- Datasets return `mel_spectrograms` but model needs `mel_codes`
- `AmharicTTSDatasetWithCodes` created but not used
- Conditioning audio extraction not standardized
- Speaker IDs not assigned

**Impact:** Data format mismatches cause training issues

**Current State:**
```python
# Dataset returns:
batch = {
    'mel_spectrograms': torch.Tensor,  # Continuous
    ...
}

# But model expects:
model.forward(
    mel_codes=torch.LongTensor,  # Discrete codes!
)
```

**Fix Required:**
1. Switch all training to `AmharicTTSDatasetWithCodes`
2. Standardize conditioning extraction
3. Add speaker ID field to manifests
4. Create unified data loading utility

---

### 7. Training-Inference Consistency Missing

**Problem:**
- Text normalization differs between train/inference
- Tokenization code paths separate
- Segment splitting only in inference
- No roundtrip validation

**Impact:** Inference quality degradation from normalization drift

**Example Gap:**
```python
# Training uses:
tokenizer.normalizer.normalize(text)  # Custom rules

# But inference uses:
self.normalizer.normalize(text)  # Different normalizer instance!
# Rules may have changed!
```

**Fix Required:**
- Serialize normalizer state with checkpoints
- Add roundtrip test: text → train → infer → text
- Unify tokenization code paths
- Validate preprocessing matches

---

### 8. Conditioning Pipeline Inconsistent

**Problem:**
- `get_conditioning()` expects specific shapes
- Emotion conditioning optional but no clear fallback
- Speed control parameter unused
- Multi-speaker not properly integrated

**Impact:** Training instability, inference errors

**Current Issues:**
```python
# Unclear fallback logic:
if emo_cond_emb is None:
    emo_cond_emb = spk_cond_emb  # Is this correct?

# Speed never used:
use_speed = torch.zeros(...)  # Always zero!
```

**Fix Required:**
- Clarify conditioning requirements
- Document fallback behavior
- Implement speed control properly
- Add speaker embedding extraction

---

## 🟢 MEDIUM PRIORITY (Quality Improvements)

### 9. Data Preprocessing Incomplete

**Missing Features:**
- Vocabulary coverage not checked on training data
- UNK token ratio unmeasured pre-training
- Audio quality validation absent (clipping, silence)
- Duration information not extracted
- No phoneme coverage analysis

**Impact:** Poor data quality goes undetected

**Needed:**
```python
# Add to prepare_amharic_data.py:
- check_vocabulary_coverage(texts, vocab)
- validate_audio_quality(audio_paths)
- extract_duration_info(manifest)
- analyze_phoneme_distribution(texts)
```

---

### 10. Data Augmentation Not Implemented

**Problem:**
- Speed/pitch perturbation mentioned but not applied
- SpecAugment referenced but missing
- No noise injection
- No reverb/room simulation

**Impact:** Model overfits to training acoustics

**Needed Implementation:**
```python
class AmharicAudioAugmentation:
    def __call__(self, audio):
        # Speed perturbation (0.9x - 1.1x)
        audio = speed_perturb(audio, factor=random.uniform(0.9, 1.1))
        
        # SpecAugment on mel features
        mel = mel_spectrogram(audio)
        mel = spec_augment(mel, freq_mask=30, time_mask=40)
        
        return mel
```

---

### 11. Configuration Management Fragmented

**Problem:**
- Multiple config files with overlapping settings
- Hardcoded values in scripts override configs
- No single source of truth
- Version compatibility unclear

**Affected Files:**
- `configs/amharic_config.yaml`
- `configs/amharic_200hr_config.yaml`
- `configs/amharic_config_lora_backup.yaml`

**Fix Required:**
- Consolidate to single Amharic config
- Remove hardcoded values
- Add config validation
- Document parameter purposes

---

### 12. Error Handling Insufficient

**Problem:**
- Silent failures common (wrong vocab loads)
- No graceful degradation
- Stack traces unhelpful for users
- No automatic recovery

**Examples:**
```python
# Current: Silent failure
try:
    model.load_state_dict(checkpoint)
except Exception as e:
    logger.error(f"Error: {e}")  # What error? Where?
    return None  # Continue with broken model!
```

**Fix Required:**
- Add specific exception types
- Provide actionable error messages
- Implement automatic fallbacks
- Add detailed logging

---

## 🔵 LOW PRIORITY (Nice to Have)

### 13. Testing Infrastructure Missing

**Gaps:**
- No unit tests for Amharic components
- No integration tests
- No regression tests
- No CI/CD pipeline

**Needed:**
```python
# tests/test_amharic_tokenizer.py
def test_tokenization_invertible():
    text = "ሰላም ዓለም"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    assert decoded == text
```

---

### 14. Documentation Outdated

**Issues:**
- README doesn't reflect current implementation
- API documentation missing
- Configuration options undocumented
- Tutorial notebooks absent

---

### 15. Performance Optimization Missing

**Opportunities:**
- No mixed precision training benchmarks
- Flash attention not integrated
- Gradient checkpointing optional
- Multi-GPU training untested

---

## Implementation Priority Matrix

| Priority | Blocker | Effort | Impact | Timeline |
|----------|---------|--------|--------|----------|
| 🔴 DAC Integration | Yes | High | Critical | 2-4 weeks |
| 🔴 Amharic Inference | Yes | Medium | Critical | 1-2 weeks |
| 🔴 Checkpoint Validation | Yes | Low | High | 3-5 days |
| 🔴 Quality Metrics | Yes | Medium | High | 1-2 weeks |
| 🟡 Enhanced Model | No | Medium | Medium | 2-3 weeks |
| 🟡 Dataset Integration | No | Medium | High | 1-2 weeks |
| 🟡 Consistency Checks | No | Low | Medium | 1 week |
| 🟡 Conditioning Fix | No | Medium | Medium | 1-2 weeks |
| 🟢 Preprocessing | No | Low | Low | 3-5 days |
| 🟢 Augmentation | No | Medium | Medium | 1 week |

---

## Recommended Implementation Order

### Phase 1: Make Training Work (4-6 weeks)
1. ✅ Fix loss functions (DONE)
2. **Integrate DAC encoder** or use alternative mel quantization
3. **Update datasets** to load quantized codes
4. **Add checkpoint validation**
5. **Test end-to-end**: data prep → train → checkpoint

### Phase 2: Make Inference Work (2-3 weeks)
6. **Create Amharic inference** (`infer_amharic.py`)
7. **Integrate AmharicTextTokenizer**
8. **Add consistency validation** (train ↔ inference)
9. **Test end-to-end**: Amharic text → audio

### Phase 3: Add Quality Measurement (2-3 weeks)
10. **Implement real evaluation** (remove mocks)
11. **Add Amharic ASR** for WER/CER
12. **Integrate audio metrics** (PESQ, STOI)
13. **Create listening test framework**

### Phase 4: Optimization (3-4 weeks)
14. **Integrate enhanced model**
15. **Add data augmentation**
16. **Implement 3-stage training**
17. **Optimize performance**

---

## Detailed Fix Examples

### Example 1: Integrating DAC Encoder

**File:** `scripts/prepare_amharic_mel_codes.py`

```python
# Add actual DAC model loading:
def load_dac_model(model_path, device='cuda'):
    from indextts.s2mel.dac.model.base import DAC
    
    # Load DAC configuration
    config = load_dac_config(model_path)
    
    # Initialize model
    dac_model = DAC(
        encoder_dim=config['encoder_dim'],
        encoder_rates=config['encoder_rates'],
        decoder_dim=config['decoder_dim'],
        decoder_rates=config['decoder_rates'],
        n_codebooks=config['n_codebooks'],
        codebook_size=config['codebook_size']
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location='cpu')
    dac_model.load_state_dict(checkpoint['model_state_dict'])
    dac_model = dac_model.to(device)
    dac_model.eval()
    
    return dac_model

# Use in encoding:
dac_model = load_dac_model('checkpoints/dac.pt')
for audio_file in audio_files:
    audio = load_audio(audio_file)
    codes = dac_model.encode(audio)  # Get discrete codes
    save_dac_file(codes, output_path)
```

### Example 2: Amharic Inference Integration

**File:** `indextts/infer_amharic.py`

```python
class IndexTTSAmharic(IndexTTS):
    """Amharic-specific inference for IndexTTS"""
    
    def __init__(self, amharic_vocab_file, **kwargs):
        # Initialize base class
        super().__init__(**kwargs)
        
        # Override with Amharic tokenizer
        from indextts.utils.amharic_front import (
            AmharicTextTokenizer, 
            AmharicTextNormalizer
        )
        
        self.normalizer = AmharicTextNormalizer()
        self.tokenizer = AmharicTextTokenizer(
            vocab_file=amharic_vocab_file,
            normalizer=self.normalizer
        )
        
        logger.info(f"Amharic tokenizer loaded: {amharic_vocab_file}")
    
    def split_amharic_segments(self, text, max_tokens=120):
        """Split Amharic text using proper punctuation"""
        # Amharic sentence endings
        sentences = re.split(r'[።፤]', text)
        
        segments = []
        for sent in sentences:
            if not sent.strip():
                continue
            
            tokens = self.tokenizer.encode(sent)
            if len(tokens) <= max_tokens:
                segments.append(tokens)
            else:
                # Further split on commas
                sub_sents = re.split(r'[፣،]', sent)
                for sub in sub_sents:
                    segments.append(self.tokenizer.encode(sub))
        
        return segments
```

### Example 3: Checkpoint Validation

**File:** `indextts/utils/checkpoint_validator.py`

```python
class CheckpointValidator:
    """Validates checkpoint compatibility"""
    
    @staticmethod
    def validate_amharic_checkpoint(checkpoint_path, tokenizer):
        """Validate Amharic fine-tuned checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Check required fields
        required_fields = [
            'model_state_dict',
            'vocab_size',
            'vocab_file',
            'normalizer_config'
        ]
        
        for field in required_fields:
            if field not in checkpoint:
                raise ValueError(f"Checkpoint missing required field: {field}")
        
        # Validate vocabulary
        if checkpoint['vocab_size'] != tokenizer.vocab_size:
            raise ValueError(
                f"Vocab size mismatch: "
                f"checkpoint={checkpoint['vocab_size']}, "
                f"tokenizer={tokenizer.vocab_size}"
            )
        
        # Validate normalizer
        normalizer_type = checkpoint['normalizer_config'].get('type')
        if normalizer_type != 'AmharicTextNormalizer':
            raise ValueError(
                f"Expected AmharicTextNormalizer, got {normalizer_type}"
            )
        
        # Check model architecture
        state_dict = checkpoint['model_state_dict']
        expected_keys = [
            'text_embedding.weight',
            'mel_embedding.weight',
            'text_head.weight'
        ]
        
        for key in expected_keys:
            if key not in state_dict:
                raise ValueError(f"Model missing layer: {key}")
        
        # Validate dimensions
        text_emb_size = state_dict['text_embedding.weight'].shape[0]
        if text_emb_size != tokenizer.vocab_size:
            raise ValueError(
                f"Text embedding size mismatch: {text_emb_size} != {tokenizer.vocab_size}"
            )
        
        logger.info("✅ Checkpoint validation passed")
        return True
```

---

## Summary

**Total Issues Identified:** 15 major gaps

**Critical Blockers:** 4 (must fix for production)
**High Priority:** 4 (needed for reliable training)
**Medium Priority:** 4 (quality improvements)
**Low Priority:** 3 (nice to have)

**Estimated Timeline to Production:** 12-16 weeks with dedicated effort

**Key Insight:** Implementation is 70% complete but fragmented. Critical 30% is integration work, not new features. Focus on connecting existing pieces before adding more functionality.

---

## Next Steps

1. **Immediate:** Fix critical blockers (Phases 1-2)
2. **Short-term:** Add quality measurement (Phase 3)
3. **Long-term:** Optimize and enhance (Phase 4)

**Remember:** Working end-to-end pipeline > perfect individual components
