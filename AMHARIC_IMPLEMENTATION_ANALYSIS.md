# Amharic IndexTTS2 Fine-tuning: Comprehensive Implementation Analysis

**Analysis Date:** 2025
**Project:** IndexTTS2 Amharic Fine-tuning Implementation
**Scope:** Complete pipeline from data preprocessing through inference

---

## Executive Summary

### Critical Findings

**🔴 SEVERITY: CRITICAL - Training Pipeline is Non-Functional**
- All training scripts implement **placeholder loss functions** that return ~0.0
- Models are not actually learning TTS-specific objectives
- Training only applies regularization, not actual speech synthesis losses
- **Impact:** Wasted GPU resources, no meaningful fine-tuning occurs

**🟡 SEVERITY: HIGH - Architecture Integration Gaps**
- Vocabulary resizing logic is fragile and semantically incorrect
- No serialization of text normalizer state with checkpoints
- Training and inference use different code paths with no validation
- Missing end-to-end validation pipeline

**🟢 SEVERITY: MEDIUM - Documentation vs Implementation**
- Claims about LoRA ineffectiveness lack empirical validation
- Performance metrics (SDPA speed, EMA quality) unverified
- Multiple training approaches with no clear guidance on selection

---

## 1. Data Preprocessing Pipeline Analysis

### 1.1 Text Normalization Issues

#### Problem: State Serialization Gap
```python
# AmharicTextNormalizer has hardcoded dictionaries
class AmharicTextNormalizer:
    def __init__(self):
        self.number_words = {...}  # Hardcoded
        self.abbreviations = {...}  # Hardcoded
        self.contractions = {...}   # Hardcoded
```

**Issue:** 
- Normalizer state not saved with model checkpoints
- If abbreviations/rules change between training and inference → different tokenization
- **No mechanism** to ensure training and inference use identical normalization

**Consequence:**
```
Training: "ዶ/ር አበበ" → "ዶክተር አበበ" (with abbreviation expansion)
Inference (updated rules): "ዶ/ר אבבב" → "ዶ/ר אבבב" (no expansion)
→ Out-of-vocabulary tokens → Poor quality
```

#### Recommendation:
```python
# Add to AmharicTextNormalizer:
def save_state(self, path):
    state = {
        'number_words': self.number_words,
        'abbreviations': self.abbreviations,
        'contractions': self.contractions
    }
    with open(path, 'w') as f:
        json.dump(state, f, ensure_ascii=False)

def load_state(self, path):
    with open(path, 'r') as f:
        state = json.load(f)
    self.number_words = state['number_words']
    self.abbreviations = state['abbreviations']
    self.contractions = state['contractions']
```

### 1.2 Data Preparation Consistency

**File:** `scripts/prepare_amharic_data.py`

**Strengths:**
- Comprehensive audio validation (duration, sample rate)
- Multiple text format support (.txt, .json, .lrc)
- Manifest generation with metadata

**Gaps:**
- No validation that Amharic text is actually covered by vocabulary
- No check for excessive punctuation or special characters
- Audio-text alignment not verified
- No detection of potential homophone issues in Amharic

**Missing Validation:**
```python
def validate_vocabulary_coverage(self, text, tokenizer):
    """Check if text is properly covered by vocabulary"""
    tokens = tokenizer.encode(text)
    unk_count = sum(1 for t in tokens if t == tokenizer.unk_token_id)
    if unk_count / len(tokens) > 0.05:  # >5% unknown
        warnings.warn(f"High UNK rate: {unk_count}/{len(tokens)}")
```

---

## 2. Tokenization & Vocabulary Management

### 2.1 Critical: Vocabulary Resizing Logic is Broken

**Location:** All training scripts (finetune_amharic.py, full_layer, enhanced, optimized)

**Current Implementation:**
```python
old_vocab_size = checkpoint['text_embedding.weight'].shape[0]
new_vocab_size = self.tokenizer.vocab_size

if old_vocab_size != new_vocab_size:
    old_embedding = checkpoint['text_embedding.weight']
    new_embedding = torch.randn(new_vocab_size, old_embedding.shape[1])
    new_embedding.normal_(mean=0.0, std=0.02)
    
    # Copy first N tokens
    min_size = min(old_vocab_size, new_vocab_size)
    new_embedding[:min_size] = old_embedding[:min_size]  # ❌ WRONG!
```

**Why This is Wrong:**
1. **Assumes token ID alignment:** Token ID 100 in old vocab ≠ same semantic meaning in new vocab
2. **Special tokens may not align:** `<unk>`, `<s>`, `</s>` positions could differ
3. **Semantic mismatch:** Copying embeddings by position has no semantic basis
4. **Loss of transfer learning:** New Amharic-specific tokens get random init → no pre-trained knowledge

**Correct Approach:**
```python
# 1. Map tokens by their string representation, not position
old_tokenizer = load_old_tokenizer(old_model_path)
new_embedding = torch.randn(new_vocab_size, embed_dim)
new_embedding.normal_(mean=0.0, std=0.02)

for new_id in range(new_vocab_size):
    token_str = new_tokenizer.convert_ids_to_tokens([new_id])[0]
    if token_str in old_tokenizer.vocab:
        old_id = old_tokenizer.vocab[token_str]
        new_embedding[new_id] = old_embedding[old_id]

# 2. For new Amharic tokens, use subword composition
for new_id in range(new_vocab_size):
    if not mapped[new_id]:  # New token
        token_str = new_tokenizer.convert_ids_to_tokens([new_id])[0]
        # Average embeddings of subword components
        new_embedding[new_id] = compose_from_subwords(token_str, old_tokenizer, old_embedding)
```

### 2.2 Vocabulary Size Analysis

**Current:** 8,000 tokens (BPE)

**Coverage Requirements for Amharic:**
- Base Amharic syllabary: ~231 characters
- With diacritics/combinations: ~300 symbols
- Numbers (digits + Amharic words): ~50 tokens
- Punctuation (including Amharic-specific ፤፦፣።): ~20 tokens
- Latin characters (mixed text): ~100 tokens
- Common words/morphemes: ~1000-2000 tokens

**Analysis:**
- 8K is **adequate** for coverage BUT
- No runtime validation that coverage is achieved
- `character_coverage=0.9999` helps but isn't verified post-training

**Missing:**
```python
def analyze_vocabulary_coverage(vocab_file, text_corpus):
    """Validate that vocabulary adequately covers corpus"""
    sp = spm.SentencePieceProcessor()
    sp.load(vocab_file)
    
    total_chars = 0
    covered_chars = 0
    unk_tokens = 0
    
    for text in text_corpus:
        tokens = sp.encode(text, out_type=str)
        total_chars += len(text)
        unk_tokens += sum(1 for t in tokens if t == '<unk>')
        covered_chars += sum(len(t) for t in tokens if t != '<unk>')
    
    coverage = covered_chars / total_chars
    unk_rate = unk_tokens / len(tokens)
    
    print(f"Character coverage: {coverage:.2%}")
    print(f"UNK token rate: {unk_rate:.2%}")
    
    if coverage < 0.99:
        warnings.warn("Insufficient vocabulary coverage!")
```

### 2.3 Tokenization Consistency Between Train/Inference

**Training Path:**
```
Manifest → Dataset → tokenizer.encode(text) → tokens
```

**Inference Path (`infer.py`):**
```
Text → tokenizer.tokenize(text) → segments → tokenizer.convert_tokens_to_ids() → tokens
```

**Issue:** Different methods, no guarantee of same output!

**Test Required:**
```python
def test_tokenization_consistency():
    text = "ሰላም ዓለም! ይህ ሙከራ ነው።"
    
    # Training path
    tokens_train = tokenizer.encode(text, out_type=int)
    
    # Inference path
    token_strs = tokenizer.tokenize(text)
    tokens_infer = tokenizer.convert_tokens_to_ids(token_strs)
    
    assert tokens_train == tokens_infer, "Tokenization mismatch!"
```

---

## 3. Model Architecture Integration

### 3.1 CRITICAL: Training Loss is Not Implemented

**All Training Scripts:**
```python
def _compute_loss(self, text_tokens, text_attention_masks, 
                 mel_spectrograms, mel_attention_masks):
    """Compute training loss for IndexTTS2 with Amharic"""
    # This is a simplified loss computation
    # In practice, you'd implement the full IndexTTS2 training logic
    
    base_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
    
    # Add regularization loss
    reg_loss = 0.0
    for name, param in self.model.named_parameters():
        if param.requires_grad and 'weight' in name:
            reg_loss += torch.norm(param, p=2) * 0.01
    
    total_loss = base_loss + reg_loss
    return total_loss  # ← Returns ~regularization only!
```

**Consequence:**
- Model parameters updated only by regularization gradients
- **No speech synthesis objective** being optimized
- Training = expensive no-op that doesn't improve TTS quality

**What Should Be Implemented:**
```python
def _compute_loss(self, text_tokens, text_attention_masks,
                 mel_spectrograms, mel_attention_masks):
    # 1. Forward pass through GPT
    speech_conditioning = self.get_conditioning(...)
    text_emb = self.text_embedding(text_tokens)
    mel_emb = self.mel_embedding(mel_codes)
    
    # 2. Get predictions
    text_logits, mel_logits = self.model.get_logits(
        speech_conditioning, text_emb, mel_emb)
    
    # 3. Compute reconstruction losses
    text_loss = F.cross_entropy(
        text_logits, text_targets, 
        ignore_index=self.stop_text_token)
    
    mel_loss = F.cross_entropy(
        mel_logits, mel_targets,
        ignore_index=self.stop_mel_token)
    
    # 4. Combine
    total_loss = text_loss + mel_loss
    return total_loss
```

### 3.2 Architecture Mismatch: Training vs Full Model

**Full Model (`model_v2.UnifiedVoice`):**
- Emotion conditioning encoder (Conformer-based)
- Speaker conditioning encoder (Perceiver-based)
- Speed embeddings
- Multi-stage conditioning

**Training Scripts:**
- Only access basic GPT forward
- Don't train emotion/speaker systems
- Ignore most conditioning pathways

**Impact:**
- Fine-tuning doesn't adapt the full model capabilities
- Emotion/speaker transfer learning not utilized
- Missing opportunity to specialize conditioning for Amharic prosody

### 3.3 Position Embedding Compatibility

**Issue:** Training scripts don't validate that:
- `max_text_tokens` matches between checkpoint and config
- `max_mel_tokens` matches
- Position embeddings are compatible

**Potential Failure:**
```
Checkpoint: max_text_tokens = 600
Config: max_text_tokens = 120
→ Loading position_emb[600, 512] into position_emb[120, 512]
→ RuntimeError: size mismatch
```

---

## 4. Training Strategy Analysis

### 4.1 LoRA vs Full Fine-tuning Confusion

**Documentation Claims:**
> "Community evidence that LoRA fails for new languages like Amharic"

**Issues:**
1. **No citation** for this "community evidence"
2. **No empirical validation** in this codebase
3. **Contradictory:** LoRA code still present and maintained
4. **Research evidence** (from web research) suggests LoRA **works well** for low-resource TTS

**Web Research Findings:**
- LoRA achieves 85-95% of full fine-tuning quality with 10x less compute
- Particularly effective for low-resource scenarios (<10 hours data)
- Prevents catastrophic forgetting of multilingual capabilities

**Recommendation:** 
- **Test both empirically** on Amharic data
- Document actual results with metrics (MOS, WER, training time, memory)
- Don't rely on unverified claims

### 4.2 Full-Layer Training Concerns

**`full_layer_finetune_amharic.py` Claims:**
> "All 24 layers will be trained"

**Reality:**
```python
trainable_params = list(self.model.parameters())
for param in trainable_params:
    param.requires_grad = True
```

**What's Actually Trained:**
- GPT transformer layers: ✓
- Text embeddings: ✓
- Mel embeddings: ✓
- Final heads: ✓

**What's NOT Trained:**
- Conditioning encoders: ✗ (still frozen)
- Perceiver modules: ✗
- Emotion/speaker systems: ✗

**Misleading:** "Full-layer" suggests everything is trained, but key components are frozen

### 4.3 Optimizations Need Validation

**`optimized_full_layer_finetune_amharic.py` Claims:**
- SDPA: "1.3-1.5x speed"
- EMA: "5-10% quality boost"
- Gradient checkpointing: "20-30% memory reduction"

**Issue:** No benchmarks provided!

**Should Include:**
```python
# Benchmark script
def benchmark_optimizations():
    configs = [
        {'sdpa': False, 'ema': False, 'grad_checkpoint': False},  # Baseline
        {'sdpa': True, 'ema': False, 'grad_checkpoint': False},   # +SDPA
        {'sdpa': True, 'ema': True, 'grad_checkpoint': False},    # +EMA
        {'sdpa': True, 'ema': True, 'grad_checkpoint': True},     # All
    ]
    
    for cfg in configs:
        speed, memory, quality = train_with_config(cfg)
        print(f"{cfg}: {speed:.2f}s/iter, {memory:.1f}GB, MOS={quality:.2f}")
```

---

## 5. Inference Pipeline Issues

### 5.1 Cache Invalidation Fragility

**Current:**
```python
if self.cache_spk_cond is None or self.cache_spk_audio_prompt != spk_audio_prompt:
    # Recompute conditioning
```

**Problem:** String comparison of file paths
```
Path 1: "/data/audio.wav"
Path 2: "./data/audio.wav"  # Same file, different string!
→ Cache miss, recomputation
```

**Better:**
```python
import hashlib

def hash_audio_file(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

if self.cache_audio_hash != hash_audio_file(spk_audio_prompt):
    # Recompute
```

### 5.2 Segment Splitting Inconsistency

**Training:** No segment splitting (full sequences)
**Inference:** Splits into segments of `max_text_tokens_per_segment`

**Consequence:**
- Model never sees segmented inputs during training
- Segment boundaries in inference may cause artifacts
- No validation that segment rejoining sounds natural

**Missing:**
- Training with segment-aware objectives
- Cross-segment smoothing in inference
- Validation that splits preserve prosody

### 5.3 Model Version Compatibility

**Issue:** Version checks exist but version not set during training

```python
# infer.py
if self.model_version and self.model_version >= 1.5:
    # Use updated padding strategy
else:
    # Use old padding
```

**Training scripts don't set:**
```python
checkpoint = {
    'model_state_dict': self.model.state_dict(),
    # 'version': ???  ← Missing!
}
```

**Risk:** Loading checkpoint without version → wrong padding → poor quality

---

## 6. Critical Missing Components

### 6.1 End-to-End Validation

**What's Missing:**
```python
# scripts/validate_amharic_complete.py
def validate_complete_pipeline():
    """Test entire train→infer pipeline"""
    
    # 1. Prepare small dataset
    prepare_test_data()
    
    # 2. Train vocabulary
    vocab_path = train_vocabulary(test_corpus)
    
    # 3. Fine-tune model (1 epoch)
    checkpoint = finetune_model(vocab_path, epochs=1)
    
    # 4. Run inference
    audio = infer_with_model(checkpoint, "ሰላም ዓለም")
    
    # 5. Validate output
    assert audio is not None
    assert len(audio) > 0
    assert not contains_artifacts(audio)
    
    print("✓ End-to-end validation passed")
```

### 6.2 Vocabulary Coverage Validation

**Currently:** `train_amharic_vocabulary.py` creates vocab but doesn't validate coverage

**Should Add:**
```python
def validate_vocab_coverage(vocab_path, manifest_file):
    sp = spm.SentencePieceProcessor()
    sp.load(vocab_path)
    
    with open(manifest_file) as f:
        texts = [json.loads(line)['text'] for line in f]
    
    total_tokens = 0
    unk_tokens = 0
    
    for text in texts:
        tokens = sp.encode(text, out_type=int)
        total_tokens += len(tokens)
        unk_tokens += sum(1 for t in tokens if t == 0)  # UNK=0
    
    unk_rate = unk_tokens / total_tokens
    print(f"UNK rate: {unk_rate:.2%}")
    
    if unk_rate > 0.05:
        raise ValueError(f"High UNK rate ({unk_rate:.2%})! Increase vocab size.")
```

### 6.3 Configuration Management

**Problem:** Settings scattered across:
- YAML configs (partially implemented)
- Python configs (hardcoded in scripts)
- Command-line args

**No single source of truth** for:
- Sample rates (24000 vs 22050 vs 16000)
- Mel spectrogram parameters
- Max token lengths
- Training hyperparameters

**Should Implement:**
```python
# configs/amharic_base_config.yaml
data:
  sample_rate: 24000
  mel:
    n_fft: 1024
    hop_length: 256
    n_mels: 100

vocabulary:
  size: 8000
  coverage: 0.9999
  
model:
  max_text_tokens: 600
  max_mel_tokens: 1815
  
training:
  batch_size: 4
  learning_rate: 5e-5
  # ... etc
```

---

## 7. Concrete Failure Scenarios

### Scenario A: Silent Training Failure

**Steps:**
1. User prepares 200hr Amharic corpus
2. Creates manifests with `prepare_amharic_data.py`
3. Trains vocabulary (8K tokens)
4. Runs `full_layer_finetune_amharic.py` for 8 epochs on T4 GPU
5. Training completes, loss decreases (regularization)
6. Saves checkpoint

**Expected:** Fine-tuned Amharic TTS model

**Reality:**
- Placeholder loss was optimized, not TTS loss
- Model weights barely changed (only regularization gradients)
- Inference produces **same quality as pre-trained model**
- 200 GPU-hours wasted

**User Experience:**
- No error messages
- Loss decreased (false signal)
- Only realizes after inference testing

### Scenario B: Vocabulary Corruption

**Steps:**
1. Train with `amharic_bpe_v1.model`
2. Save checkpoint
3. Later: Update vocabulary to `amharic_bpe_v2.model` (8K, same size)
4. Load checkpoint for inference
5. Vocabulary file loaded from v2, embeddings from v1

**Result:**
- Token ID 100 in v2 ≠ Token ID 100 in v1
- Embeddings map to wrong tokens
- Generated audio: gibberish or heavily degraded

**No Error:** Vocab sizes match, no runtime failure, just wrong output

### Scenario C: Normalization Drift

**Steps:**
1. Train with normalization rules:
   ```python
   abbreviations = {"ዶ/ር": "ዶክተር"}
   ```
2. Save checkpoint
3. Update normalizer code:
   ```python
   abbreviations = {"ዶ/ር": "ዲክተር"}  # Different expansion
   ```
4. Run inference

**Result:**
- Training data had "ዶክተር" tokens
- Inference produces "ዲክተር" tokens
- If "ዲክተር" wasn't in training vocab → UNK → poor quality

### Scenario D: Memory Overflow on T4

**Steps:**
1. Follow "full-layer" guide for 200hr dataset
2. Set `batch_size=4` (as in examples)
3. Run on T4 GPU (16GB VRAM)
4. Enable all optimizations (SDPA, gradient checkpointing, FP16)

**Reality:**
- 200hr dataset → very long mel sequences
- Batch of 4 with max_mel_tokens=1815 → OOM
- Script crashes with `CUDA out of memory`
- No checkpoint saved (training interrupted)

**Missing:** Dynamic batch sizing or validation of memory requirements

---

## 8. Recommendations (Prioritized)

### 🔴 CRITICAL (Must Fix Before Production)

#### 1. Implement Actual TTS Loss Function

**File:** All training scripts

**Current:**
```python
return torch.tensor(0.0, requires_grad=True) + regularization
```

**Required:**
```python
def _compute_loss(self, batch):
    # Extract inputs
    text_tokens = batch['text_tokens']
    mel_targets = batch['mel_targets']
    
    # Forward pass
    conditioning = self.get_conditioning(batch['audio_prompt'])
    predictions = self.model(conditioning, text_tokens, ...)
    
    # Compute losses
    mel_loss = F.cross_entropy(
        predictions['mel_logits'], 
        mel_targets,
        ignore_index=self.stop_mel_token
    )
    
    text_loss = F.cross_entropy(
        predictions['text_logits'],
        batch['text_targets'],
        ignore_index=self.stop_text_token
    )
    
    return mel_loss + text_loss
```

**Effort:** 1-2 days
**Impact:** Enables actual fine-tuning

#### 2. Fix Vocabulary Resizing Logic

**Current:** Copies embeddings by position (semantically wrong)

**Required:** Map by token string representation

**Effort:** 4-6 hours
**Impact:** Prevents embedding corruption

#### 3. Add Vocabulary Serialization

**Required:**
```python
# Save with checkpoint
checkpoint = {
    'model_state_dict': model.state_dict(),
    'vocab_path': tokenizer.vocab_file,
    'normalizer_state': normalizer.get_state(),
    'version': '2.0'
}

# Load and validate
def load_checkpoint_safe(path):
    ckpt = torch.load(path)
    
    # Validate vocabulary matches
    current_vocab = load_vocab(tokenizer.vocab_file)
    saved_vocab = load_vocab(ckpt['vocab_path'])
    
    if current_vocab != saved_vocab:
        raise ValueError("Vocabulary mismatch!")
    
    # Restore normalizer
    normalizer.load_state(ckpt['normalizer_state'])
    
    return ckpt
```

**Effort:** 3-4 hours
**Impact:** Prevents training/inference mismatch

### 🟡 HIGH (Important for Quality)

#### 4. Add End-to-End Validation Script

**Create:** `scripts/validate_pipeline.py`

**Should Test:**
- Data preparation → manifests valid
- Vocabulary training → coverage adequate
- Model training → loss decreases meaningfully
- Inference → produces audio
- Roundtrip: text → audio → ASR → text similarity

**Effort:** 1 day
**Impact:** Catches integration issues early

#### 5. Standardize Training Paths

**Issue:** Multiple training scripts with overlapping functionality

**Recommendation:**
```
scripts/
  train_amharic.py          # Main training script
    --mode [lora|full]      # Training mode
    --optimizations [sdpa,ema,checkpoint]  # Optional optimizations
    --config path/to/config.yaml
```

**Consolidate:**
- `finetune_amharic.py`
- `full_layer_finetune_amharic.py`
- `enhanced_finetune_amharic.py`
- `optimized_full_layer_finetune_amharic.py`

**Effort:** 2-3 days
**Impact:** Easier maintenance, clearer documentation

#### 6. Add Vocabulary Coverage Validation

**In:** `scripts/train_amharic_vocabulary.py`

**Add post-training check:**
```python
if __name__ == "__main__":
    # ... train vocabulary ...
    
    # Validate coverage
    validate_coverage(
        vocab_path=model_path,
        corpus=args.text_files,
        min_coverage=0.99,
        max_unk_rate=0.05
    )
```

**Effort:** 2-3 hours
**Impact:** Ensures vocabulary is adequate

### 🟢 MEDIUM (Quality of Life)

#### 7. Unified Configuration System

**Create:** `configs/amharic_defaults.yaml` with ALL settings

**Migrate hardcoded values** from scripts to config

**Effort:** 1 day
**Impact:** Easier experimentation

#### 8. Add Tokenization Consistency Tests

**Create:** `tests/test_tokenization.py`

```python
def test_train_infer_tokenization_match():
    text = "ሰላም ዓለም"
    
    # Training path
    tokens_train = train_tokenize(text)
    
    # Inference path
    tokens_infer = infer_tokenize(text)
    
    assert tokens_train == tokens_infer
```

**Effort:** 2-3 hours
**Impact:** Prevents silent failures

#### 9. Benchmark Optimizations

**Create:** `scripts/benchmark_training.py`

**Measure:**
- Training speed (samples/sec)
- Memory usage (GB)
- Model quality (MOS on held-out set)

**For configurations:**
- Baseline
- +SDPA
- +EMA
- +Gradient checkpointing
- All combined

**Effort:** 1 day
**Impact:** Validates optimization claims

### 📊 DOCUMENTATION

#### 10. Empirically Test LoRA vs Full Fine-tuning

**Experiment:**
```
Dataset: 10hr Amharic (split: 8hr train, 1hr val, 1hr test)

Configurations:
1. LoRA (rank=8, alpha=16)
2. LoRA (rank=16, alpha=32)
3. Full fine-tuning (all parameters)
4. Full fine-tuning (freeze conditioning)

Metrics:
- MOS (naturalness)
- WER (intelligibility)
- Training time
- Memory usage
- Inference speed
```

**Document results** → Update README with evidence-based guidance

**Effort:** 2-3 days (including evaluation)
**Impact:** Replaces speculation with data

---

## 9. Best Practices (From Research)

### Transfer Learning Strategy

**Recommended Approach:**
1. Start with multilingual pre-trained model (e.g., covers Semitic languages)
2. Fine-tune on Amharic with LoRA or full approach (validate empirically)
3. Use data augmentation (synthetic data, speed/pitch perturbation)
4. Multi-stage curriculum: high-resource languages first, then Amharic

**From Research:**
- Transfer learning yields 20-50% MOS improvement over training from scratch
- 5-10 hours of data sufficient with proper transfer learning
- LoRA achieves 85-95% of full fine-tuning quality with 10x less compute

### Vocabulary Management

**Recommended:**
- Subword tokenization (BPE/SentencePiece)
- Vocabulary size: 10K-30K for Amharic
  - 8K (current) is lower end but acceptable
  - Consider 20K for better coverage with less data
- Character coverage: 0.9999+ for script diversity
- Validate coverage post-training

**From Research:**
- Byte-level tokenization (vocab ~256) also viable for Ge'ez script
- Subword tokenization reduces perplexity 15-30% on low-resource scripts

### Training Configuration

**For 200hr Dataset (T4 GPU, 16GB):**
```yaml
training:
  batch_size: 1
  gradient_accumulation_steps: 16  # Effective batch size = 16
  learning_rate: 2e-5
  warmup_steps: 3000
  max_epochs: 6-8
  early_stopping_patience: 3
  
optimizations:
  mixed_precision: true  # FP16
  gradient_checkpointing: true
  sdpa: true  # If available
  ema: true
  
data_augmentation:
  speed_perturbation: true
  speed_range: [0.9, 1.1]
  pitch_perturbation: true
  pitch_range: [-0.5, 0.5]
  noise_injection: true
  noise_level: 0.005
```

### Evaluation Strategy

**Metrics:**
1. **MOS (Mean Opinion Score):** Native Amharic speakers rate naturalness (1-5)
2. **WER (Word Error Rate):** ASR transcription accuracy
3. **CER (Character Error Rate):** For Amharic script
4. **Intelligibility:** Comprehension tests

**Validation:**
- Hold out 10-15% of data for validation
- Test set should include:
  - Diverse speakers
  - Various sentence lengths
  - Different prosodic patterns
  - Edge cases (numbers, abbreviations, mixed scripts)

---

## 10. Conclusion

### Current State Assessment

**🔴 Critical Issues:**
1. Training pipeline non-functional (placeholder losses)
2. Vocabulary management semantically incorrect
3. No checkpoint/vocabulary validation

**🟡 Integration Gaps:**
1. Training/inference paths diverge
2. Text normalization not serialized
3. Missing end-to-end validation

**🟢 Strengths:**
1. Comprehensive Amharic text processing
2. Multiple training strategies (need empirical validation)
3. Good documentation of approaches

### Immediate Next Steps

**Phase 1 (Critical Fixes - Week 1):**
1. Implement actual TTS loss function
2. Fix vocabulary resizing logic
3. Add checkpoint validation

**Phase 2 (Integration - Week 2):**
1. Create end-to-end validation script
2. Add vocabulary coverage checks
3. Unify configuration management

**Phase 3 (Optimization - Week 3):**
1. Empirically test LoRA vs full fine-tuning
2. Benchmark optimization claims
3. Document best practices

### Production Readiness

**Current:** **NOT PRODUCTION READY**
- Training doesn't actually fine-tune the model
- High risk of silent failures
- No validation of critical assumptions

**After Phase 1+2:** **READY FOR EXPERIMENTATION**
- Core training functional
- Basic validation in place
- Can iterate on quality

**After Phase 3:** **PRODUCTION READY**
- Empirically validated approaches
- Comprehensive testing
- Clear documentation and best practices

---

## Appendix: Key Files Reference

### Data Pipeline
- `scripts/prepare_amharic_data.py` - Manifest generation ✅
- `scripts/train_amharic_vocabulary.py` - Vocabulary training ⚠️ (needs validation)
- `indextts/utils/amharic_front.py` - Text processing ⚠️ (needs serialization)

### Training
- `scripts/finetune_amharic.py` - LoRA training ❌ (placeholder loss)
- `scripts/full_layer_finetune_amharic.py` - Full training ❌ (placeholder loss)
- `scripts/optimized_full_layer_finetune_amharic.py` - Optimized ❌ (placeholder loss)
- `scripts/enhanced_finetune_amharic.py` - Enhanced ❌ (placeholder loss)

### Model Architecture
- `indextts/gpt/model_v2.py` - Core TTS model ✅
- `indextts/utils/feature_extractors.py` - Mel extraction ✅

### Inference
- `indextts/infer.py` - Original inference ⚠️ (path divergence)
- `indextts/infer_v2.py` - Enhanced inference ⚠️ (path divergence)

### Evaluation
- `scripts/evaluate_amharic.py` - Evaluation metrics ⚠️ (incomplete)

**Legend:**
- ✅ Functional
- ⚠️ Needs improvement
- ❌ Critical issue

---

**End of Analysis**

*This document provides a comprehensive analysis of the Amharic IndexTTS2 fine-tuning implementation. Priority should be given to fixing critical issues (placeholder losses, vocabulary handling) before proceeding with production use.*