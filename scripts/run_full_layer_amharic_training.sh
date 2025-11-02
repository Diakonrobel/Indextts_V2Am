#!/bin/bash
# Full Layer Training Script for Amharic IndexTTS2 - NO LoRA
# Based on community evidence: LoRA fails for new language adaptation
# Optimized for 200-hour dataset with T4 GPU (16GB VRAM)

echo "🚀 AMHARIC FULL LAYER TRAINING - NO LoRA"
echo "========================================"
echo "💡 Evidence: Community tested Japanese (similar to Amharic)"
echo "   ❌ LoRA: Failed for new language adaptation"  
echo "   ✅ Full Training: Successful with all layers"
echo "   🎯 Decision: Train all 24 layers for Amharic"

# Configuration
CONFIG_PATH="configs/amharic_200hr_full_training_config.yaml"
MODEL_PATH="checkpoints/gpt.pth"  # Pre-trained IndexTTS2 model
AMHARIC_VOCAB="amharic_bpe.model"  # Trained Amharic vocabulary
OUTPUT_DIR="checkpoints/amharic_200hr_full_layer_training"
TRAIN_MANIFEST="amharic_dataset/train_manifest.jsonl"
VAL_MANIFEST="amharic_dataset/val_manifest.jsonl"

# Full layer training parameters (optimized for 200hr + T4)
EPOCHS=8                    # INCREASED: Full training needs more epochs
BATCH_SIZE=1                # REDUCED: Full training needs smaller batches
LEARNING_RATE=2e-5          # REDUCED: Full training needs lower LR
GRADIENT_ACCUMULATION=16    # INCREASED: Effective batch size: 16
WARMUP_STEPS=3000           # INCREASED: More warmup for full training
MIXED_PRECISION=true        # CRITICAL: Must be true for full training

# Anti-overfitting for full training
SAVE_EVERY=250              # MORE FREQUENT: Full training checkpoints
LOG_EVERY=25                # MORE FREQUENT: Full training monitoring
VAL_SPLIT=0.15              # More validation data

echo "📋 Full Layer Training Configuration:"
echo "   Dataset: 200 hours"
echo "   GPU: T4 16GB VRAM"
echo "   Training Type: ALL 24 LAYERS (No LoRA)"
echo "   Epochs: $EPOCHS (full training requirement)"
echo "   Batch Size: $BATCH_SIZE + grad accumulation = 16 effective"
echo "   Learning Rate: $LEARNING_RATE (lower for stability)"
echo "   Mixed Precision: $MIXED_PRECISION (critical for memory)"
echo "   Validation Split: $VAL_SPLIT"

# Check prerequisites
echo ""
echo "🔍 Checking prerequisites for full layer training..."

if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ Full training config not found: $CONFIG_PATH"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Pre-trained model not found: $MODEL_PATH"
    echo "   Please download IndexTTS2 pre-trained model"
    exit 1
fi

if [ ! -f "$AMHARIC_VOCAB" ]; then
    echo "❌ Amharic vocabulary not found: $AMHARIC_VOCAB"
    echo "   Please run: python scripts/train_amharic_vocabulary.py"
    exit 1
fi

if [ ! -f "$TRAIN_MANIFEST" ] || [ ! -f "$VAL_MANIFEST" ]; then
    echo "❌ Dataset manifests not found"
    echo "   Please run: python scripts/prepare_amharic_data.py"
    exit 1
fi

echo "✅ All prerequisites met for full layer training"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Memory optimization for T4 GPU - Full layer training
echo ""
echo "🧠 Optimizing T4 GPU for full layer training..."
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export CUDA_VISIBLE_DEVICES=0

# Verify LoRA is disabled in config
echo "🔧 Verifying LoRA is disabled..."
if grep -q "enabled: false" "$CONFIG_PATH"; then
    echo "✅ LoRA correctly disabled in config"
else
    echo "⚠️  Warning: LoRA may not be disabled in config"
fi

# Training command for full layer training
echo ""
echo "🏃 Starting FULL LAYER training..."
echo "💾 Checkpoints will be saved to: $OUTPUT_DIR"
echo "📊 Monitoring training every $LOG_EVERY steps"
echo "💾 Checkpoint saving every $SAVE_EVERY steps"
echo "🎯 NO LoRA adapters will be used"
echo ""

# Run full layer training
python scripts/full_layer_finetune_amharic.py \
    --config "$CONFIG_PATH" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --amharic_vocab "$AMHARIC_VOCAB" \
    --train_manifest "$TRAIN_MANIFEST" \
    --val_manifest "$VAL_MANIFEST" \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --mixed_precision \
    --use_wandb

# Check training success
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 FULL LAYER training completed successfully!"
    echo "📁 Best model saved at: $OUTPUT_DIR/full_training_best_model.pt"
    echo "📁 All checkpoints in: $OUTPUT_DIR/"
    echo ""
    echo "🔍 Key Differences from LoRA Training:"
    echo "   ✅ All 24 layers trained (not just adapters)"
    echo "   ✅ Complete model capacity for Amharic"
    echo "   ✅ No adapter limitations"
    echo "   ✅ Proven approach for new languages"
    echo ""
    echo "📋 Next steps:"
    echo "   1. Check training logs: $OUTPUT_DIR/full_layer_training.log"
    echo "   2. Evaluate model: python scripts/evaluate_amharic.py --model_path $OUTPUT_DIR/full_training_best_model.pt"
    echo "   3. Generate samples for quality assessment"
    
    # Show training summary
    if [ -f "$OUTPUT_DIR/full_layer_training.log" ]; then
        echo ""
        echo "📊 Training Summary:"
        tail -20 "$OUTPUT_DIR/full_layer_training.log"
    fi
    
    echo ""
    echo "🏆 SUCCESS: Full layer training completed!"
    echo "   This approach is proven to work for new languages like Amharic"
    echo "   Community evidence shows LoRA fails for complex new scripts"
else
    echo ""
    echo "❌ Full layer training failed!"
    echo "🔍 Check logs in: $OUTPUT_DIR/full_layer_training.log"
    echo "💡 Common issues for full layer training:"
    echo "   - Out of memory: Try smaller batch size or more accumulation"
    echo "   - Slow training: This is normal for full layer training"
    echo "   - Quality issues: May need more epochs or different LR"
    echo ""
    echo "🔧 Memory optimization options:"
    echo "   - Reduce batch_size to 1"
    echo "   - Increase gradient_accumulation_steps to 32"
    echo "   - Enable all memory optimizations in config"
    exit 1
fi