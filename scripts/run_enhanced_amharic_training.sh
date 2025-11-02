#!/bin/bash
# Enhanced Amharic Training Script for 200-hour dataset
# Optimized for T4 GPU (16GB VRAM) with anti-overfitting measures

echo "🚀 Enhanced Amharic IndexTTS2 Training - 200hr Dataset"
echo "=================================================="

# Configuration
CONFIG_PATH="configs/amharic_200hr_config.yaml"
MODEL_PATH="checkpoints/gpt.pth"  # Pre-trained IndexTTS2 model
AMHARIC_VOCAB="amharic_bpe.model"  # Trained Amharic vocabulary
OUTPUT_DIR="checkpoints/amharic_200hr_enhanced"
TRAIN_MANIFEST="amharic_dataset/train_manifest.jsonl"
VAL_MANIFEST="amharic_dataset/val_manifest.jsonl"

# Training parameters optimized for 200hr dataset + T4 GPU
EPOCHS=6                    # Reduced to prevent overfitting
BATCH_SIZE=2               # T4 GPU optimized
LEARNING_RATE=5e-5         # Lower LR for stability
GRADIENT_ACCUMULATION=8    # Effective batch size: 16
WARMUP_STEPS=2000          # Increased for larger dataset
LORA_RANK=8                # T4 memory efficient
LORA_DROPOUT=0.1           # Anti-overfitting dropout

# Anti-overfitting specific settings
SAVE_EVERY=500             # Frequent checkpoints
LOG_EVERY=50               # Detailed monitoring
VAL_SPLIT=0.15             # More validation data

echo "📋 Configuration Summary:"
echo "   Dataset: 200 hours"
echo "   GPU: T4 16GB VRAM"
echo "   Epochs: $EPOCHS (anti-overfitting)"
echo "   Batch Size: $BATCH_SIZE + grad accumulation = 16 effective"
echo "   Learning Rate: $LEARNING_RATE"
echo "   LoRA Rank: $LORA_RANK (memory optimized)"
echo "   Validation Split: $VAL_SPLIT"

# Check prerequisites
echo ""
echo "🔍 Checking prerequisites..."

if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ Config file not found: $CONFIG_PATH"
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

echo "✅ All prerequisites met"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Memory optimization for T4 GPU
echo ""
echo "🧠 Optimizing T4 GPU memory settings..."
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=0

# Training command with anti-overfitting
echo ""
echo "🏃 Starting enhanced training..."
echo "💾 Checkpoints will be saved to: $OUTPUT_DIR"
echo "📊 Monitoring training every $LOG_EVERY steps"
echo "💾 Checkpoint saving every $SAVE_EVERY steps"
echo ""

# Run enhanced training with comprehensive monitoring
python scripts/enhanced_finetune_amharic.py \
    --config "$CONFIG_PATH" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --amharic_vocab "$AMHARIC_VOCAB" \
    --train_manifest "$TRAIN_MANIFEST" \
    --val_manifest "$VAL_MANIFEST" \
    --use_lora \
    --lora_rank $LORA_RANK \
    --lora_alpha 16.0 \
    --lora_dropout $LORA_DROPOUT \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --use_wandb

# Check training success
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Enhanced training completed successfully!"
    echo "📁 Best model saved at: $OUTPUT_DIR/enhanced_best_model.pt"
    echo "📁 All checkpoints in: $OUTPUT_DIR/"
    echo ""
    echo "🔍 Next steps:"
    echo "   1. Check training logs in: $OUTPUT_DIR/enhanced_training.log"
    echo "   2. Evaluate model: python scripts/evaluate_amharic.py --model_path $OUTPUT_DIR/enhanced_best_model.pt"
    echo "   3. Generate samples for quality assessment"
    
    # Show training summary
    if [ -f "$OUTPUT_DIR/enhanced_training.log" ]; then
        echo ""
        echo "📊 Training Summary:"
        tail -20 "$OUTPUT_DIR/enhanced_training.log"
    fi
else
    echo ""
    echo "❌ Training failed!"
    echo "🔍 Check logs in: $OUTPUT_DIR/enhanced_training.log"
    echo "💡 Common issues:"
    echo "   - Out of memory: Reduce batch size or enable more memory optimization"
    echo "   - Overfitting detected: Early stopping activated (good!)"
    echo "   - Low quality data: Check manifest files and audio quality"
    exit 1
fi