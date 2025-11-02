#!/bin/bash

# Amharic IndexTTS2 Complete Training Pipeline Script
# This script orchestrates the entire Amharic fine-tuning process

set -e

echo "=========================================="
echo "Amharic IndexTTS2 Fine-tuning Pipeline"
echo "=========================================="

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIGS_DIR="$PROJECT_ROOT/configs"
MODELS_DIR="$PROJECT_ROOT/amharic_models"
DATA_DIR="$PROJECT_ROOT/amharic_dataset"
OUTPUT_DIR="$PROJECT_ROOT/checkpoints/amharic"
LOGS_DIR="$PROJECT_ROOT/logs/amharic"

# Create necessary directories
mkdir -p "$MODELS_DIR" "$DATA_DIR" "$OUTPUT_DIR" "$LOGS_DIR"

# Default parameters
BATCH_SIZE=4
NUM_EPOCHS=15
LEARNING_RATE=5e-5
LORA_RANK=16
LORA_ALPHA=16.0
USE_LORA=true
USE_WANDB=false

# Help function
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --audio-dir DIR           Directory containing Amharic audio files (required)"
    echo "  --text-dir DIR            Directory containing Amharic text files (required)"
    echo "  --pretrained-model PATH   Path to pre-trained IndexTTS2 model (required)"
    echo "  --config FILE             Configuration file (default: configs/amharic_config.yaml)"
    echo "  --batch-size SIZE         Training batch size (default: 4)"
    echo "  --epochs NUM              Number of training epochs (default: 15)"
    echo "  --lr RATE                Learning rate (default: 5e-5)"
    echo "  --lora-rank RANK          LoRA rank (default: 16)"
    echo "  --lora-alpha ALPHA        LoRA alpha (default: 16.0)"
    echo "  --no-lora                 Disable LoRA adapters"
    echo "  --wandb                   Enable Weights & Biases logging"
    echo "  --help                    Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --audio-dir ./amharic_audio --text-dir ./amharic_text --pretrained-model ./checkpoints/gpt.pth"
}

# Parse arguments
AUDIO_DIR=""
TEXT_DIR=""
PRETRAINED_MODEL=""
CONFIG_FILE="$CONFIGS_DIR/amharic_config.yaml"

while [[ $# -gt 0 ]]; do
    case $1 in
        --audio-dir)
            AUDIO_DIR="$2"
            shift 2
            ;;
        --text-dir)
            TEXT_DIR="$2"
            shift 2
            ;;
        --pretrained-model)
            PRETRAINED_MODEL="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --lora-rank)
            LORA_RANK="$2"
            shift 2
            ;;
        --lora-alpha)
            LORA_ALPHA="$2"
            shift 2
            ;;
        --no-lora)
            USE_LORA=false
            shift
            ;;
        --wandb)
            USE_WANDB=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "$AUDIO_DIR" ]]; then
    echo "Error: --audio-dir is required"
    show_help
    exit 1
fi

if [[ -z "$TEXT_DIR" ]]; then
    echo "Error: --text-dir is required"
    show_help
    exit 1
fi

if [[ -z "$PRETRAINED_MODEL" ]]; then
    echo "Error: --pretrained-model is required"
    show_help
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Verify input directories exist
if [[ ! -d "$AUDIO_DIR" ]]; then
    echo "Error: Audio directory not found: $AUDIO_DIR"
    exit 1
fi

if [[ ! -d "$TEXT_DIR" ]]; then
    echo "Error: Text directory not found: $TEXT_DIR"
    exit 1
fi

if [[ ! -f "$PRETRAINED_MODEL" ]]; then
    echo "Error: Pre-trained model not found: $PRETRAINED_MODEL"
    exit 1
fi

echo "Configuration:"
echo "  Audio directory: $AUDIO_DIR"
echo "  Text directory: $TEXT_DIR"
echo "  Pre-trained model: $PRETRAINED_MODEL"
echo "  Config file: $CONFIG_FILE"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Learning rate: $LEARNING_RATE"
echo "  LoRA enabled: $USE_LORA"
echo "  LoRA rank: $LORA_RANK"
echo "  LoRA alpha: $LORA_ALPHA"
echo "  Weights & Biases: $USE_WANDB"
echo ""

# Step 1: Prepare Amharic Dataset
echo "Step 1: Preparing Amharic dataset..."
python scripts/prepare_amharic_data.py \
    --audio_dir "$AUDIO_DIR" \
    --text_dir "$TEXT_DIR" \
    --output_dir "$DATA_DIR" \
    --min_duration 1.5 \
    --max_duration 25.0 \
    --min_text_length 8 \
    --max_text_length 400 \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --random_seed 42

if [[ $? -ne 0 ]]; then
    echo "Error: Dataset preparation failed"
    exit 1
fi

TRAIN_MANIFEST="$DATA_DIR/train_manifest.jsonl"
VAL_MANIFEST="$DATA_DIR/val_manifest.jsonl"

if [[ ! -f "$TRAIN_MANIFEST" ]] || [[ ! -f "$VAL_MANIFEST" ]]; then
    echo "Error: Manifest files not created"
    exit 1
fi

echo "Dataset preparation completed successfully"
echo ""

# Step 2: Create Amharic Vocabulary
echo "Step 2: Creating Amharic vocabulary..."

# Collect all text files for vocabulary training
TEXT_FILES=()
while IFS= read -r -d '' file; do
    TEXT_FILES+=("$file")
done < <(find "$TEXT_DIR" -type f \( -name "*.txt" -o -name "*.json" -o -name "*.lrc" \) -print0)

if [[ ${#TEXT_FILES[@]} -eq 0 ]]; then
    echo "Error: No text files found in $TEXT_DIR"
    exit 1
fi

# Train vocabulary
python scripts/train_amharic_vocabulary.py \
    --text_files "${TEXT_FILES[@]}" \
    --output_dir "$MODELS_DIR" \
    --model_prefix "amharic_bpe" \
    --character_coverage 0.9999

if [[ $? -ne 0 ]]; then
    echo "Error: Vocabulary training failed"
    exit 1
fi

AMHARIC_VOCAB="$MODELS_DIR/amharic_bpe.model"

if [[ ! -f "$AMHARIC_VOCAB" ]]; then
    echo "Error: Amharic vocabulary model not created"
    exit 1
fi

echo "Amharic vocabulary creation completed successfully"
echo ""

# Step 3: Fine-tune Model
echo "Step 3: Starting Amharic fine-tuning..."

# Set up LoRA parameters
LORA_PARAMS=""
if [[ "$USE_LORA" == "true" ]]; then
    LORA_PARAMS="--use_lora --lora_rank $LORA_RANK --lora_alpha $LORA_ALPHA"
fi

# Set up WandB
WANDB_PARAMS=""
if [[ "$USE_WANDB" == "true" ]]; then
    WANDB_PARAMS="--use_wandb"
fi

# Start training
python scripts/finetune_amharic.py \
    --config "$CONFIG_FILE" \
    --model_path "$PRETRAINED_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --amharic_vocab "$AMHARIC_VOCAB" \
    --train_manifest "$TRAIN_MANIFEST" \
    --val_manifest "$VAL_MANIFEST" \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    $LORA_PARAMS \
    $WANDB_PARAMS

if [[ $? -ne 0 ]]; then
    echo "Error: Model fine-tuning failed"
    exit 1
fi

# Check if model was saved
BEST_MODEL="$OUTPUT_DIR/best_amharic_model.pt"
if [[ ! -f "$BEST_MODEL" ]]; then
    echo "Error: Fine-tuned model not saved"
    exit 1
fi

echo "Amharic fine-tuning completed successfully"
echo ""

# Step 4: Generate Sample Outputs
echo "Step 4: Generating sample outputs..."

# Create sample directory
SAMPLES_DIR="$OUTPUT_DIR/samples"
mkdir -p "$SAMPLES_DIR"

# Generate evaluation report
python scripts/evaluate_amharic.py \
    --config "$CONFIG_FILE" \
    --model_path "$BEST_MODEL" \
    --amharic_vocab "$AMHARIC_VOCAB" \
    --output_dir "$OUTPUT_DIR/evaluation" \
    --test_texts_file "$DATA_DIR/train_manifest.jsonl"

echo "Sample outputs generated successfully"
echo ""

# Final Summary
echo "=========================================="
echo "Amharic IndexTTS2 Training Complete!"
echo "=========================================="
echo ""
echo "Output locations:"
echo "  Dataset manifests: $DATA_DIR"
echo "  Vocabulary model: $AMHARIC_VOCAB"
echo "  Fine-tuned model: $BEST_MODEL"
echo "  Training outputs: $OUTPUT_DIR"
echo "  Evaluation report: $OUTPUT_DIR/evaluation"
echo ""
echo "Next steps:"
echo "  1. Review the evaluation report in $OUTPUT_DIR/evaluation"
echo "  2. Test the fine-tuned model with new Amharic text"
echo "  3. Adjust hyperparameters if needed and retrain"
echo ""

if [[ "$USE_WANDB" == "true" ]]; then
    echo "  4. Check training logs in Weights & Biases dashboard"
fi

echo ""
echo "Training pipeline completed successfully! 🎉"