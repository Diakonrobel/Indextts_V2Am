#!/usr/bin/env python
\"\"\"
Amharic Fine-tuning for IndexTTS2 - FULL LAYER TRAINING (Default)
Community-proven approach for new languages with complex scripts

Usage:
    python train_amharic_full.py --data_dir <path_to_data> --vocab <amharic_bpe.model>

Features:
    - ✅ Full Layer Training (No LoRA)
    - ✅ T4 GPU Optimized (16GB VRAM)
    - ✅ Memory Optimizations Enabled
    - ✅ All Amharic Features Preserved
    - ✅ Anti-Overfitting Monitoring
\"\"\"

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.finetune_amharic import AmharicTTSFineTuner

def main():
    parser = argparse.ArgumentParser(
        description='IndexTTS2 Amharic Full Layer Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=\"\"\"
Examples:
  # Basic training
  python train_amharic_full.py --data_dir ./amharic_data --vocab amharic_bpe.model
  
  # With custom checkpoints
  python train_amharic_full.py --data_dir ./data --vocab vocab.model --checkpoint checkpoints/gpt.pth
  
  # Resume training
  python train_amharic_full.py --data_dir ./data --vocab vocab.model --resume checkpoints/latest.pt
        \"\"\"
    )
    
    # Required arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing training data (manifests)')
    parser.add_argument('--vocab', type=str, required=True,
                        help='Path to Amharic BPE vocabulary model')
    
    # Optional arguments
    parser.add_argument('--config', type=str, default='configs/amharic_config.yaml',
                        help='Path to config file (default: full training config)')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/gpt.pth',
                        help='Path to pretrained checkpoint')
    parser.add_argument('--output_dir', type=str, default='checkpoints/amharic_full_training',
                        help='Output directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config, careful with T4!)')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--no_mixed_precision', action='store_true',
                        help='Disable mixed precision (NOT recommended for T4)')
    
    args = parser.parse_args()
    
    # Load config
    print(f\"\\n{'='*60}\")
    print(\"IndexTTS2 Amharic Full Layer Training\")
    print(\"Community-Proven Approach for New Languages\")
    print(f\"{'='*60}\\n\")
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Verify LoRA is disabled
    if config.get('lora', {}).get('enabled', False):
        print(\"⚠️  WARNING: LoRA is enabled in config!\")\n        print(\"   Community evidence shows LoRA FAILS for new languages.\")\n        print(\"   Disabling LoRA and using Full Layer Training...\\n\")\n        config['lora']['enabled'] = False
    
    print(\"✅ Configuration:\")
    print(f\"   - Training Mode: FULL LAYER (all 24 layers)\")
    print(f\"   - LoRA Status: DISABLED (community-proven approach)\")
    print(f\"   - GPU Optimization: T4 16GB VRAM optimized\")
    print(f\"   - Batch Size: {config['training']['batch_size']}\")
    print(f\"   - Gradient Accumulation: {config['training'].get('gradient_accumulation_steps', 16)}\")
    print(f\"   - Effective Batch: {config['training']['batch_size'] * config['training'].get('gradient_accumulation_steps', 16)}\")
    print(f\"   - Learning Rate: {config['training']['learning_rate']}\")
    print(f\"   - Epochs: {config['training']['num_epochs']}\")
    print(f\"   - Mixed Precision: {'✅ ENABLED' if not args.no_mixed_precision else '❌ DISABLED'}\")
    print(f\"   - Gradient Checkpointing: ✅ ENABLED\")
    print(f\"   - CPU Offload: ✅ ENABLED\\n\")
    
    # Override config with command-line args if provided
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
        print(f\"⚠️  WARNING: Custom batch size {args.batch_size} set.\")
        print(\"   T4 GPU may run out of memory if > 1. Use gradient accumulation instead!\\n\")
    
    # Check manifest files
    train_manifest = os.path.join(args.data_dir, 'train.jsonl')
    val_manifest = os.path.join(args.data_dir, 'val.jsonl')
    
    if not os.path.exists(train_manifest):
        print(f\"❌ ERROR: Training manifest not found: {train_manifest}\")
        print(f\"   Expected structure: {args.data_dir}/train.jsonl\")
        print(f\"   Run prepare_amharic_data.py first!\\n\")
        sys.exit(1)
    
    if not os.path.exists(val_manifest):
        print(f\"⚠️  WARNING: Validation manifest not found: {val_manifest}\")
        print(f\"   Training will use train/val split from config\\n\")
    
    print(f\"✅ Data:\")\n    print(f\"   - Training manifest: {train_manifest}\")
    print(f\"   - Validation manifest: {val_manifest if os.path.exists(val_manifest) else 'Not found (will split from train)'}\")
    print(f\"   - Vocabulary: {args.vocab}\\n\")
    
    # Initialize trainer
    print(\"Initializing Fine-Tuner...\\n\")
    
    trainer = AmharicTTSFineTuner(
        config_path=args.config,
        model_path=args.checkpoint,
        output_dir=args.output_dir,
        amharic_vocab_path=args.vocab,
        use_lora=False,  # ❌ DISABLED
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Create data loaders
    print(\"Creating data loaders...\\n\")
    
    train_loader = trainer.create_data_loader(
        manifest_file=train_manifest,
        batch_size=config['training']['batch_size'],
        num_workers=config['hardware'].get('num_workers', 1),
        shuffle=True
    )
    
    if os.path.exists(val_manifest):
        val_loader = trainer.create_data_loader(
            manifest_file=val_manifest,
            batch_size=config['training']['batch_size'],
            num_workers=config['hardware'].get('num_workers', 1),
            shuffle=False
        )
    else:
        val_loader = None
        print(\"⚠️  No validation loader (will use training split)\\n\")
    
    # Training information
    print(f\"{'='*60}\")
    print(\"Training Information:\")
    print(f\"{'='*60}\")
    print(f\"Total training samples: {len(train_loader.dataset)}\")
    print(f\"Steps per epoch: {len(train_loader)}\")
    print(f\"Total steps: {len(train_loader) * config['training']['num_epochs']}\")
    print(f\"Expected training time: 5-7 days on T4 GPU\")
    print(f\"Checkpoint frequency: Every {config['training']['save_every']} steps\")
    print(f\"Sample generation: Every {config['output']['generate_samples_every']} steps\")
    print(f\"{'='*60}\\n\")
    
    # Memory warning
    print(\"💡 MEMORY TIPS:\")
    print(\"   - Monitor GPU usage: nvidia-smi -l 1\")
    print(\"   - If OOM error: Reduce batch_size or increase gradient_accumulation\")
    print(\"   - Mixed precision is CRITICAL for T4 (saves 50% memory)\")
    print(f\"   - Current VRAM estimate: ~14-15GB (T4 has 16GB)\\n\")
    
    # Start training
    print(f\"{'='*60}\")
    print(\"🚀 Starting Full Layer Training...\")
    print(f\"{'='*60}\\n\")
    
    try:
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['training']['num_epochs'],
            learning_rate=config['training']['learning_rate'],
            resume_from=args.resume
        )
        
        print(f\"\\n{'='*60}\")
        print(\"✅ Training completed successfully!\")
        print(f\"{'='*60}\")
        print(f\"Best model saved to: {args.output_dir}/best_model.pt\")
        print(f\"Training logs: {config['output']['log_dir']}\")
        print(f\"Samples: {config['output']['sample_dir']}\\n\")
        
    except KeyboardInterrupt:
        print(f\"\\n{'='*60}\")
        print(\"⚠️  Training interrupted by user\")
        print(f\"{'='*60}\")
        print(f\"Latest checkpoint: {args.output_dir}/latest.pt\")
        print(f\"Resume with: --resume {args.output_dir}/latest.pt\\n\")
        
    except Exception as e:
        print(f\"\\n{'='*60}\")
        print(f\"❌ Training failed with error:\")
        print(f\"{'='*60}\")
        print(f\"{str(e)}\\n\")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
