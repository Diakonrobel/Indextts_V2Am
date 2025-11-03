"""Checkpoint validation utilities for safe model loading"""
import torch
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class CheckpointValidator:
    """Validates checkpoint compatibility before loading"""
    
    @staticmethod
    def validate(
        checkpoint_path: str,
        expected_vocab_size: int,
        expected_normalizer: str = 'AmharicTextNormalizer',
        strict: bool = True
    ) -> Dict:
        """
        Validate checkpoint before loading
        
        Args:
            checkpoint_path: Path to checkpoint file
            expected_vocab_size: Expected vocabulary size
            expected_normalizer: Expected normalizer type
            strict: If True, raise errors; if False, only warn
            
        Returns:
            Loaded checkpoint if valid
            
        Raises:
            ValueError: If checkpoint is invalid (when strict=True)
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        except Exception as e:
            raise ValueError(f"Cannot load checkpoint from {checkpoint_path}: {e}")
        
        errors = []
        warnings_list = []
        
        # Check required fields
        if 'model_state_dict' not in checkpoint:
            errors.append("No 'model_state_dict' in checkpoint")
        
        # Check vocabulary size
        if 'vocab_size' in checkpoint:
            if checkpoint['vocab_size'] != expected_vocab_size:
                error_msg = (
                    f"Vocabulary size mismatch:\n"
                    f"  Checkpoint: {checkpoint['vocab_size']}\n"
                    f"  Expected: {expected_vocab_size}\n"
                    f"  This checkpoint is incompatible!"
                )
                errors.append(error_msg)
        else:
            warnings_list.append("No 'vocab_size' in checkpoint - cannot verify vocabulary")
        
        # Check normalizer
        if 'normalizer_config' in checkpoint:
            norm_type = checkpoint['normalizer_config'].get('type')
            if norm_type != expected_normalizer:
                error_msg = (
                    f"Normalizer type mismatch:\n"
                    f"  Checkpoint: {norm_type}\n"
                    f"  Expected: {expected_normalizer}"
                )
                if strict:
                    errors.append(error_msg)
                else:
                    warnings_list.append(error_msg)
        else:
            warnings_list.append("No 'normalizer_config' - cannot verify normalizer")
        
        # Check vocab file
        if 'vocab_file' in checkpoint:
            vocab_file = Path(checkpoint['vocab_file'])
            if not vocab_file.exists():
                warnings_list.append(f"Vocab file not found: {vocab_file}")
        else:
            warnings_list.append("No 'vocab_file' path saved in checkpoint")
        
        # Check model state dict structure
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            expected_keys = [
                'text_embedding.weight',
                'mel_embedding.weight',
                'text_head.weight'
            ]
            
            for key in expected_keys:
                if key not in state_dict:
                    warnings_list.append(f"Model missing expected layer: {key}")
            
            # Validate text embedding size
            if 'text_embedding.weight' in state_dict:
                text_emb_size = state_dict['text_embedding.weight'].shape[0]
                if text_emb_size != expected_vocab_size:
                    error_msg = (
                        f"Text embedding size mismatch:\n"
                        f"  Checkpoint: {text_emb_size}\n"
                        f"  Expected vocab size: {expected_vocab_size}"
                    )
                    errors.append(error_msg)
        
        # Print results
        if warnings_list:
            logger.warning("Checkpoint validation warnings:")
            for warn in warnings_list:
                logger.warning(f"  ⚠️  {warn}")
        
        if errors:
            error_msg = "Checkpoint validation failed:\n" + "\n".join(f"  ❌ {e}" for e in errors)
            if strict:
                raise ValueError(error_msg)
            else:
                logger.error(error_msg)
                logger.error("Continuing anyway (strict=False)")
        else:
            logger.info("✅ Checkpoint validation passed")
        
        return checkpoint
    
    @staticmethod
    def get_checkpoint_info(checkpoint_path: str) -> Dict:
        """Get checkpoint metadata without loading full state dict
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            Dictionary with checkpoint metadata
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            info = {
                'step': checkpoint.get('step'),
                'epoch': checkpoint.get('epoch'),
                'loss': checkpoint.get('loss'),
                'vocab_size': checkpoint.get('vocab_size'),
                'vocab_file': checkpoint.get('vocab_file'),
                'training_type': checkpoint.get('training_type'),
                'has_normalizer_config': 'normalizer_config' in checkpoint,
                'has_model_state': 'model_state_dict' in checkpoint
            }
            
            # Get model size
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                total_params = sum(p.numel() for p in state_dict.values())
                info['total_parameters'] = total_params
                info['model_size_mb'] = total_params * 4 / (1024 * 1024)  # Assuming float32
            
            return info
            
        except Exception as e:
            logger.error(f"Error reading checkpoint info: {e}")
            return {}


def main():
    parser = argparse.ArgumentParser(description="Amharic TTS Inference")
    parser.add_argument("--prompt_audio", required=True,
                       help="Reference audio file (speaker voice)")
    parser.add_argument("--text", required=True,
                       help="Amharic text to synthesize")
    parser.add_argument("--output", required=True,
                       help="Output audio path (WAV)")
    parser.add_argument("--amharic_vocab", required=True,
                       help="Path to Amharic SentencePiece model (.model file)")
    parser.add_argument("--model_dir", default="checkpoints",
                       help="Model checkpoint directory")
    parser.add_argument("--config", default=None,
                       help="Config YAML file (default: model_dir/config.yaml)")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed information")
    parser.add_argument("--device", default=None,
                       help="Device (cuda/cpu, default: auto)")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature (default: 0.8)")
    parser.add_argument("--top_p", type=float, default=0.8,
                       help="Nucleus sampling (default: 0.8)")
    parser.add_argument("--max_mel_tokens", type=int, default=600,
                       help="Max mel tokens to generate (default: 600)")
    
    args = parser.parse_args()
    
    # Determine config path
    if args.config is None:
        args.config = str(Path(args.model_dir) / "config.yaml")
    
    # Check files exist
    if not Path(args.prompt_audio).exists():
        print(f"❌ Reference audio not found: {args.prompt_audio}")
        return 1
    
    if not Path(args.amharic_vocab).exists():
        print(f"❌ Amharic vocabulary not found: {args.amharic_vocab}")
        return 1
    
    if not Path(args.config).exists():
        print(f"❌ Config file not found: {args.config}")
        return 1
    
    # Initialize Amharic TTS
    print("Initializing Amharic TTS...")
    try:
        tts = AmharicTTS(
            amharic_vocab_path=args.amharic_vocab,
            cfg_path=args.config,
            model_dir=args.model_dir,
            use_fp16=True,
            device=args.device
        )
    except Exception as e:
        print(f"❌ Failed to initialize TTS: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Generate speech
    print(f"\nGenerating speech...")
    print(f"Text: {args.text}")
    
    try:
        result = tts.infer(
            audio_prompt=args.prompt_audio,
            text=args.text,
            output_path=args.output,
            verbose=args.verbose,
            temperature=args.temperature,
            top_p=args.top_p,
            max_mel_tokens=args.max_mel_tokens
        )
        
        if args.output:
            print(f"\n✅ Generated audio saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
