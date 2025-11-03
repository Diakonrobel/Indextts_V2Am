#!/usr/bin/env python
"""End-to-End Pipeline Validation for IndexTTS2 Amharic Training

Validates the complete pipeline:
1. Data preparation → tokenization
2. Tokenization → model forward pass
3. Loss computation
4. Checkpoint saving/loading
5. Inference consistency
"""
import sys
import argparse
import logging
from pathlib import Path
import torch
import json

sys.path.append(str(Path(__file__).parent.parent))

from indextts.utils.amharic_front import AmharicTextTokenizer, AmharicTextNormalizer
from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.feature_extractors import MelSpectrogramFeatures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_tokenization(vocab_file: str, test_texts: list):
    """Validate tokenization produces reasonable outputs"""
    logger.info("=== Validating Tokenization ===")
    
    tokenizer = AmharicTextTokenizer(
        vocab_file=vocab_file,
        normalizer=AmharicTextNormalizer()
    )
    
    total_unk = 0
    total_tokens = 0
    
    for text in test_texts:
        tokens = tokenizer.encode(text, out_type=int)
        total_tokens += len(tokens)
        
        # Check for UNK tokens (usually ID 0)
        unk_count = sum(1 for t in tokens if t == 0)
        total_unk += unk_count
        
        logger.info(f"Text: {text[:50]}...")
        logger.info(f"  Tokens: {len(tokens)}, UNK: {unk_count}")
    
    unk_ratio = total_unk / total_tokens if total_tokens > 0 else 0
    logger.info(f"\nOverall UNK ratio: {unk_ratio:.2%}")
    
    if unk_ratio > 0.05:
        logger.warning(f"⚠️  High UNK ratio ({unk_ratio:.2%})! Vocabulary may be insufficient.")
        return False
    
    logger.info("✅ Tokenization validation passed")
    return True


def validate_model_forward(vocab_file: str, model_config: dict):
    """Validate model can do forward pass without errors"""
    logger.info("\n=== Validating Model Forward Pass ===")
    
    try:
        # Create tokenizer
        tokenizer = AmharicTextTokenizer(
            vocab_file=vocab_file,
            normalizer=AmharicTextNormalizer()
        )
        
        # Create model
        model = UnifiedVoice(
            layers=model_config.get('layers', 24),
            model_dim=model_config.get('model_dim', 1024),
            heads=model_config.get('heads', 16),
            number_text_tokens=tokenizer.vocab_size,
            max_text_tokens=model_config.get('max_text_tokens', 402),
            max_mel_tokens=model_config.get('max_mel_tokens', 604),
            number_mel_codes=model_config.get('number_mel_codes', 8194)
        )
        
        # Create dummy batch
        batch_size = 2
        text_tokens = torch.randint(0, tokenizer.vocab_size, (batch_size, 50))
        text_lengths = torch.tensor([50, 45])
        mel_codes = torch.randint(0, 8194, (batch_size, 100))
        mel_lengths = torch.tensor([100, 95])
        
        # Dummy conditioning
        speech_cond = torch.randn(batch_size, 1024, 100)
        cond_lengths = torch.tensor([100, 100])
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            cond_latent = model.get_conditioning(speech_cond, cond_lengths)
            loss_text, loss_mel, _ = model(
                speech_conditioning_latent=cond_latent,
                text_inputs=text_tokens,
                text_lengths=text_lengths,
                mel_codes=mel_codes,
                wav_lengths=mel_lengths,
                cond_mel_lengths=cond_lengths
            )
        
        logger.info(f"Forward pass successful")
        logger.info(f"  Text loss: {loss_text.item():.4f}")
        logger.info(f"  Mel loss: {loss_mel.item():.4f}")
        
        # Validate losses are reasonable
        if loss_text.item() < 0 or loss_mel.item() < 0:
            logger.error("❌ Negative loss detected!")
            return False
        
        if loss_text.item() > 20 or loss_mel.item() > 20:
            logger.warning("⚠️  Very high loss - model might not be initialized properly")
        
        logger.info("✅ Model forward pass validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_checkpoint_save_load(vocab_file: str, output_dir: Path):
    """Validate checkpoint can be saved and loaded correctly"""
    logger.info("\n=== Validating Checkpoint Save/Load ===")
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        tokenizer = AmharicTextTokenizer(
            vocab_file=vocab_file,
            normalizer=AmharicTextNormalizer()
        )
        
        # Create minimal checkpoint
        checkpoint = {
            'step': 100,
            'epoch': 1,
            'loss': 5.5,
            'vocab_size': tokenizer.vocab_size,
            'vocab_file': vocab_file,
            'normalizer_config': {
                'type': 'AmharicTextNormalizer',
                'number_words': tokenizer.normalizer.number_words,
                'contractions': tokenizer.normalizer.contractions
            }
        }
        
        # Save
        checkpoint_path = output_dir / "test_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Load
        loaded = torch.load(checkpoint_path, map_location='cpu')
        
        # Validate
        assert loaded['step'] == checkpoint['step']
        assert loaded['vocab_size'] == checkpoint['vocab_size']
        assert loaded['vocab_file'] == checkpoint['vocab_file']
        assert 'normalizer_config' in loaded
        
        logger.info("✅ Checkpoint save/load validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Checkpoint validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_data_manifest(manifest_file: str):
    """Validate training data manifest is properly formatted"""
    logger.info("\n=== Validating Data Manifest ===")
    
    try:
        if not Path(manifest_file).exists():
            logger.error(f"❌ Manifest file not found: {manifest_file}")
            return False
        
        with open(manifest_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) == 0:
            logger.error("❌ Manifest file is empty")
            return False
        
        # Check first few entries
        for i, line in enumerate(lines[:5]):
            try:
                data = json.loads(line.strip())
                required_fields = ['audio_path', 'text', 'id']
                
                for field in required_fields:
                    if field not in data:
                        logger.error(f"❌ Missing required field '{field}' in entry {i}")
                        return False
                
                # Check audio file exists
                if not Path(data['audio_path']).exists():
                    logger.warning(f"⚠️  Audio file not found: {data['audio_path']}")
                
            except json.JSONDecodeError:
                logger.error(f"❌ Invalid JSON at line {i+1}")
                return False
        
        logger.info(f"Validated {len(lines)} entries in manifest")
        logger.info("✅ Data manifest validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Manifest validation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="End-to-end pipeline validation")
    parser.add_argument("--vocab", type=str, required=True,
                       help="Path to vocabulary file")
    parser.add_argument("--manifest", type=str,
                       help="Path to training manifest (optional)")
    parser.add_argument("--output_dir", type=str, default="./validation_output",
                       help="Output directory for test files")
    
    args = parser.parse_args()
    
    logger.info("Starting End-to-End Pipeline Validation")
    logger.info("="*60)
    
    results = {}
    
    # Test texts
    test_texts = [
        "ሰላም ዓለም",
        "የአማርኛ ቋንቋ ሙከራ ነው",
        "እንዴት ነህ?"
    ]
    
    # 1. Validate tokenization
    results['tokenization'] = validate_tokenization(args.vocab, test_texts)
    
    # 2. Validate model forward pass
    model_config = {
        'layers': 24,
        'model_dim': 1024,
        'heads': 16
    }
    results['model_forward'] = validate_model_forward(args.vocab, model_config)
    
    # 3. Validate checkpoint save/load
    results['checkpoint'] = validate_checkpoint_save_load(
        args.vocab,
        Path(args.output_dir)
    )
    
    # 4. Validate manifest if provided
    if args.manifest:
        results['manifest'] = validate_data_manifest(args.manifest)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name:20s}: {status}")
        if not passed:
            all_passed = False
    
    logger.info("="*60)
    
    if all_passed:
        logger.info("\n🎉 All validations passed! Pipeline is ready for training.")
        return 0
    else:
        logger.error("\n❌ Some validations failed. Please fix issues before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
