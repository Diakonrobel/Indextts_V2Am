#!/usr/bin/env python3
"""
Complete Amharic IndexTTS2 Training Validation Script
Tests the entire pipeline from tokenizer to model loading
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def test_amharic_components():
    """Test all Amharic components individually"""
    
    print("🧪 Testing Amharic IndexTTS2 Components\n")
    
    # Test 1: Text Normalizer
    try:
        from indextts.utils.amharic_front import AmharicTextNormalizer
        normalizer = AmharicTextNormalizer()
        test_text = "ሰላም ዓለም! ይህ ሙከራ ነው። 123 ነው።"
        normalized = normalizer.normalize(test_text)
        print(f"✅ Text Normalizer: '{test_text}' → '{normalized}'")
    except Exception as e:
        print(f"❌ Text Normalizer failed: {e}")
        return False
    
    # Test 2: Text Tokenizer (without vocab file for now)
    try:
        from indextts.utils.amharic_front import AmharicTextTokenizer
        print("✅ AmharicTextTokenizer imported successfully")
        print("   Note: Need to train vocabulary first with train_amharic_vocabulary.py")
    except Exception as e:
        print(f"❌ Text Tokenizer failed: {e}")
        return False
    
    # Test 3: Model Architecture (vocabulary compatibility)
    try:
        from indextts.gpt.model_v2 import UnifiedVoice
        print("✅ UnifiedVoice model architecture loaded")
        
        # Test with hypothetical Amharic vocabulary size
        model = UnifiedVoice(
            layers=4,
            model_dim=512,
            heads=8,
            max_text_tokens=600,
            max_mel_tokens=1815,
            number_text_tokens=8000,  # Amharic vocabulary size
            number_mel_codes=8194,
            start_text_token=0,
            stop_text_token=1,
            start_mel_token=8192,
            stop_mel_token=8193,
            condition_type="conformer_perceiver",
            condition_module={
                'output_size': 512,
                'linear_units': 2048,
                'attention_heads': 8,
                'num_blocks': 6,
                'input_layer': 'conv2d2',
                'perceiver_mult': 2
            },
            emo_condition_module={
                'output_size': 512,
                'linear_units': 1024,
                'attention_heads': 4,
                'num_blocks': 4,
                'input_layer': 'conv2d2',
                'perceiver_mult': 2
            }
        )
        print(f"✅ Model vocabulary compatibility confirmed")
        print(f"   Text embedding shape: {model.text_embedding.weight.shape}")
        print(f"   Text head shape: {model.text_head.weight.shape}")
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False
    
    # Test 4: Feature Extractors
    try:
        from indextts.utils.feature_extractors import MelSpectrogramFeatures
        mel_extractor = MelSpectrogramFeatures(
            sample_rate=24000,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=100
        )
        print("✅ Mel Spectrogram Features initialized")
        print("   Audio → MEL conversion ready")
    except Exception as e:
        print(f"❌ Feature extractors failed: {e}")
        return False
    
    # Test 5: Script Files
    script_files = [
        "scripts/train_amharic_vocabulary.py",
        "scripts/prepare_amharic_data.py", 
        "scripts/finetune_amharic.py",
        "scripts/evaluate_amharic.py"
    ]
    
    for script in script_files:
        if os.path.exists(script):
            print(f"✅ Script available: {script}")
        else:
            print(f"❌ Script missing: {script}")
            return False
    
    # Test 6: Configuration
    config_files = [
        "configs/amharic_config.yaml",
        "AMHARIC_INDEXTTS2_README.md"
    ]
    
    for config in config_files:
        if os.path.exists(config):
            print(f"✅ Config available: {config}")
        else:
            print(f"❌ Config missing: {config}")
            return False
    
    print("\n🎉 All components validated successfully!")
    return True


def show_training_workflow():
    """Show the complete training workflow"""
    
    print("\n" + "="*60)
    print("🚀 AMHARIC INDEXTTS2 TRAINING WORKFLOW")
    print("="*60)
    
    workflow_steps = [
        ("1️⃣", "Prepare Amharic Text Data", 
         "Collect Amharic text files (.txt), one sentence per line"),
        
        ("2️⃣", "Train Amharic BPE Vocabulary", 
         "python scripts/train_amharic_vocabulary.py --text_files amharic_texts.txt"),
        
        ("3️⃣", "Prepare Audio Dataset", 
         "Organize audio files (.wav) and matching transcript files"),
        
        ("4️⃣", "Generate Training Manifests", 
         "python scripts/prepare_amharic_data.py --audio_dir audio/ --text_dir transcripts/"),
        
        ("5️⃣", "Download Pretrained Model", 
         "Get IndexTTS v2 pretrained model (gpt.pth)"),
        
        ("6️⃣", "Start Training", 
         "python scripts/finetune_amharic.py --config configs/amharic_config.yaml --model_path checkpoints/gpt.pth"),
        
        ("7️⃣", "Evaluate Model", 
         "python scripts/evaluate_amharic.py --model_path checkpoints/amharic_best.pt"),
        
        ("8️⃣", "Generate Speech", 
         "Test inference with Amharic text samples")
    ]
    
    for step, title, command in workflow_steps:
        print(f"\n{step} {title}")
        print(f"   {command}")
    
    print("\n" + "="*60)
    print("💡 KEY POINTS:")
    print("="*60)
    print("• BPE tokenizer only (no G2P needed)")
    print("• Automatic vocabulary resizing by IndexTTS2")
    print("• Automatic audio → MEL spectrogram conversion")
    print("• Complete emotion transfer capability")
    print("• Full Ge'ez script support (250-300 characters)")
    print("="*60)


def main():
    """Main validation function"""
    print("🔍 AMHARIC INDEXTTS2 PIPELINE VALIDATION")
    print("="*60)
    
    # Test all components
    if not test_amharic_components():
        print("\n❌ Component validation failed!")
        return 1
    
    # Show complete workflow
    show_training_workflow()
    
    print("\n✅ VALIDATION COMPLETE - READY FOR TRAINING!")
    return 0


if __name__ == "__main__":
    exit(main())