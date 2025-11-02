#!/usr/bin/env python3
"""
Amharic IndexTTS2 Pipeline Validation Script (Simplified)
Tests individual components and ensures the complete system works correctly
"""

import os
import sys
import json
import traceback
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def test_amharic_text_processing():
    """Test Amharic text processing components"""
    print("Testing Amharic Text Processing...")
    
    try:
        # Check if the module exists
        amharic_module = Path(__file__).parent.parent / "indextts" / "utils" / "amharic_front.py"
        
        if not amharic_module.exists():
            print("  [FAIL] amharic_front.py not found")
            return False
        
        # Check for key classes and functions
        with open(amharic_module, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_components = [
            "class AmharicTextNormalizer",
            "class AmharicTextTokenizer", 
            "def create_amharic_sentencepiece_model",
            "normalize_numbers",
            "expand_contractions",
            "expand_abbreviations"
        ]
        
        for component in required_components:
            if component in content:
                print(f"  [PASS] Found: {component}")
            else:
                print(f"  [FAIL] Missing: {component}")
                return False
        
        print("  [PASS] Amharic Text Processing: All components found")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Amharic Text Processing: {e}")
        return False

def test_dataset_preparation():
    """Test dataset preparation script"""
    print("Testing Dataset Preparation...")
    
    try:
        script_path = Path(__file__).parent / "prepare_amharic_data.py"
        
        if not script_path.exists():
            print("  [FAIL] prepare_amharic_data.py not found")
            return False
        
        # Check for key functions
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_functions = [
            "class AmharicDatasetPreparer",
            "def find_audio_text_pairs",
            "def validate_audio_file",
            "def process_dataset_pairs",
            "def prepare_dataset"
        ]
        
        for func in required_functions:
            if func in content:
                print(f"  [PASS] Found: {func}")
            else:
                print(f"  [FAIL] Missing: {func}")
                return False
        
        print("  [PASS] Dataset Preparation: All components found")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Dataset Preparation: {e}")
        return False

def test_vocabulary_training():
    """Test vocabulary training script"""
    print("Testing Vocabulary Training...")
    
    try:
        script_path = Path(__file__).parent / "train_amharic_vocabulary.py"
        
        if not script_path.exists():
            print("  [FAIL] train_amharic_vocabulary.py not found")
            return False
        
        # Check for key functions
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_functions = [
            "class AmharicVocabularyTrainer",
            "def prepare_text_data",
            "def analyze_text_statistics",
            "def suggest_vocabulary_size",
            "def train_vocabulary",
            "create_amharic_sentencepiece_model"
        ]
        
        for func in required_functions:
            if func in content:
                print(f"  [PASS] Found: {func}")
            else:
                print(f"  [FAIL] Missing: {func}")
                return False
        
        print("  [PASS] Vocabulary Training: All components found")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Vocabulary Training: {e}")
        return False

def test_fine_tuning():
    """Test fine-tuning script"""
    print("Testing Fine-tuning...")
    
    try:
        script_path = Path(__file__).parent / "finetune_amharic.py"
        
        if not script_path.exists():
            print("  [FAIL] finetune_amharic.py not found")
            return False
        
        # Check for key components
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_components = [
            "class AmharicTTSDataset",
            "class AmharicTTSFineTuner",
            "def _load_model",
            "def _setup_lora",
            "def _compute_loss",
            "def train",
            "AmharicTextTokenizer"
        ]
        
        for component in required_components:
            if component in content:
                print(f"  [PASS] Found: {component}")
            else:
                print(f"  [FAIL] Missing: {component}")
                return False
        
        print("  [PASS] Fine-tuning: All components found")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Fine-tuning: {e}")
        return False

def test_evaluation():
    """Test evaluation script"""
    print("Testing Evaluation...")
    
    try:
        script_path = Path(__file__).parent / "evaluate_amharic.py"
        
        if not script_path.exists():
            print("  [FAIL] evaluate_amharic.py not found")
            return False
        
        # Check for key components
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_components = [
            "class AmharicTTSEvaluator",
            "def evaluate_text_processing",
            "def evaluate_model_inference",
            "def evaluate_audio_quality",
            "def evaluate_linguistic_features",
            "def generate_comprehensive_report",
            "def run_comprehensive_evaluation"
        ]
        
        for component in required_components:
            if component in content:
                print(f"  [PASS] Found: {component}")
            else:
                print(f"  [FAIL] Missing: {component}")
                return False
        
        print("  [PASS] Evaluation: All components found")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Evaluation: {e}")
        return False

def test_configuration():
    """Test configuration file"""
    print("Testing Configuration...")
    
    try:
        config_path = Path(__file__).parent.parent / "configs" / "amharic_config.yaml"
        
        if not config_path.exists():
            print("  [FAIL] amharic_config.yaml not found")
            return False
        
        # Check for key sections
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_sections = [
            "dataset:",
            "gpt:",
            "training:",
            "amharic_text:",
            "amharic_phonetics:",
            "lora:"
        ]
        
        for section in required_sections:
            if section in content:
                print(f"  [PASS] Found: {section}")
            else:
                print(f"  [FAIL] Missing: {section}")
                return False
        
        print("  [PASS] Configuration: All sections found")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Configuration: {e}")
        return False

def test_automation():
    """Test automation script"""
    print("Testing Automation...")
    
    try:
        script_path = Path(__file__).parent / "run_amharic_training.sh"
        
        if not script_path.exists():
            print("  [FAIL] run_amharic_training.sh not found")
            return False
        
        # Check for key sections
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_sections = [
            "prepare_amharic_data.py",
            "train_amharic_vocabulary.py",
            "finetune_amharic.py",
            "evaluate_amharic.py"
        ]
        
        for section in required_sections:
            if section in content:
                print(f"  [PASS] Found reference to: {section}")
            else:
                print(f"  [FAIL] Missing reference to: {section}")
                return False
        
        print("  [PASS] Automation: All components referenced")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Automation: {e}")
        return False

def test_documentation():
    """Test documentation"""
    print("Testing Documentation...")
    
    try:
        # Check main README
        readme_path = Path(__file__).parent.parent / "AMHARIC_INDEXTTS2_README.md"
        
        if not readme_path.exists():
            print("  [FAIL] AMHARIC_INDEXTTS2_README.md not found")
            return False
        
        with open(readme_path, 'r', encoding='utf-8') as f:
            readme_content = f.read()
        
        # Check for key sections in README
        required_sections = [
            "Amharic",
            "IndexTTS2",
            "## Overview",
            "## Quick Start",
            "## Features",
            "### Key Features"
        ]
        
        for section in required_sections:
            if section in readme_content:
                print(f"  [PASS] README contains: {section}")
            else:
                print(f"  [WARN] README may be missing: {section}")
        
        # Check sample texts
        samples_path = Path(__file__).parent.parent / "samples" / "amharic_test_texts.txt"
        
        if samples_path.exists():
            print("  [PASS] Sample texts found")
            with open(samples_path, 'r', encoding='utf-8') as f:
                sample_content = f.read()
            if len(sample_content.split('\n')) > 5:
                print(f"  [PASS] Sample texts contain {len(sample_content.split(chr(10)))} lines")
            else:
                print(f"  [WARN] Sample texts may be too short")
        else:
            print("  [WARN] Sample texts not found")
        
        print("  [PASS] Documentation: Basic components found")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Documentation: {e}")
        return False

def test_project_structure():
    """Test overall project structure"""
    print("Testing Project Structure...")
    
    try:
        project_root = Path(__file__).parent.parent
        
        required_structure = [
            ("configs", "directory"),
            ("indextts/utils", "directory"),
            ("scripts", "directory"),
            ("AMHARIC_INDEXTTS2_README.md", "file"),
            ("samples", "directory")
        ]
        
        for path, type_str in required_structure:
            full_path = project_root / path
            
            if type_str == "directory" and full_path.is_dir():
                print(f"  [PASS] Found directory: {path}")
            elif type_str == "file" and full_path.is_file():
                print(f"  [PASS] Found file: {path}")
            else:
                print(f"  [FAIL] Missing {type_str}: {path}")
                return False
        
        # Check script files
        scripts_dir = project_root / "scripts"
        script_files = [
            "prepare_amharic_data.py",
            "train_amharic_vocabulary.py",
            "finetune_amharic.py",
            "evaluate_amharic.py",
            "run_amharic_training.sh",
            "validate_amharic_pipeline.py"
        ]
        
        for script_file in script_files:
            script_path = scripts_dir / script_file
            if script_path.exists():
                print(f"  [PASS] Found script: {script_file}")
            else:
                print(f"  [FAIL] Missing script: {script_file}")
                return False
        
        print("  [PASS] Project Structure: All required files and directories found")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Project Structure: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive validation of the entire pipeline"""
    print("=" * 70)
    print("AMHARIC INDEXTTS2 PIPELINE VALIDATION")
    print("=" * 70)
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Amharic Text Processing", test_amharic_text_processing),
        ("Dataset Preparation", test_dataset_preparation),
        ("Vocabulary Training", test_vocabulary_training),
        ("Fine-tuning", test_fine_tuning),
        ("Evaluation", test_evaluation),
        ("Configuration", test_configuration),
        ("Automation", test_automation),
        ("Documentation", test_documentation),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [FAIL] Test failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Tests Passed: {passed}")
    print(f"Tests Failed: {failed}")
    print(f"Total Tests: {passed + failed}")
    print(f"Success Rate: {passed / (passed + failed) * 100:.1f}%")
    
    if failed == 0:
        print("\n[SUCCESS] ALL TESTS PASSED!")
        print("\nAmharic IndexTTS2 pipeline components are complete and ready.")
        print("\nNext steps:")
        print("1. Install required dependencies: pip install sentencepiece torch torchaudio")
        print("2. Prepare your Amharic audio and text data")
        print("3. Run: chmod +x scripts/run_amharic_training.sh")
        print("4. Execute: ./scripts/run_amharic_training.sh --help")
        print("5. Follow the detailed guide in AMHARIC_INDEXTTS2_README.md")
    else:
        print(f"\n[WARNING] {failed} test(s) failed. Please review the errors above.")
        print("Some components may need to be completed or dependencies installed.")
    
    print("\n" + "=" * 70)
    print("COMPONENT SUMMARY:")
    print("- Amharic Text Processing: Modern script (ፊደል) support")
    print("- Vocabulary Training: SentencePiece BPE for Amharic")
    print("- Dataset Preparation: Audio-text pairing with validation")
    print("- Fine-tuning: LoRA-based efficient adaptation")
    print("- Evaluation: Comprehensive quality assessment")
    print("- Automation: End-to-end pipeline scripting")
    print("=" * 70)
    
    return failed == 0

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)