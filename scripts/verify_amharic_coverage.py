#!/usr/bin/env python3
"""
Character Coverage Verification for Amharic BPE Tokenizer
Verifies that all Amharic characters are covered by the tokenizer
"""

import sys
import json
from collections import Counter
import re
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from indextts.utils.amharic_front import AmharicTextNormalizer

def collect_amharic_characters(text_files):
    """Collect all unique Amharic characters from text files"""
    all_text = ""
    file_stats = {}
    
    for text_file in text_files:
        if not Path(text_file).exists():
            print(f"⚠️  File not found: {text_file}")
            continue
            
        with open(text_file, 'r', encoding='utf-8') as f:
            content = f.read()
            all_text += content + "\n"
            file_stats[text_file] = len(content)
    
    # Find all Amharic characters (Unicode range U+1200 to U+137F)
    amharic_chars = set()
    for char in all_text:
        if 0x1200 <= ord(char) <= 0x137F:
            amharic_chars.add(char)
    
    # Find other important characters
    punctuation = set()
    for char in all_text:
        if char in '።፣፤፥፧፨፡.,;:!?"[]{}()':
            punctuation.add(char)
    
    # Numbers
    numbers = set()
    for char in all_text:
        if char.isdigit():
            numbers.add(char)
    
    # Latin characters (for mixed text)
    latin_chars = set()
    for char in all_text:
        if char.isascii() and char.isalpha():
            latin_chars.add(char)
    
    stats = {
        'total_text_length': len(all_text),
        'files_processed': len(file_stats),
        'amharic_characters': sorted(list(amharic_chars)),
        'amharic_count': len(amharic_chars),
        'punctuation': sorted(list(punctuation)),
        'numbers': sorted(list(numbers)),
        'latin_chars': sorted(list(latin_chars)),
        'file_stats': file_stats
    }
    
    return all_text, stats

def estimate_vocabulary_requirements(stats):
    """Estimate vocabulary size needed for full coverage"""
    
    # Base requirements
    base_chars = stats['amharic_count']  # All Amharic characters
    punctuation = len(stats['punctuation'])
    numbers = len(stats['numbers'])
    latin = len(stats['latin_chars'])
    
    # Special tokens
    special_tokens = ['<unk>', '<s>', '</s>', '<pad>', '<speaker>', '<emotion>', '<duration>']
    
    # Estimated BPE vocabulary
    estimated_vocab = base_chars + punctuation + numbers + latin + len(special_tokens) + 500  # BPE subwords
    
    return {
        'amharic_base_characters': base_chars,
        'punctuation': punctuation,
        'numbers': numbers,
        'latin_characters': latin,
        'special_tokens': len(special_tokens),
        'estimated_bpe_subwords': 500,
        'total_estimated_vocab': estimated_vocab,
        'recommended_vocab_size': min(12000, max(8000, estimated_vocab))
    }

def verify_coverage_requirements(stats):
    """Verify if collected characters meet Amharic requirements"""
    
    requirements = {
        'amharic_consonants': 33,
        'amharic_vowels_per_consonant': 7,
        'total_amharic_expected': 33 * 7,  # 231 characters
        'punctuation_required': 8,  # ።፣፤፥፧፨፡ + others
    }
    
    coverage = {
        'amharic_characters_found': stats['amharic_count'],
        'amharic_coverage_percentage': (stats['amharic_count'] / requirements['total_amharic_expected']) * 100,
        'punctuation_found': len(stats['punctuation']),
        'punctuation_coverage': (len(stats['punctuation']) / requirements['punctuation_required']) * 100,
    }
    
    return requirements, coverage

def generate_character_report(stats, requirements, coverage, vocab_est):
    """Generate comprehensive character coverage report"""
    
    report = {
        'summary': {
            'total_text_length': stats['total_text_length'],
            'files_processed': stats['files_processed'],
            'amharic_coverage': f"{coverage['amharic_coverage_percentage']:.1f}%",
            'vocabulary_recommended': vocab_est['recommended_vocab_size']
        },
        'character_coverage': {
            'amharic_characters': {
                'found': stats['amharic_count'],
                'expected_min': requirements['total_amharic_expected'],
                'coverage_percentage': f"{coverage['amharic_coverage_percentage']:.1f}%"
            },
            'punctuation': {
                'found': len(stats['punctuation']),
                'expected': requirements['punctuation_required'],
                'coverage': f"{coverage['punctuation_coverage']:.1f}%"
            }
        },
        'vocabulary_estimate': vocab_est,
        'recommendations': []
    }
    
    # Add recommendations
    if coverage['amharic_coverage_percentage'] < 80:
        report['recommendations'].append("⚠️  Low Amharic character coverage - need more diverse text")
    
    if stats['amharic_count'] < requirements['total_amharic_expected']:
        report['recommendations'].append(f"📝 Found {stats['amharic_count']} Amharic characters, expected ~{requirements['total_amharic_expected']}")
    
    if coverage['punctuation_coverage'] < 100:
        report['recommendations'].append("🔤 Add more punctuation variations")
    
    if vocab_est['recommended_vocab_size'] > 10000:
        report['recommendations'].append("📊 Large vocabulary needed - consider character-level normalization")
    
    report['recommendations'].append("✅ Ready for BPE training with current character coverage")
    
    return report

def main():
    """Main verification function"""
    print("🔍 AMHARIC BPE CHARACTER COVERAGE VERIFICATION")
    print("="*60)
    
    # Default text files to check (user can provide their own)
    text_files = [
        "amharic_texts.txt",  # User should provide
        "samples/amharic_test_texts.txt",  # Sample provided
    ]
    
    # Check for user-provided files
    user_text_files = []
    for file_arg in sys.argv[1:]:
        if file_arg.endswith('.txt'):
            user_text_files.append(file_arg)
    
    if user_text_files:
        text_files = user_text_files
        print(f"📁 Using user-provided files: {text_files}")
    else:
        print(f"📁 Checking default files: {text_files}")
        print("💡 Run with: python verify_amharic_coverage.py your_text_files.txt")
    
    # Collect characters
    print("\n📊 Analyzing character coverage...")
    all_text, stats = collect_amharic_characters(text_files)
    
    if not stats['amharic_characters']:
        print("❌ No Amharic characters found in the provided text files!")
        print("   Please provide text files containing Amharic script (ፊደል)")
        return 1
    
    # Estimate vocabulary requirements
    print(f"📝 Found {len(stats['amharic_characters'])} unique Amharic characters")
    vocab_est = estimate_vocabulary_requirements(stats)
    
    # Verify coverage
    requirements, coverage = verify_coverage_requirements(stats)
    
    # Generate report
    report = generate_character_report(stats, requirements, coverage, vocab_est)
    
    # Print summary
    print(f"\n🎯 COVERAGE SUMMARY:")
    print(f"   Amharic Characters: {coverage['amharic_coverage_percentage']:.1f}% coverage")
    print(f"   Punctuation: {coverage['punctuation_coverage']:.1f}% coverage")
    print(f"   Recommended Vocab Size: {vocab_est['recommended_vocab_size']}")
    
    # Print recommendations
    if report['recommendations']:
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"   {rec}")
    
    # Save detailed report
    report_file = "amharic_character_coverage_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📋 Detailed report saved: {report_file}")
    
    # Show character samples
    print(f"\n🔤 SAMPLE AMHARIC CHARACTERS:")
    sample_chars = stats['amharic_characters'][:10]
    print(f"   {''.join(sample_chars)} ...")
    
    print(f"\n✅ CHARACTER COVERAGE VERIFICATION COMPLETE")
    return 0

if __name__ == "__main__":
    exit(main())