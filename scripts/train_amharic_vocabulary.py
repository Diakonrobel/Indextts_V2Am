"""
Amharic vocabulary training script for SentencePiece BPE model
Creates optimized vocabulary for Amharic IndexTTS2 fine-tuning
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict
import json
import re
from collections import Counter
import sentencepiece as spm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from indextts.utils.amharic_front import create_amharic_sentencepiece_model, AmharicTextNormalizer


class AmharicVocabularyTrainer:
    """Train and optimize Amharic vocabulary for IndexTTS2"""
    
    def __init__(self, output_dir: str = "amharic_models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.normalizer = AmharicTextNormalizer()
        self.logger = self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'vocabulary_training.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def prepare_text_data(self, text_files: List[str], output_file: str) -> str:
        """
        Prepare and clean Amharic text data for vocabulary training
        
        Args:
            text_files: List of paths to Amharic text files
            output_file: Path to output cleaned text file
        
        Returns:
            Path to cleaned text file
        """
        self.logger.info("Preparing Amharic text data...")
        
        cleaned_texts = []
        total_lines = 0
        valid_lines = 0
        
        for text_file in text_files:
            self.logger.info(f"Processing {text_file}")
            
            if not os.path.exists(text_file):
                self.logger.warning(f"File not found: {text_file}")
                continue
            
            with open(text_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    total_lines += 1
                    line = line.strip()
                    
                    # Skip empty lines
                    if not line:
                        continue
                    
                    # Basic validation for Amharic text
                    # Check if line contains Amharic characters or reasonable text
                    if len(line) < 3:  # Too short
                        continue
                    
                    # Check for reasonable Amharic text patterns
                    if not re.search(r'[ሀ-፿]', line) and not re.search(r'[a-zA-Z]', line):
                        # No Amharic or Latin characters - likely invalid
                        continue
                    
                    # Normalize the text
                    normalized_line = self.normalizer.normalize(line)
                    
                    if normalized_line:
                        cleaned_texts.append(normalized_line)
                        valid_lines += 1
                    
                    # Progress logging
                    if line_num % 1000 == 0:
                        self.logger.info(f"Processed {line_num} lines from {text_file}")
        
        self.logger.info(f"Text preparation complete: {valid_lines}/{total_lines} valid lines")
        
        # Write cleaned texts to file
        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            for text in cleaned_texts:
                f.write(text + '\n')
        
        self.logger.info(f"Cleaned text saved to: {output_path}")
        return str(output_path)
    
    def analyze_text_statistics(self, text_file: str) -> Dict:
        """Analyze text statistics for vocabulary optimization"""
        self.logger.info(f"Analyzing text statistics for {text_file}")
        
        char_count = 0
        word_count = 0
        line_count = 0
        unique_chars = set()
        amharic_chars = set()
        latin_chars = set()
        
        with open(text_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                line_count += 1
                char_count += len(line)
                word_count += len(line.split())
                
                # Character analysis
                for char in line:
                    unique_chars.add(char)
                    if re.match(r'[ሀ-፿]', char):  # Amharic range
                        amharic_chars.add(char)
                    elif re.match(r'[a-zA-Z]', char):  # Latin
                        latin_chars.add(char)
        
        stats = {
            'total_lines': line_count,
            'total_characters': char_count,
            'total_words': word_count,
            'unique_characters': len(unique_chars),
            'amharic_characters': len(amharic_chars),
            'latin_characters': len(latin_chars),
            'avg_line_length': char_count / line_count if line_count > 0 else 0,
            'avg_word_length': char_count / word_count if word_count > 0 else 0
        }
        
        self.logger.info(f"Text Statistics: {json.dumps(stats, indent=2)}")
        return stats
    
    def suggest_vocabulary_size(self, stats: Dict) -> int:
        """Suggest optimal vocabulary size based on text statistics"""
        # Base vocabulary size on text size and character diversity
        
        base_size = 2000  # Minimum for basic coverage
        
        # Adjust based on text size
        if stats['total_words'] < 10000:
            vocab_size = base_size
        elif stats['total_words'] < 50000:
            vocab_size = base_size + 2000
        elif stats['total_words'] < 100000:
            vocab_size = base_size + 4000
        elif stats['total_words'] < 500000:
            vocab_size = base_size + 6000
        else:
            vocab_size = base_size + 8000
        
        # Adjust based on character diversity
        if stats['amharic_characters'] > 300:
            vocab_size += 1000  # Rich Amharic character set
        if stats['latin_characters'] > 100:
            vocab_size += 500  # Mixed language content
        
        # Cap vocabulary size for efficiency
        vocab_size = min(vocab_size, 12000)
        
        suggested_size = vocab_size
        self.logger.info(f"Suggested vocabulary size: {suggested_size}")
        return suggested_size
    
    def train_vocabulary(
        self,
        text_file: str,
        vocab_size: int = None,
        model_prefix: str = "amharic_bpe",
        character_coverage: float = 0.9999
    ) -> str:
        """
        Train SentencePiece vocabulary model
        
        Args:
            text_file: Path to prepared text file
            vocab_size: Vocabulary size (auto-determined if None)
            model_prefix: Prefix for output model
            character_coverage: Character coverage ratio
        
        Returns:
            Path to trained model file
        """
        self.logger.info("Starting vocabulary training...")
        
        # Analyze text and suggest vocabulary size if not provided
        if vocab_size is None:
            stats = self.analyze_text_statistics(text_file)
            vocab_size = self.suggest_vocabulary_size(stats)
        
        # Train the model
        model_path = create_amharic_sentencepiece_model(
            text_file=text_file,
            vocab_size=vocab_size,
            model_prefix=str(self.output_dir / model_prefix),
            character_coverage=character_coverage
        )
        
        # Copy model to output directory if needed
        if not model_path.startswith(str(self.output_dir)):
            import shutil
            final_model_path = self.output_dir / Path(model_path).name
            shutil.copy2(model_path, final_model_path)
            model_path = str(final_model_path)
        
        self.logger.info(f"Vocabulary model trained: {model_path}")
        
        # Save training configuration
        config = {
            'vocab_size': vocab_size,
            'character_coverage': character_coverage,
            'model_prefix': model_prefix,
            'model_path': model_path,
            'training_date': str(Path().cwd())
        }
        
        config_path = self.output_dir / f"{model_prefix}_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Training configuration saved: {config_path}")
        return model_path
    
    def evaluate_vocabulary(self, model_path: str, test_texts: List[str]) -> Dict:
        """Evaluate vocabulary coverage and quality"""
        self.logger.info("Evaluating vocabulary quality...")
        
        # Load the trained model
        sp_model = sentencepiece.SentencePieceProcessor()
        sp_model.load(model_path)
        
        total_tokens = 0
        unknown_tokens = 0
        vocab_size = sp_model.get_piece_size()
        
        results = {
            'vocab_size': vocab_size,
            'evaluation_samples': len(test_texts),
            'total_tokens_processed': 0,
            'unknown_tokens': 0,
            'coverage_rate': 0.0,
            'avg_tokens_per_text': 0.0
        }
        
        for text in test_texts:
            if not text.strip():
                continue
            
            # Normalize text
            normalized_text = self.normalizer.normalize(text)
            
            # Tokenize
            tokens = sp_model.encode(normalized_text, out_type=str)
            total_tokens += len(tokens)
            
            # Count unknown tokens (these would be tokenized as single chars)
            for token in tokens:
                if len(token) == 1 and not re.match(r'[ሀ-፿a-zA-Z]', token):
                    unknown_tokens += 1
        
        results['total_tokens_processed'] = total_tokens
        results['unknown_tokens'] = unknown_tokens
        results['coverage_rate'] = (total_tokens - unknown_tokens) / total_tokens if total_tokens > 0 else 0
        results['avg_tokens_per_text'] = total_tokens / len(test_texts) if test_texts else 0
        
        self.logger.info(f"Vocabulary evaluation results: {json.dumps(results, indent=2)}")
        return results


def main():
    parser = argparse.ArgumentParser(description="Train Amharic vocabulary for IndexTTS2")
    parser.add_argument("--text_files", nargs='+', required=True, 
                       help="Path(s) to Amharic text files")
    parser.add_argument("--output_dir", type=str, default="amharic_models",
                       help="Output directory for vocabulary models")
    parser.add_argument("--vocab_size", type=int, default=None,
                       help="Vocabulary size (auto-determined if not specified)")
    parser.add_argument("--model_prefix", type=str, default="amharic_bpe",
                       help="Prefix for vocabulary model files")
    parser.add_argument("--character_coverage", type=float, default=0.9999,
                       help="Character coverage ratio")
    parser.add_argument("--test_texts", nargs='+', default=[],
                       help="Test texts for vocabulary evaluation")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = AmharicVocabularyTrainer(output_dir=args.output_dir)
    
    # Prepare text data
    cleaned_text_file = trainer.prepare_text_data(
        args.text_files, 
        "amharic_cleaned_texts.txt"
    )
    
    # Train vocabulary
    model_path = trainer.train_vocabulary(
        text_file=cleaned_text_file,
        vocab_size=args.vocab_size,
        model_prefix=args.model_prefix,
        character_coverage=args.character_coverage
    )
    
    # Evaluate vocabulary if test texts provided
    if args.test_texts:
        evaluation_results = trainer.evaluate_vocabulary(model_path, args.test_texts)
        
        # Save evaluation results
        eval_path = Path(args.output_dir) / "vocabulary_evaluation.json"
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        
        print(f"Vocabulary evaluation saved to: {eval_path}")
    
    print(f"Amharic vocabulary training complete!")
    print(f"Model path: {model_path}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()