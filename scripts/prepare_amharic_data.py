"""
Amharic dataset preparation script for IndexTTS2 fine-tuning
Generates training manifests from Amharic audio-text pairs
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random
import hashlib
import re

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from indextts.utils.amharic_front import AmharicTextNormalizer


class AmharicDatasetPreparer:
    """Prepare Amharic dataset for IndexTTS2 fine-tuning"""
    
    def __init__(
        self,
        audio_dir: str,
        text_dir: str,
        output_dir: str,
        sample_rate: int = 24000,
        min_duration: float = 1.0,
        max_duration: float = 30.0,
        min_text_length: int = 5,
        max_text_length: int = 500
    ):
        self.audio_dir = Path(audio_dir)
        self.text_dir = Path(text_dir)
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self.normalizer = AmharicTextNormalizer()
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logging()
        
        # Audio format configurations
        self.supported_audio_formats = {'.wav', '.flac', '.m4a', '.mp3', '.ogg'}
        self.supported_text_formats = {'.txt', '.json', '.lrc'}
    
    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'dataset_preparation.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def find_audio_text_pairs(self) -> List[Dict]:
        """Find matching audio-text file pairs"""
        self.logger.info("Finding audio-text file pairs...")
        
        pairs = []
        audio_files = []
        text_files = []
        
        # Collect audio files
        for ext in self.supported_audio_formats:
            audio_files.extend(self.audio_dir.rglob(f"*{ext}"))
            audio_files.extend(self.audio_dir.rglob(f"*{ext.upper()}"))
        
        # Collect text files
        for ext in self.supported_text_formats:
            text_files.extend(self.text_dir.rglob(f"*{ext}"))
            text_files.extend(self.text_dir.rglob(f"*{ext.upper()}"))
        
        self.logger.info(f"Found {len(audio_files)} audio files and {len(text_files)} text files")
        
        # Try to match files by filename stem
        audio_stems = {f.stem.lower(): f for f in audio_files}
        text_stems = {f.stem.lower(): f for f in text_files}
        
        matched_pairs = []
        for stem, audio_file in audio_stems.items():
            if stem in text_stems:
                text_file = text_stems[stem]
                matched_pairs.append({
                    'audio_file': str(audio_file),
                    'text_file': str(text_file),
                    'base_name': stem
                })
        
        self.logger.info(f"Matched {len(matched_pairs)} audio-text pairs")
        
        # If no matches found, try alternative matching strategies
        if not matched_pairs:
            self.logger.warning("No direct matches found, trying alternative strategies...")
            
            # Try directory-based matching
            audio_dirs = {f.parent: [] for f in audio_files}
            for f in audio_files:
                audio_dirs[f.parent].append(f)
            
            for text_file in text_files:
                # Look for audio in same directory or parent directory
                search_dirs = [text_file.parent, text_file.parent.parent]
                for search_dir in search_dirs:
                    if search_dir in audio_dirs:
                        # Take first audio file in directory
                        audio_files_in_dir = audio_dirs[search_dir]
                        if audio_files_in_dir:
                            audio_file = audio_files_in_dir[0]  # Take first available
                            matched_pairs.append({
                                'audio_file': str(audio_file),
                                'text_file': str(text_file),
                                'base_name': text_file.stem.lower()
                            })
                            break
        
        return matched_pairs
    
    def load_text_content(self, text_file: Path) -> str:
        """Load and process text content from various formats"""
        try:
            if text_file.suffix.lower() == '.json':
                with open(text_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        # Try common keys for text content
                        for key in ['text', 'transcript', 'content', 'sentence', 'utterance']:
                            if key in data and isinstance(data[key], str):
                                return data[key]
                        # If no common key, use the first string value
                        for value in data.values():
                            if isinstance(value, str) and value.strip():
                                return value
                    elif isinstance(data, list) and data:
                        # First string in list
                        for item in data:
                            if isinstance(item, str) and item.strip():
                                return item
                return ""
            
            elif text_file.suffix.lower() == '.lrc':
                with open(text_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    # Extract text from LRC format (skip time stamps)
                    text_lines = []
                    for line in lines:
                        line = line.strip()
                        # Skip lines with time stamps like [00:12.34]
                        if not re.match(r'^\[\d+:\d+\.\d+\]', line):
                            if line:
                                text_lines.append(line)
                    return ' '.join(text_lines)
            
            else:  # .txt files
                with open(text_file, 'r', encoding='utf-8') as f:
                    return f.read().strip()
                    
        except Exception as e:
            self.logger.error(f"Error loading text file {text_file}: {e}")
            return ""
    
    def validate_audio_file(self, audio_file: Path) -> Tuple[bool, float, str]:
        """Validate audio file and get its duration"""
        try:
            # Check if torch is available for audio loading
            try:
                import torch
                import torchaudio
            except ImportError:
                self.logger.warning("torch/torchaudio not available, skipping audio validation")
                return True, 5.0, "Audio validation skipped (torch not available)"
            
            # Load audio file
            audio, sr = torchaudio.load(str(audio_file))
            
            # Convert to mono if needed
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                audio = resampler(audio)
                sr = self.sample_rate
            
            # Calculate duration
            duration = audio.shape[1] / sr
            
            # Validate duration
            if duration < self.min_duration:
                return False, duration, f"Too short: {duration:.2f}s < {self.min_duration}s"
            
            if duration > self.max_duration:
                return False, duration, f"Too long: {duration:.2f}s > {self.max_duration}s"
            
            return True, duration, "OK"
            
        except Exception as e:
            return False, 0.0, f"Error loading audio: {e}"
    
    def process_dataset_pairs(self, pairs: List[Dict]) -> List[Dict]:
        """Process audio-text pairs and create dataset entries"""
        self.logger.info("Processing audio-text pairs...")
        
        valid_entries = []
        invalid_count = 0
        
        for i, pair in enumerate(pairs):
            try:
                # Load text content
                text_content = self.load_text_content(Path(pair['text_file']))
                
                if not text_content or len(text_content.strip()) < self.min_text_length:
                    self.logger.debug(f"Skipping {pair['base_name']}: Insufficient text")
                    invalid_count += 1
                    continue
                
                # Normalize text
                normalized_text = self.normalizer.normalize(text_content)
                
                if len(normalized_text) > self.max_text_length:
                    # Truncate long text
                    normalized_text = normalized_text[:self.max_text_length]
                    self.logger.debug(f"Truncated text for {pair['base_name']}")
                
                # Validate audio file
                audio_valid, duration, reason = self.validate_audio_file(Path(pair['audio_file']))
                
                if not audio_valid:
                    self.logger.debug(f"Skipping {pair['base_name']}: {reason}")
                    invalid_count += 1
                    continue
                
                # Create unique ID
                file_id = hashlib.md5(pair['audio_file'].encode()).hexdigest()[:16]
                
                # Create dataset entry
                entry = {
                    'id': file_id,
                    'audio_path': pair['audio_file'],
                    'text': normalized_text,
                    'original_text': text_content,
                    'duration': duration,
                    'text_length': len(normalized_text),
                    'base_name': pair['base_name'],
                    'sample_rate': self.sample_rate
                }
                
                valid_entries.append(entry)
                
            except Exception as e:
                self.logger.error(f"Error processing pair {pair}: {e}")
                invalid_count += 1
                continue
        
        self.logger.info(f"Dataset processing complete: {len(valid_entries)} valid, {invalid_count} invalid")
        return valid_entries
    
    def split_dataset(self, entries: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict:
        """Split dataset into train/val/test sets"""
        self.logger.info("Splitting dataset...")
        
        # Shuffle entries
        random.shuffle(entries)
        
        total = len(entries)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        test_size = total - train_size - val_size
        
        train_entries = entries[:train_size]
        val_entries = entries[train_size:train_size + val_size]
        test_entries = entries[train_size + val_size:]
        
        splits = {
            'train': train_entries,
            'val': val_entries,
            'test': test_entries
        }
        
        self.logger.info(f"Dataset split: {len(train_entries)} train, {len(val_entries)} val, {len(test_entries)} test")
        return splits
    
    def save_manifests(self, splits: Dict) -> Dict[str, str]:
        """Save manifest files for each split"""
        self.logger.info("Saving manifest files...")
        
        manifest_paths = {}
        
        for split_name, entries in splits.items():
            manifest_path = self.output_dir / f"{split_name}_manifest.jsonl"
            
            with open(manifest_path, 'w', encoding='utf-8') as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            manifest_paths[split_name] = str(manifest_path)
            self.logger.info(f"Saved {split_name} manifest: {manifest_path}")
        
        return manifest_paths
    
    def generate_statistics(self, splits: Dict) -> Dict:
        """Generate dataset statistics"""
        self.logger.info("Generating dataset statistics...")
        
        stats = {}
        
        for split_name, entries in splits.items():
            if not entries:
                stats[split_name] = {'count': 0}
                continue
            
            durations = [entry['duration'] for entry in entries]
            text_lengths = [entry['text_length'] for entry in entries]
            
            split_stats = {
                'count': len(entries),
                'total_duration': sum(durations),
                'avg_duration': sum(durations) / len(durations),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'avg_text_length': sum(text_lengths) / len(text_lengths),
                'min_text_length': min(text_lengths),
                'max_text_length': max(text_lengths)
            }
            
            stats[split_name] = split_stats
        
        # Overall statistics
        all_entries = []
        for entries in splits.values():
            all_entries.extend(entries)
        
        if all_entries:
            all_durations = [entry['duration'] for entry in all_entries]
            all_text_lengths = [entry['text_length'] for entry in all_entries]
            
            stats['overall'] = {
                'total_samples': len(all_entries),
                'total_duration': sum(all_durations),
                'avg_duration': sum(all_durations) / len(all_durations),
                'avg_text_length': sum(all_text_lengths) / len(all_text_lengths),
                'unique_audio_files': len(set(entry['audio_path'] for entry in all_entries)),
                'unique_text_files': len(set(entry['base_name'] for entry in all_entries))
            }
        
        return stats
    
    def prepare_dataset(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        random_seed: int = 42
    ) -> Dict[str, str]:
        """Main dataset preparation pipeline"""
        self.logger.info("Starting Amharic dataset preparation...")
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        # Find audio-text pairs
        pairs = self.find_audio_text_pairs()
        
        if not pairs:
            raise ValueError("No audio-text pairs found!")
        
        # Process pairs
        valid_entries = self.process_dataset_pairs(pairs)
        
        if not valid_entries:
            raise ValueError("No valid audio-text pairs found after processing!")
        
        # Split dataset
        splits = self.split_dataset(valid_entries, train_ratio, val_ratio)
        
        # Save manifests
        manifest_paths = self.save_manifests(splits)
        
        # Generate and save statistics
        stats = self.generate_statistics(splits)
        stats_path = self.output_dir / "dataset_statistics.json"
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Dataset statistics saved: {stats_path}")
        
        # Save preparation summary
        summary = {
            'preparation_date': str(Path().cwd()),
            'audio_dir': str(self.audio_dir),
            'text_dir': str(self.text_dir),
            'output_dir': str(self.output_dir),
            'parameters': {
                'sample_rate': self.sample_rate,
                'min_duration': self.min_duration,
                'max_duration': self.max_duration,
                'min_text_length': self.min_text_length,
                'max_text_length': self.max_text_length,
                'train_ratio': train_ratio,
                'val_ratio': val_ratio,
                'random_seed': random_seed
            },
            'statistics': stats,
            'manifest_paths': manifest_paths
        }
        
        summary_path = self.output_dir / "preparation_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Preparation summary saved: {summary_path}")
        self.logger.info("Amharic dataset preparation complete!")
        
        return manifest_paths


def main():
    parser = argparse.ArgumentParser(description="Prepare Amharic dataset for IndexTTS2")
    parser.add_argument("--audio_dir", type=str, required=True,
                       help="Directory containing audio files")
    parser.add_argument("--text_dir", type=str, required=True,
                       help="Directory containing text files")
    parser.add_argument("--output_dir", type=str, default="amharic_dataset",
                       help="Output directory for prepared dataset")
    parser.add_argument("--sample_rate", type=int, default=24000,
                       help="Target sample rate for audio files")
    parser.add_argument("--min_duration", type=float, default=1.0,
                       help="Minimum audio duration in seconds")
    parser.add_argument("--max_duration", type=float, default=30.0,
                       help="Maximum audio duration in seconds")
    parser.add_argument("--min_text_length", type=int, default=5,
                       help="Minimum text length in characters")
    parser.add_argument("--max_text_length", type=int, default=500,
                       help="Maximum text length in characters")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                       help="Validation set ratio")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Initialize preparer
    preparer = AmharicDatasetPreparer(
        audio_dir=args.audio_dir,
        text_dir=args.text_dir,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        min_text_length=args.min_text_length,
        max_text_length=args.max_text_length
    )
    
    # Prepare dataset
    manifest_paths = preparer.prepare_dataset(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_seed=args.random_seed
    )
    
    print("Amharic dataset preparation complete!")
    print("Manifest files:")
    for split, path in manifest_paths.items():
        print(f"  {split}: {path}")


if __name__ == "__main__":
    main()