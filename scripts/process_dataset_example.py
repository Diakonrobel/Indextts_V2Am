#!/usr/bin/env python3
"""Example script for comprehensive dataset processing"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from indextts.utils.dataset_processor import ComprehensiveDatasetProcessor


def example_srt_vtt_processing():
    """Example: Process media file with SRT/VTT subtitles"""
    print("=" * 60)
    print("Example 1: SRT/VTT + Media Processing")
    print("=" * 60)
    
    processor = ComprehensiveDatasetProcessor(
        output_dir="output/processed_dataset",
        sample_rate=24000,
        min_snr_db=20.0,
        validate_quality=True
    )
    
    # Process video with subtitles
    samples, stats = processor.process_from_srt_vtt(
        media_path="path/to/video.mp4",
        subtitle_path="path/to/subtitles.srt",
        dataset_name="my_dataset"
    )
    
    print(f"\n✅ Processed {stats.processed} samples")
    print(f"📊 Total duration: {stats.total_duration:.1f}s")
    print(f"⭐ Average quality: {stats.avg_quality_score:.1f}/10")
    print(f"📈 Pass rate: {stats.pass_rate:.1%}")


def example_web_url_processing():
    """Example: Process web URLs"""
    print("\n" + "=" * 60)
    print("Example 2: Web URL Processing")
    print("=" * 60)
    
    processor = ComprehensiveDatasetProcessor(
        output_dir="output/web_dataset",
        sample_rate=24000,
        validate_quality=True
    )
    
    # Process YouTube videos
    urls = [
        'https://www.youtube.com/watch?v=example1',
        'https://www.youtube.com/watch?v=example2',
    ]
    
    samples, stats = processor.process_from_urls(
        urls=urls,
        dataset_name="youtube_dataset",
        max_workers=4,
        extract_subtitles=True
    )
    
    print(f"\n✅ Processed {stats.processed} samples from {len(urls)} URLs")
    print(f"📊 Total duration: {stats.total_duration:.1f}s")
    print(f"⭐ Average quality: {stats.avg_quality_score:.1f}/10")


def example_dataset_splitting():
    """Example: Split dataset into train/val/test"""
    print("\n" + "=" * 60)
    print("Example 3: Dataset Splitting")
    print("=" * 60)
    
    processor = ComprehensiveDatasetProcessor(
        output_dir="output/processed_dataset"
    )
    
    # Split dataset
    split_paths = processor.split_dataset(
        manifest_path="output/processed_dataset/manifests/my_dataset_manifest.jsonl",
        train_ratio=0.8,
        val_ratio=0.1,
        random_seed=42
    )
    
    print("\n📁 Split manifests created:")
    for split_name, path in split_paths.items():
        print(f"  {split_name}: {path}")


if __name__ == "__main__":
    print("\n🎙️ IndexTTS2 Dataset Processing Examples\n")
    
    # Uncomment the example you want to run:
    
    # example_srt_vtt_processing()
    # example_web_url_processing()
    # example_dataset_splitting()
    
    print("\n" + "=" * 60)
    print("To use these examples:")
    print("1. Uncomment one of the example functions above")
    print("2. Update the file paths/URLs to match your data")
    print("3. Run: python scripts/process_dataset_example.py")
    print("=" * 60)
