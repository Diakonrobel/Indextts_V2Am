#!/usr/bin/env python
"""Prepare Amharic audio data by encoding to mel codes using DAC codec

This script processes raw audio files and generates discrete mel codes
that can be used directly in training, avoiding on-the-fly quantization.
"""
import sys
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import torch

sys.path.append(str(Path(__file__).parent.parent))

from indextts.s2mel.dac.model.base import DACFile
from audiotools import AudioSignal


def load_dac_model(model_path: str, device: str = 'cuda'):
    """Load DAC model for encoding"""
    # TODO: Implement proper DAC model loading
    # For now, return None and note that this needs implementation
    print("⚠️  DAC model loading not implemented yet")
    print("   This script shows the structure but needs:")
    print("   1. DAC model checkpoint")
    print("   2. Proper model.load() implementation")
    return None


def encode_audio_to_codes(audio_path: str, dac_model, output_dir: Path):
    """Encode single audio file to discrete codes
    
    Args:
        audio_path: Path to audio file
        dac_model: Loaded DAC model
        output_dir: Directory to save .dac files
        
    Returns:
        Path to saved .dac file or None if failed
    """
    try:
        # Load audio
        signal = AudioSignal.load_from_file_with_ffmpeg(audio_path)
        
        # Compress to codes using DAC
        if dac_model is not None:
            dac_file = dac_model.compress(
                signal,
                win_duration=1.0,
                normalize_db=-16,
                verbose=False
            )
            
            # Save .dac file
            audio_name = Path(audio_path).stem
            output_path = output_dir / f"{audio_name}.dac"
            dac_file.save(output_path)
            
            return output_path
        else:
            print(f"Skipping {audio_path} - no DAC model")
            return None
            
    except Exception as e:
        print(f"Error encoding {audio_path}: {e}")
        return None


def process_manifest(
    manifest_file: str,
    output_dir: str,
    dac_model_path: str = None,
    device: str = 'cuda'
):
    """Process all audio files in manifest and generate mel codes
    
    Args:
        manifest_file: Input manifest with audio_path and text
        output_dir: Output directory for .dac files and updated manifest
        dac_model_path: Path to DAC model checkpoint
        device: Device for encoding
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load DAC model
    dac_model = load_dac_model(dac_model_path, device) if dac_model_path else None
    
    # Load manifest
    with open(manifest_file, 'r', encoding='utf-8') as f:
        entries = [json.loads(line) for line in f]
    
    print(f"Processing {len(entries)} audio files...")
    
    # Process each entry
    updated_entries = []
    for entry in tqdm(entries):
        audio_path = entry['audio_path']
        
        # Encode to codes
        dac_path = encode_audio_to_codes(audio_path, dac_model, output_dir)
        
        # Update entry
        if dac_path:
            entry['mel_codes_path'] = str(dac_path)
            entry['has_mel_codes'] = True
        else:
            entry['has_mel_codes'] = False
        
        updated_entries.append(entry)
    
    # Save updated manifest
    output_manifest = output_dir / 'manifest_with_codes.jsonl'
    with open(output_manifest, 'w', encoding='utf-8') as f:
        for entry in updated_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\nProcessing complete!")
    print(f"Updated manifest saved to: {output_manifest}")
    print(f"Encoded files: {sum(1 for e in updated_entries if e['has_mel_codes'])}")
    
    if dac_model is None:
        print("\n⚠️  WARNING: No DAC model provided")
        print("   To actually encode files, provide --dac_model path")
        print("   For now, training scripts will use random codes")


def main():
    parser = argparse.ArgumentParser(description="Prepare mel codes for Amharic training")
    parser.add_argument("--manifest", type=str, required=True,
                       help="Input manifest file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for codes")
    parser.add_argument("--dac_model", type=str,
                       help="Path to DAC model checkpoint (optional)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device for encoding")
    
    args = parser.parse_args()
    
    process_manifest(
        manifest_file=args.manifest,
        output_dir=args.output_dir,
        dac_model_path=args.dac_model,
        device=args.device
    )


if __name__ == "__main__":
    main()
