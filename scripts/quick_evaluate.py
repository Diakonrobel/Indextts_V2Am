#!/usr/bin/env python
"""Quick audio quality evaluation for Amharic TTS

Provides fast quality checks without requiring full evaluation suite.
"""
import argparse
import sys
from pathlib import Path
import torch
import torchaudio
import numpy as np


def quick_audio_quality_check(audio_path: str) -> bool:
    """Fast audio quality checks
    
    Returns:
        True if audio passes basic quality checks
    """
    try:
        audio, sr = torchaudio.load(audio_path)
    except Exception as e:
        print(f"❌ Cannot load audio: {e}")
        return False
    
    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    
    audio_np = audio.numpy().flatten()
    
    # Basic metrics
    duration = len(audio_np) / sr
    rms = float(np.sqrt(np.mean(audio_np ** 2)))
    peak = float(np.max(np.abs(audio_np)))
    
    # Zero crossing rate
    zcr = float(np.sum(np.diff(np.sign(audio_np)) != 0) / len(audio_np))
    
    # Quality flags
    flags = []
    passed = True
    
    if peak > 0.99:
        flags.append("⚠️  CLIPPING DETECTED (peak > 0.99)")
        passed = False
    
    if rms < 0.01:
        flags.append("⚠️  TOO QUIET (RMS < 0.01) - possibly silent")
        passed = False
    
    if duration < 0.5:
        flags.append("⚠️  VERY SHORT (<0.5s)")
    
    if duration > 30:
        flags.append("⚠️  VERY LONG (>30s)")
    
    if zcr > 0.5:
        flags.append("⚠️  HIGH NOISE (ZCR > 0.5)")
        passed = False
    
    # Print results
    print(f"\n📊 Audio Quality Metrics")
    print(f"{'='*40}")
    print(f"Duration:    {duration:.2f}s")
    print(f"RMS Energy:  {rms:.4f}")
    print(f"Peak Level:  {peak:.4f}")
    print(f"ZCR:         {zcr:.4f}")
    print(f"Sample Rate: {sr} Hz")
    
    if flags:
        print(f"\n⚠️  Quality Issues Detected:")
        for flag in flags:
            print(f"  {flag}")
    
    if passed:
        print(f"\n✅ Basic quality checks PASSED")
    else:
        print(f"\n❌ Basic quality checks FAILED")
    
    return passed


def quick_intelligibility_check(text_input: str, audio_path: str) -> bool:
    """Check if audio duration roughly matches expected duration
    
    Returns:
        True if duration is reasonable for text length
    """
    try:
        audio, sr = torchaudio.load(audio_path)
    except Exception as e:
        print(f"❌ Cannot load audio: {e}")
        return False
    
    duration = audio.shape[1] / sr
    
    # Rough estimate: Amharic ~10 characters/second
    # (varies by speaker, but good ballpark)
    expected_duration = len(text_input) / 10
    
    ratio = duration / expected_duration if expected_duration > 0 else 0
    
    print(f"\n📏 Duration Check")
    print(f"{'='*40}")
    print(f"Text length:      {len(text_input)} characters")
    print(f"Audio duration:   {duration:.1f}s")
    print(f"Expected (~10ch/s): {expected_duration:.1f}s")
    print(f"Ratio:            {ratio:.2f}x")
    
    # Reasonable range: 0.5x to 2.0x expected
    if 0.5 < ratio < 2.0:
        print(f"\n✅ Duration reasonable for text length")
        return True
    else:
        print(f"\n⚠️  Duration suspicious (too fast or too slow)")
        print(f"   This may indicate:")
        if ratio < 0.5:
            print(f"   - Very fast speech (>20 chars/sec)")
            print(f"   - Possible truncation")
        else:
            print(f"   - Very slow speech (<5 chars/sec)")
            print(f"   - Possible silence or padding")
        return False


def compare_with_reference(generated_path: str, reference_path: str):
    """Compare generated audio with reference
    
    Simple comparison of duration and energy
    """
    try:
        gen_audio, gen_sr = torchaudio.load(generated_path)
        ref_audio, ref_sr = torchaudio.load(reference_path)
    except Exception as e:
        print(f"❌ Cannot load audio files: {e}")
        return
    
    gen_duration = gen_audio.shape[1] / gen_sr
    ref_duration = ref_audio.shape[1] / ref_sr
    
    gen_rms = float(np.sqrt(np.mean(gen_audio.numpy() ** 2)))
    ref_rms = float(np.sqrt(np.mean(ref_audio.numpy() ** 2)))
    
    print(f"\n🔄 Comparison with Reference")
    print(f"{'='*40}")
    print(f"Generated duration: {gen_duration:.2f}s")
    print(f"Reference duration: {ref_duration:.2f}s")
    print(f"Duration ratio:     {gen_duration/ref_duration:.2f}x")
    print(f"\nGenerated RMS:      {gen_rms:.4f}")
    print(f"Reference RMS:      {ref_rms:.4f}")
    print(f"Energy ratio:       {gen_rms/ref_rms:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Quick audio quality evaluation")
    parser.add_argument("--audio", required=True,
                       help="Generated audio file to evaluate")
    parser.add_argument("--text", default=None,
                       help="Original text (for intelligibility check)")
    parser.add_argument("--reference", default=None,
                       help="Reference audio for comparison")
    
    args = parser.parse_args()
    
    if not Path(args.audio).exists():
        print(f"❌ Audio file not found: {args.audio}")
        return 1
    
    print(f"Evaluating: {args.audio}")
    
    # Run quality check
    quality_passed = quick_audio_quality_check(args.audio)
    
    # Run intelligibility check if text provided
    intel_passed = True
    if args.text:
        intel_passed = quick_intelligibility_check(args.text, args.audio)
    
    # Compare with reference if provided
    if args.reference:
        if Path(args.reference).exists():
            compare_with_reference(args.audio, args.reference)
        else:
            print(f"⚠️  Reference audio not found: {args.reference}")
    
    # Final verdict
    print(f"\n{'='*40}")
    if quality_passed and intel_passed:
        print(f"✅ Overall: PASSED")
        return 0
    else:
        print(f"⚠️  Overall: ISSUES DETECTED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
