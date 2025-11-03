#!/usr/bin/env python
"""Amharic TTS Inference for IndexTTS2

Provides Amharic text-to-speech generation using fine-tuned models.
"""
import sys
import re
import argparse
from pathlib import Path
import warnings

sys.path.append(str(Path(__file__).parent.parent))

from indextts.infer import IndexTTS
from indextts.utils.amharic_front import AmharicTextTokenizer, AmharicTextNormalizer


class AmharicTTS(IndexTTS):
    """Amharic-specific TTS inference"""
    
    def __init__(self, amharic_vocab_path: str, **kwargs):
        """
        Initialize Amharic TTS
        
        Args:
            amharic_vocab_path: Path to Amharic SentencePiece model
            **kwargs: Arguments for base IndexTTS class
        """
        # Initialize base class
        super().__init__(**kwargs)
        
        # Override with Amharic tokenizer
        self.normalizer = AmharicTextNormalizer()
        self.amharic_tokenizer = AmharicTextTokenizer(
            vocab_file=amharic_vocab_path,
            normalizer=self.normalizer
        )
        
        # Keep original for comparison
        self.original_tokenizer = self.tokenizer
        
        # Use Amharic tokenizer
        self.tokenizer = self.amharic_tokenizer
        
        print(f"✅ Loaded Amharic tokenizer: {amharic_vocab_path}")
        print(f"   Vocabulary size: {self.tokenizer.vocab_size}")
    

    
    def infer(self, audio_prompt: str, text: str, output_path: str = None, 
              verbose: bool = False, **generation_kwargs):
        """
        Generate Amharic speech from text
        
        Args:
            audio_prompt: Path to reference audio file
            text: Amharic text to synthesize
            output_path: Where to save generated audio
            verbose: Print detailed information
            **generation_kwargs: Generation parameters
        """
        # Validate Amharic text
        if not self._is_amharic_text(text):
            warnings.warn(
                f"Input text may not be Amharic (detected few Ethiopic characters). "
                f"Results may be poor.",
                UserWarning
            )
        
        if verbose:
            print(f"Original text: {text}")
            print(f"Text length: {len(text)} characters")
        
        # Use parent class inference with Amharic tokenizer
        return super().infer(
            audio_prompt=audio_prompt,
            text=text,
            output_path=output_path,
            verbose=verbose,
            **generation_kwargs
        )
    
    def _is_amharic_text(self, text: str) -> bool:
        """Check if text contains Amharic characters"""
        # Amharic Unicode range: U+1200 to U+137F
        amharic_chars = sum(1 for c in text if '\u1200' <= c <= '\u137F')
        total_chars = sum(1 for c in text if c.isalpha())
        
        if total_chars == 0:
            return False
        
        return (amharic_chars / total_chars) > 0.5


def main():
    parser = argparse.ArgumentParser(description="Amharic TTS Inference")
    parser.add_argument("--prompt_audio", required=True,
                       help="Reference audio file (WAV format)")
    parser.add_argument("--text", required=True,
                       help="Amharic text to synthesize")
    parser.add_argument("--output", required=True,
                       help="Output audio path")
    parser.add_argument("--amharic_vocab", required=True,
                       help="Path to Amharic SentencePiece model")
    parser.add_argument("--model_dir", default="checkpoints",
                       help="Model directory")
    parser.add_argument("--config", default=None,
                       help="Config file (default: model_dir/config.yaml)")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed information")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.8,
                       help="Nucleus sampling threshold")
    parser.add_argument("--max_mel_tokens", type=int, default=600,
                       help="Maximum mel tokens to generate")
    
    args = parser.parse_args()
    
    # Determine config path
    if args.config is None:
        args.config = str(Path(args.model_dir) / "config.yaml")
    
    # Initialize Amharic TTS
    print("Initializing Amharic TTS...")
    tts = AmharicTTS(
        amharic_vocab_path=args.amharic_vocab,
        cfg_path=args.config,
        model_dir=args.model_dir,
        use_fp16=True
    )
    
    # Generate speech
    print(f"Generating speech for: {args.text[:50]}...")
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
    
    return result


if __name__ == "__main__":
    main()
