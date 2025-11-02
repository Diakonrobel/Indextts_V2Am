"""
Amharic-specific text processing for IndexTTS2
Modern Amharic script (ፊደል) support with comprehensive linguistic handling
"""
import re
import unicodedata
from typing import List, Dict, Optional
import sentencepiece as spm
from .front import TextTokenizer


class AmharicTextNormalizer:
    """Amharic text normalizer with modern script (ፊደል) support"""
    
    def __init__(self):
        # Amharic number words (modern Amharic)
        self.number_words = {
            # Basic numbers 0-20
            "ትር": "0", "ዜሮ": "0", "ዜሮን": "0",
            "አንድ": "1", "ሰባት": "1",
            "ሁለት": "2", "ክልተ": "2", "ኪልት": "2", 
            "ሶስት": "3", "ሶስተኛ": "3",
            "አራት": "4", "አራተኛ": "4",
            "አምስት": "5", "አምስተኛ": "5",
            "ስድስት": "6", "ስድስተኛ": "6",
            "ሰባት": "7", "ሰባተኛ": "7", "ሰባብ": "7",
            "ስምንት": "8", "ስምንተኛ": "8",
            "ዘጠኝ": "9", "ዘጠነኛ": "9",
            "አስር": "10", "አስረኛ": "10",
            "አስከፍለት": "11", "አስከፈለት": "11",
            "አስራሁለት": "12", "ሰደስት": "12",
            "አስራሶስት": "13", "አስራአራት": "14", "አስራአምስት": "15",
            "አስራስድስት": "16", "አስራሰባት": "17", "አስራስምንት": "18", "አስራዘጠኝ": "19",
            "የአስር": "20", "ዐስር": "20",
            
            # Tens
            "ሰላሳ": "30", "አርባ": "40", "አምሳ": "50", 
            "ስድሳ": "60", "ሰባታ": "70", "ሰማንያ": "80", "ዘጠና": "90",
            
            # Hundreds
            "አንድ መቶ": "100", "ሁለት መቶ": "200", "ሶስት መቶ": "300",
            "አራት መቶ": "400", "አምስት መቶ": "500",
            
            # Large numbers
            "ሽቅብ": "1000", "ሚሊዮን": "1000000", "ቢሊዮን": "1000000000"
        }
        
        # Ordinal numbers
        self.ordinal_words = {
            "መጀመሪያ": "1.", "የመጀመሪያ": "1.", "አንደኛ": "1.", "የአንደኛ": "1.",
            "ሁለተኛ": "2.", "የሁለተኛ": "2.", "ክልተ": "2.", "የክልት": "2.",
            "ሶስተኛ": "3.", "የሶስተኛ": "3.",
            "አራተኛ": "4.", "የአራተኛ": "4.",
            "አምስተኛ": "5.", "የአምስተኛ": "5.",
            "ስድስተኛ": "6.", "የስድስተኛ": "6.",
            "ሰባተኛ": "7.", "የሰባተኛ": "7.",
            "ስምንተኛ": "8.", "የስምንተኛ": "8.",
            "ዘጠነኛ": "9.", "የዘጠነኛ": "9.",
            "አስረኛ": "10.", "የአስረኛ": "10."
        }
        
        # Common Amharic abbreviations
        self.abbreviations = {
            "ም.ም.": "ምሳሌ ምሳሌ",
            "ም.ክ.": "ምክትል ክብረ ትልቅ አበቃ",
            "ዶ/ር": "ዶክተር",
            "ፕ/ር": "ፕሮፌሰር",
            "ዶ/ር": "ዶክተር",
            "ፀሃይ": "ፀሃይ",
            "ሐረገ": "ሐረገ",
            "ሐውለት": "ሐውለት",
            "ሐረጋት": "ሐረጋት",
            "መ.ስፈር": "መስፈር",
            "ም.የልህት": "ምክትል የልህት ትምህርት",
            "ት.ም.ሐ.ወ": "ትምህርት ምክትል ሐረገ ወንጌል",
            "ስራ": "ስራ",
            "የስራ": "የስራ"
        }
        
        # Amharic contractions and abbreviations in modern usage
        self.contractions = {
            "ከሆነ": "ከ ሆነ",
            "እንደሆነ": "እንደ ሆነ",
            "እንዲሁም": "እንዲሁ እንዲሁ",
            "ምክንያቱም": "ምክንያቱ ምክንያቱ",
            "ከላይ": "ከ ላይ",
            "ከታች": "ከ ታች",
            "ከሰሜን": "ከ ሰሜን",
            "ከደቡብ": "ከ ደቡብ",
            "ከምስራቅ": "ከ ምስራቅ",
            "ከምዕራብ": "ከ ምዕራብ"
        }
    
    def normalize_numbers(self, text: str) -> str:
        """Convert Amharic number words to digits"""
        words = text.split()
        result = []
        
        for word in words:
            word_lower = word.strip()
            if word_lower in self.number_words:
                result.append(self.number_words[word_lower])
            elif word_lower in self.ordinal_words:
                result.append(self.ordinal_words[word_lower])
            else:
                result.append(word)
        
        return " ".join(result)
    
    def expand_contractions(self, text: str) -> str:
        """Expand Amharic contractions"""
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        return text
    
    def expand_abbreviations(self, text: str) -> str:
        """Expand common Amharic abbreviations"""
        for abbr, expansion in self.abbreviations.items():
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', expansion, text)
        return text
    
    def normalize_punctuation(self, text: str) -> str:
        """Normalize Amharic punctuation"""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Normalize various quote styles to standard quotes
        text = re.sub(r'[""„"]', '"', text)
        text = re.sub(r'[""‚"]', "'", text)
        # Normalize dashes
        text = re.sub(r'[–—]', '-', text)
        # Normalize ellipsis
        text = re.sub(r'\.{3,}', '...', text)
        return text.strip()
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters for Amharic"""
        # Normalize to NFC form (Canonical Composition) for Amharic
        text = unicodedata.normalize('NFC', text)
        return text
    
    def handle_amharic_script(self, text: str) -> str:
        """Handle modern Amharic script (ፊደል) specific processing"""
        # Ensure proper handling of Amharic syllabic structure
        # Modern Amharic uses syllabic writing where each character represents a consonant+vowel combination
        
        # Clean up common script issues
        # Handle cases where characters might be improperly combined or separated
        
        # Keep Amharic characters as is - don't transliterate to Latin
        # This maintains the proper script for modern Amharic TTS
        
        return text
    
    def normalize(self, text: str) -> str:
        """Main normalization pipeline for Amharic text"""
        if not text or not text.strip():
            return ""
        
        # Unicode normalization for Amharic
        text = self.normalize_unicode(text)
        
        # Handle Amharic script specifics
        text = self.handle_amharic_script(text)
        
        # Expand contractions
        text = self.expand_contractions(text)
        
        # Expand abbreviations
        text = self.expand_abbreviations(text)
        
        # Normalize numbers
        text = self.normalize_numbers(text)
        
        # Normalize punctuation
        text = self.normalize_punctuation(text)
        
        return text


class AmharicTextTokenizer(TextTokenizer):
    """Amharic-specific text tokenizer extending the base TextTokenizer"""
    
    def __init__(self, vocab_file: str, normalizer: Optional[AmharicTextNormalizer] = None):
        super().__init__(vocab_file, normalizer)
        if normalizer is None:
            self.normalizer = AmharicTextNormalizer()
    
    def preprocess_amharic_text(self, text: str) -> str:
        """Amharic-specific text preprocessing"""
        # Modern Amharic script (ፊደል) preservation
        # Keep the original script intact for proper pronunciation
        
        # Handle whitespace normalization for Amharic
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Ensure proper sentence boundaries
        text = re.sub(r'\s+([.!?])', r'\1', text)
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Override tokenize to include Amharic preprocessing"""
        if not text or not text.strip():
            return []
        
        # Apply Amharic-specific preprocessing
        text = self.preprocess_amharic_text(text)
        
        # Apply normalization
        if self.normalizer:
            text = self.normalizer.normalize(text)
        
        # Apply pre-tokenizers (skip CJK char tokenization for Amharic)
        # Amharic uses a different script system
        
        # Tokenize with SentencePiece
        return self.sp_model.Encode(text, out_type=str)
    
    def encode(self, text: str, **kwargs):
        """Override encode to include Amharic preprocessing"""
        if not text or not text.strip():
            return []
        
        # Apply Amharic-specific preprocessing
        text = self.preprocess_amharic_text(text)
        
        # Apply normalization
        if self.normalizer:
            text = self.normalizer.normalize(text)
        
        return self.sp_model.Encode(text, out_type=kwargs.pop("out_type", int), **kwargs)


def create_amharic_sentencepiece_model(
    text_file: str,
    vocab_size: int = 8000,  # Smaller vocabulary for Amharic
    model_prefix: str = "amharic_bpe",
    character_coverage: float = 0.9999  # Higher coverage for Amharic script
) -> str:
    """
    Create an Amharic SentencePiece model from text data
    
    Args:
        text_file: Path to text file with Amharic sentences
        vocab_size: Vocabulary size for the model
        model_prefix: Prefix for output model files
        character_coverage: Character coverage for the model
    
    Returns:
        Path to the created model file
    """
    import sentencepiece as spm
    
    # SentencePiece training parameters optimized for Amharic
    spm.SentencePieceTrainer.train(
        input=text_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type='bpe',
        max_sentence_length=8192,
        shuffle_input_sentence=True,
        input_sentence_size=500000,  # Smaller dataset for Amharic
        seed_sentencepiece_size=500000,
        shrinking_factor=0.75,
        num_threads=16,
        num_sub_iterations=2,
        max_sentencepiece_length=16,
        split_by_unicode_script=True,  # Important for Amharic script
        split_by_whitespace=True,
        split_by_number=True,
        treat_whitespace_as_suffix=False,
        allow_whitespace_only_pieces=False,
        split_digits=True,
        vocabulary_output_piece_score=True,
        hard_vocab_limit=True,
        use_all_vocab=False,
        byte_fallback=True,
        vocabulary_threshold=10,  # Lower threshold for Amharic
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=-1,
        unk_piece='<unk>',
        bos_piece='<s>',
        eos_piece='</s>',
        pad_piece='<pad>',
        unk_surface='<unk>',
        train_extremely_large_corpus=False,
        enable_differential_privacy=False,
        differential_privacy_noise_level=0.0,
        differential_privacy_clipping_threshold=0.0,
        user_defined_symbols=['<speaker>', '<emotion>', '<duration>', '<amharic>']
    )
    
    return f"{model_prefix}.model"


if __name__ == "__main__":
    # Example usage
    normalizer = AmharicTextNormalizer()
    test_text = "ሰላም ዓለም! ይህ ሙከራ ነው። 1+1=2 ነው።"
    normalized = normalizer.normalize(test_text)
    print(f"Original: {test_text}")
    print(f"Normalized: {normalized}")
    
    # Test vocabulary creation (requires Amharic text file)
    # model_path = create_amharic_sentencepiece_model("amharic_texts.txt")
    # print(f"Created model: {model_path}")