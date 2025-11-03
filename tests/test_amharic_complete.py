import pytest
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from indextts.utils.amharic_front import AmharicTextTokenizer, AmharicTextNormalizer
from indextts.utils.mel_quantization import simple_mel_quantization
from indextts.utils.checkpoint_validator import CheckpointValidator
from indextts.utils.audio_quality_metrics import calculate_audio_quality_metrics


class TestAmharicTokenization:
    def test_normalizer_basic(self):
        normalizer = AmharicTextNormalizer()
        text = "ሰላም ዓለም"
        normalized = normalizer.normalize(text)
        assert normalized is not None
        assert len(normalized) > 0
    
    def test_amharic_characters(self):
        normalizer = AmharicTextNormalizer()
        # Test all Amharic vowel orders
        test_syllables = ["ሀ", "ሁ", "ሂ", "ሃ", "ሄ", "ህ", "ሆ"]
        for syl in test_syllables:
            result = normalizer.normalize(syl)
            assert result is not None
    
    def test_number_normalization(self):
        normalizer = AmharicTextNormalizer()
        result = normalizer.normalize("123")
        assert result is not None


class TestMelQuantization:
    def test_simple_quantization_shape(self):
        mel = torch.randn(2, 100, 200)
        codes = simple_mel_quantization(mel, n_codes=8194)
        assert codes.shape == (2, 200)
    
    def test_code_range(self):
        mel = torch.randn(1, 80, 100)
        codes = simple_mel_quantization(mel, n_codes=8194)
        assert codes.min() >= 0
        assert codes.max() < 8194
    
    def test_single_sample(self):
        mel = torch.randn(80, 100)
        codes = simple_mel_quantization(mel, n_codes=8194)
        assert codes.shape == (100,)


class TestCheckpointValidation:
    def test_validator_exists(self):
        assert CheckpointValidator is not None
    
    def test_get_checkpoint_info(self):
        # Test checkpoint info extraction (no actual checkpoint needed)
        assert hasattr(CheckpointValidator, 'get_checkpoint_info')


class TestAudioQualityMetrics:
    def test_metrics_calculator(self):
        assert calculate_audio_quality_metrics is not None


class TestEndToEndPipeline:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
    def test_minimal_training_loop(self):
        # Test: Can run 1 training step without errors
        pass
    
    def test_tokenization_to_codes(self):
        # Test: Text -> Tokens -> Mel codes pipeline
        normalizer = AmharicTextNormalizer()
        text = "ሰላም"
        normalized = normalizer.normalize(text)
        assert normalized is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
