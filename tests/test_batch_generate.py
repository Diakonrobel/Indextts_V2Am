import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

sys.path.append(str(Path(__file__).parent.parent))

from amharic_gradio_app import AmharicTTSGradioApp


class TestBatchGenerate:
    """Test cases for the batch_generate function"""
    
    @pytest.fixture
    def tts_instance(self):
        """Create a TTS instance for testing"""
        return AmharicTTSGradioApp()
    
    def test_batch_generate_no_model_loaded(self, tts_instance):
        """Test that batch_generate returns a single error string when no model is loaded"""
        # Ensure no model is loaded
        tts_instance.current_model = None
        tts_instance.current_tokenizer = None
        
        # Call batch_generate with valid texts
        result = tts_instance.batch_generate(
            texts="Hello\nWorld",
            voice_id="default",
            emotion="neutral",
            speed=1.0,
            sample_rate=22050
        )
        
        # Assert single error string is returned
        assert isinstance(result, str)
        assert "❌" in result
        assert "load a model first" in result.lower()
    
    def test_batch_generate_no_texts_provided(self, tts_instance):
        """Test that batch_generate returns a single error string when no texts are provided"""
        # Mock a loaded model
        tts_instance.current_model = Mock()
        tts_instance.current_tokenizer = Mock()
        
        # Test with None
        result = tts_instance.batch_generate(
            texts=None,
            voice_id="default",
            emotion="neutral",
            speed=1.0,
            sample_rate=22050
        )
        
        # Assert single error string is returned
        assert isinstance(result, str)
        assert "❌" in result
        assert "provide texts" in result.lower()
        
        # Test with empty string
        result_empty = tts_instance.batch_generate(
            texts="",
            voice_id="default",
            emotion="neutral",
            speed=1.0,
            sample_rate=22050
        )
        
        assert isinstance(result_empty, str)
        assert "❌" in result_empty
        assert "provide texts" in result_empty.lower()
    
    def test_batch_generate_successful(self, tts_instance):
        """Test that batch_generate returns success message and audio paths when successful"""
        # Mock a loaded model and tokenizer
        tts_instance.current_model = Mock()
        tts_instance.current_tokenizer = Mock()
        
        # Mock the generate_speech method to return successful results
        mock_audio_path_1 = "/path/to/audio1.wav"
        mock_audio_path_2 = "/path/to/audio2.wav"
        
        def mock_generate_speech(text, voice_id, emotion, speed, pitch, temp, tokens, sr):
            if "first" in text.lower():
                return mock_audio_path_1, "✅ Generated successfully"
            elif "second" in text.lower():
                return mock_audio_path_2, "✅ Generated successfully"
            return None, "Error"
        
        tts_instance.generate_speech = Mock(side_effect=mock_generate_speech)
        
        # Call batch_generate with multiple texts
        result = tts_instance.batch_generate(
            texts="First text\nSecond text",
            voice_id="default",
            emotion="neutral",
            speed=1.0,
            sample_rate=22050
        )
        
        # Assert result is a single string
        assert isinstance(result, str)
        
        # Assert success message is included
        assert "✅" in result
        assert "Generated" in result
        assert "2" in result  # Number of files generated
        
        # Assert audio paths are included in the result
        assert mock_audio_path_1 in result
        assert mock_audio_path_2 in result
        
        # Assert format matches expected pattern
        assert "Audio 1:" in result
        assert "Audio 2:" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
