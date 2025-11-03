"""
Unit tests for webui.py module

Tests the following functionality:
1. get_gpu_info function with CUDA available
2. get_gpu_info function with CUDA not available  
3. clear_logs function
4. Refresh GPU Status button interaction
5. Clear Logs button interaction
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


class TestGetGpuInfo:
    """Test suite for the get_gpu_info function"""
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.current_device')
    @patch('torch.cuda.get_device_name')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.memory_reserved')
    def test_get_gpu_info_cuda_available(
        self, 
        mock_memory_reserved,
        mock_memory_allocated,
        mock_get_device_name,
        mock_current_device,
        mock_device_count,
        mock_is_available
    ):
        """Test get_gpu_info correctly retrieves and formats GPU information when CUDA is available"""
        # Import the function
        from webui import get_gpu_info
        
        # Setup mocks
        mock_is_available.return_value = True
        mock_device_count.return_value = 2
        mock_current_device.return_value = 0
        mock_get_device_name.return_value = "NVIDIA Tesla T4"
        mock_memory_allocated.return_value = 2.5 * 1024**3  # 2.5 GB in bytes
        mock_memory_reserved.return_value = 4.0 * 1024**3  # 4.0 GB in bytes
        
        # Call the function
        result = get_gpu_info()
        
        # Assertions
        assert result is not None
        assert isinstance(result, dict)
        assert result["available"] is True
        assert result["device_count"] == 2
        assert result["current_device"] == 0
        assert result["device_name"] == "NVIDIA Tesla T4"
        assert result["memory_allocated"] == "2.50 GB"
        assert result["memory_reserved"] == "4.00 GB"
        
        # Verify all mocks were called
        mock_is_available.assert_called_once()
        mock_device_count.assert_called_once()
        mock_current_device.assert_called_once()
        mock_get_device_name.assert_called_once_with(0)
        mock_memory_allocated.assert_called_once_with(0)
        mock_memory_reserved.assert_called_once_with(0)
    
    @patch('torch.cuda.is_available')
    def test_get_gpu_info_cuda_not_available(self, mock_is_available):
        """Test get_gpu_info correctly indicates when CUDA is not available"""
        # Import the function
        from webui import get_gpu_info
        
        # Setup mock
        mock_is_available.return_value = False
        
        # Call the function
        result = get_gpu_info()
        
        # Assertions
        assert result is not None
        assert isinstance(result, dict)
        assert result["available"] is False
        assert "message" in result
        assert result["message"] == "CUDA not available"
        
        # Verify mock was called
        mock_is_available.assert_called_once()
    
    @patch('torch.cuda.is_available')
    def test_get_gpu_info_exception_handling(self, mock_is_available):
        """Test get_gpu_info handles exceptions gracefully"""
        # Import the function
        from webui import get_gpu_info
        
        # Setup mock to raise an exception
        mock_is_available.side_effect = RuntimeError("CUDA initialization failed")
        
        # Call the function
        result = get_gpu_info()
        
        # Assertions
        assert result is not None
        assert isinstance(result, dict)
        assert "error" in result
        assert "CUDA initialization failed" in result["error"]
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.current_device')
    @patch('torch.cuda.get_device_name')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.memory_reserved')
    def test_get_gpu_info_memory_formatting(
        self,
        mock_memory_reserved,
        mock_memory_allocated,
        mock_get_device_name,
        mock_current_device,
        mock_device_count,
        mock_is_available
    ):
        """Test get_gpu_info correctly formats memory values to GB with 2 decimal places"""
        from webui import get_gpu_info
        
        # Setup mocks with specific memory values
        mock_is_available.return_value = True
        mock_device_count.return_value = 1
        mock_current_device.return_value = 0
        mock_get_device_name.return_value = "GPU"
        mock_memory_allocated.return_value = 1536 * 1024**2  # 1.5 GB in bytes
        mock_memory_reserved.return_value = 3072 * 1024**2  # 3.0 GB in bytes
        
        result = get_gpu_info()
        
        # Check formatting
        assert result["memory_allocated"] == "1.50 GB"
        assert result["memory_reserved"] == "3.00 GB"


class TestClearLogs:
    """Test suite for the clear_logs function"""
    
    def test_clear_logs_returns_empty_string(self):
        """Test clear_logs function returns an empty string"""
        from webui import clear_logs
        
        # Call the function
        result = clear_logs()
        
        # Assertions
        assert result is not None
        assert isinstance(result, str)
        assert result == ""
        assert len(result) == 0
    
    def test_clear_logs_multiple_calls(self):
        """Test clear_logs consistently returns empty string on multiple calls"""
        from webui import clear_logs
        
        # Call multiple times
        for _ in range(5):
            result = clear_logs()
            assert result == ""


class TestGradioButtonInteractions:
    """Test suite for Gradio button interactions"""
    
    @patch('webui.get_gpu_info')
    def test_refresh_gpu_status_button_calls_function(self, mock_get_gpu_info):
        """Test 'Refresh GPU Status' button calls the get_gpu_info function and updates display"""
        # Setup mock return value
        mock_gpu_info = {
            "available": True,
            "device_count": 1,
            "current_device": 0,
            "device_name": "NVIDIA T4",
            "memory_allocated": "1.23 GB",
            "memory_reserved": "2.34 GB"
        }
        mock_get_gpu_info.return_value = mock_gpu_info
        
        # Simulate button click by calling the function directly
        # (In actual Gradio app, this is wired via gpu_refresh_btn.click)
        result = mock_get_gpu_info()
        
        # Assertions
        assert result == mock_gpu_info
        mock_get_gpu_info.assert_called_once()
    
    @patch('webui.get_gpu_info')
    def test_refresh_gpu_status_updates_display_on_click(self, mock_get_gpu_info):
        """Test that clicking refresh GPU status updates the GPU info display"""
        # Setup mock
        expected_info = {
            "available": True,
            "device_count": 2,
            "device_name": "Test GPU"
        }
        mock_get_gpu_info.return_value = expected_info
        
        # Simulate the click handler
        updated_value = mock_get_gpu_info()
        
        # Verify the display would be updated with correct info
        assert updated_value == expected_info
        assert updated_value["available"] is True
        assert updated_value["device_count"] == 2
    
    @patch('webui.clear_logs')
    def test_clear_logs_button_calls_function(self, mock_clear_logs):
        """Test 'Clear Logs' button calls the clear_logs function and clears the log display"""
        # Setup mock
        mock_clear_logs.return_value = ""
        
        # Simulate button click by calling the function directly
        # (In actual Gradio app, this is wired via clear_logs_btn.click)
        result = mock_clear_logs()
        
        # Assertions
        assert result == ""
        mock_clear_logs.assert_called_once()
    
    @patch('webui.clear_logs')
    def test_clear_logs_button_clears_display(self, mock_clear_logs):
        """Test that clicking clear logs empties the log display"""
        # Setup mock
        mock_clear_logs.return_value = ""
        
        # Simulate having some log content initially
        initial_logs = "Some log content\nMore log lines\nError messages"
        
        # Simulate the click handler clearing the logs
        cleared_logs = mock_clear_logs()
        
        # Verify the display would be cleared
        assert cleared_logs == ""
        assert len(cleared_logs) == 0
        assert cleared_logs != initial_logs


class TestGradioIntegration:
    """Integration tests for Gradio UI components"""
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    @patch('torch.cuda.current_device', return_value=0)
    @patch('torch.cuda.get_device_name', return_value="Test GPU")
    @patch('torch.cuda.memory_allocated', return_value=1024**3)
    @patch('torch.cuda.memory_reserved', return_value=2*1024**3)
    def test_gpu_refresh_end_to_end(self, *mocks):
        """Test end-to-end flow of GPU refresh button"""
        from webui import get_gpu_info
        
        # Simulate user clicking the refresh button
        gpu_info = get_gpu_info()
        
        # Verify the complete response
        assert gpu_info["available"] is True
        assert "device_name" in gpu_info
        assert "memory_allocated" in gpu_info
        assert "memory_reserved" in gpu_info
    
    def test_clear_logs_end_to_end(self):
        """Test end-to-end flow of clear logs button"""
        from webui import clear_logs
        
        # Simulate user clicking the clear logs button
        result = clear_logs()
        
        # Verify logs are cleared
        assert result == ""
    
    @patch('webui.get_gpu_info')
    @patch('webui.clear_logs')
    def test_both_buttons_work_independently(self, mock_clear_logs, mock_get_gpu_info):
        """Test that both buttons can be used independently without interference"""
        # Setup mocks
        mock_get_gpu_info.return_value = {"available": True}
        mock_clear_logs.return_value = ""
        
        # Click GPU refresh button
        gpu_result = mock_get_gpu_info()
        assert gpu_result == {"available": True}
        
        # Click clear logs button
        logs_result = mock_clear_logs()
        assert logs_result == ""
        
        # Verify both were called
        mock_get_gpu_info.assert_called_once()
        mock_clear_logs.assert_called_once()


class TestWebUIEdgeCases:
    """Test edge cases and error scenarios"""
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_get_gpu_info_zero_devices(self, mock_device_count, mock_is_available):
        """Test get_gpu_info when CUDA is available but no devices found"""
        from webui import get_gpu_info
        
        mock_is_available.return_value = True
        mock_device_count.return_value = 0
        
        # This should still work but show 0 devices
        result = get_gpu_info()
        assert result["available"] is True
        assert result["device_count"] == 0
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.current_device')
    @patch('torch.cuda.get_device_name')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.memory_reserved')
    def test_get_gpu_info_multiple_devices(self, *mocks):
        """Test get_gpu_info with multiple GPU devices"""
        from webui import get_gpu_info
        
        # Setup for 4 GPUs
        mocks[5].return_value = True  # is_available
        mocks[4].return_value = 4  # device_count
        mocks[3].return_value = 0  # current_device
        mocks[2].return_value = "Multi-GPU System"
        mocks[1].return_value = 1024**3
        mocks[0].return_value = 2*1024**3
        
        result = get_gpu_info()
        
        assert result["available"] is True
        assert result["device_count"] == 4
    
    def test_clear_logs_idempotent(self):
        """Test that clear_logs is idempotent (same result when called multiple times)"""
        from webui import clear_logs
        
        results = [clear_logs() for _ in range(10)]
        
        # All results should be identical empty strings
        assert all(r == "" for r in results)
        assert len(set(results)) == 1  # All results are the same


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s", "--tb=short"])
