"""
Unit tests for Amharic IndexTTS2 Gradio App Dark Theme Components

Tests the following functionality:
1. Gradio application initialized with the new "dark_theme"
2. Main header element renders with gradient background and text shadow
3. Tab navigation buttons apply hover effects and selected states correctly
4. Primary action buttons display gradient style and hover effects
5. Secondary action buttons display outline style and hover effects
"""

import pytest
import sys
import re
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from bs4 import BeautifulSoup

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


class TestDarkThemeInitialization:
    """Test suite for dark theme initialization"""
    
    @patch('amharic_gradio_app.gr.themes.Base')
    def test_dark_theme_initialized_with_correct_hues(self, mock_base_theme):
        """Test that the Gradio application is initialized with the correct dark theme hues"""
        # Setup mock theme
        mock_theme_instance = MagicMock()
        mock_set_return = MagicMock()
        mock_theme_instance.set.return_value = mock_set_return
        mock_base_theme.return_value = mock_theme_instance
        
        # Import and create app
        from amharic_gradio_app import AmharicTTSGradioApp
        app = AmharicTTSGradioApp()
        
        # Create interface to trigger theme creation
        with patch('amharic_gradio_app.gr.Blocks'):
            app.create_interface()
        
        # Verify gr.themes.Base was called with correct hues
        mock_base_theme.assert_called_once_with(
            primary_hue="violet",
            secondary_hue="purple",
            neutral_hue="slate"
        )
    
    @patch('amharic_gradio_app.gr.themes.Base')
    def test_dark_theme_sets_correct_background_colors(self, mock_base_theme):
        """Test that dark theme sets the correct background colors"""
        # Setup mock theme
        mock_theme_instance = MagicMock()
        mock_set_return = MagicMock()
        mock_theme_instance.set.return_value = mock_set_return
        mock_base_theme.return_value = mock_theme_instance
        
        # Import and create app
        from amharic_gradio_app import AmharicTTSGradioApp
        app = AmharicTTSGradioApp()
        
        # Create interface
        with patch('amharic_gradio_app.gr.Blocks'):
            app.create_interface()
        
        # Verify set() was called with dark color settings
        mock_theme_instance.set.assert_called_once()
        call_kwargs = mock_theme_instance.set.call_args[1]
        
        assert call_kwargs['body_background_fill'] == "*neutral_950"
        assert call_kwargs['body_background_fill_dark'] == "*neutral_950"
        assert call_kwargs['background_fill_primary'] == "*neutral_900"
        assert call_kwargs['background_fill_primary_dark'] == "*neutral_900"
        assert call_kwargs['background_fill_secondary'] == "*neutral_800"
        assert call_kwargs['background_fill_secondary_dark'] == "*neutral_800"
        assert call_kwargs['border_color_primary'] == "*neutral_700"
        assert call_kwargs['border_color_primary_dark'] == "*neutral_700"
    
    @patch('amharic_gradio_app.gr.themes.Base')
    @patch('amharic_gradio_app.gr.Blocks')
    def test_dark_theme_applied_to_gradio_blocks(self, mock_blocks, mock_base_theme):
        """Test that dark theme is passed to Gradio Blocks"""
        # Setup mocks
        mock_theme_instance = MagicMock()
        mock_set_return = MagicMock()
        mock_theme_instance.set.return_value = mock_set_return
        mock_base_theme.return_value = mock_theme_instance
        
        mock_blocks_instance = MagicMock()
        mock_blocks.return_value.__enter__ = MagicMock(return_value=mock_blocks_instance)
        mock_blocks.return_value.__exit__ = MagicMock(return_value=False)
        
        # Import and create app
        from amharic_gradio_app import AmharicTTSGradioApp
        app = AmharicTTSGradioApp()
        app.create_interface()
        
        # Verify Blocks was called with the theme
        call_args = mock_blocks.call_args
        assert 'theme' in call_args[1]
        assert call_args[1]['theme'] == mock_set_return


class TestMainHeaderGradientAndShadow:
    """Test suite for main header element styling"""
    
    def test_main_header_has_gradient_background(self):
        """Test that main header has the specified gradient background"""
        from amharic_gradio_app import AmharicTTSGradioApp
        
        app = AmharicTTSGradioApp()
        
        # Get the CSS from the app
        with patch('amharic_gradio_app.gr.Blocks') as mock_blocks:
            with patch('amharic_gradio_app.gr.HTML') as mock_html:
                with patch('amharic_gradio_app.gr.Tabs'):
                    with patch('amharic_gradio_app.gr.themes.Base'):
                        app.create_interface()
                
                # Find the main header HTML call
                header_html = None
                for call in mock_html.call_args_list:
                    if call[0] and 'main-header' in call[0][0]:
                        header_html = call[0][0]
                        break
                
                assert header_html is not None, "Main header HTML not found"
                
                # Check for gradient background
                assert 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' in header_html
    
    def test_main_header_has_text_shadow(self):
        """Test that main header h1 has the specified text shadow"""
        from amharic_gradio_app import AmharicTTSGradioApp
        
        app = AmharicTTSGradioApp()
        
        with patch('amharic_gradio_app.gr.Blocks'):
            with patch('amharic_gradio_app.gr.HTML') as mock_html:
                with patch('amharic_gradio_app.gr.Tabs'):
                    with patch('amharic_gradio_app.gr.themes.Base'):
                        app.create_interface()
                
                # Find main header HTML
                header_html = None
                for call in mock_html.call_args_list:
                    if call[0] and 'main-header' in call[0][0]:
                        header_html = call[0][0]
                        break
                
                assert header_html is not None
                
                # Check for text shadow with correct rgba values
                assert 'text-shadow: 0 2px 20px rgba(102, 126, 234, 0.5)' in header_html
    
    def test_main_header_contains_all_required_elements(self):
        """Test that main header contains all required elements (h1, h2, p, features div)"""
        from amharic_gradio_app import AmharicTTSGradioApp
        
        app = AmharicTTSGradioApp()
        
        with patch('amharic_gradio_app.gr.Blocks'):
            with patch('amharic_gradio_app.gr.HTML') as mock_html:
                with patch('amharic_gradio_app.gr.Tabs'):
                    with patch('amharic_gradio_app.gr.themes.Base'):
                        app.create_interface()
                
                # Find main header HTML
                header_html = None
                for call in mock_html.call_args_list:
                    if call[0] and 'main-header' in call[0][0]:
                        header_html = call[0][0]
                        break
                
                assert header_html is not None
                
                # Parse HTML
                soup = BeautifulSoup(header_html, 'html.parser')
                
                # Check for required elements
                assert soup.find('h1') is not None, "h1 element not found"
                assert soup.find('h2') is not None, "h2 element not found"
                assert len(soup.find_all('p')) >= 2, "Not enough p elements found"
                assert soup.find('div', class_='main-header') is not None, "main-header div not found"
    
    def test_css_contains_main_header_styles(self):
        """Test that CSS contains proper styling for .main-header class"""
        from amharic_gradio_app import AmharicTTSGradioApp
        
        app = AmharicTTSGradioApp()
        
        with patch('amharic_gradio_app.gr.Blocks') as mock_blocks:
            with patch('amharic_gradio_app.gr.themes.Base'):
                app.create_interface()
            
            # Get CSS from Blocks call
            css_content = mock_blocks.call_args[1]['css']
            
            # Check for main-header class definition
            assert '.main-header' in css_content
            assert 'background: linear-gradient(135deg, #667eea 0%, #764ba2 100%)' in css_content
            assert 'box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3)' in css_content
            assert 'backdrop-filter: blur(10px)' in css_content


class TestTabNavigationStyles:
    """Test suite for tab navigation button styles"""
    
    def test_tab_nav_button_default_style(self):
        """Test that tab navigation buttons have correct default styling"""
        from amharic_gradio_app import AmharicTTSGradioApp
        
        app = AmharicTTSGradioApp()
        
        with patch('amharic_gradio_app.gr.Blocks') as mock_blocks:
            with patch('amharic_gradio_app.gr.themes.Base'):
                app.create_interface()
            
            css_content = mock_blocks.call_args[1]['css']
            
            # Check for tab-nav button styles
            assert '.tab-nav button' in css_content
            assert 'background: rgba(26, 31, 46, 0.6) !important' in css_content
            assert 'color: var(--text-primary) !important' in css_content
            assert 'border: 1px solid var(--border-color) !important' in css_content
    
    def test_tab_nav_button_hover_effect(self):
        """Test that tab navigation buttons have correct hover effects"""
        from amharic_gradio_app import AmharicTTSGradioApp
        
        app = AmharicTTSGradioApp()
        
        with patch('amharic_gradio_app.gr.Blocks') as mock_blocks:
            with patch('amharic_gradio_app.gr.themes.Base'):
                app.create_interface()
            
            css_content = mock_blocks.call_args[1]['css']
            
            # Check for hover state
            assert '.tab-nav button:hover' in css_content
            assert 'background: rgba(102, 126, 234, 0.2) !important' in css_content
            assert 'border-color: var(--accent-primary) !important' in css_content
            assert 'transform: translateY(-2px)' in css_content
            assert 'box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important' in css_content
    
    def test_tab_nav_button_selected_state(self):
        """Test that tab navigation buttons have correct selected state styling"""
        from amharic_gradio_app import AmharicTTSGradioApp
        
        app = AmharicTTSGradioApp()
        
        with patch('amharic_gradio_app.gr.Blocks') as mock_blocks:
            with patch('amharic_gradio_app.gr.themes.Base'):
                app.create_interface()
            
            css_content = mock_blocks.call_args[1]['css']
            
            # Check for selected state
            assert '.tab-nav button.selected' in css_content
            assert 'background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important' in css_content
            assert 'border-color: var(--accent-primary) !important' in css_content
            assert 'box-shadow: 0 4px 20px rgba(102, 126, 234, 0.5) !important' in css_content
    
    def test_tab_nav_button_transition(self):
        """Test that tab navigation buttons have smooth transitions"""
        from amharic_gradio_app import AmharicTTSGradioApp
        
        app = AmharicTTSGradioApp()
        
        with patch('amharic_gradio_app.gr.Blocks') as mock_blocks:
            with patch('amharic_gradio_app.gr.themes.Base'):
                app.create_interface()
            
            css_content = mock_blocks.call_args[1]['css']
            
            # Check for transition
            assert 'transition: all 0.3s ease !important' in css_content


class TestPrimaryButtonStyles:
    """Test suite for primary action button styles"""
    
    def test_primary_button_gradient_background(self):
        """Test that primary buttons have gradient background"""
        from amharic_gradio_app import AmharicTTSGradioApp
        
        app = AmharicTTSGradioApp()
        
        with patch('amharic_gradio_app.gr.Blocks') as mock_blocks:
            with patch('amharic_gradio_app.gr.themes.Base'):
                app.create_interface()
            
            css_content = mock_blocks.call_args[1]['css']
            
            # Check for primary button gradient
            assert '.primary' in css_content
            assert 'background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important' in css_content
            assert 'border: none !important' in css_content
            assert 'color: white !important' in css_content
    
    def test_primary_button_box_shadow(self):
        """Test that primary buttons have correct box shadow"""
        from amharic_gradio_app import AmharicTTSGradioApp
        
        app = AmharicTTSGradioApp()
        
        with patch('amharic_gradio_app.gr.Blocks') as mock_blocks:
            with patch('amharic_gradio_app.gr.themes.Base'):
                app.create_interface()
            
            css_content = mock_blocks.call_args[1]['css']
            
            # Check for box shadow
            assert 'box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important' in css_content
    
    def test_primary_button_hover_effect(self):
        """Test that primary buttons have correct hover effects"""
        from amharic_gradio_app import AmharicTTSGradioApp
        
        app = AmharicTTSGradioApp()
        
        with patch('amharic_gradio_app.gr.Blocks') as mock_blocks:
            with patch('amharic_gradio_app.gr.themes.Base'):
                app.create_interface()
            
            css_content = mock_blocks.call_args[1]['css']
            
            # Check for hover effect
            assert '.primary:hover' in css_content
            assert 'transform: translateY(-2px)' in css_content
            assert 'box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important' in css_content
    
    def test_button_base_styles(self):
        """Test that all buttons have base rounded corners and transitions"""
        from amharic_gradio_app import AmharicTTSGradioApp
        
        app = AmharicTTSGradioApp()
        
        with patch('amharic_gradio_app.gr.Blocks') as mock_blocks:
            with patch('amharic_gradio_app.gr.themes.Base'):
                app.create_interface()
            
            css_content = mock_blocks.call_args[1]['css']
            
            # Check for base button styles
            assert 'button {' in css_content
            assert 'border-radius: 8px !important' in css_content
            assert 'transition: all 0.3s ease !important' in css_content


class TestSecondaryButtonStyles:
    """Test suite for secondary action button styles"""
    
    def test_secondary_button_outline_style(self):
        """Test that secondary buttons have correct outline styling"""
        from amharic_gradio_app import AmharicTTSGradioApp
        
        app = AmharicTTSGradioApp()
        
        with patch('amharic_gradio_app.gr.Blocks') as mock_blocks:
            with patch('amharic_gradio_app.gr.themes.Base'):
                app.create_interface()
            
            css_content = mock_blocks.call_args[1]['css']
            
            # Check for secondary button styles
            assert '.secondary' in css_content
            assert 'background: rgba(102, 126, 234, 0.2) !important' in css_content
            assert 'border: 1px solid var(--accent-primary) !important' in css_content
            assert 'color: var(--accent-primary) !important' in css_content
    
    def test_secondary_button_hover_effect(self):
        """Test that secondary buttons have correct hover effects"""
        from amharic_gradio_app import AmharicTTSGradioApp
        
        app = AmharicTTSGradioApp()
        
        with patch('amharic_gradio_app.gr.Blocks') as mock_blocks:
            with patch('amharic_gradio_app.gr.themes.Base'):
                app.create_interface()
            
            css_content = mock_blocks.call_args[1]['css']
            
            # Check for secondary hover effect
            assert '.secondary:hover' in css_content
            assert 'background: rgba(102, 126, 234, 0.3) !important' in css_content
    
    def test_secondary_button_transition(self):
        """Test that secondary buttons inherit transition from base button styles"""
        from amharic_gradio_app import AmharicTTSGradioApp
        
        app = AmharicTTSGradioApp()
        
        with patch('amharic_gradio_app.gr.Blocks') as mock_blocks:
            with patch('amharic_gradio_app.gr.themes.Base'):
                app.create_interface()
            
            css_content = mock_blocks.call_args[1]['css']
            
            # Verify base button transition exists (inherited by secondary)
            assert 'button {' in css_content
            assert 'transition: all 0.3s ease !important' in css_content


class TestCSSVariables:
    """Test suite for CSS custom properties (variables)"""
    
    def test_css_variables_defined(self):
        """Test that all required CSS variables are defined"""
        from amharic_gradio_app import AmharicTTSGradioApp
        
        app = AmharicTTSGradioApp()
        
        with patch('amharic_gradio_app.gr.Blocks') as mock_blocks:
            with patch('amharic_gradio_app.gr.themes.Base'):
                app.create_interface()
            
            css_content = mock_blocks.call_args[1]['css']
            
            # Check for all CSS variables
            assert '--primary-bg: #0f1419' in css_content
            assert '--secondary-bg: #1a1f2e' in css_content
            assert '--card-bg: #1e2533' in css_content
            assert '--accent-primary: #667eea' in css_content
            assert '--accent-secondary: #764ba2' in css_content
            assert '--accent-success: #4CAF50' in css_content
            assert '--accent-warning: #FFC107' in css_content
            assert '--accent-error: #F44336' in css_content
            assert '--text-primary: #e8eaed' in css_content
            assert '--text-secondary: #9aa0a6' in css_content
            assert '--border-color: #2d3748' in css_content
    
    def test_css_root_selector_exists(self):
        """Test that :root selector is present for CSS variables"""
        from amharic_gradio_app import AmharicTTSGradioApp
        
        app = AmharicTTSGradioApp()
        
        with patch('amharic_gradio_app.gr.Blocks') as mock_blocks:
            with patch('amharic_gradio_app.gr.themes.Base'):
                app.create_interface()
            
            css_content = mock_blocks.call_args[1]['css']
            
            # Check for :root selector
            assert ':root {' in css_content


class TestIntegration:
    """Integration tests for complete dark theme implementation"""
    
    @patch('amharic_gradio_app.gr.Blocks')
    @patch('amharic_gradio_app.gr.themes.Base')
    def test_create_interface_returns_gradio_app(self, mock_theme, mock_blocks):
        """Test that create_interface returns a Gradio app instance"""
        from amharic_gradio_app import AmharicTTSGradioApp
        
        # Setup mocks
        mock_theme_instance = MagicMock()
        mock_set_return = MagicMock()
        mock_theme_instance.set.return_value = mock_set_return
        mock_theme.return_value = mock_theme_instance
        
        mock_app_instance = MagicMock()
        mock_blocks.return_value.__enter__ = MagicMock(return_value=mock_app_instance)
        mock_blocks.return_value.__exit__ = MagicMock(return_value=False)
        
        # Create app
        app = AmharicTTSGradioApp()
        interface = app.create_interface()
        
        # Verify interface is created
        assert interface is not None
    
    def test_all_css_sections_present(self):
        """Test that all major CSS sections are present in the stylesheet"""
        from amharic_gradio_app import AmharicTTSGradioApp
        
        app = AmharicTTSGradioApp()
        
        with patch('amharic_gradio_app.gr.Blocks') as mock_blocks:
            with patch('amharic_gradio_app.gr.themes.Base'):
                app.create_interface()
            
            css_content = mock_blocks.call_args[1]['css']
            
            # Check for all major sections
            required_sections = [
                ':root',
                '.gradio-container',
                '.main-header',
                '.tab-nav button',
                '.primary',
                '.secondary',
                'input, textarea, select',
                '::-webkit-scrollbar'
            ]
            
            for section in required_sections:
                assert section in css_content, f"CSS section '{section}' not found"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s", "--tb=short"])
