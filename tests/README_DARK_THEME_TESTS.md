# Dark Theme Unit Tests - Amharic IndexTTS2 Gradio App

## Overview

This document describes the comprehensive unit tests for the dark theme implementation in the Amharic IndexTTS2 Gradio application (`amharic_gradio_app.py`).

## Test File

`test_amharic_gradio_dark_theme.py`

## Test Coverage

### 1. Dark Theme Initialization (`TestDarkThemeInitialization`)

Tests that verify the Gradio application is properly initialized with the dark theme.

**Test Cases:**
- `test_dark_theme_initialized_with_correct_hues` - Verifies `gr.themes.Base` is called with:
  - `primary_hue="violet"`
  - `secondary_hue="purple"`
  - `neutral_hue="slate"`

- `test_dark_theme_sets_correct_background_colors` - Verifies theme configuration includes:
  - `body_background_fill="*neutral_950"`
  - `background_fill_primary="*neutral_900"`
  - `background_fill_secondary="*neutral_800"`
  - `border_color_primary="*neutral_700"`

- `test_dark_theme_applied_to_gradio_blocks` - Verifies the theme is passed to `gr.Blocks`

### 2. Main Header Gradient and Shadow (`TestMainHeaderGradientAndShadow`)

Tests that verify the main header element renders correctly with gradient and text effects.

**Test Cases:**
- `test_main_header_has_gradient_background` - Verifies header contains:
  - `linear-gradient(135deg, #667eea 0%, #764ba2 100%)`

- `test_main_header_has_text_shadow` - Verifies h1 contains:
  - `text-shadow: 0 2px 20px rgba(102, 126, 234, 0.5)`

- `test_main_header_contains_all_required_elements` - Verifies presence of:
  - `<h1>` element
  - `<h2>` element
  - Multiple `<p>` elements
  - `.main-header` div wrapper

- `test_css_contains_main_header_styles` - Verifies CSS contains:
  - `.main-header` class definition
  - Gradient background
  - Box shadow with purple glow
  - Backdrop filter blur effect

### 3. Tab Navigation Styles (`TestTabNavigationStyles`)

Tests that verify tab navigation buttons apply correct styles for different states.

**Test Cases:**
- `test_tab_nav_button_default_style` - Verifies default state:
  - `background: rgba(26, 31, 46, 0.6)`
  - `color: var(--text-primary)`
  - `border: 1px solid var(--border-color)`

- `test_tab_nav_button_hover_effect` - Verifies hover state:
  - `background: rgba(102, 126, 234, 0.2)`
  - `border-color: var(--accent-primary)`
  - `transform: translateY(-2px)`
  - `box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3)`

- `test_tab_nav_button_selected_state` - Verifies selected state:
  - `background: linear-gradient(135deg, #667eea 0%, #764ba2 100%)`
  - Enhanced box shadow

- `test_tab_nav_button_transition` - Verifies smooth transitions:
  - `transition: all 0.3s ease`

### 4. Primary Button Styles (`TestPrimaryButtonStyles`)

Tests that verify primary action buttons display the gradient style and effects.

**Test Cases:**
- `test_primary_button_gradient_background` - Verifies:
  - `background: linear-gradient(135deg, #667eea 0%, #764ba2 100%)`
  - `border: none`
  - `color: white`

- `test_primary_button_box_shadow` - Verifies:
  - `box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4)`

- `test_primary_button_hover_effect` - Verifies hover state:
  - `transform: translateY(-2px)`
  - `box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6)`

- `test_button_base_styles` - Verifies all buttons have:
  - `border-radius: 8px`
  - `transition: all 0.3s ease`

### 5. Secondary Button Styles (`TestSecondaryButtonStyles`)

Tests that verify secondary action buttons display the outline style and effects.

**Test Cases:**
- `test_secondary_button_outline_style` - Verifies:
  - `background: rgba(102, 126, 234, 0.2)`
  - `border: 1px solid var(--accent-primary)`
  - `color: var(--accent-primary)`

- `test_secondary_button_hover_effect` - Verifies hover state:
  - `background: rgba(102, 126, 234, 0.3)`

- `test_secondary_button_transition` - Verifies transition inheritance from base styles

### 6. CSS Variables (`TestCSSVariables`)

Tests that verify all CSS custom properties are properly defined.

**Test Cases:**
- `test_css_variables_defined` - Verifies all variables in `:root`:
  - `--primary-bg: #0f1419`
  - `--secondary-bg: #1a1f2e`
  - `--card-bg: #1e2533`
  - `--accent-primary: #667eea`
  - `--accent-secondary: #764ba2`
  - `--accent-success: #4CAF50`
  - `--accent-warning: #FFC107`
  - `--accent-error: #F44336`
  - `--text-primary: #e8eaed`
  - `--text-secondary: #9aa0a6`
  - `--border-color: #2d3748`

- `test_css_root_selector_exists` - Verifies `:root` selector presence

### 7. Integration Tests (`TestIntegration`)

Tests that verify the complete dark theme implementation works as a whole.

**Test Cases:**
- `test_create_interface_returns_gradio_app` - Verifies `create_interface()` returns valid instance

- `test_all_css_sections_present` - Verifies all major CSS sections exist:
  - `:root`
  - `.gradio-container`
  - `.main-header`
  - `.tab-nav button`
  - `.primary`
  - `.secondary`
  - `input, textarea, select`
  - `::-webkit-scrollbar`

## Running the Tests

### Run All Dark Theme Tests
```bash
pytest tests/test_amharic_gradio_dark_theme.py -v
```

### Run Specific Test Class
```bash
pytest tests/test_amharic_gradio_dark_theme.py::TestDarkThemeInitialization -v
pytest tests/test_amharic_gradio_dark_theme.py::TestMainHeaderGradientAndShadow -v
pytest tests/test_amharic_gradio_dark_theme.py::TestTabNavigationStyles -v
pytest tests/test_amharic_gradio_dark_theme.py::TestPrimaryButtonStyles -v
pytest tests/test_amharic_gradio_dark_theme.py::TestSecondaryButtonStyles -v
```

### Run Specific Test
```bash
pytest tests/test_amharic_gradio_dark_theme.py::TestDarkThemeInitialization::test_dark_theme_initialized_with_correct_hues -v
```

### Run with Coverage
```bash
pytest tests/test_amharic_gradio_dark_theme.py --cov=amharic_gradio_app --cov-report=html
```

## Dependencies

The tests require the following packages:
- `pytest` - Test framework
- `beautifulsoup4` - HTML parsing for header element tests
- `unittest.mock` - Mocking Gradio components (part of Python standard library)

Install dependencies:
```bash
pip install pytest beautifulsoup4
```

## Test Statistics

- **Total Test Classes:** 7
- **Total Test Cases:** 22
- **Coverage Areas:**
  - Theme initialization: 3 tests
  - Header styling: 4 tests
  - Tab navigation: 4 tests
  - Primary buttons: 4 tests
  - Secondary buttons: 3 tests
  - CSS variables: 2 tests
  - Integration: 2 tests

## Key Testing Patterns

### 1. Mocking Gradio Components
```python
@patch('amharic_gradio_app.gr.Blocks')
@patch('amharic_gradio_app.gr.themes.Base')
def test_example(mock_blocks, mock_theme):
    # Test implementation
```

### 2. CSS Content Verification
```python
css_content = mock_blocks.call_args[1]['css']
assert 'expected-css-rule' in css_content
```

### 3. HTML Element Verification
```python
soup = BeautifulSoup(header_html, 'html.parser')
assert soup.find('h1') is not None
```

## Success Criteria

All tests should pass, verifying:
1. ✅ Dark theme properly initialized with violet/purple/slate hues
2. ✅ Main header has gradient background and text shadow
3. ✅ Tab navigation buttons have hover and selected state styles
4. ✅ Primary buttons display gradient and lift on hover
5. ✅ Secondary buttons display outline style and hover effects
6. ✅ All CSS variables defined
7. ✅ Complete integration works correctly

## Troubleshooting

### Import Errors
If you see `ModuleNotFoundError: No module named 'amharic_gradio_app'`:
- Ensure you're running tests from the project root directory
- Verify `amharic_gradio_app.py` exists in the project root

### Mock-Related Failures
If mocks aren't working as expected:
- Check that patch targets match actual import paths
- Verify mock setup in setUp/fixture methods

### BeautifulSoup Errors
If HTML parsing fails:
- Install beautifulsoup4: `pip install beautifulsoup4`
- Verify HTML content is valid

## Related Files

- `amharic_gradio_app.py` - Main application file being tested
- `DARK_THEME_GUIDE.md` - Comprehensive guide to the dark theme design
- `test_webui.py` - Other WebUI tests

## Maintenance

When updating the dark theme:
1. Update CSS in `amharic_gradio_app.py`
2. Update corresponding tests in `test_amharic_gradio_dark_theme.py`
3. Update `DARK_THEME_GUIDE.md` documentation
4. Run tests to verify changes: `pytest tests/test_amharic_gradio_dark_theme.py -v`

## Author Notes

These tests use **mock-based testing** to avoid launching the full Gradio app, making tests:
- ✅ Fast (no UI rendering)
- ✅ Isolated (no external dependencies)
- ✅ Reliable (deterministic outcomes)
- ✅ CI/CD friendly (no display server needed)

The tests focus on **structural and style verification** rather than visual testing, ensuring:
- CSS rules are properly defined
- HTML structure is correct
- Theme configuration is applied
- Component styling follows design specifications
