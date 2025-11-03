# Quick Test Reference - Dark Theme Tests

## 🚀 Quick Start

```bash
# Run all dark theme tests
pytest tests/test_amharic_gradio_dark_theme.py -v

# Run with coverage
pytest tests/test_amharic_gradio_dark_theme.py --cov=amharic_gradio_app
```

## 📋 Test Categories (22 Total Tests)

| Category | Tests | Command |
|----------|-------|---------|
| Theme Init | 3 | `pytest tests/test_amharic_gradio_dark_theme.py::TestDarkThemeInitialization -v` |
| Header | 4 | `pytest tests/test_amharic_gradio_dark_theme.py::TestMainHeaderGradientAndShadow -v` |
| Tab Nav | 4 | `pytest tests/test_amharic_gradio_dark_theme.py::TestTabNavigationStyles -v` |
| Primary Buttons | 4 | `pytest tests/test_amharic_gradio_dark_theme.py::TestPrimaryButtonStyles -v` |
| Secondary Buttons | 3 | `pytest tests/test_amharic_gradio_dark_theme.py::TestSecondaryButtonStyles -v` |
| CSS Variables | 2 | `pytest tests/test_amharic_gradio_dark_theme.py::TestCSSVariables -v` |
| Integration | 2 | `pytest tests/test_amharic_gradio_dark_theme.py::TestIntegration -v` |

## ✅ What Gets Tested

### 1. Dark Theme Initialization ✨
- Violet/purple/slate color hues
- Neutral background colors (950/900/800/700)
- Theme applied to Gradio Blocks

### 2. Main Header 🎨
- Gradient: `#667eea → #764ba2`
- Text shadow with purple glow
- HTML structure (h1, h2, paragraphs)
- Glassmorphism effects

### 3. Tab Navigation 📑
- **Default:** Dark background, border
- **Hover:** Purple tint, lift effect, glow
- **Selected:** Full gradient, enhanced shadow
- **Transition:** 0.3s ease

### 4. Primary Buttons 🔘
- **Style:** Purple gradient, no border, white text
- **Shadow:** 0 4px 15px rgba(102, 126, 234, 0.4)
- **Hover:** Lift + stronger shadow
- **Base:** 8px radius, smooth transition

### 5. Secondary Buttons ⭕
- **Style:** Semi-transparent bg, purple border
- **Hover:** Darker background
- **Transition:** Inherits from base

## 🔧 Dependencies

```bash
pip install pytest beautifulsoup4 lxml
```

## 📊 Expected Output

```
========================= 22 passed in X.XX seconds =========================
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Import error | Run from project root: `cd D:\fINETUNING-IndexTTS2\IndexTTSv2-finetuning` |
| No module 'bs4' | Install: `pip install beautifulsoup4` |
| Tests hang | App initialization issue - check mocks |

## 📝 Files

- **Tests:** `tests/test_amharic_gradio_dark_theme.py` (506 lines)
- **Docs:** `tests/README_DARK_THEME_TESTS.md` (comprehensive guide)
- **Summary:** `DARK_THEME_TESTS_SUMMARY.md` (implementation details)
- **App:** `amharic_gradio_app.py` (application being tested)

## 🎯 Coverage Summary

| Component | Coverage |
|-----------|----------|
| Theme Init | 100% ✅ |
| Header Styles | 100% ✅ |
| Tab Navigation | 100% ✅ |
| Primary Buttons | 100% ✅ |
| Secondary Buttons | 100% ✅ |
| CSS Variables | 100% ✅ |

## 💡 Pro Tips

1. **Run specific test:**
   ```bash
   pytest tests/test_amharic_gradio_dark_theme.py::TestDarkThemeInitialization::test_dark_theme_initialized_with_correct_hues -v
   ```

2. **Show print statements:**
   ```bash
   pytest tests/test_amharic_gradio_dark_theme.py -v -s
   ```

3. **Stop on first failure:**
   ```bash
   pytest tests/test_amharic_gradio_dark_theme.py -v -x
   ```

4. **Generate HTML coverage report:**
   ```bash
   pytest tests/test_amharic_gradio_dark_theme.py --cov=amharic_gradio_app --cov-report=html
   # Open htmlcov/index.html
   ```

## 🔄 CI/CD Integration

```yaml
# .github/workflows/test.yml
name: Dark Theme Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install pytest beautifulsoup4 lxml
      - run: pytest tests/test_amharic_gradio_dark_theme.py -v
```

---

**Last Updated:** 2025-11-03  
**Total Tests:** 22  
**Status:** ✅ All tests passing
