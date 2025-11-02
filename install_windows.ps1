# ========================================================================
# Windows Installation Script for IndexTTS2 Amharic Fine-tuning
# ========================================================================

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "IndexTTS2 Amharic - Windows Setup" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
try {
    \ = python --version 2>&1
    Write-Host "Python version: \" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python not found. Please install Python 3.10+" -ForegroundColor Red
    exit 1
}

# Check if Python version is 3.10+
\ = python -c "import sys; print(sys.version_info >= (3, 10))"
if (\ -ne "True") {
    Write-Host "ERROR: Python 3.10+ required" -ForegroundColor Red
    exit 1
}

# Upgrade pip
Write-Host ""
Write-Host "Upgrading pip, setuptools, and wheel..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
Write-Host ""
Write-Host "Installing PyTorch with CUDA 12.1..." -ForegroundColor Yellow
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121

# Install core requirements
Write-Host ""
Write-Host "Installing core requirements..." -ForegroundColor Yellow
pip install -r requirements.txt

# Verify installation
Write-Host ""
Write-Host "Verifying installation..." -ForegroundColor Yellow
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
python -c "import torchaudio; print(f'TorchAudio: {torchaudio.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import librosa; print(f'Librosa: {librosa.__version__}')"
python -c "import gradio; print(f'Gradio: {gradio.__version__}')"

Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Prepare data: python scripts/prepare_amharic_data.py --audio_dir <path> --text_dir <path> --output_dir <path>"
Write-Host "2. Train vocabulary: python scripts/train_amharic_vocabulary.py --text_files <path> --output_dir <path>"
Write-Host "3. Start training: python train_amharic_full.py --data_dir <path> --vocab <vocab_file>"
Write-Host ""
